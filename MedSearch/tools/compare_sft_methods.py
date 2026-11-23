"""Compare the eight saved CE/Top32 dev runs without regenerating answers."""
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import wandb
import yaml


def paired(first, second):
    gained = [a['id'] for a, b in zip(first, second) if a['verifier_acc'] and not b['verifier_acc']]
    lost = [a['id'] for a, b in zip(first, second) if b['verifier_acc'] and not a['verifier_acc']]
    n = len(gained) + len(lost)
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(len(gained), len(lost)) + 1)) / 2**n) if n else 1.
    return {'gained_ids': gained, 'lost_ids': lost, 'net_correct_change': len(gained) - len(lost),
            'descriptive_mcnemar_p': p}


def main():
    roots = {'top32': Path('MedSearch/outputs/distillation/sft_epoch_comparison_20261013'),
             'ce': Path('MedSearch/outputs/distillation/sft_ce_epoch_comparison_20261013')}
    output = Path('MedSearch/outputs/distillation/sft_loss_comparison_20261013')
    if output.exists():
        raise FileExistsError(output)
    expected = [str(r['id']) for r in pq.read_table('MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet').to_pylist()]
    configs = [yaml.safe_load(Path(f'MedSearch/configs/sft_{m}_epoch1.yaml').read_text()) for m in ['top32', 'ce']]
    for config in configs:
        config.pop('data_path')
        for key in ['output_dir', 'run_name']:
            config['training'].pop(key)
    assert configs[0] == configs[1]
    initial_reports = [json.loads(Path(f'MedSearch/outputs/distillation/sft_{m}_epoch1_20261013/run_report.json').read_text())
                       for m in ['top32', 'ce']]
    assert initial_reports[0]['source_sha256'] == initial_reports[1]['source_sha256']
    assert initial_reports[0]['git_commit'] == initial_reports[1]['git_commit']
    schedules = [json.loads((r / 'training/run_report.json').read_text())['schedule_extension'] for r in roots.values()]
    assert schedules[0] == schedules[1]
    ce_history = json.loads((roots['ce'] / 'training/run_report.json').read_text())['log_history']
    assert all(row['top32_loss'] == 0 for row in ce_history if 'top32_loss' in row)
    reference_manifest = json.loads((roots['top32'] / 'epoch1/rollout_manifest.json').read_text())
    reference_summary = json.loads((roots['top32'] / 'epoch1/summary.json').read_text())
    rows, scores, raw = [], {}, {}
    for method, root in roots.items():
        assert json.loads((root / 'pipeline_status.json').read_text())['status'] == 'complete'
        assert json.loads((root / 'selected_verification.json').read_text())['status'] == 'passed'
        comparison = json.loads((root / 'comparison.json').read_text())
        for summary in comparison['summaries']:
            epoch = summary['epoch']
            folder = root / f'epoch{epoch}'
            manifest = json.loads((folder / 'rollout_manifest.json').read_text())
            assert manifest['status'] == 'complete' and not manifest['retrieval_error_ids']
            for key in ['dev_sha256', 'model', 'generation_config', 'max_action_length', 'batch_size', 'stop', 'include_stop_str_in_output']:
                assert manifest[key] == reference_manifest[key], (method, epoch, key)
            for key in ['verifier_model', 'verifier_device', 'verifier_dtype', 'verifier_max_length', 'verifier_threshold', 'verifier_batch_size']:
                assert summary[key] == reference_summary[key], (method, epoch, key)
            for name, digest in [('rollouts.jsonl', summary['rollout_sha256']), ('scores.jsonl', summary['scores_sha256'])]:
                assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
            scores[method, epoch] = [json.loads(line) for line in (folder / 'scores.jsonl').read_text().splitlines()]
            raw[method, epoch] = [json.loads(line) for line in (folder / 'rollouts.jsonl').read_text().splitlines()]
            assert [r['id'] for r in scores[method, epoch]] == [r['id'] for r in raw[method, epoch]] == expected
            assert summary['total'] == 396
            assert summary['verifier_correct'] == sum(r['verifier_acc'] for r in scores[method, epoch])
            assert summary['rule_correct'] == sum(r['rule_acc'] for r in scores[method, epoch])
            rows.append({**summary, 'method': method})
    assert len(rows) == 8
    reference_raw = raw['top32', 1]
    for key, records in raw.items():
        assert [(r['question'], r['ground_truth'], r['prompt_ids']) for r in records] == [
            (r['question'], r['ground_truth'], r['prompt_ids']) for r in reference_raw], key
    # 主指标同分时，先选较早epoch，再优先不需要概率缓存的CE。
    best = max(rows, key=lambda r: (r['verifier_correct'], -r['epoch'], r['method'] == 'ce'))
    method, epoch = best['method'], best['epoch']
    adapter = (Path(f'MedSearch/outputs/distillation/sft_{method}_epoch1_20261013/adapter') if epoch == 1
               else roots[method] / 'training' / f'checkpoint-{125 * epoch}')
    selected = {'method': method, 'selected_epoch': epoch, 'base_model': 'models/Qwen2.5-3B-Instruct',
                'adapter_path': str(adapter.resolve()),
                'adapter_sha256': hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest(),
                'verifier_correct': best['verifier_correct'], 'dev_total': 396,
                'verifier_accuracy': best['verifier_accuracy'],
                'selection_rule': 'highest verifier correct; ties earlier epoch, then CE',
                'purpose': 'RL initialization candidate; RL not started', 'independent_test_evaluated': False,
                'schedule_caveat': 'both: epoch1 cosine to125; resume extended cosine to1000'}
    same_epoch = [{'epoch': ep, 'first': 'top32', 'second': 'ce',
                   **paired(scores['top32', ep], scores['ce', ep])} for ep in [1, 3, 5, 8]]
    family_best = {m: max((r for r in rows if r['method'] == m), key=lambda r: (r['verifier_correct'], -r['epoch']))
                   for m in roots}
    a, b = family_best['top32'], family_best['ce']
    best_pair = {'first': f"top32/epoch{a['epoch']}", 'second': f"ce/epoch{b['epoch']}",
                 **paired(scores['top32', a['epoch']], scores['ce', b['epoch']])}
    cases = []
    changed = set(best_pair['gained_ids'] + best_pair['lost_ids'])
    for left, right, ls, rs in zip(raw['top32', a['epoch']], raw['ce', b['epoch']],
                                   scores['top32', a['epoch']], scores['ce', b['epoch']]):
        if left['id'] in changed:
            cases.append({'id': left['id'], 'question': left['question'], 'references': left['ground_truth'],
                          'top32_epoch': a['epoch'], 'ce_epoch': b['epoch'],
                          'top32_answer': left['final_answer'], 'ce_answer': right['final_answer'],
                          'top32_verifier_prob': ls['verifier_prob'], 'ce_verifier_prob': rs['verifier_prob'],
                          'top32_rule_acc': ls['rule_acc'], 'ce_rule_acc': rs['rule_acc']})
    training_runs = {}
    api = wandb.Api(timeout=30)
    for name, folder in roots.items():
        training = json.loads((folder / 'training/run_report.json').read_text())
        server = api.run(training['wandb_url'].replace('https://wandb.ai/', '').replace('/runs/', '/'))
        assert server.state == 'finished' and server.summary.get('global_step') == 1000
        training_runs[name] = {'url': training['wandb_url'], 'server_state': server.state, 'global_step': 1000}
    run = wandb.init(project='medsearch_r1', name='sft-ce-vs-top32-epochs1-3-5-8-s20261013',
                     config={'epochs': [1, 3, 5, 8], 'dev_rows': 396, 'training_seed': 20261013,
                             'same_trajectories': True, 'same_schedule': True,
                             'selection_rule': selected['selection_rule']})
    run.define_metric('epoch')
    run.define_metric('dev/*', step_metric='epoch')
    for ep in [1, 3, 5, 8]:
        metrics = {'epoch': ep}
        for row in rows:
            if row['epoch'] == ep:
                for key in ['verifier_accuracy', 'rule_accuracy', 'closed_answer_rate', 'average_searches',
                            'average_action_tokens', 'invalid_action_episodes']:
                    metrics[f"dev/{row['method']}/{key}"] = row[key]
        run.log(metrics)
    run.summary.update({'selected_epoch': epoch, 'selected_method': method,
                        'selected_verifier_accuracy': best['verifier_accuracy']})
    report = {'summaries': rows, 'same_epoch_top32_vs_ce': same_epoch, 'best_top32_vs_best_ce': best_pair,
              'selected': selected, 'wandb_url': run.url,
              'training_wandb_verified': training_runs,
              'fairness_verified': {'same_training_config': True, 'same_resume_schedule': True,
                                    'same_generation_and_scoring': True, 'ce_top32_loss_zero': True},
              'limitations': 'single seed, dev selection, automatic verifier; no independent test or downstream RL'}
    output.mkdir(parents=True)
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'selected_checkpoint.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'changed_cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cases), encoding='utf-8')
    run.finish()
    print(json.dumps({'selected': selected, 'wandb_url': run.url}), flush=True)


if __name__ == '__main__':
    main()
