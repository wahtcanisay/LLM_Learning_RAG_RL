"""Read back saved dev scores, compare paired outcomes, and publish metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['top32', 'ce'], default='top32')
    args = parser.parse_args()
    root=Path('MedSearch/outputs/distillation/sft_epoch_comparison_20261013' if args.method == 'top32'
              else 'MedSearch/outputs/distillation/sft_ce_epoch_comparison_20261013')
    dev=pq.read_table('MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet').to_pylist()
    expected=[str(row['id']) for row in dev]
    summaries=[]
    scores={}
    raw={}
    for epoch in [1,3,5,8]:
        out=root/f'epoch{epoch}'
        summary=json.loads((out/'summary.json').read_text())
        manifest=json.loads((out/'rollout_manifest.json').read_text())
        for name,digest in [('rollouts.jsonl',summary['rollout_sha256']),('scores.jsonl',summary['scores_sha256'])]:
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
        scores[epoch]=[json.loads(line) for line in (out/'scores.jsonl').read_text().splitlines()]
        raw[epoch]=[json.loads(line) for line in (out/'rollouts.jsonl').read_text().splitlines()]
        assert [row['id'] for row in scores[epoch]]==[row['id'] for row in raw[epoch]]==expected
        assert summary['total']==396 and manifest['status']=='complete'
        summary['average_action_tokens']=sum(sum(row['response_mask']) for row in raw[epoch])/396
        summaries.append(summary)
    best=max(summaries,key=lambda row:(row['verifier_correct'],-row['epoch']))
    paired=[]
    for epoch in [3,5,8]:
        gain=[a['id'] for a,b in zip(scores[epoch],scores[1]) if a['verifier_acc'] and not b['verifier_acc']]
        loss=[a['id'] for a,b in zip(scores[epoch],scores[1]) if b['verifier_acc'] and not a['verifier_acc']]
        discordant=len(gain)+len(loss)
        pvalue=min(1.,2*sum(math.comb(discordant,k) for k in range(min(len(gain),len(loss))+1))/2**discordant) if discordant else 1.
        paired.append({'epoch':epoch,'vs_epoch':1,'gained_ids':gain,'lost_ids':loss,
                       'net_correct_change':len(gain)-len(loss),'mcnemar_exact_two_sided_p':pvalue})
    selected=best['epoch']
    adapter=(Path(f'MedSearch/outputs/distillation/sft_{args.method}_epoch1_20261013/adapter') if selected==1 else root/'training'/f'checkpoint-{125*selected}')
    candidate={'selected_epoch':selected,'base_model':'models/Qwen2.5-3B-Instruct','adapter_path':str(adapter.resolve()),
               'adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
               'verifier_correct':best['verifier_correct'],'dev_total':396,'verifier_accuracy':best['verifier_accuracy'],
               'selection_rule':'highest fixed-dev verifier correct count; exact ties prefer earlier epoch',
               'purpose':'RL initialization candidate; RL not started','independent_test_evaluated':False,
               'schedule_caveat':'epoch1 single-epoch cosine; epochs2-8 resume with extended cosine horizon'}
    candidate['method'] = args.method
    report={'summaries':summaries,'paired_vs_epoch1':paired,'selected':candidate}
    run=wandb.init(project='medsearch_r1',name=f'sft-{args.method}-dev-epochs1-3-5-8-s20261013',
                   config={'epochs':[1,3,5,8],'dev_rows':396,'seed':42,'temperature':0,'top_k':5,'max_search_turns':2,
                           'observation_tokens':700,'response_budget':4096,'max_action_tokens':500,'verifier_threshold':.5,
                           'selection_rule':candidate['selection_rule']})
    run.define_metric('epoch')
    run.define_metric('dev/*',step_metric='epoch')
    for row in summaries:
        run.log({'epoch':row['epoch'],'dev/verifier_accuracy':row['verifier_accuracy'],
                 'dev/rule_accuracy':row['rule_accuracy'],'dev/closed_answer_rate':row['closed_answer_rate'],
                 'dev/average_searches':row['average_searches'],'dev/average_response_tokens':row['average_response_tokens'],
                 'dev/average_action_tokens':row['average_action_tokens'],
                 'dev/invalid_action_episodes':row['invalid_action_episodes'],'dev/exhausted_episodes':row['exhausted_episodes']})
    run.summary.update({'selected_epoch':selected,'selected_verifier_accuracy':best['verifier_accuracy']})
    report['wandb_url']=run.url
    (root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'selected_checkpoint.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf-8')
    cases=[]
    for item in paired:
        changed=set(item['gained_ids']+item['lost_ids'])
        for before,after,old_score,new_score in zip(raw[1],raw[item['epoch']],scores[1],scores[item['epoch']]):
            if before['id'] in changed:
                cases.append({'id':before['id'],'epoch':item['epoch'],'question':before['question'],'references':before['ground_truth'],
                              'epoch1_answer':before['final_answer'],'candidate_answer':after['final_answer'],
                              'epoch1_verifier_prob':old_score['verifier_prob'],'candidate_verifier_prob':new_score['verifier_prob'],
                              'epoch1_rule_acc':old_score['rule_acc'],'candidate_rule_acc':new_score['rule_acc']})
    (root/'changed_cases.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in cases),encoding='utf-8')
    print(json.dumps({'summaries':summaries,'selected':candidate,'wandb_url':run.url},ensure_ascii=False),flush=True)
    run.finish()


if __name__=='__main__':
    main()
