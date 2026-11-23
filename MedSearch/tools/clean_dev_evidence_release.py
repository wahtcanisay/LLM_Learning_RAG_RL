"""Export/verify the source-preserving dev subset from explicit review ledgers.

This module never adjudicates medical evidence or calls a teacher. Decisions
are the separately saved model-authored semantic reviews, not score thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MedSearch/data/dev_cleaning/dev_evidence_20261022'
RULE = ROOT / 'MedSearch/docs/superpowers/plans/2026-10-22-dev-evidence-cleaning.md'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')


def tsv(name):
    with (OUT / name).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def check_inputs():
    inputs = json.loads((OUT / 'input_inventory.json').read_text(encoding='utf-8'))
    for item in inputs:
        # Another chat owns mutable RL/SFT publication pointers. Our boundary is
        # the immutable data files frozen at this dev run's start, not those pointers.
        if Path(item['path']).name in {'CURRENT_RL_RELEASE.json', 'CURRENT_SFT_RELEASE.json'}:
            continue
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    isolation = json.loads((OUT / 'isolation_manifest.json').read_text(encoding='utf-8'))
    for item in isolation['references'].values():
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    return inputs, isolation


def resolve():
    source = pq.read_table(ROOT / 'MedSearch/data/rl/dev.parquet')
    rows = source.to_pylist()
    assert len(rows) == 396 and rows == read_jsonl(OUT / 'source_dev.jsonl')
    assert len({r['id'] for r in rows}) == len(rows)
    questions = {int(r['ordinal']): r for r in tsv('question_adjudications.tsv')}
    evidence_history = tsv('evidence_adjudications.tsv')
    evidence = {int(r['ordinal']): r for r in evidence_history}
    isolation = {r['source_index']: r for r in tsv('isolation_adjudications.tsv')}
    queue = json.loads((OUT / 'isolation_review_queue.json').read_text(encoding='utf-8'))
    assert {r['candidate'].split(':')[-1] for r in queue} == set(isolation)
    assert not json.loads((OUT / 'internal_review_queue.json').read_text(encoding='utf-8'))
    assert not json.loads((OUT / 'exact_isolation.json').read_text(encoding='utf-8'))['conflicts']
    ledger = []
    for ordinal, row in enumerate(rows, 1):
        q, e = questions.get(ordinal), evidence.get(ordinal)
        assert q or e, ordinal
        decision = isolation.get(row['id'].split(':')[-1])
        reasons = [d['reason'] if 'reason' in d else d['rationale'] for d in [q, e] if d]
        if decision and decision['decision'] == 'exclude_same_task':
            final, primary = 'excluded', 'same_task_boundary_conflict'
            reasons.insert(0, decision['rationale'])
        elif q and q['decision'] == 'reject':
            final, primary = 'excluded', 'question_gold_conflict'
        elif e and e['decision'] == 'accept':
            # Only this explicit source-resolved uncertainty supersedes a question hold.
            assert q is None or ordinal == 359, ordinal
            final, primary = 'accepted', 'source_supported_question_gold'
        else:
            final = 'quarantine'
            primary = 'question_or_gold_uncertain' if q else 'support_not_fully_confirmed'
        ledger.append(dict(ordinal=ordinal, question_id=row['id'], question=row['question'],
            golden_answers=row['golden_answers'], outcome=final, primary_reason=primary,
            reasons=reasons, question_review=q, evidence_review=e,
            evidence_review_history=[d for d in evidence_history if int(d['ordinal']) == ordinal],
            isolation_review=decision or dict(decision='not_flagged_by_exact_lexical_mpnet_screen'),
            reviewer='current_chat_primary_agent', review_method='model_authored_semantic_review',
            teacher_student_outputs_used_for_admission=False, verifier_used_for_admission=False,
            clinical_expert_review=False))
    return source, ledger


def export():
    inputs, isolation = check_inputs()
    source, ledger = resolve()
    accepted = [r for r in ledger if r['outcome'] == 'accepted']
    wanted = {did for r in accepted for did in r['evidence_review']['document_ids'].split(',')}
    corpus = {}
    with (ROOT / 'MedSearch/data/retrieval/medrag_textbooks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            d = json.loads(line)
            if d['id'] in wanted:
                corpus[d['id']] = d
    assert set(corpus) == wanted
    overrides = json.loads((OUT / 'rubric_overrides.json').read_text(encoding='utf-8'))
    evidence, rubrics, diagnostics, qrels = [], [], [], []
    for row in accepted:
        i, qid = row['ordinal'], row['question_id']
        passages = []
        for did in row['evidence_review']['document_ids'].split(','):
            d = corpus[did]
            passages.append(dict(document_id=did, title=d['title'], quote=d['content'],
                quote_start=0, quote_end=len(d['content']), quote_scope='complete_frozen_corpus_paragraph',
                content_sha256=hashlib.sha256(d['content'].encode()).hexdigest(),
                row_sha256=hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                source_path='MedSearch/data/retrieval/medrag_textbooks.jsonl'))
            qrels.append(f'{qid}\t{did}\t1\n')
        worked_example = any('correct answer' in p['quote'].lower() for p in passages)
        ev = dict(ordinal=i, question_id=qid, golden_answers=row['golden_answers'],
            support_rationale=row['evidence_review']['rationale'], passages=passages,
            evidence_location_method='gold_aware_offline_audit_plus_manual_corpus_lookup',
            gold_aware=True, is_policy_retrieval=False,
            contains_worked_example_or_answer_key=worked_example,
            qrels_exhaustive=False, evidence_level='full_answer_support_by_primary_agent_review')
        evidence.append(ev)
        extra = overrides.get(str(i), {})
        rubric = dict(question_id=qid, original_gold=row['golden_answers'],
            required_points=extra.get('required_points', row['golden_answers']),
            accepted_aliases=list(dict.fromkeys(row['golden_answers'] + extra.get('aliases', []))),
            reject_boundary=extra.get('reject_boundary', '必须回答题目所问对象；矛盾回答不等价，任何未列合理替代回答须语义复核，不能仅因不匹配gold判医学错误。'),
            scoring_method='semantic_checklist_not_executable_verifier',
            aliases_exhaustive=False, reviewer='current_chat_primary_agent',
            caveat='模型审阅用答案要点，未经医学专家核验；不给作答策略作为输入。')
        rubric.update({k:v for k,v in extra.items() if k in ['derivation', 'alias_provenance']})
        rubrics.append(rubric)
        rec = json.loads((OUT / 'retrieval' / f'{i:03d}.json').read_text(encoding='utf-8'))
        assert rec['id'] == qid and rec['question'] == row['question']
        blind = next(q for q in rec['queries'] if q['mode'] == 'question_only')
        assert not blind['gold_aware'] and blind['query'] == row['question'] and blind['top_k'] == 3
        hits = blind['response']['result'][0]
        hit_ids = [str(h['document']['id']) for h in hits]
        for query in rec['queries']:
            for hit in query['response']['result'][0]:
                did = str(hit['document']['id'])
                if did in corpus:
                    assert hit['document'] == corpus[did], did
        known_hits = [p['document_id'] for p in passages if p['document_id'] in hit_ids]
        visible = [p['document_id'] for p in passages if p['quote'] in blind['window_500_tokens']]
        diagnostics.append(dict(question_id=qid, query=blind['query'], gold_aware=False, top_k=3,
            returned_document_ids=hit_ids, selected_known_positive_document_ids=[p['document_id'] for p in passages],
            selected_known_positive_document_hits=known_hits, selected_known_positive_doc_hit_at3=bool(known_hits),
            selected_complete_corpus_paragraphs_visible_in_500_tokens=visible,
            all_selected_complete_paragraphs_visible=len(visible)==len(passages),
            metric_note='固定原题BM25；不是Agent召回。qrels非穷尽，未命中所选文档不等于无有效证据；完整段落可见是保守字符串检查，不是语义窗口充分性评分。'))
    indices = [r['ordinal']-1 for r in accepted]
    table = source.take(pa.array(indices, type=pa.int64()))
    release = OUT / 'release'
    release.mkdir(exist_ok=True)
    pq.write_table(table, release / 'dev.parquet', compression='zstd')
    assert pq.read_table(release / 'dev.parquet').to_pylist() == table.to_pylist()
    write_jsonl(release / 'accepted.jsonl', table.to_pylist())
    write_jsonl(release / 'evidence.jsonl', evidence)
    write_jsonl(release / 'answer_rubrics.jsonl', rubrics)
    write_jsonl(release / 'retrieval_diagnostics.jsonl', diagnostics)
    (release / 'known_positive_qrels.tsv').write_text('query_id\tdoc_id\trelevance\n' + ''.join(qrels), encoding='utf-8')
    write_jsonl(OUT / 'decisions.jsonl', ledger)
    for kind in ['excluded', 'quarantine']:
        write_jsonl(OUT / f'{kind}.jsonl', [r for r in ledger if r['outcome'] == kind])
    counts = dict(Counter(r['outcome'] for r in ledger))
    write_json(OUT / 'frozen_inputs.json', [dict(path=r['path'], sha256=r['sha256'], bytes=r['bytes'], rows=r.get('rows')) for r in inputs])
    write_json(release / 'manifest.json', dict(status='exported_pending_loader_verification',
        run_id=OUT.name, original_dev_rows=len(ledger), outcomes=counts,
        selection='question_gold_and_corpus_support; independent of teacher/student performance',
        original_dev_sha256=sha(ROOT/'MedSearch/data/rl/dev.parquet'),
        corpus_sha256=sha(ROOT/'MedSearch/data/retrieval/medrag_textbooks.jsonl'),
        corpus_rows=125847, source_rows_unchanged=True, rule_path=str(RULE.relative_to(ROOT)), rule_sha256=sha(RULE),
        reviewer='current_chat_primary_agent', clinical_expert_review=False,
        teacher_inference_calls=0, student_inference_calls=0, training_steps=0,
        dev_only=True, independent_final_test=False, training_ready=False,
        scoring_rubrics_are_manual_checklists=True, existing_verifier_not_validated_against_these_rubrics=True,
        isolation=isolation, adjudicated_same_task_exclusions=13,
        training_boundary_scope='immutable start-of-run data files: SFT1000, RL312, historical RL and frozen test; mutable training pointers owned by other work and not reviewed or changed',
        worked_example_evidence_rows=sum(r['contains_worked_example_or_answer_key'] for r in evidence),
        fixed_question_bm25_diagnostics=dict(rows=len(diagnostics),
            selected_known_positive_doc_hit_at3=sum(r['selected_known_positive_doc_hit_at3'] for r in diagnostics),
            all_selected_complete_paragraphs_visible=sum(r['all_selected_complete_paragraphs_visible'] for r in diagnostics),
            not_agent_recall=True, qrels_exhaustive=False),
        script_path=str(Path(__file__).relative_to(ROOT)), script_sha256=sha(Path(__file__)),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        review_artifact_sha256={name:sha(OUT/name) for name in ['question_adjudications.tsv','evidence_adjudications.tsv',
            'isolation_adjudications.tsv','rubric_overrides.json','decisions.jsonl']},
        artifact_sha256={p.name:sha(p) for p in release.iterdir() if p.is_file() and p.name not in {'manifest.json','verification.json'}}))
    print(json.dumps(dict(exported=counts, accepted_rows=len(accepted)), ensure_ascii=False))


def verify():
    check_inputs()
    source, ledger = resolve()
    release=OUT/'release'
    manifest=json.loads((release/'manifest.json').read_text(encoding='utf-8'))
    for name,value in manifest['artifact_sha256'].items(): assert sha(release/name)==value, name
    for name,value in manifest['review_artifact_sha256'].items(): assert sha(OUT/name)==value, name
    assert sha(RULE)==manifest['rule_sha256'] and sha(Path(__file__))==manifest['script_sha256']
    expected=[source.to_pylist()[r['ordinal']-1] for r in ledger if r['outcome']=='accepted']
    actual=pq.read_table(release/'dev.parquet').to_pylist()
    assert actual==expected==read_jsonl(release/'accepted.jsonl')
    assert pq.read_schema(release/'dev.parquet')==source.schema
    assert read_jsonl(OUT/'decisions.jsonl')==ledger
    ids=[r['id'] for r in actual]
    ev=read_jsonl(release/'evidence.jsonl')
    rubrics=read_jsonl(release/'answer_rubrics.jsonl')
    assert ids==[r['question_id'] for r in ev]==[r['question_id'] for r in rubrics]
    requested={p['document_id'] for r in ev for p in r['passages']}
    corpus={}
    with (ROOT/'MedSearch/data/retrieval/medrag_textbooks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            d=json.loads(line)
            if d['id'] in requested:corpus[d['id']]=d
    citations=0
    for r in ev:
        assert r['passages'] and r['gold_aware'] and not r['is_policy_retrieval']
        for p in r['passages']:
            d=corpus[p['document_id']]
            assert p['quote']==d['content'][p['quote_start']:p['quote_end']]
            assert p['content_sha256']==hashlib.sha256(d['content'].encode()).hexdigest()
            citations+=1
    sys.path.insert(0,str(ROOT/'MedSearch/.deps/verl'))
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-3B-Instruct',local_files_only=True)
    lengths=[len(tokenizer.apply_chat_template(r['prompt'],add_generation_prompt=True,tokenize=True,return_dict=False)) for r in actual]
    assert max(lengths)<=512, max(lengths)
    config=OmegaConf.create(dict(max_prompt_length=512,return_raw_chat=True,
        filter_overlong_prompts=True,filter_overlong_prompts_workers=1,truncation='error',cache_dir=str(OUT/'loader_cache')))
    dataset=RLHFDataset(str(release/'dev.parquet'),tokenizer,config)
    assert len(dataset)==len(actual)
    for i,r in enumerate(actual):
        loaded=dataset[i]
        assert loaded['reward_model']['ground_truth']['target']==r['golden_answers']
        assert list(loaded['raw_prompt'])==r['prompt']
        assert r['prompt']==source.to_pylist()[next(x['ordinal']-1 for x in ledger if x['question_id']==r['id'])]['prompt']
    collated=collate_fn([dataset[i] for i in range(min(4,len(dataset)))])
    assert len(collated['reward_model'])==min(4,len(dataset))
    check_inputs()
    counts=dict(Counter(r['outcome'] for r in ledger))
    assert sum(counts.values())==396 and counts==manifest['outcomes']
    receipt=dict(status='passed',original_rows=396,outcomes=counts,source_question_gold_prompt_unchanged=True,
        citations_exactly_bound=citations,unique_corpus_paragraphs=len(corpus),
        loader='verl.utils.dataset.rl_dataset.RLHFDataset',loader_rows=len(dataset),collated_rows=min(4,len(dataset)),
        max_prompt_tokens=max(lengths),max_prompt_length=512,tokenizer='models/Qwen2.5-3B-Instruct',
        loader_sha256=sha(ROOT/'MedSearch/.deps/verl/verl/utils/dataset/rl_dataset.py'),
        rejected_or_quarantine_overlap=False,original_and_protected_inputs_unchanged=True,
        selected_parquet_sha256=sha(release/'dev.parquet'),semantic_correctness_claim='model_review_not_clinical_expert')
    write_json(release/'verification.json',receipt)
    manifest['status']='published_dev_subset_verified'
    manifest['evaluation_ready']=True
    manifest['verification_sha256']=sha(release/'verification.json')
    write_json(release/'manifest.json',manifest)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify: verify()
    else: export()
