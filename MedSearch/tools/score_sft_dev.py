"""Score already saved dev rollouts with the frozen local verifier."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from MedSearch.rl.reward import compute_outcome_reward
from MedSearch.tools.evaluate_agent_rollout import verifier_score
from MedSearch.tools.run_cpu_verifier_server import TEMPLATE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--output-root',default='MedSearch/outputs/distillation/sft_epoch_comparison_20261013')
    parser.add_argument('--cpu-crosscheck',type=int,default=0)
    args=parser.parse_args()
    out=Path(args.output_root)/f'epoch{args.epoch}'
    manifest=json.loads((out/'rollout_manifest.json').read_text())
    assert manifest['status']=='complete' and not manifest['retrieval_error_ids']
    raw=out/'rollouts.jsonl'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==manifest['rollout_sha256']
    rows=[json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    assert len(rows)==manifest['processed'] and len({row['id'] for row in rows})==len(rows)
    if (out/'scores.jsonl').exists():
        raise FileExistsError(out/'scores.jsonl')
    started=time.monotonic()
    model_path='MedSearch/models/external_baselines/medical_o1_verifier_3B'
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_path,dtype=torch.bfloat16,
                                                             local_files_only=True).to('cuda').eval()
    probabilities=[0.]*len(rows)
    pairs=[]
    for index,row in enumerate(rows):
        answers=[a.strip() for a in re.findall(r'<answer>(.*?)</answer>',row['response'],re.DOTALL)
                 if a.strip() not in {'answer','and'}]
        if answers:
            for target in row['ground_truth']:
                pairs.append((index,TEMPLATE.format(answers[-1],target,tokenizer.eos_token or '')))
    with torch.inference_mode():
        for start in range(0,len(pairs),8):
            batch=pairs[start:start+8]
            inputs=tokenizer([prompt for _,prompt in batch],padding=True,truncation=True,
                             max_length=1024,return_tensors='pt').to('cuda')
            probs=torch.softmax(model(**inputs).logits.float(),dim=-1)[:,1].cpu().tolist()
            for (index,_),prob in zip(batch,probs,strict=True):
                probabilities[index]=max(probabilities[index],prob)
            print(json.dumps({'epoch':args.epoch,'verifier_pairs_scored':start+len(batch),'total_pairs':len(pairs)}),flush=True)
    scored=[]
    for row,prob in zip(rows,probabilities,strict=True):
        reward=compute_outcome_reward('medsearch_medical_verifiable',row['response'],{'target':row['ground_truth']})
        scored.append({'id':row['id'],'rule_acc':int(reward['acc']),'verifier_prob':prob,
                       'verifier_acc':int(prob>.5),'answered':row['answered'],'searches':row['searches'],
                       'invalid_actions':row['invalid_actions'],'response_tokens':row['response_tokens'],
                       'exhausted':row['exhausted']})
    crosscheck=[]
    for row,prob in list(zip(rows,probabilities))[:args.cpu_crosscheck]:
        cpu=verifier_score(row['response'],row['ground_truth'],'http://127.0.0.1:8002/verify')
        crosscheck.append({'id':row['id'],'cpu_prob':cpu,'gpu_prob':prob,'same_decision':(cpu>.5)==(prob>.5)})
    score_path=out/'scores.jsonl'
    score_path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in scored),encoding='utf-8')
    total=len(rows)
    summary={'epoch':args.epoch,'total':total,'verifier_correct':sum(row['verifier_acc'] for row in scored),
             'rule_correct':sum(row['rule_acc'] for row in scored),
             'closed_answers':sum(row['answered'] for row in scored),
             'invalid_action_episodes':sum(row['invalid_actions']>0 for row in scored),
             'exhausted_episodes':sum(row['exhausted'] for row in scored),
             'average_searches':sum(row['searches'] for row in scored)/total,
             'average_response_tokens':sum(row['response_tokens'] for row in scored)/total,
             'verifier_accuracy':sum(row['verifier_acc'] for row in scored)/total,
             'rule_accuracy':sum(row['rule_acc'] for row in scored)/total,
             'closed_answer_rate':sum(row['answered'] for row in scored)/total,
             'verifier_model':model_path,'verifier_device':'cuda','verifier_dtype':'bfloat16',
             'verifier_max_length':1024,'verifier_threshold':.5,'verifier_batch_size':8,
             'cpu_crosscheck':crosscheck,'rollout_sha256':manifest['rollout_sha256'],
             'scores_sha256':hashlib.sha256(score_path.read_bytes()).hexdigest(),
             'scoring_elapsed_seconds':time.monotonic()-started,
             'rollout_elapsed_seconds':manifest['elapsed_seconds'],'status':'complete'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
