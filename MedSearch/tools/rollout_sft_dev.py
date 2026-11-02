"""Save fixed-dev trajectories first; no verifier runs during generation."""
import argparse
import functools
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from vllm import LLM
from vllm.lora.request import LoRARequest

from MedSearch.tools.evaluate_agent_rollout import load_states, run_agent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output-root',default='MedSearch/outputs/distillation/sft_epoch_comparison_20261013')
    opts=parser.parse_args()
    out=Path(opts.output_root)/f'epoch{opts.epoch}'
    if out.exists():
        raise FileExistsError(out)
    adapter=Path(opts.adapter).resolve()
    dev=Path('MedSearch/data/distillation/pools_v2_20260926_devclean/dev.parquet').resolve()
    args=SimpleNamespace(distillation_dir=True,response_length=4096,max_model_len=4608,max_turns=2,
                         observation_length=700,top_k=5,search_url='http://127.0.0.1:8001/retrieve',
                         temperature=0.,top_p=1.,seed=42)
    manifest={'status':'running','epoch':opts.epoch,'adapter':str(adapter),
              'adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
              'dev':str(dev),'dev_sha256':hashlib.sha256(dev.read_bytes()).hexdigest(),
              'model':'models/Qwen2.5-3B-Instruct','generation_config':vars(args),
              'max_action_length':500,'batch_size':8,'stop':['</search>','</answer>'],
              'include_stop_str_in_output':True,'verifier_during_rollout':False,
              'processed':0,'limit':opts.limit}
    out.mkdir(parents=True)
    manifest_path=out/'rollout_manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save_manifest()
    started=time.monotonic()
    try:
        llm=LLM(model=manifest['model'],dtype='bfloat16',max_model_len=args.max_model_len,
                gpu_memory_utilization=.70,enforce_eager=True,tensor_parallel_size=1,
                max_num_seqs=8,max_num_batched_tokens=8192,
                enable_lora=True,max_lora_rank=8,max_loras=1)
        tokenizer=llm.get_tokenizer()
        # 原生 LoRARequest 绑定到同一个 generate；复用已有搜索循环。
        llm.generate=functools.partial(llm.generate,lora_request=LoRARequest(f'epoch{opts.epoch}',1,str(adapter)))
        states=load_states(dev,tokenizer,opts.limit)
        assert opts.limit is not None or len(states)==396
        raw=out/'rollouts.jsonl'
        with raw.open('w',encoding='utf-8') as stream:
            for start in range(0,len(states),8):
                batch=states[start:start+8]
                run_agent(llm,tokenizer,batch,args)
                for state in batch:
                    row={'id':state.item_id,'question':state.question,'ground_truth':state.targets,
                         'prompt_ids':state.runtime_ids[:state.prompt_length],
                         'response_ids':state.response_ids,'response_mask':state.response_mask,
                         'segments':state.segments,'retrieval_records':state.search_records,
                         'response':tokenizer.decode(state.response_ids,skip_special_tokens=True),
                         'final_answer':state.final_answer,'answered':state.answered,
                         'searches':state.searches,'invalid_actions':state.invalid_actions,
                         'response_tokens':len(state.response_ids),'exhausted':state.exhausted,
                         'stop_reason':state.stop_reason}
                    assert len(row['response_ids'])==len(row['response_mask'])
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()
                manifest['processed']=start+len(batch)
                save_manifest()
                print(json.dumps({'epoch':opts.epoch,'processed':manifest['processed'],'total':len(states)}),flush=True)
        assert len({state.item_id for state in states})==len(states)
        errors=[state.item_id for state in states if str(state.stop_reason).startswith('retrieval_error:')]
        manifest.update(status='complete',elapsed_seconds=time.monotonic()-started,
                        retrieval_error_ids=errors,rollout_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        save_manifest()
        if errors:
            raise RuntimeError(f'Retrieval infrastructure errors: {errors}')
    except BaseException as error:
        manifest.update(status='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-started)
        save_manifest()
        raise


if __name__=='__main__':
    main()
