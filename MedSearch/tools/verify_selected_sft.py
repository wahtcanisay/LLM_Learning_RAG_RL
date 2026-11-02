"""Check the chosen adapter is active after reload and generates one legal action."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import wandb
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from MedSearch.rl.protocol import parse_action
from MedSearch.sft.data import load_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='MedSearch/outputs/distillation/sft_epoch_comparison_20261013')
    args = parser.parse_args()
    root=Path(args.output_root)
    selected=json.loads((root/'selected_checkpoint.json').read_text())
    adapter=Path(selected['adapter_path'])
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==selected['adapter_sha256']
    tokenizer=AutoTokenizer.from_pretrained(selected['base_model'],local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(selected['base_model'],dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,adapter).to('cuda').eval()
    row=load_samples('MedSearch/data/distillation/top32_sft1000_20261011/train.parquet')[0]
    prefix=row['prompt_ids']
    ids=torch.tensor([prefix],device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        active=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1].float()
        with model.disable_adapter():
            disabled=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1].float()
    logit_difference=(active-disabled).abs().max().item()
    assert logit_difference>0
    class EndAction(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            text=tokenizer.decode(input_ids[0,len(prefix):][-16:],skip_special_tokens=False)
            return torch.full((input_ids.shape[0],),'</search>' in text or '</answer>' in text,
                              device=input_ids.device,dtype=torch.bool)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=500,
                                 do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([EndAction()]))
    text=tokenizer.decode(generated[0,len(prefix):],skip_special_tokens=True)
    action=parse_action(text)
    assert action is not None,text
    tensors=load_file(str(adapter/'adapter_model.safetensors'))
    norm=sum(t.float().square().sum().item() for key,t in tensors.items() if 'lora_B' in key)**.5
    comparison=json.loads((root/'comparison.json').read_text())
    api=wandb.Api(timeout=30)
    run=api.run(comparison['wandb_url'].replace('https://wandb.ai/','').replace('/runs/','/'))
    assert run.state=='finished' and run.summary.get('selected_epoch')==selected['selected_epoch']
    if 'selected_method' in run.summary:
        assert run.summary['selected_method'] == selected['method']
    result={'status':'passed','selected_epoch':selected['selected_epoch'],'adapter_hash_verified':True,
            'active_vs_disabled_logits_max_abs_difference':logit_difference,'lora_b_norm':norm,
            'action_kind':action.kind,'action_text':text,'inference_context':'training question initial prompt; no teacher history or gold',
            'wandb_server_state':run.state,'wandb_selected_epoch':run.summary.get('selected_epoch'),
            'rl_started':False,'verl_initialization_not_executed':True}
    (root/'selected_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
