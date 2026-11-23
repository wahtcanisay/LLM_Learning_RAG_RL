import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainerCallback, TrainingArguments
from MedSearch.sft.distillation_trainer import DistillationTrainer


def test_extended_resume_updates_on_first_resumed_step(tmp_path):
    from MedSearch.tools.continue_sft_epochs import ExtendResumeSchedule
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,
                                     num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                                     pad_token_id=0,bos_token_id=1,eos_token_id=2))
    batch={'input_ids':torch.tensor([[1,3,4,5]]),'attention_mask':torch.ones(1,4,dtype=torch.long),
           'labels':torch.tensor([[-100,3,4,5]])}
    def trainer(max_steps):
        args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,max_steps=max_steps,
                               learning_rate=1e-3,lr_scheduler_type='cosine',save_strategy='steps',save_steps=1,
                               report_to='none',disable_tqdm=True,remove_unused_columns=False)
        return DistillationTrainer(model=model,args=args,train_dataset=[batch]*3,
                                   data_collator=lambda rows:rows[0])
    first=trainer(1)
    first.train()
    assert first.optimizer.param_groups[0]['lr']==0
    before=model.lm_head.weight.detach().clone()
    resumed=trainer(3)
    extension=ExtendResumeSchedule(resumed)
    resumed.add_callback(extension)
    snapshots=[]
    class Capture(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            snapshots.append(model.lm_head.weight.detach().clone())
    resumed.add_callback(Capture())
    result=resumed.train(resume_from_checkpoint=str(tmp_path/'checkpoint-1'))
    assert extension.details['loaded_learning_rate']==0
    assert extension.details['resumed_learning_rate']>0
    assert not torch.equal(before,snapshots[0])
    assert result.global_step==3 and resumed.optimizer.param_groups[0]['lr']==0
