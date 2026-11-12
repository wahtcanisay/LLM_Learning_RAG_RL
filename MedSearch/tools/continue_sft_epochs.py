"""Resume the accepted epoch-1 checkpoint to cumulative epoch 8."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import wandb
import yaml
from transformers import AutoTokenizer, TrainerCallback, set_seed
from safetensors.torch import load_file

from MedSearch.sft.train import PROJECT_ROOT, load_config, build_dataloader, load_base_model, attach_lora, build_trainer, save_adapter


class ExtendResumeSchedule(TrainerCallback):
    """Re-evaluate the new native cosine schedule at the restored step."""
    def __init__(self,trainer):
        self.trainer=trainer
        self.details={}

    def on_train_begin(self,args,state,control,**kwargs):
        optimizer=self.trainer.optimizer
        self.details['loaded_learning_rate']=optimizer.param_groups[0]['lr']
        self.trainer.lr_scheduler.step(state.global_step)
        self.details.update(resumed_learning_rate=optimizer.param_groups[0]['lr'],resume_step=state.global_step,
                            planned_total_steps=state.max_steps)
        assert self.details['resumed_learning_rate']>0
        print(json.dumps({'schedule_extension':self.details}),flush=True)


class StopAfterResumeStep(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        control.should_training_stop=True
        control.should_save=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-resume',action='store_true')
    parser.add_argument('--method', choices=['top32', 'ce'], default='top32')
    args=parser.parse_args()
    root=PROJECT_ROOT/('MedSearch/outputs/distillation/sft_epoch_comparison_20261013' if args.method == 'top32'
                       else 'MedSearch/outputs/distillation/sft_ce_epoch_comparison_20261013')
    output=root/('resume_smoke' if args.smoke_resume else 'training')
    if output.exists():
        raise FileExistsError(output)
    checkpoint=PROJECT_ROOT/f'MedSearch/outputs/distillation/sft_{args.method}_epoch1_20261013/checkpoint-125'
    config=load_config(PROJECT_ROOT/f'MedSearch/configs/sft_{args.method}_epoch1.yaml')
    config.update(adapter_path=str(checkpoint),resume_from_checkpoint=str(checkpoint))
    config['training'].update(output_dir=str(output),num_train_epochs=8,save_total_limit=None,
                              run_name=f'sft-{args.method}-resume-epoch2to8-s20261013',report_to='none' if args.smoke_resume else 'wandb')
    set_seed(config['training']['seed'])
    tokenizer=AutoTokenizer.from_pretrained(PROJECT_ROOT/config['model_path'],local_files_only=True)
    tokenizer.padding_side='right'
    dataloader=build_dataloader(config,tokenizer)
    model=attach_lora(load_base_model(config,dtype=torch.bfloat16),config)
    trainer=build_trainer(config,model,tokenizer,dataloader)
    extension=ExtendResumeSchedule(trainer)
    trainer.add_callback(extension)
    if args.smoke_resume:
        trainer.add_callback(StopAfterResumeStep())
    else:
        gate=json.loads((root/'resume_smoke/run_report.json').read_text())
        assert gate['status']=='complete' and gate['adapter_delta_sq']>0
        assert json.loads((root/'epoch1/summary.json').read_text())['total']==396
    output.mkdir(parents=True,exist_ok=True)  # Trainer 已创建 output_dir；入口处仍拒绝覆盖已有运行。
    (output/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True),encoding='utf-8')
    run=None if args.smoke_resume else wandb.init(project='medsearch_r1',name=config['training']['run_name'],
                                                config={'config':config,'resume_step':125,'schedule_policy':'extend cosine horizon to total1000; re-evaluate LR at step125'})
    report={'status':'running','config':config,'resume_checkpoint':str(checkpoint),
            'initial_adapter_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (PROJECT_ROOT/'MedSearch/sft').glob('*.py')},
            'wandb_url':run.url if run else None}
    report_path=output/'run_report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    try:
        result=trainer.train(resume_from_checkpoint=str(checkpoint))
        expected=126 if args.smoke_resume else 1000
        assert result.global_step==expected
        save_adapter(trainer.model,tokenizer,output/'adapter')
        previous=load_file(str(checkpoint/'adapter_model.safetensors'))
        actual=load_file(str(output/'adapter/adapter_model.safetensors'))
        delta_sq=sum((actual[k].float()-previous[k].float()).square().sum().item() for k in previous)
        assert delta_sq>0
        checkpoints=[126] if args.smoke_resume else [250,375,500,625,750,875,1000]
        for step in checkpoints:
            folder=output/f'checkpoint-{step}'
            assert all((folder/name).is_file() for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'])
        report.update(status='complete',metrics=result.metrics,global_step=result.global_step,
                      adapter_delta_sq=delta_sq,schedule_extension=extension.details,
                      log_history=trainer.state.log_history,checkpoints=checkpoints,
                      elapsed_seconds=time.monotonic()-started,
                      cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if run:
            run.summary.update({**result.metrics,'global_step':result.global_step,'checkpoints_saved':len(checkpoints),
                                'cuda_peak_allocated_gib':report['cuda_peak_allocated_gib']})
        print(json.dumps({k:report[k] for k in ['status','global_step','elapsed_seconds','adapter_delta_sq','schedule_extension','wandb_url']}),flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if run:
            run.finish(exit_code=1)
        raise
    if run:
        run.finish()


if __name__=='__main__':
    main()
