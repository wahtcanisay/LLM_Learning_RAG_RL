"""One-shot sequential continuation/evaluation; every subprocess keeps its log."""
import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--method', choices=['top32', 'ce'], default='top32')
    args=parser.parse_args()
    root=Path('MedSearch/outputs/distillation/sft_epoch_comparison_20261013' if args.method == 'top32'
              else 'MedSearch/outputs/distillation/sft_ce_epoch_comparison_20261013')
    if args.method == 'top32':
        assert json.loads((root/'epoch1/rollout_manifest.json').read_text())['status']=='complete'
    root.mkdir(parents=True, exist_ok=True)
    status=root/'pipeline_status.json'
    if status.exists() and not args.resume:
        raise FileExistsError(status)
    pipeline=json.loads(status.read_text()) if args.resume else {'completed_phases':[]}
    if args.resume:
        shutil.copy2(status,root/'pipeline_status.failed_attempt1.json')
        pipeline.pop('error_type',None)
    pipeline.update(status='running')
    phases=[]
    if args.method == 'ce':
        phases.extend([('prepare_ce', ['MedSearch.tools.prepare_ce_sft']),
                       ('checkpoint_smoke', ['MedSearch.tools.smoke_sft_checkpoint', '--method', 'ce']),
                       ('train_epoch1', ['MedSearch.tools.run_sft_epoch', '--method', 'ce']),
                       ('check_epoch1', ['MedSearch.tools.check_sft_epoch', '--method', 'ce']),
                       ('rollout_epoch1', ['MedSearch.tools.rollout_sft_dev', '--epoch', '1', '--adapter',
                          'MedSearch/outputs/distillation/sft_ce_epoch1_20261013/adapter', '--output-root', str(root)])])
    phases.extend([('score_epoch1',['MedSearch.tools.score_sft_dev','--epoch','1','--cpu-crosscheck','4','--output-root',str(root)]),
                   ('resume_smoke',['MedSearch.tools.continue_sft_epochs','--smoke-resume','--method',args.method]),
                   ('train_to_epoch8',['MedSearch.tools.continue_sft_epochs','--method',args.method])])
    for epoch in [3,5,8]:
        phases.extend([(f'rollout_epoch{epoch}',['MedSearch.tools.rollout_sft_dev','--epoch',str(epoch),'--adapter',
                         str(root/'training'/f'checkpoint-{125*epoch}'),'--output-root',str(root)]),
                       (f'score_epoch{epoch}',['MedSearch.tools.score_sft_dev','--epoch',str(epoch),'--output-root',str(root)])])
    phases.append(('compare',['MedSearch.tools.compare_sft_epochs','--method',args.method]))
    phases.append(('verify_selected',['MedSearch.tools.verify_selected_sft','--output-root',str(root)]))
    try:
        for name,arguments in phases:
            if name in pipeline['completed_phases']:
                continue
            pipeline['current_phase']=name
            status.write_text(json.dumps(pipeline,indent=2),encoding='utf-8')
            print('Starting',name,flush=True)
            log_path=root/f'{name}.log'
            if log_path.exists():
                shutil.copy2(log_path,root/f'{name}.failed_attempt1.log')
            with log_path.open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-u','-m',*arguments],stdout=log,stderr=subprocess.STDOUT,check=True)
            pipeline['completed_phases'].append(name)
        pipeline.update(status='complete',current_phase=None)
        status.write_text(json.dumps(pipeline,indent=2),encoding='utf-8')
        print(json.dumps(pipeline),flush=True)
    except BaseException as error:
        pipeline.update(status='failed',error_type=type(error).__name__)
        status.write_text(json.dumps(pipeline,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
