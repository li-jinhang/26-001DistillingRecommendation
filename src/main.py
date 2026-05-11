import os
import argparse
from utils_package.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', type=str, default='KDH', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--diagnose_training', action='store_true')
    parser.add_argument('--diagnose_batches', type=int, default=1)
    parser.add_argument('--disable_lr_scheduler', action='store_true')
    parser.add_argument('--disable_grad_clip', action='store_true')
    parser.add_argument('--disable_weight_decay', action='store_true')
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--debug_overfit_samples', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--stopping_step', type=int, default=None)
    parser.add_argument('--stage1_epochs', type=int, default=None)
    parser.add_argument('--stage2_epochs', type=int, default=None)

    args, _ = parser.parse_known_args()
    config_dict = {
        'gpu_id': 0,
        'diagnose_training': bool(args.diagnose_training),
        'diagnose_batches': int(args.diagnose_batches),
        'disable_lr_scheduler': bool(args.disable_lr_scheduler),
        'disable_grad_clip': bool(args.disable_grad_clip),
        'disable_weight_decay': bool(args.disable_weight_decay),
        'grad_accum_steps': int(args.grad_accum_steps),
        'debug_overfit_samples': int(args.debug_overfit_samples),
        'use_amp': bool(args.use_amp),
    }
    if args.epochs is not None:
        config_dict['epochs'] = int(args.epochs)
    if args.stopping_step is not None:
        config_dict['stopping_step'] = int(args.stopping_step)
    if args.stage1_epochs is not None or args.stage2_epochs is not None:
        stage1 = int(args.stage1_epochs) if args.stage1_epochs is not None else 1
        stage2 = int(args.stage2_epochs) if args.stage2_epochs is not None else max(stage1, config_dict.get('epochs', stage1))
        config_dict['stage_epochs'] = [stage1, stage2]

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

