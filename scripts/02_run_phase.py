"""Rerun Phases 2-5 locally under identical conditions.

The project memory flags that Phase 3 was trained for 10 epochs against Phase 2's 15, which
invalidated the enhancement-vs-baseline comparison. Rather than fixing only that one run, every
phase is retrained here on the same GPU, same library versions, same seed, same epoch count and
same batch size, so the whole results table is internally consistent.

Usage:  python scripts/02_run_phase.py --phase 3
        python scripts/02_run_phase.py --phase all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CHECKPOINTS_DIR, FIGURES_DIR
from src.data import DEFAULT_ENHANCEMENT, URICADSMultiTaskDataset, load_manifest
from src.engine import evaluate, get_device, make_loader, set_seed, train_model
from src.enhanced_cache import build_enhanced_splits
from src.model import MultiTaskNet, MultiTaskNetV2
from src.reporting import (Tee, figure_path, plot_qualitative, plot_roc,
                           plot_training_curves, render_console_png, save_phase_result)

CLASSIFICATION_ONLY = ['diagnosis', 'global_pathology', 'kidney_seg']
FULL_NO_CONSISTENCY = ['diagnosis', 'global_pathology', 'local_pathology',
                       'kidney_seg', 'lesion_seg']
FULL = FULL_NO_CONSISTENCY + ['consistency']

PHASES = {
    '2': dict(name='Phase 2 - fixed-weight multi-task baseline',
              model=MultiTaskNet, enhancement=None, active_terms=CLASSIFICATION_ONLY,
              selection='kidney_dice', checkpoint='phase2_baseline_local.pt',
              consistency='absolute'),
    '3': dict(name='Phase 3 - fixed-parameter enhancement (15 epochs, matched to Phase 2)',
              model=MultiTaskNet, enhancement=DEFAULT_ENHANCEMENT,
              active_terms=CLASSIFICATION_ONLY, selection='kidney_dice',
              checkpoint='phase3_enhanced_local.pt', consistency='absolute'),
    '4': dict(name='Phase 4 - full 5-head multi-task scaffold',
              model=MultiTaskNetV2, enhancement=None, active_terms=FULL_NO_CONSISTENCY,
              selection='combined_dice', checkpoint='phase4_multitask_local.pt',
              consistency='absolute'),
    '5': dict(name='Phase 5 - anatomical consistency (absolute-mass formulation)',
              model=MultiTaskNetV2, enhancement=None, active_terms=FULL,
              selection='combined_dice', checkpoint='phase5_anatomical_local.pt',
              consistency='absolute'),
    '5b': dict(name='Phase 5b - anatomical consistency (corrected ratio formulation)',
               model=MultiTaskNetV2, enhancement=None, active_terms=FULL,
               selection='combined_dice', checkpoint='phase5b_anatomical_ratio_local.pt',
               consistency='ratio'),
}


def run_phase(phase_key, epochs, batch_size, lr, num_workers):
    cfg = PHASES[phase_key]
    device = get_device()
    manifest = load_manifest()

    print('=' * 78)
    print(cfg['name'])
    print('=' * 78)
    print(f'epochs={epochs} batch_size={batch_size} lr={lr} seed=42 '
          f'device={torch.cuda.get_device_name(0)}')
    print(f'loss terms (equal weights): {", ".join(cfg["active_terms"])}')
    if cfg['enhancement']:
        print(f'enhancement: {cfg["enhancement"]}')
    print(flush=True)

    set_seed()
    if cfg['enhancement']:
        print('Precomputing enhanced images (one-time, pooled across CPU cores)...', flush=True)
        ds = build_enhanced_splits(manifest, cfg['enhancement'])
        train_ds, val_ds, test_ds = ds['train'], ds['val'], ds['test']
    else:
        train_ds, val_ds, test_ds = (
            URICADSMultiTaskDataset(manifest[manifest['split'] == s]) for s in
            ('train', 'val', 'test'))
    print(f'train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}\n', flush=True)

    set_seed()
    model = cfg['model']().to(device)
    history, best = train_model(
        model, train_ds, val_ds, device, epochs=epochs, batch_size=batch_size, lr=lr,
        active_terms=cfg['active_terms'], checkpoint_name=cfg['checkpoint'],
        num_workers=num_workers, selection=cfg['selection'],
        log_name=f'phase{phase_key}_history.json', consistency=cfg['consistency'])

    print(f'\nBest validation {cfg["selection"]}: {best:.4f}')
    print('Loading best checkpoint for test evaluation...', flush=True)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, cfg['checkpoint']),
                                     map_location=device))
    test_loader = make_loader(test_ds, batch_size, False, num_workers)
    metrics = evaluate(model, test_loader, device, use_amp=True)

    print('\n' + '-' * 78)
    print(f'PHASE {phase_key.upper()} TEST RESULTS')
    print('-' * 78)
    for key in ('diagnosis_auc', 'global_pathology_mean_auc', 'local_pathology_mean_auc',
                'kidney_dice', 'lesion_dice', 'lesion_dice_positive_only',
                'consistency_violation_rate'):
        print(f'  {key:32s}: {metrics[key]:.4f}')
    print(f'  {"n_test":32s}: {metrics["n_samples"]}')

    # figures
    plot_training_curves(history, figure_path(f'phase{phase_key}_curves.png'), cfg['name'])
    from src.metrics import MetricAccumulator
    acc = MetricAccumulator()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            with torch.autocast('cuda'):
                out = model(images)
            if not isinstance(out, dict):
                out = {'diagnosis': out[0], 'global_pathology': out[1], 'kidney_seg': out[2],
                       'lesion_seg': torch.full_like(out[2], -20.0),
                       'local_pathology': torch.zeros(images.shape[0], 9, device=device) - 20.0}
            acc.update(out, batch, device)
    plot_roc(acc.diag_targets, acc.diag_probs, figure_path(f'phase{phase_key}_roc.png'),
             f'Phase {phase_key} - healthy vs pathological ROC')
    lesion_idx = [i for i in range(len(test_ds)) if test_ds[i]['has_lesion'].item() > 0][:3]
    plot_qualitative(model, test_ds, device, figure_path(f'phase{phase_key}_qualitative.png'),
                     indices=lesion_idx, title=f'Phase {phase_key} - test predictions')

    save_phase_result(f'phase{phase_key}', {
        'config': {k: (v.__name__ if callable(v) else v) for k, v in cfg.items()},
        'hyperparameters': {'epochs': epochs, 'batch_size': batch_size, 'lr': lr, 'seed': 42},
        'test_metrics': {k: v for k, v in metrics.items() if not isinstance(v, dict)},
        'local_pathology_aucs': metrics.get('local_pathology_aucs'),
        'history': history,
        'best_val_score': best,
    })
    del model
    torch.cuda.empty_cache()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=0)
    args = ap.parse_args()

    phases = ['2', '3', '4', '5', '5b'] if args.phase == 'all' else args.phase.split(',')
    for key in phases:
        with Tee() as tee:
            run_phase(key, args.epochs, args.batch_size, args.lr, args.num_workers)
        render_console_png(tee.text, os.path.join(FIGURES_DIR, f'phase{key}_console.png'),
                           title=f'python scripts/02_run_phase.py --phase {key}')


if __name__ == '__main__':
    main()
