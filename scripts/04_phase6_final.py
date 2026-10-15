"""Phase 6, final step: retrain the winning configuration at full fidelity.

The search in 03_phase6_search.py runs at reduced fidelity for cost reasons. This script takes
the best configuration found and retrains it under exactly the conditions used for Phases 2-5
(512x512, 15 epochs, batch 4, seed 42), so the optimized model can be placed in the same
results table as the fixed-weight baselines without any confound.

Usage:  python scripts/04_phase6_final.py                 # uses the winning optimizer
        python scripts/04_phase6_final.py --optimizer WOA # or a specific one
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CHECKPOINTS_DIR, FIGURES_DIR
from src.engine import evaluate, get_device, make_loader, set_seed, train_model
from src.enhanced_cache import build_enhanced_splits
from src.data import load_manifest
from src.losses import LOSS_TERMS
from src.metrics import MetricAccumulator
from src.model import MultiTaskNetV2
from src.reporting import (Tee, figure_path, load_phase_result, plot_qualitative, plot_roc,
                           plot_training_curves, render_console_png, save_phase_result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--optimizer', default=None)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--consistency', default='ratio')
    ap.add_argument('--save-as', default=None,
                    help='result key; defaults to phase6_final. Use a distinct key when '
                         'retraining a non-winning optimizer to check proxy ranking.')
    args = ap.parse_args()

    search = load_phase_result('phase6_search')
    if not search:
        raise SystemExit('Run scripts/03_phase6_search.py first.')

    opt_name = args.optimizer or search['winner']
    best = search['results'][opt_name]
    loss_weights = best['loss_weights']
    enhancement = best['enhancement']

    device = get_device()
    manifest = load_manifest()

    print('=' * 78)
    print(f'PHASE 6 FINAL - optimized configuration from {opt_name}, full fidelity')
    print('=' * 78)
    print(f'Search fitness at reduced fidelity: {best["best_fitness"]:.4f} '
          f'({best["n_evaluations"]} model trainings, {best["wall_clock_hours"]:.2f} h)')
    print(f'\nOptimized loss weights (Phases 2-5 used 1.000 for every term):')
    for term in LOSS_TERMS:
        print(f'  {term:20s} {loss_weights[term]:6.3f}')
    print(f'\nOptimized enhancement parameters '
          f'(Phase 3 used clip=2.0, tile=8, h=10, gamma=1.2):')
    for k, v in enhancement.items():
        print(f'  {k:24s} {v}')
    print(f'\nTraining: {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}, '
          f'512x512, seed 42, consistency={args.consistency}\n', flush=True)

    set_seed()
    print('Precomputing enhanced images with the optimized parameters...', flush=True)
    splits = build_enhanced_splits(manifest, enhancement)
    train_ds, val_ds, test_ds = splits['train'], splits['val'], splits['test']
    print(f'train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}\n', flush=True)

    set_seed()
    model = MultiTaskNetV2().to(device)
    checkpoint = f'phase6_optimized_{opt_name}_local.pt'
    history, best_val = train_model(
        model, train_ds, val_ds, device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weights=loss_weights, checkpoint_name=checkpoint,
        selection='combined_dice', log_name=f'phase6_final_{opt_name}_history.json',
        consistency=args.consistency)

    print(f'\nBest validation combined Dice: {best_val:.4f}')
    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, checkpoint),
                                     map_location=device))
    test_loader = make_loader(test_ds, args.batch_size, False)
    metrics = evaluate(model, test_loader, device, use_amp=True)

    print('\n' + '-' * 78)
    print('PHASE 6 OPTIMIZED - TEST RESULTS')
    print('-' * 78)
    comparison = {}
    for phase in ('phase2', 'phase4', 'phase5', 'phase5b'):
        payload = load_phase_result(phase)
        if payload:
            comparison[phase] = payload['test_metrics']

    keys = ('diagnosis_auc', 'global_pathology_mean_auc', 'local_pathology_mean_auc',
            'kidney_dice', 'lesion_dice', 'lesion_dice_positive_only',
            'consistency_violation_rate')
    header = f'{"metric":32s} {"Phase 6":>9s}'
    for phase in comparison:
        header += f' {phase.replace("phase", "P"):>9s}'
    print(header)
    print('-' * len(header))
    for key in keys:
        line = f'{key:32s} {metrics[key]:9.4f}'
        for phase in comparison:
            line += f' {comparison[phase].get(key, float("nan")):9.4f}'
        print(line)

    tag = '' if args.save_as is None else f'_{opt_name}'
    plot_training_curves(history, figure_path(f'phase6_final{tag}_curves.png'),
                         f'Phase 6 - optimized configuration ({opt_name}), full fidelity')
    acc = MetricAccumulator()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            with torch.autocast('cuda'):
                out = model(images)
            acc.update(out, batch, device)
    plot_roc(acc.diag_targets, acc.diag_probs, figure_path(f'phase6_final{tag}_roc.png'),
             'Phase 6 optimized - healthy vs pathological ROC')
    lesion_idx = [i for i in range(len(test_ds)) if test_ds[i]['has_lesion'].item() > 0][:3]
    plot_qualitative(model, test_ds, device, figure_path(f'phase6_final{tag}_qualitative.png'),
                     indices=lesion_idx, title=f'Phase 6 optimized ({opt_name}) - predictions')

    save_phase_result(args.save_as or 'phase6_final', {
        'optimizer': opt_name,
        'loss_weights': loss_weights,
        'enhancement': enhancement,
        'search_fitness': best['best_fitness'],
        'hyperparameters': {'epochs': args.epochs, 'batch_size': args.batch_size,
                            'lr': args.lr, 'seed': 42, 'consistency': args.consistency},
        'test_metrics': {k: v for k, v in metrics.items() if not isinstance(v, dict)},
        'history': history,
        'best_val_score': best_val,
    })


if __name__ == '__main__':
    with Tee() as tee:
        main()
    _tag = ''
    if '--save-as' in sys.argv and '--optimizer' in sys.argv:
        _tag = '_' + sys.argv[sys.argv.index('--optimizer') + 1]
    render_console_png(tee.text,
                       os.path.join(FIGURES_DIR, f'phase6_final{_tag}_console.png'),
                       title='python scripts/04_phase6_final.py')
