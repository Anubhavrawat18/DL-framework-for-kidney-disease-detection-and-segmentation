"""Phase 7: ablations.

Three questions this answers, none of which the earlier phases can answer on their own:

  1. Does multi-task training actually help, or would a single-task model do as well?
     -> single-task variants trained with exactly one loss term active.

  2. The Phase 6 contribution has two halves - optimized loss weights and optimized
     enhancement. Which half is doing the work?
     -> optimized weights with plain images, and equal weights with optimized enhancement,
        each isolating one half against the same baseline.

  3. Does the anatomical consistency term help once it is formulated correctly?
     -> already covered by Phase 4 vs Phase 5 vs Phase 5b.

Usage:  python scripts/05_phase7_ablations.py
        python scripts/05_phase7_ablations.py --only diagnosis_only,weights_only
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
from src.reporting import (Tee, figure_path, load_phase_result, plot_training_curves,
                           render_console_png, save_phase_result)
from src.model import MultiTaskNetV2

FULL = ['diagnosis', 'global_pathology', 'local_pathology', 'kidney_seg', 'lesion_seg',
        'consistency']


def ablation_configs(search):
    """Built at runtime because two ablations need Phase 6's optimized parameters."""
    configs = {
        'diagnosis_only': dict(
            label='Single-task: diagnosis only',
            active_terms=['diagnosis'], weights=None, enhancement=None,
            selection='diagnosis_auc', consistency='ratio'),
        'kidney_seg_only': dict(
            label='Single-task: kidney segmentation only',
            active_terms=['kidney_seg'], weights=None, enhancement=None,
            selection='kidney_dice', consistency='ratio'),
        'lesion_seg_only': dict(
            label='Single-task: lesion segmentation only',
            active_terms=['lesion_seg'], weights=None, enhancement=None,
            selection='lesion_dice_positive_only', consistency='ratio'),
    }
    if search:
        winner = search['results'][search['winner']]
        configs['weights_only'] = dict(
            label='Optimized loss weights, no enhancement',
            active_terms=FULL, weights=winner['loss_weights'], enhancement=None,
            selection='combined_dice', consistency='ratio')
        configs['enhancement_only'] = dict(
            label='Optimized enhancement, equal loss weights',
            active_terms=FULL, weights=None, enhancement=winner['enhancement'],
            selection='combined_dice', consistency='ratio')
    else:
        configs['enhancement_fixed_only'] = dict(
            label='Hand-picked enhancement, equal loss weights (no Phase 6 results yet)',
            active_terms=FULL, weights=None, enhancement=DEFAULT_ENHANCEMENT,
            selection='combined_dice', consistency='ratio')
    return configs


def run_ablation(key, cfg, manifest, device, epochs, batch_size, lr):
    print('=' * 78)
    print(f'ABLATION: {cfg["label"]}')
    print('=' * 78)
    print(f'active loss terms : {", ".join(cfg["active_terms"])}')
    print(f'loss weights      : '
          f'{"optimized" if cfg["weights"] else "equal (1.0 each)"}')
    print(f'enhancement       : {cfg["enhancement"] or "none"}')
    print(f'epochs={epochs} batch_size={batch_size} lr={lr} seed=42\n', flush=True)

    set_seed()
    if cfg['enhancement']:
        print('Precomputing enhanced images...', flush=True)
        splits = build_enhanced_splits(manifest, cfg['enhancement'])
        train_ds, val_ds, test_ds = splits['train'], splits['val'], splits['test']
    else:
        train_ds, val_ds, test_ds = (
            URICADSMultiTaskDataset(manifest[manifest['split'] == s])
            for s in ('train', 'val', 'test'))

    set_seed()
    model = MultiTaskNetV2().to(device)
    checkpoint = f'phase7_{key}_local.pt'
    history, best_val = train_model(
        model, train_ds, val_ds, device, epochs=epochs, batch_size=batch_size, lr=lr,
        weights=cfg['weights'], active_terms=cfg['active_terms'],
        checkpoint_name=checkpoint, selection=cfg['selection'],
        log_name=f'phase7_{key}_history.json', consistency=cfg['consistency'])

    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, checkpoint),
                                     map_location=device))
    metrics = evaluate(model, make_loader(test_ds, batch_size, False), device, use_amp=True)

    print(f'\nTest results for "{cfg["label"]}":')
    for key_m in ('diagnosis_auc', 'kidney_dice', 'lesion_dice',
                  'lesion_dice_positive_only', 'consistency_violation_rate'):
        print(f'  {key_m:32s}: {metrics[key_m]:.4f}')
    print(flush=True)

    plot_training_curves(history, figure_path(f'phase7_{key}_curves.png'), cfg['label'])
    del model
    torch.cuda.empty_cache()
    return {'label': cfg['label'],
            'active_terms': cfg['active_terms'],
            'weights': cfg['weights'],
            'enhancement': cfg['enhancement'],
            'test_metrics': {k: v for k, v in metrics.items() if not isinstance(v, dict)},
            'history': history,
            'best_val_score': best_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    device = get_device()
    manifest = load_manifest()
    search = load_phase_result('phase6_search')
    configs = ablation_configs(search)
    if args.only:
        wanted = set(args.only.split(','))
        configs = {k: v for k, v in configs.items() if k in wanted}

    existing = load_phase_result('phase7_ablations') or {}
    results = {k: v for k, v in existing.items() if isinstance(v, dict)}

    for key, cfg in configs.items():
        results[key] = run_ablation(key, cfg, manifest, device, args.epochs,
                                    args.batch_size, args.lr)
        save_phase_result('phase7_ablations', results)

    # multi-task references already trained in earlier phases
    for phase, label in (('phase4', 'Multi-task, no consistency term (Phase 4)'),
                         ('phase5b', 'Multi-task + corrected consistency (Phase 5b)'),
                         ('phase6_final', 'Fully optimized (Phase 6)')):
        payload = load_phase_result(phase)
        if payload:
            results[phase] = {'label': label, 'test_metrics': payload['test_metrics']}

    print('=' * 78)
    print('ABLATION SUMMARY')
    print('=' * 78)
    rows = [(v['label'], v['test_metrics']) for v in results.values() if 'test_metrics' in v]
    print(f'{"configuration":48s} {"AUC":>8s} {"kDice":>8s} {"lDice+":>8s} {"viol":>8s}')
    print('-' * 84)
    for label, m in rows:
        print(f'{label[:48]:48s} {m.get("diagnosis_auc", float("nan")):8.4f} '
              f'{m.get("kidney_dice", float("nan")):8.4f} '
              f'{m.get("lesion_dice_positive_only", float("nan")):8.4f} '
              f'{m.get("consistency_violation_rate", float("nan")):8.4f}')

    save_phase_result('phase7_ablations', results)


if __name__ == '__main__':
    with Tee() as tee:
        main()
    render_console_png(tee.text, os.path.join(FIGURES_DIR, 'phase7_console.png'),
                       title='python scripts/05_phase7_ablations.py', max_lines=140)
