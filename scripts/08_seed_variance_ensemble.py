"""Seed variance and ensembling, measured on the fixed held-out split.

Two questions this answers together, both of which the project needs before any classification
claim can be defended:

  1. How much does test performance move from seed alone? Every comparison made so far used one
     seed per configuration, so without this number a 0.03 AUC gap cannot be interpreted. The
     bootstrap gives sampling error on a fixed model; this gives training variance across models.

  2. Does averaging the ensemble's predictions beat its members? Ensembling is the one classical
     accuracy lever that is nearly free here, and unlike the k-fold models these are all trained
     on the same split, so they can be averaged and scored on a genuinely held-out test set.

Each run uses the Phase 5b recipe and differs only in the seed controlling initialization and
batch order. The decision threshold is chosen on validation, never on test.

Usage:  python scripts/08_seed_variance_ensemble.py --seeds 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import CHECKPOINTS_DIR, FIGURES_DIR
from src.data import URICADSMultiTaskDataset, load_manifest
from src.engine import (_forward, evaluate, get_device, make_loader, set_seed, train_model)
from src.metrics import best_threshold, classification_metrics, safe_auc
from src.model import MultiTaskNetV2
from src.reporting import Tee, render_console_png, save_phase_result

FULL = ['diagnosis', 'global_pathology', 'local_pathology', 'kidney_seg', 'lesion_seg',
        'consistency']


@torch.no_grad()
def predict(model, dataset, device, batch_size=4):
    model.eval()
    probs, targets = [], []
    for batch in make_loader(dataset, batch_size, False):
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            out = _forward(model, batch['image'].to(device))
        probs.extend(torch.sigmoid(out['diagnosis'].float()).cpu().numpy())
        targets.extend(batch['global_diagnosis'].numpy())
    return np.asarray(targets), np.asarray(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=6)
    args = ap.parse_args()

    device = get_device()
    manifest = load_manifest()
    val_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'val'])
    test_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'test'])

    print('=' * 78)
    print(f'SEED VARIANCE AND ENSEMBLING - {args.seeds} runs of the same recipe')
    print('=' * 78)
    print('Recipe: Phase 5b (equal weights, ratio consistency, no augmentation),')
    print(f'        {args.epochs} epochs, batch {args.batch_size}, Dice-selected checkpoint.')
    print('Only the seed differs between runs.\n', flush=True)

    val_probs, test_probs, targets_v, targets_t = [], [], None, None
    per_seed = []

    for i in range(args.seeds):
        seed = 42 + i * 101
        print('-' * 78)
        print(f'RUN {i + 1}/{args.seeds}  (seed {seed})')
        print('-' * 78, flush=True)

        set_seed(seed)
        train_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'train'])
        model = MultiTaskNetV2().to(device)
        checkpoint = f'seed{seed}_local.pt'
        train_model(model, train_ds, val_ds, device, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr, active_terms=FULL,
                    consistency='ratio', checkpoint_name=checkpoint,
                    num_workers=args.num_workers, selection='combined_dice',
                    log_name=f'seed{seed}_history.json', verbose=False)

        model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, checkpoint),
                                         map_location=device))
        metrics = evaluate(model, make_loader(test_ds, args.batch_size, False), device,
                           use_amp=True)
        targets_v, vp = predict(model, val_ds, device, args.batch_size)
        targets_t, tp = predict(model, test_ds, device, args.batch_size)
        val_probs.append(vp)
        test_probs.append(tp)

        thr = best_threshold(targets_v, vp, 'balanced_accuracy')
        tuned = classification_metrics(targets_t, tp, threshold=thr)
        per_seed.append({
            'seed': seed,
            'auc': metrics['diagnosis_auc'],
            'kidney_dice': metrics['kidney_dice'],
            'lesion_dice_positive_only': metrics['lesion_dice_positive_only'],
            'threshold': thr,
            'balanced_accuracy': tuned['balanced_accuracy'],
            'sensitivity': tuned['sensitivity'],
            'specificity': tuned['specificity'],
        })
        print(f'  AUC={metrics["diagnosis_auc"]:.4f}  kidney_dice={metrics["kidney_dice"]:.4f}  '
              f'tuned bal_acc={tuned["balanced_accuracy"]:.4f}', flush=True)
        del model
        torch.cuda.empty_cache()

    aucs = np.array([r['auc'] for r in per_seed])
    print('\n' + '=' * 78)
    print('SEED VARIANCE (same recipe, different seeds)')
    print('=' * 78)
    for key in ('auc', 'kidney_dice', 'lesion_dice_positive_only', 'balanced_accuracy'):
        vals = np.array([r[key] for r in per_seed], dtype=float)
        print(f'  {key:28s} mean={vals.mean():.4f}  std={vals.std():.4f}  '
              f'min={vals.min():.4f}  max={vals.max():.4f}  spread={vals.max()-vals.min():.4f}')

    ens_val = np.mean(val_probs, axis=0)
    ens_test = np.mean(test_probs, axis=0)
    ens_auc = safe_auc(targets_t, ens_test)
    ens_thr = best_threshold(targets_v, ens_val, 'balanced_accuracy')
    ens_tuned = classification_metrics(targets_t, ens_test, threshold=ens_thr)

    print('\n' + '=' * 78)
    print(f'ENSEMBLE OF {args.seeds} MODELS (mean predicted probability)')
    print('=' * 78)
    print(f'  {"metric":28s} {"members mean":>13s} {"members best":>13s} {"ensemble":>10s}')
    print('-' * 68)
    print(f'  {"AUC":28s} {aucs.mean():13.4f} {aucs.max():13.4f} {ens_auc:10.4f}')
    for key in ('balanced_accuracy', 'sensitivity', 'specificity'):
        vals = np.array([r[key] for r in per_seed], dtype=float)
        print(f'  {key:28s} {vals.mean():13.4f} {vals.max():13.4f} {ens_tuned[key]:10.4f}')
    print(f'\n  Ensemble threshold (validation): {ens_thr:.3f}')
    print(f'  Ensemble gain over member mean : {ens_auc - aucs.mean():+.4f} AUC')
    print(f'  Ensemble gain over best member : {ens_auc - aucs.max():+.4f} AUC')

    print('\nInterpretation: the seed spread above is the floor for any single-split comparison.')
    print('Differences between configurations smaller than that spread carry no information.')

    save_phase_result('seed_variance_ensemble', {
        'seeds': args.seeds,
        'epochs': args.epochs,
        'per_seed': per_seed,
        'seed_auc_mean': float(aucs.mean()),
        'seed_auc_std': float(aucs.std()),
        'seed_auc_spread': float(aucs.max() - aucs.min()),
        'ensemble': {
            'auc': float(ens_auc), 'threshold': float(ens_thr),
            **{k: v for k, v in ens_tuned.items() if k != 'confusion'},
            'confusion': ens_tuned['confusion'],
        },
        'test_probs': [p.tolist() for p in test_probs],
        'test_targets': targets_t.tolist(),
    })


if __name__ == '__main__':
    with Tee() as tee:
        main()
    render_console_png(tee.text, os.path.join(FIGURES_DIR, 'seed_variance_console.png'),
                       title='python scripts/08_seed_variance_ensemble.py', max_lines=120)
