"""Phase 8: 5-fold cross-validation, matching the protocol URI-CADS reports.

The published URI-CADS system reports 0.8741 AUC for healthy/pathological and 0.8190 AUC for
multi-pathological, both from 5-fold cross-validation over the whole dataset. Phases 2-7 here
use a single fixed 70/15/15 split, so their numbers are not directly comparable to those.
This script reruns the chosen configuration under the same 5-fold protocol so that the headline
comparison in the paper is like-for-like, and so that the result comes with a variance estimate
rather than a single point value.

Usage:  python scripts/06_phase8_kfold.py --config optimized
        python scripts/06_phase8_kfold.py --config baseline   # equal weights, no enhancement
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CHECKPOINTS_DIR, FIGURES_DIR, SEED
from src.data import URICADSMultiTaskDataset, load_manifest
from src.engine import evaluate, get_device, make_loader, set_seed, train_model
from src.enhanced_cache import build_enhanced_splits
from src.model import MultiTaskNetV2, MultiTaskNetV3
from src.metrics import best_threshold, classification_metrics
from src.reporting import (Tee, load_phase_result, render_console_png, save_phase_result)


@torch.no_grad()
def _diagnosis_probs(model, dataset, device, batch_size):
    from src.engine import _forward
    model.eval()
    probs, targets = [], []
    for batch in make_loader(dataset, batch_size, False):
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            out = _forward(model, batch['image'].to(device))
        probs.extend(torch.sigmoid(out['diagnosis'].float()).cpu().numpy())
        targets.extend(batch['global_diagnosis'].numpy())
    return np.asarray(targets), np.asarray(probs)

URICADS_PUBLISHED = {'healthy_pathological_auc': 0.8741, 'multi_pathological_auc': 0.8190}

REPORT_KEYS = ('diagnosis_auc', 'diagnosis_auprc', 'diagnosis_accuracy',
               'diagnosis_balanced_accuracy', 'diagnosis_sensitivity',
               'diagnosis_specificity', 'diagnosis_f1',
               'global_pathology_mean_auc', 'local_pathology_mean_auc',
               'kidney_dice', 'lesion_dice', 'lesion_dice_positive_only',
               'consistency_violation_rate')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='optimized', choices=('optimized', 'baseline'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--consistency', default='ratio')
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--augment', action='store_true', help='enable training augmentation')
    ap.add_argument('--scheduler', default='none', choices=('none', 'cosine'))
    ap.add_argument('--diagnosis-loss', default='bce',
                    choices=('bce', 'balanced', 'focal'))
    ap.add_argument('--selection', default='combined_dice')
    ap.add_argument('--tag', default=None, help='result key suffix; defaults to --config')
    ap.add_argument('--arch', default='v2', choices=('v2', 'v3'),
                    help='v3 adds lesion-guided pooling to the local-pathology head')
    ap.add_argument('--ensemble-seeds', type=int, default=1,
                    help='models trained per fold; >1 averages their predicted probabilities')
    args = ap.parse_args()

    diag_cfg = {'bce': None,
                'balanced': {'type': 'weighted_bce', 'pos_weight': 'balanced'},
                'focal': {'type': 'focal', 'alpha': 0.25, 'gamma': 2.0}}[args.diagnosis_loss]
    tag = args.tag or args.config
    arch_cls = {'v2': MultiTaskNetV2, 'v3': MultiTaskNetV3}[args.arch]

    device = get_device()
    manifest = load_manifest().reset_index(drop=True)

    loss_weights, enhancement = None, None
    if args.config == 'optimized':
        search = load_phase_result('phase6_search')
        if not search:
            raise SystemExit('Run scripts/03_phase6_search.py first, or use --config baseline.')
        winner = search['results'][search['winner']]
        loss_weights, enhancement = winner['loss_weights'], winner['enhancement']

    print('=' * 78)
    print(f'PHASE 8 - {args.folds}-FOLD CROSS-VALIDATION ({args.config} configuration)')
    print('=' * 78)
    print(f'Protocol matched to URI-CADS: stratified {args.folds}-fold over all '
          f'{len(manifest)} images.')
    print(f'loss weights : {"optimized" if loss_weights else "equal (1.0 each)"}')
    print(f'enhancement  : {enhancement or "none"}')
    print(f'epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} seed={SEED}')
    print(f'architecture={args.arch}  models per fold={args.ensemble_seeds}')
    print(f'augment={args.augment} scheduler={args.scheduler} '
          f'diagnosis_loss={args.diagnosis_loss} selection={args.selection}\n',
          flush=True)

    # Enhancement is fixed across folds, so it is computed once over the whole dataset.
    enhanced_images = None
    if enhancement:
        from src.enhanced_cache import precompute_enhanced
        print('Precomputing enhanced images for the full dataset...', flush=True)
        enhanced_images = precompute_enhanced(manifest['image_path'].tolist(), enhancement,
                                              desc='all folds')

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    labels = manifest['global_diagnosis'].values
    fold_metrics, tuned_metrics, fold_predictions = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(manifest, labels), start=1):
        # carve a validation slice out of the training portion for checkpoint selection
        rng = np.random.default_rng(SEED + fold)
        shuffled = rng.permutation(train_idx)
        n_val = int(0.15 * len(shuffled))
        val_idx, tr_idx = shuffled[:n_val], shuffled[n_val:]

        print('-' * 78)
        print(f'FOLD {fold}/{args.folds} | train={len(tr_idx)} val={len(val_idx)} '
              f'test={len(test_idx)}')
        print('-' * 78, flush=True)

        def make_ds(indices, augment=False):
            sub = manifest.iloc[indices]
            if enhanced_images is None:
                return URICADSMultiTaskDataset(sub, augment=augment, seed=SEED + fold)
            from src.enhanced_cache import PrecomputedEnhancedDataset
            return PrecomputedEnhancedDataset(sub, enhanced_images[indices],
                                              augment=augment, seed=SEED + fold)

        train_ds = make_ds(tr_idx, augment=args.augment)
        val_ds, test_ds = make_ds(val_idx), make_ds(test_idx)

        member_val, member_test = [], []
        for member in range(args.ensemble_seeds):
            member_seed = SEED + fold + member * 1000
            set_seed(member_seed)
            model = arch_cls().to(device)
            checkpoint = f'phase8_{tag}_fold{fold}_m{member}.pt'
            train_model(model, train_ds, val_ds, device, epochs=args.epochs,
                        batch_size=args.batch_size, lr=args.lr, weights=loss_weights,
                        checkpoint_name=checkpoint, selection=args.selection,
                        num_workers=args.num_workers,
                        log_name=f'phase8_{tag}_fold{fold}_m{member}_history.json',
                        consistency=args.consistency, diagnosis_loss_cfg=diag_cfg,
                        scheduler=args.scheduler, verbose=args.ensemble_seeds == 1)
            model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, checkpoint),
                                             map_location=device))
            if args.ensemble_seeds > 1:
                vt_m, vp_m = _diagnosis_probs(model, val_ds, device, args.batch_size)
                tt_m, tp_m = _diagnosis_probs(model, test_ds, device, args.batch_size)
                member_val.append(vp_m)
                member_test.append(tp_m)
                print(f'  fold {fold} member {member + 1}/{args.ensemble_seeds} '
                      f'(seed {member_seed}) trained', flush=True)
            if member < args.ensemble_seeds - 1:
                del model
                torch.cuda.empty_cache()

        # Segmentation metrics come from the last member; the ensemble applies to the
        # diagnosis probabilities, which is where averaging is meaningful.
        metrics = evaluate(model, make_loader(test_ds, args.batch_size, False), device,
                           use_amp=True)

        # Operating point chosen on this fold's validation slice, applied to this fold's test
        # slice - never on test itself. The ensemble override must happen BEFORE the metrics
        # are snapshotted into fold_metrics, or the summary reports one member's AUC while the
        # per-fold line reports the ensemble's.
        if args.ensemble_seeds > 1:
            v_t, _ = _diagnosis_probs(model, val_ds, device, args.batch_size)
            t_t, _ = _diagnosis_probs(model, test_ds, device, args.batch_size)
            v_p = np.mean(member_val, axis=0)
            t_p = np.mean(member_test, axis=0)
            from src.metrics import safe_auc
            metrics['diagnosis_auc'] = safe_auc(t_t, t_p)
            metrics['diagnosis_auprc'] = classification_metrics(t_t, t_p)['auprc']
        else:
            v_t, v_p = _diagnosis_probs(model, val_ds, device, args.batch_size)
            t_t, t_p = _diagnosis_probs(model, test_ds, device, args.batch_size)

        fold_metrics.append({k: v for k, v in metrics.items() if not isinstance(v, dict)})
        thr = best_threshold(v_t, v_p, 'balanced_accuracy')
        tuned = classification_metrics(t_t, t_p, threshold=thr)
        tuned_metrics.append({k: v for k, v in tuned.items() if k != 'confusion'})
        fold_predictions.append({'test_index': test_idx.tolist(),
                                 'targets': t_t.tolist(), 'probs': t_p.tolist(),
                                 'threshold': thr})
        print(f'Fold {fold} tuned (thr={thr:.3f}): '
              f'bal_acc={tuned["balanced_accuracy"]:.4f} '
              f'sens={tuned["sensitivity"]:.4f} spec={tuned["specificity"]:.4f}')

        print(f'\nFold {fold} test: ' + ' | '.join(
            f'{k}={metrics[k]:.4f}' for k in ('diagnosis_auc', 'kidney_dice',
                                              'lesion_dice_positive_only')))
        print(flush=True)
        del model
        torch.cuda.empty_cache()

    print('=' * 78)
    print(f'{args.folds}-FOLD CROSS-VALIDATION SUMMARY ({args.config})')
    print('=' * 78)
    summary = {}
    print(f'{"metric":34s} {"mean":>9s} {"std":>9s} ' +
          ' '.join(f'{"f"+str(i+1):>8s}' for i in range(args.folds)))
    print('-' * (34 + 20 + 9 * args.folds))
    for key in REPORT_KEYS:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[key] = {'mean': float(np.nanmean(values)), 'std': float(np.nanstd(values)),
                        'folds': values.tolist()}
        print(f'{key:34s} {np.nanmean(values):9.4f} {np.nanstd(values):9.4f} ' +
              ' '.join(f'{v:8.4f}' for v in values))

    print('\n' + '=' * 78)
    print(f'TUNED-THRESHOLD SUMMARY ({args.folds} folds, threshold chosen per fold on validation)')
    print('=' * 78)
    tuned_summary = {}
    for key in ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1'):
        vals = np.array([t[key] for t in tuned_metrics], dtype=float)
        tuned_summary[key] = {'mean': float(np.nanmean(vals)), 'std': float(np.nanstd(vals)),
                              'folds': vals.tolist()}
        print(f'{key:34s} {np.nanmean(vals):9.4f} {np.nanstd(vals):9.4f} ' +
              ' '.join(f'{v:8.4f}' for v in vals))

    print('\n' + '=' * 78)
    print('COMPARISON WITH PUBLISHED URI-CADS RESULTS (same 5-fold protocol)')
    print('=' * 78)
    ours = summary['diagnosis_auc']['mean']
    theirs = URICADS_PUBLISHED['healthy_pathological_auc']
    print(f'  Healthy/pathological AUC   ours: {ours:.4f} +/- '
          f'{summary["diagnosis_auc"]["std"]:.4f}   URI-CADS: {theirs:.4f}   '
          f'delta: {ours - theirs:+.4f}')
    ours_multi = summary['local_pathology_mean_auc']['mean']
    theirs_multi = URICADS_PUBLISHED['multi_pathological_auc']
    print(f'  Multi-pathological AUC     ours: {ours_multi:.4f} +/- '
          f'{summary["local_pathology_mean_auc"]["std"]:.4f}   URI-CADS: {theirs_multi:.4f}   '
          f'delta: {ours_multi - theirs_multi:+.4f}')
    print('\nNote: URI-CADS uses a dual Mask R-CNN + Faster R-CNN pipeline with a ResNet-50 '
          'backbone. The model here is a single ResNet-34 encoder with lightweight decoders.')

    save_phase_result(f'phase8_kfold_{tag}', {
        'config': args.config,
        'arch': args.arch,
        'ensemble_seeds': args.ensemble_seeds,
        'folds': args.folds,
        'loss_weights': loss_weights,
        'enhancement': enhancement,
        'hyperparameters': {'epochs': args.epochs, 'batch_size': args.batch_size,
                            'lr': args.lr, 'seed': SEED},
        'summary': summary,
        'tuned_summary': tuned_summary,
        'per_fold': fold_metrics,
        'per_fold_tuned': tuned_metrics,
        'fold_predictions': fold_predictions,
        'uricads_published': URICADS_PUBLISHED,
    })


if __name__ == '__main__':
    with Tee() as tee:
        main()
    import sys as _sys
    cfg = 'optimized' if '--config' not in _sys.argv else _sys.argv[_sys.argv.index('--config') + 1]
    render_console_png(tee.text, os.path.join(FIGURES_DIR, f'phase8_kfold_{cfg}_console.png'),
                       title=f'python scripts/06_phase8_kfold.py --config {cfg}',
                       max_lines=140)
