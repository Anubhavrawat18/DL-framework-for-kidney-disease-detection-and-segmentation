"""Classification-first experiments.

Phases 2-6 optimised a fitness that was half segmentation, and every checkpoint was selected on
Dice. Diagnosis AUC consequently sat at ~0.90 across seven configurations and never moved. This
script targets classification directly, using the levers that were never tried:

  * data augmentation (none was used anywhere in Phases 2-6)
  * a cosine learning-rate schedule with warm-up, and a longer budget
  * class-imbalance handling, since healthy is only 23% of the data
  * checkpoint selection on validation AUC rather than Dice
  * a decision threshold chosen on validation instead of assumed to be 0.5

Every variant keeps the Phase 5b architecture and the corrected ratio consistency term, so the
only things changing are the ones named above.

Usage:  python scripts/07_classification.py --variant aug_sched_balanced
        python scripts/07_classification.py --variant all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import CHECKPOINTS_DIR, FIGURES_DIR
from src.data import (GEOMETRIC_AUGMENTATION, MILD_AUGMENTATION,
                      URICADSMultiTaskDataset, load_manifest)
from src.engine import evaluate, get_device, make_loader, set_seed, train_model, _forward
from src.metrics import best_threshold, classification_metrics
from src.model import MultiTaskNetV2
from src.reporting import (Tee, figure_path, plot_training_curves, render_console_png,
                           save_phase_result)

FULL = ['diagnosis', 'global_pathology', 'local_pathology', 'kidney_seg', 'lesion_seg',
        'consistency']

VARIANTS = {
    # Control: changes only the checkpoint-selection metric (Dice -> AUC) against Phase 5b, so
    # the effect of selection is separated from the effect of augmentation.
    'control': dict(
        label='No augmentation, AUC-selected checkpoint',
        augment=False, augment_params=None, epochs=15, scheduler='none',
        diagnosis_loss_cfg=None, selection='diagnosis_auc'),
    'geo': dict(
        label='Geometric augmentation only',
        augment=True, augment_params=GEOMETRIC_AUGMENTATION, epochs=15, scheduler='none',
        diagnosis_loss_cfg=None, selection='combined_dice'),
    'geo_sched': dict(
        label='Geometric augmentation + cosine schedule, 25 epochs',
        augment=True, augment_params=GEOMETRIC_AUGMENTATION, epochs=25, scheduler='cosine',
        diagnosis_loss_cfg=None, selection='combined_dice'),
    'geo_sched_balanced': dict(
        label='Geometric + schedule + class-balanced BCE',
        augment=True, augment_params=GEOMETRIC_AUGMENTATION, epochs=25, scheduler='cosine',
        diagnosis_loss_cfg={'type': 'weighted_bce', 'pos_weight': 'balanced'},
        selection='combined_dice'),
    'geo_sched_focal': dict(
        label='Geometric + schedule + focal loss',
        augment=True, augment_params=GEOMETRIC_AUGMENTATION, epochs=25, scheduler='cosine',
        diagnosis_loss_cfg={'type': 'focal', 'alpha': 0.25, 'gamma': 2.0},
        selection='combined_dice'),
    'mild': dict(
        label='Geometric + mild intensity jitter + schedule',
        augment=True, augment_params=MILD_AUGMENTATION, epochs=25, scheduler='cosine',
        diagnosis_loss_cfg=None, selection='combined_dice'),
}

CLS_KEYS = ('diagnosis_auc', 'diagnosis_auprc', 'diagnosis_accuracy',
            'diagnosis_balanced_accuracy', 'diagnosis_sensitivity', 'diagnosis_specificity',
            'diagnosis_f1')


@torch.no_grad()
def diagnosis_probs(model, dataset, device, batch_size=4, num_workers=0):
    model.eval()
    probs, targets = [], []
    for batch in make_loader(dataset, batch_size, False, num_workers):
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            out = _forward(model, batch['image'].to(device))
        probs.extend(torch.sigmoid(out['diagnosis'].float()).cpu().numpy())
        targets.extend(batch['global_diagnosis'].numpy())
    return np.asarray(targets), np.asarray(probs)


def run_variant(key, cfg, manifest, device, batch_size, lr, num_workers):
    print('=' * 78)
    print(f'CLASSIFICATION VARIANT: {cfg["label"]}')
    print('=' * 78)
    ap_ = cfg.get('augment_params') or {}
    print(f'augment={cfg["augment"]}  epochs={cfg["epochs"]}  scheduler={cfg["scheduler"]}')
    if cfg['augment']:
        print(f'  geometric: hflip={ap_.get("hflip_prob")} rot={ap_.get("rotate_deg")} '
              f'scale={ap_.get("scale_jitter")} shift={ap_.get("translate_frac")}')
        print(f'  intensity: brightness={ap_.get("brightness")} contrast={ap_.get("contrast")} '
              f'gamma={ap_.get("gamma_jitter")}')
    print(f'diagnosis loss={cfg["diagnosis_loss_cfg"] or "plain BCE"}')
    print('checkpoint selected on validation ' + cfg.get('selection', 'combined_dice') + '\n', flush=True)

    set_seed()
    train_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'train'],
                                       augment=cfg['augment'],
                                       augment_params=cfg.get('augment_params'), seed=42)
    val_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'val'])
    test_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'test'])

    set_seed()
    model = MultiTaskNetV2().to(device)
    checkpoint = f'cls_{key}_local.pt'
    history, best_val = train_model(
        model, train_ds, val_ds, device, epochs=cfg['epochs'], batch_size=batch_size, lr=lr,
        active_terms=FULL, consistency='ratio', checkpoint_name=checkpoint,
        num_workers=num_workers, selection=cfg.get('selection', 'combined_dice'),
        log_name=f'cls_{key}_history.json',
        diagnosis_loss_cfg=cfg['diagnosis_loss_cfg'], scheduler=cfg['scheduler'])

    model.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, checkpoint),
                                     map_location=device))
    metrics = evaluate(model, make_loader(test_ds, batch_size, False), device, use_amp=True)

    # The operating point is chosen on validation only, then applied unchanged to test.
    v_targets, v_probs = diagnosis_probs(model, val_ds, device, batch_size)
    t_targets, t_probs = diagnosis_probs(model, test_ds, device, batch_size)
    thr = best_threshold(v_targets, v_probs, 'balanced_accuracy')
    tuned = classification_metrics(t_targets, t_probs, threshold=thr)

    print(f'\nBest validation selection score: {best_val:.4f}')
    print('-' * 78)
    print(f'TEST RESULTS - {cfg["label"]}')
    print('-' * 78)
    for k in CLS_KEYS:
        print(f'  {k:32s}: {metrics[k]:.4f}')
    print(f'  {"kidney_dice":32s}: {metrics["kidney_dice"]:.4f}')
    print(f'  {"lesion_dice_positive_only":32s}: {metrics["lesion_dice_positive_only"]:.4f}')
    print(f'\n  Threshold tuned on validation  : {thr:.3f}')
    for k in ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1'):
        print(f'  tuned {k:26s}: {tuned[k]:.4f}')
    print(f'  tuned confusion                 : {tuned["confusion"]}', flush=True)

    plot_training_curves(history, figure_path(f'cls_{key}_curves.png'), cfg['label'])
    payload = {
        'label': cfg['label'],
        'config': {k: v for k, v in cfg.items() if k != 'label'},
        'hyperparameters': {'batch_size': batch_size, 'lr': lr, 'seed': 42,
                            'selection': cfg.get('selection', 'combined_dice')},
        'test_metrics': {k: v for k, v in metrics.items() if not isinstance(v, dict)},
        'confusion_at_0.5': metrics.get('diagnosis_confusion'),
        'tuned_threshold': thr,
        'tuned_metrics': {k: v for k, v in tuned.items() if k != 'confusion'},
        'tuned_confusion': tuned['confusion'],
        'history': history,
        'best_val_auc': best_val,
    }
    save_phase_result(f'classification_{key}', payload)
    del model
    torch.cuda.empty_cache()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='all')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=6)
    args = ap.parse_args()

    device = get_device()
    manifest = load_manifest()
    keys = list(VARIANTS) if args.variant == 'all' else args.variant.split(',')

    results = {}
    for key in keys:
        results[key] = run_variant(key, VARIANTS[key], manifest, device,
                                   args.batch_size, args.lr, args.num_workers)

    print('=' * 78)
    print('CLASSIFICATION SUMMARY (test set)')
    print('=' * 78)
    print(f'{"variant":42s} {"AUC":>8s} {"bal.acc":>8s} {"sens":>7s} {"spec":>7s}')
    print('-' * 76)
    print(f'{"Phase 5b reference (no aug, no schedule)":42s} {0.9014:8.4f} {0.8335:8.4f} '
          f'{0.8435:7.4f} {0.8235:7.4f}')
    for key, p in results.items():
        t = p['tuned_metrics']
        print(f'{p["label"][:42]:42s} {p["test_metrics"]["diagnosis_auc"]:8.4f} '
              f'{t["balanced_accuracy"]:8.4f} {t["sensitivity"]:7.4f} {t["specificity"]:7.4f}')
    print('\nAUC is threshold-free; the other three use the validation-tuned threshold.')


if __name__ == '__main__':
    with Tee() as tee:
        main()
    render_console_png(tee.text, os.path.join(FIGURES_DIR, 'classification_console.png'),
                       title='python scripts/07_classification.py', max_lines=140)
