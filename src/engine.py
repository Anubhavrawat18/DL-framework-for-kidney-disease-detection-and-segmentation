"""Training / evaluation loops shared by every phase."""
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CHECKPOINTS_DIR, LOGS_DIR, SEED
from .losses import compute_loss
from .metrics import MetricAccumulator


def get_device(require_cuda=True):
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError('CUDA not available - refusing to train on CPU.')
        return torch.device('cpu')
    return torch.device('cuda')


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    # Windows spawns workers by pickling the dataset. A dataset holding precomputed images in
    # RAM is far too large to push through that pipe, and gains nothing from workers anyway
    # since its images need no per-item decoding.
    if not getattr(dataset, 'supports_workers', True):
        num_workers = 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)


def _forward(model, images):
    """Normalizes both architectures to the V2 dict interface."""
    out = model(images)
    if isinstance(out, dict):
        return out
    diagnosis, global_pathology, seg = out
    return {'diagnosis': diagnosis, 'global_pathology': global_pathology,
            'kidney_seg': seg, 'local_pathology': None, 'lesion_seg': None}


def run_epoch(model, loader, device, optimizer=None, weights=None, scaler=None,
              active_terms=None, consistency='absolute', diagnosis_loss=None):
    """active_terms: subset of loss terms to use (Phase 2/3 has no lesion/local heads)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    totals, n = {}, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=scaler is not None):
                outputs = _forward(model, images)
                if active_terms is not None:
                    loss, breakdown = _partial_loss(outputs, batch, device, active_terms,
                                                    weights, consistency, diagnosis_loss)
                else:
                    loss, breakdown = compute_loss(outputs, batch, device, weights, consistency,
                                                   diagnosis_loss)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            totals['loss'] = totals.get('loss', 0.0) + loss.item()
            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

    return {k: v / n for k, v in totals.items()}


def _partial_loss(outputs, batch, device, active_terms, weights, consistency='absolute',
                  diagnosis_loss=None):
    from .losses import CONSISTENCY_FORMS, bce_logits, dice_loss
    diag_loss_fn = diagnosis_loss or bce_logits
    kidney_target = batch['kidney_mask'].to(device)
    available = {
        'diagnosis': lambda: diag_loss_fn(outputs['diagnosis'],
                                          batch['global_diagnosis'].to(device)),
        'global_pathology': lambda: bce_logits(outputs['global_pathology'],
                                               batch['global_pathologies'].to(device)),
        'local_pathology': lambda: bce_logits(outputs['local_pathology'],
                                              batch['local_pathologies'].to(device)),
        'kidney_seg': lambda: dice_loss(outputs['kidney_seg'], kidney_target),
        'lesion_seg': lambda: dice_loss(outputs['lesion_seg'],
                                        batch['lesion_mask'].to(device)),
        'consistency': lambda: CONSISTENCY_FORMS[consistency](outputs['lesion_seg'],
                                                             kidney_target),
    }
    terms = {t: available[t]() for t in active_terms}
    if weights is None:
        w = {t: 1.0 for t in active_terms}
    elif isinstance(weights, dict):
        w = {t: float(weights.get(t, 1.0)) for t in active_terms}
    else:
        from .losses import LOSS_TERMS
        full = dict(zip(LOSS_TERMS, weights))
        w = {t: float(full[t]) for t in active_terms}
    total = sum(w[t] * terms[t] for t in active_terms)
    return total, {t: v.item() for t, v in terms.items()}


@torch.no_grad()
def _fill_absent_heads(outputs, batch, device):
    """Substitutes a strongly negative constant for any head the architecture does not have.

    A disabled head must not silently contribute a real-looking score. A constant logit yields
    an all-zero prediction and a chance-level AUC, which is the honest reading of "this model
    does not predict that", and it keeps the metric accumulator's shapes valid.
    """
    if all(v is not None for v in outputs.values()):
        return outputs
    filled = dict(outputs)
    n = batch['kidney_mask'].shape[0]
    h, w = batch['kidney_mask'].shape[-2:]
    shapes = {'diagnosis': (n,), 'global_pathology': (n, batch['global_pathologies'].shape[1]),
              'local_pathology': (n, batch['local_pathologies'].shape[1]),
              'kidney_seg': (n, h, w), 'lesion_seg': (n, h, w)}
    for key, shape in shapes.items():
        if filled.get(key) is None:
            filled[key] = torch.full(shape, -20.0, device=device)
    return filled


def evaluate(model, loader, device, use_amp=False):
    model.eval()
    acc = MetricAccumulator()
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        with torch.autocast('cuda', enabled=use_amp):
            outputs = _forward(model, images)
        acc.update(_fill_absent_heads(outputs, batch, device), batch, device)
    return acc.compute()


def train_model(model, train_ds, val_ds, device, *, epochs, batch_size=4, lr=1e-4,
                weights=None, active_terms=None, checkpoint_name=None, num_workers=0,
                use_amp=True, log_name=None, selection='combined_dice', verbose=True,
                consistency='absolute', diagnosis_loss_cfg=None, scheduler='none',
                weight_decay=0.0):
    """Trains, checkpointing on the best validation score. Returns (history, best_score)."""
    train_loader = make_loader(train_ds, batch_size, True, num_workers)
    val_loader = make_loader(val_ds, batch_size, False, num_workers)
    if weight_decay:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    from .losses import make_diagnosis_loss
    diagnosis_loss = make_diagnosis_loss(diagnosis_loss_cfg)
    # Cosine annealing with a short warm-up: the flat 1e-4 used in Phases 2-6 leaves the model
    # still moving at epoch 15, so the final weights are wherever the last step happened to land.
    lr_sched = None
    if scheduler == 'cosine':
        warmup = max(1, int(0.1 * epochs))

        def lr_lambda(e):
            if e < warmup:
                return (e + 1) / warmup
            progress = (e - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name) if checkpoint_name else None
    history, best_score = [], -float('inf')

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)
        train_stats = run_epoch(model, train_loader, device, optimizer=optimizer,
                                weights=weights, scaler=scaler, active_terms=active_terms,
                                consistency=consistency, diagnosis_loss=diagnosis_loss)
        if lr_sched is not None:
            lr_sched.step()
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        elapsed = time.time() - t0

        if selection == 'combined_dice':
            score = (val_metrics['kidney_dice'] + val_metrics['lesion_dice']) / 2
        elif selection == 'kidney_dice':
            score = val_metrics['kidney_dice']
        else:
            score = val_metrics[selection]

        row = {'epoch': epoch, 'train_loss': train_stats['loss'],
               'elapsed_s': elapsed, 'selection_score': score, **
               {f'val_{k}': v for k, v in val_metrics.items()
                if not isinstance(v, dict)},
               **{f'train_{k}': v for k, v in train_stats.items() if k != 'loss'}}
        history.append(row)

        if verbose:
            print(f'Epoch {epoch:2d}/{epochs} | train_loss={train_stats["loss"]:.4f} '
                  f'| val_auc={val_metrics["diagnosis_auc"]:.4f} '
                  f'kidney_dice={val_metrics["kidney_dice"]:.4f} '
                  f'lesion_dice={val_metrics["lesion_dice"]:.4f} '
                  f'| {elapsed:.1f}s', flush=True)

        if score > best_score:
            best_score = score
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
                if verbose:
                    print(f'  -> new best ({selection}={score:.4f}); saved {checkpoint_name}',
                          flush=True)

    if log_name:
        with open(os.path.join(LOGS_DIR, log_name), 'w') as fh:
            json.dump(history, fh, indent=2)

    return history, best_score
