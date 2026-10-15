"""Phase 6: the search space, the fitness function, and a resumable evaluation driver.

DESIGN NOTES (these were open decisions in the project memory; here is what was settled and why)

1. Fitness function
   A single scalar to MAXIMIZE on the validation split:

       F = 0.30*AUC_diagnosis
         + 0.15*AUC_local_pathology
         + 0.25*Dice_kidney
         + 0.25*Dice_lesion_positive_only
         + 0.05*(1 - consistency_violation_rate)

   All five components live in [0, 1] and are "higher is better", so the weighted sum needs
   no rescaling. Rationale for the split: healthy/pathological AUC is the headline clinical
   number and the one URI-CADS reports, so it carries the largest single share; the two
   segmentation tasks are weighted equally and together dominate (0.50) because segmentation
   quality is what the multi-task design is supposed to buy; local pathology and anatomical
   consistency are secondary objectives that should shape ties rather than drive the search.

   CRITICAL: lesion Dice enters as `lesion_dice_positive_only`, NOT the raw mean. About a
   third of images contain no lesion, and on those a correct all-zeros prediction scores ~1.0.
   Optimizing the raw mean would reward a degenerate "predict nothing" model. Restricting to
   lesion-bearing images closes that loophole.

2. Search fidelity
   Each fitness evaluation trains a model from scratch, so a full-fidelity evaluation is far
   too expensive to run a few hundred times. The search therefore runs at reduced fidelity
   (smaller images, fewer epochs, cheap denoiser) and the single best configuration per
   optimizer is then retrained at full fidelity. This low-fidelity-proxy pattern is standard
   in hyper-parameter optimization; what matters is that the proxy RANKS configurations
   consistently with full fidelity, not that it reproduces absolute values.

3. Reproducibility / resumability
   Every evaluation is cached to disk keyed by the rounded parameter vector, so an interrupted
   run resumes without repeating GPU work, and every evaluation is appended to a JSONL trail
   for the paper's audit record.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

from .config import LOGS_DIR, SEED
from .engine import evaluate, make_loader, set_seed, train_model
from .enhanced_cache import EnhancementPool, build_enhanced_splits
from .losses import LOSS_TERMS
from .model import MultiTaskNetV2

# --- search space: 6 loss weights + 4 enhancement parameters -----------------------------
PARAM_NAMES = list(LOSS_TERMS) + [
    'clahe_clip_limit', 'clahe_tile_grid_size', 'denoise_h', 'gamma']

LOWER_BOUNDS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  1.0,  4.0,  0.0, 0.5])
UPPER_BOUNDS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0,  5.0, 16.0, 20.0, 2.0])

FITNESS_WEIGHTS = {
    'diagnosis_auc': 0.30,
    'local_pathology_mean_auc': 0.15,
    'kidney_dice': 0.25,
    'lesion_dice_positive_only': 0.25,
    'consistency': 0.05,
}

# The profile above puts half the weight on segmentation, so the search correctly concluded
# that starving the classification heads was a good trade. When classification accuracy is the
# objective, that is the wrong instruction. The 'classification' profile below makes diagnosis
# dominant and keeps the segmentation heads only as auxiliary supervision.
FITNESS_PROFILES = {
    'balanced': FITNESS_WEIGHTS,
    'classification': {
        'diagnosis_auc': 0.45,
        'diagnosis_balanced_accuracy': 0.25,
        'local_pathology_mean_auc': 0.15,
        'kidney_dice': 0.075,
        'lesion_dice_positive_only': 0.075,
    },
}

# Any task falling below its equal-weight Phase 5b test value is penalized, so the optimizer
# cannot buy classification gains by abandoning a head entirely.
REGRESSION_GUARD = {
    'kidney_dice': 0.87,
    'lesion_dice_positive_only': 0.45,
    'local_pathology_mean_auc': 0.70,
}
REGRESSION_PENALTY = 0.5


def decode(vector):
    """Raw optimizer vector -> (loss weight dict, enhancement param dict)."""
    v = np.asarray(vector, dtype=float)
    loss_weights = {name: float(v[i]) for i, name in enumerate(LOSS_TERMS)}
    enhancement = {
        'clahe_clip_limit': float(v[6]),
        'clahe_tile_grid_size': int(round(v[7])),
        'denoise_h': float(v[8]),
        'gamma': float(v[9]),
    }
    return loss_weights, enhancement


def fitness_from_metrics(metrics, profile='balanced', guard=False):
    """Weighted scalarization of validation metrics. Higher is better."""
    weights = FITNESS_PROFILES[profile] if isinstance(profile, str) else profile
    parts = {
        'diagnosis_auc': metrics['diagnosis_auc'],
        'diagnosis_balanced_accuracy': metrics.get('diagnosis_balanced_accuracy'),
        'local_pathology_mean_auc': metrics['local_pathology_mean_auc'],
        'kidney_dice': metrics['kidney_dice'],
        'lesion_dice_positive_only': metrics['lesion_dice_positive_only'],
        'consistency': 1.0 - metrics['consistency_violation_rate'],
    }
    score = 0.0
    for key, w in weights.items():
        value = parts.get(key)
        # A NaN component (e.g. an AUC column with one class present) contributes nothing
        # rather than poisoning the whole score.
        score += w * (0.0 if value is None or np.isnan(value) else float(value))

    if guard:
        for key, floor in REGRESSION_GUARD.items():
            value = parts.get(key)
            if value is not None and not np.isnan(value) and value < floor:
                score -= REGRESSION_PENALTY * (floor - float(value))
    return score, parts


def _key(vector):
    rounded = np.round(np.asarray(vector, dtype=float), 4).tolist()
    return hashlib.md5(json.dumps(rounded).encode()).hexdigest()[:16]


class Phase6Evaluator:
    """Trains one candidate configuration and returns its validation fitness."""

    def __init__(self, manifest, device, *, img_size=(256, 256), epochs=3, batch_size=16,
                 lr=1e-4, fast_denoise=False, run_name='phase6', num_workers=0, seed=SEED,
                 consistency='ratio', enhance_workers=None, fitness_profile='balanced',
                 guard=False, augment=False, scheduler='none', diagnosis_loss_cfg=None):
        self.manifest = manifest
        self.device = device
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.fast_denoise = fast_denoise
        self.num_workers = num_workers
        self.seed = seed
        self.run_name = run_name
        self.consistency = consistency
        self.enhance_workers = enhance_workers
        self.fitness_profile = fitness_profile
        self.guard = guard
        self.augment = augment
        self.scheduler = scheduler
        self.diagnosis_loss_cfg = diagnosis_loss_cfg

        self.pool = EnhancementPool(workers=enhance_workers)
        self.cache_path = os.path.join(LOGS_DIR, f'{run_name}_eval_cache.json')
        self.trail_path = os.path.join(LOGS_DIR, f'{run_name}_evaluations.jsonl')
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as fh:
                self.cache = json.load(fh)
        self.current_optimizer = None

    def _save_cache(self):
        with open(self.cache_path, 'w') as fh:
            json.dump(self.cache, fh, indent=2)

    def __call__(self, vector):
        """Returns COST (negative fitness) so the minimizing optimizers work directly."""
        key = _key(vector)
        if key in self.cache:
            return -self.cache[key]['fitness']

        loss_weights, enhancement = decode(vector)
        t0 = time.time()
        set_seed(self.seed)

        # Enhancement is deterministic for a fixed parameter set, so it is computed once here
        # across a process pool and reused for every epoch of this candidate's training.
        splits = build_enhanced_splits(self.manifest, enhancement, splits=('train', 'val'),
                                       img_size=self.img_size,
                                       fast_denoise=self.fast_denoise,
                                       pool=self.pool, quiet=True,
                                       augment_train=self.augment)
        train_ds, val_ds = splits['train'], splits['val']

        model = MultiTaskNetV2().to(self.device)
        train_model(model, train_ds, val_ds, self.device, epochs=self.epochs,
                    batch_size=self.batch_size, lr=self.lr, weights=loss_weights,
                    checkpoint_name=None, num_workers=self.num_workers,
                    log_name=None, verbose=False, consistency=self.consistency,
                    diagnosis_loss_cfg=self.diagnosis_loss_cfg, scheduler=self.scheduler,
                    selection='diagnosis_auc' if self.fitness_profile == 'classification'
                    else 'combined_dice')

        val_loader = make_loader(val_ds, self.batch_size, False, self.num_workers)
        metrics = evaluate(model, val_loader, self.device, use_amp=True)
        fitness, parts = fitness_from_metrics(metrics, self.fitness_profile, self.guard)

        del model
        torch.cuda.empty_cache()

        record = {
            'key': key,
            'optimizer': self.current_optimizer,
            'vector': np.asarray(vector, dtype=float).tolist(),
            'loss_weights': loss_weights,
            'enhancement': enhancement,
            'fitness': fitness,
            'fitness_parts': parts,
            'val_metrics': {k: v for k, v in metrics.items() if not isinstance(v, dict)},
            'seconds': round(time.time() - t0, 1),
        }
        self.cache[key] = record
        self._save_cache()
        with open(self.trail_path, 'a') as fh:
            fh.write(json.dumps(record) + '\n')

        print(f'    eval[{self.current_optimizer}] fitness={fitness:.4f} '
              f'(auc={metrics["diagnosis_auc"]:.3f} kdice={metrics["kidney_dice"]:.3f} '
              f'ldice+={metrics["lesion_dice_positive_only"]:.3f}) '
              f'{record["seconds"]:.0f}s', flush=True)
        return -fitness
