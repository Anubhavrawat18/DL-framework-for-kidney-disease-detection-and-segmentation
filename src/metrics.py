"""Evaluation metrics.

Dice is accumulated per sample rather than per batch. The notebooks averaged batch means,
which silently weights a short final batch as heavily as a full one; per-sample averaging
removes that and is what the paper should report.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def dice_per_sample(pred_logits, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred_flat, target_flat = pred.flatten(1), target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).detach().cpu().numpy()


def violation_per_sample(lesion_logits, kidney_mask_target, eps=1e-6):
    """Fraction of predicted lesion probability mass falling outside the true kidney."""
    lesion_prob = torch.sigmoid(lesion_logits)
    total_mass = lesion_prob.flatten(1).sum(dim=1) + eps
    outside = (lesion_prob * (1.0 - kidney_mask_target)).flatten(1).sum(dim=1)
    return (outside / total_mass).detach().cpu().numpy()


def safe_auc(targets, probs):
    targets = np.asarray(targets)
    if len(np.unique(targets)) < 2:
        return float('nan')
    return roc_auc_score(targets, probs)


def classification_metrics(targets, probs, threshold=0.5):
    """Threshold-dependent metrics the paper needs alongside AUC.

    AUC is threshold-free, so it cannot answer "what is the accuracy". With 23% healthy and
    77% pathological, plain accuracy is also misleading on its own: always predicting
    "pathological" already scores 0.77. Balanced accuracy (the mean of sensitivity and
    specificity) is the honest headline number for this dataset, and sensitivity/specificity
    are what a clinical reader will actually ask for.
    """
    targets = np.asarray(targets).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)

    tp = int(((pred == 1) & (targets == 1)).sum())
    tn = int(((pred == 0) & (targets == 0)).sum())
    fp = int(((pred == 1) & (targets == 0)).sum())
    fn = int(((pred == 0) & (targets == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else float('nan')
    specificity = tn / (tn + fp) if (tn + fp) else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision == precision and sensitivity == sensitivity
          and (precision + sensitivity) else float('nan'))

    return {
        'accuracy': (tp + tn) / len(targets) if len(targets) else float('nan'),
        'balanced_accuracy': (sensitivity + specificity) / 2,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auprc': (average_precision_score(targets, probs)
                  if len(np.unique(targets)) > 1 else float('nan')),
        'threshold': threshold,
        'majority_class_accuracy': max(targets.mean(), 1 - targets.mean()),
        'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }


def best_threshold(targets, probs, criterion='balanced_accuracy'):
    """Threshold maximizing the chosen criterion, selected on validation data only."""
    targets = np.asarray(targets).astype(int)
    if len(np.unique(targets)) < 2:
        return 0.5
    candidates = np.unique(np.round(np.asarray(probs, dtype=float), 3))
    best, best_score = 0.5, -np.inf
    for t in candidates:
        score = classification_metrics(targets, probs, threshold=float(t))[criterion]
        if score == score and score > best_score:
            best, best_score = float(t), score
    return best


def multilabel_auc(targets, probs):
    """Per-column AUC, skipping columns without both classes present."""
    targets, probs = np.asarray(targets), np.asarray(probs)
    aucs = {}
    for c in range(targets.shape[1]):
        aucs[c] = safe_auc(targets[:, c], probs[:, c])
    valid = [v for v in aucs.values() if not np.isnan(v)]
    return aucs, (float(np.mean(valid)) if valid else float('nan'))


class MetricAccumulator:
    """Collects per-sample predictions across an evaluation pass."""

    def __init__(self):
        self.diag_probs, self.diag_targets = [], []
        self.global_probs, self.global_targets = [], []
        self.local_probs, self.local_targets = [], []
        self.kidney_dice, self.lesion_dice, self.violation = [], [], []
        self.has_lesion = []

    def update(self, outputs, batch, device):
        kidney_target = batch['kidney_mask'].to(device)
        lesion_target = batch['lesion_mask'].to(device)

        self.diag_probs.extend(torch.sigmoid(outputs['diagnosis']).detach().cpu().numpy())
        self.diag_targets.extend(batch['global_diagnosis'].numpy())
        self.global_probs.extend(torch.sigmoid(outputs['global_pathology']).detach().cpu().numpy())
        self.global_targets.extend(batch['global_pathologies'].numpy())
        self.local_probs.extend(torch.sigmoid(outputs['local_pathology']).detach().cpu().numpy())
        self.local_targets.extend(batch['local_pathologies'].numpy())

        self.kidney_dice.extend(dice_per_sample(outputs['kidney_seg'], kidney_target))
        self.lesion_dice.extend(dice_per_sample(outputs['lesion_seg'], lesion_target))
        self.violation.extend(violation_per_sample(outputs['lesion_seg'], kidney_target))
        self.has_lesion.extend(batch['has_lesion'].numpy())

    def compute(self):
        has_lesion = np.asarray(self.has_lesion, dtype=bool)
        lesion_dice = np.asarray(self.lesion_dice)
        local_aucs, local_mean_auc = multilabel_auc(self.local_targets, self.local_probs)
        _, global_mean_auc = multilabel_auc(self.global_targets, self.global_probs)

        cls = classification_metrics(self.diag_targets, self.diag_probs)
        return {
            'diagnosis_auc': safe_auc(self.diag_targets, self.diag_probs),
            'diagnosis_accuracy': cls['accuracy'],
            'diagnosis_balanced_accuracy': cls['balanced_accuracy'],
            'diagnosis_sensitivity': cls['sensitivity'],
            'diagnosis_specificity': cls['specificity'],
            'diagnosis_f1': cls['f1'],
            'diagnosis_auprc': cls['auprc'],
            'majority_class_accuracy': cls['majority_class_accuracy'],
            'diagnosis_confusion': cls['confusion'],
            'global_pathology_mean_auc': global_mean_auc,
            'local_pathology_mean_auc': local_mean_auc,
            'local_pathology_aucs': {k: v for k, v in local_aucs.items()},
            'kidney_dice': float(np.mean(self.kidney_dice)),
            'lesion_dice': float(np.mean(lesion_dice)),
            # The headline lesion Dice is inflated by the ~1/3 of images with no lesion, where a
            # correct empty prediction scores ~1.0. This is the number that reflects real
            # detection quality on cases that actually contain a lesion.
            'lesion_dice_positive_only': (float(np.mean(lesion_dice[has_lesion]))
                                          if has_lesion.any() else float('nan')),
            'n_lesion_positive': int(has_lesion.sum()),
            'consistency_violation_rate': float(np.mean(self.violation)),
            'n_samples': len(self.diag_targets),
        }
