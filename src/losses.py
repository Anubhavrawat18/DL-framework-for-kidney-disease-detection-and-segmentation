"""Loss terms. Phases 2-5 use equal weights; Phase 6 supplies an optimized weight vector."""
import torch
import torch.nn as nn

LOSS_TERMS = ('diagnosis', 'global_pathology', 'local_pathology',
              'kidney_seg', 'lesion_seg', 'consistency')

bce_logits = nn.BCEWithLogitsLoss()


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    pred_flat, target_flat = pred.flatten(1), target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()


def anatomical_consistency_loss(lesion_logits, kidney_mask_target):
    """Phase 5 formulation: absolute lesion probability mass outside the true kidney.

    Kept as-is for reproducibility. Note this does NOT minimize the violation RATE that gets
    reported: it is minimized by suppressing lesion probability everywhere, which shrinks the
    reported ratio's numerator and denominator together. Measured effect is a 47% WORSE
    violation rate than the unconstrained Phase 4 model. Use the ratio form below instead.
    """
    return (torch.sigmoid(lesion_logits) * (1.0 - kidney_mask_target)).mean()


def anatomical_consistency_ratio_loss(lesion_logits, kidney_mask_target, eps=1e-6):
    """Corrected formulation: directly minimizes the reported violation rate.

    Optimizing the ratio rather than the absolute mass removes the degenerate solution, since
    globally scaling lesion probability down leaves the ratio unchanged.
    """
    lesion_prob = torch.sigmoid(lesion_logits)
    outside = (lesion_prob * (1.0 - kidney_mask_target)).flatten(1).sum(dim=1)
    total = lesion_prob.flatten(1).sum(dim=1) + eps
    return (outside / total).mean()


CONSISTENCY_FORMS = {
    'absolute': anatomical_consistency_loss,
    'ratio': anatomical_consistency_ratio_loss,
}

# Healthy is the minority class (450 of 1,985, about 23%), so plain BCE is minimized by
# leaning toward "pathological" - which is exactly the behaviour observed: specificity 0.46
# against sensitivity 0.96. These give the optimizer a way to correct that during training
# rather than only afterwards by moving the decision threshold.
DIAGNOSIS_POS_WEIGHT_BALANCED = 450.0 / 1535.0


def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    """Down-weights easy examples so the rare, hard healthy cases dominate the gradient."""
    prob = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def make_diagnosis_loss(cfg=None):
    """cfg: None/'bce' | {'type': 'weighted_bce', 'pos_weight': float|'balanced'}
            | {'type': 'focal', 'alpha': float, 'gamma': float}"""
    if cfg is None or cfg == 'bce' or (isinstance(cfg, dict) and cfg.get('type') == 'bce'):
        return bce_logits
    if isinstance(cfg, str):
        cfg = {'type': cfg}
    kind = cfg.get('type')
    if kind == 'weighted_bce':
        pw = cfg.get('pos_weight', 'balanced')
        pw = DIAGNOSIS_POS_WEIGHT_BALANCED if pw == 'balanced' else float(pw)

        def weighted(logits, targets):
            return nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=torch.tensor(pw, device=logits.device))
        return weighted
    if kind == 'focal':
        alpha, gamma = cfg.get('alpha', 0.25), cfg.get('gamma', 2.0)
        return lambda logits, targets: focal_bce(logits, targets, alpha, gamma)
    raise ValueError(f'unknown diagnosis loss config: {cfg}')


def compute_loss(outputs, batch, device, weights=None, consistency='absolute',
                 diagnosis_loss=None):
    """weights: dict or 6-vector over LOSS_TERMS. None -> the unweighted Phase 2-5 sum."""
    if weights is None:
        weights = {t: 1.0 for t in LOSS_TERMS}
    elif not isinstance(weights, dict):
        weights = dict(zip(LOSS_TERMS, weights))
    diag_loss_fn = diagnosis_loss or bce_logits

    kidney_target = batch['kidney_mask'].to(device)
    terms = {
        'diagnosis': diag_loss_fn(outputs['diagnosis'], batch['global_diagnosis'].to(device)),
        'global_pathology': bce_logits(outputs['global_pathology'],
                                       batch['global_pathologies'].to(device)),
        'local_pathology': bce_logits(outputs['local_pathology'],
                                      batch['local_pathologies'].to(device)),
        'kidney_seg': dice_loss(outputs['kidney_seg'], kidney_target),
        'lesion_seg': dice_loss(outputs['lesion_seg'], batch['lesion_mask'].to(device)),
        'consistency': CONSISTENCY_FORMS[consistency](outputs['lesion_seg'], kidney_target),
    }
    total = sum(float(weights[t]) * terms[t] for t in LOSS_TERMS)
    return total, {t: v.item() for t, v in terms.items()}
