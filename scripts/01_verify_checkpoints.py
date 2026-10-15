"""Re-evaluate the four existing Colab checkpoints with the local code.

Two jobs at once:
  1. PORT VERIFICATION - if local test metrics reproduce the numbers the Colab notebooks
     reported, the local reimplementation is faithful and every later result is trustworthy.
  2. Closes the open item from the project memory: Phase 4's consistency violation rate was
     never measured, so the Phase 4 vs Phase 5 anatomical-constraint comparison was incomplete.
     No retraining needed - just a forward pass with the saved weights.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CHECKPOINTS_DIR, FIGURES_DIR
from src.data import DEFAULT_ENHANCEMENT, URICADSMultiTaskDataset, load_manifest
from src.engine import evaluate, get_device, make_loader, set_seed
from src.model import MultiTaskNet, MultiTaskNetV2
from src.reporting import Tee, render_console_png, save_phase_result

# What the Colab notebooks reported, for side-by-side comparison.
COLAB_REPORTED = {
    'phase2_baseline_best.pt': {'diagnosis_auc': 0.9003, 'kidney_dice': 0.8735},
    'phase3_enhanced_best.pt': {'diagnosis_auc': 0.8712, 'kidney_dice': 0.8891},
    'phase4_multitask_best.pt': {'diagnosis_auc': 0.8880, 'kidney_dice': 0.8966,
                                 'lesion_dice': 0.3388, 'local_pathology_mean_auc': 0.7724},
    'phase5_anatomical_best.pt': {'diagnosis_auc': 0.8758, 'kidney_dice': 0.8970,
                                  'lesion_dice': 0.3425, 'consistency_violation_rate': 0.3790},
}

CHECKPOINTS = [
    ('phase2_baseline_best.pt', MultiTaskNet, None),
    ('phase3_enhanced_best.pt', MultiTaskNet, DEFAULT_ENHANCEMENT),
    ('phase4_multitask_best.pt', MultiTaskNetV2, None),
    ('phase5_anatomical_best.pt', MultiTaskNetV2, None),
]


def main():
    device = get_device()
    manifest = load_manifest()
    results = {}

    for ckpt_name, model_cls, enhancement in CHECKPOINTS:
        path = os.path.join(CHECKPOINTS_DIR, ckpt_name)
        if not os.path.exists(path):
            print(f'SKIP {ckpt_name} - not found')
            continue

        print('\n' + '=' * 72)
        print(f'{ckpt_name}  ({model_cls.__name__}'
              f'{", enhanced input" if enhancement else ""})')
        print('=' * 72, flush=True)

        set_seed()
        model = model_cls(pretrained=False).to(device)
        state = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f'  state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected')

        test_ds = URICADSMultiTaskDataset(manifest[manifest['split'] == 'test'],
                                          enhancement=enhancement)
        metrics = evaluate(model, make_loader(test_ds, 4, False), device, use_amp=True)

        reported = COLAB_REPORTED.get(ckpt_name, {})
        print(f'{"metric":34s} {"local":>10s} {"colab":>10s} {"delta":>10s}')
        print('-' * 68)
        for key in ('diagnosis_auc', 'kidney_dice', 'lesion_dice',
                    'lesion_dice_positive_only', 'local_pathology_mean_auc',
                    'global_pathology_mean_auc', 'consistency_violation_rate'):
            if key not in metrics:
                continue
            local_v = metrics[key]
            colab_v = reported.get(key)
            if colab_v is None:
                print(f'{key:34s} {local_v:10.4f} {"-":>10s} {"(new)":>10s}')
            else:
                print(f'{key:34s} {local_v:10.4f} {colab_v:10.4f} {local_v - colab_v:+10.4f}')
        print(f'{"n_test_samples":34s} {metrics["n_samples"]:10d}')
        print(f'{"n_lesion_positive":34s} {metrics["n_lesion_positive"]:10d}')

        results[ckpt_name] = {'local': {k: v for k, v in metrics.items()
                                        if not isinstance(v, dict)},
                              'colab_reported': reported}
        del model
        torch.cuda.empty_cache()

    print('\n' + '=' * 72)
    print('PHASE 4 vs PHASE 5 - the anatomical consistency comparison, now complete')
    print('=' * 72)
    p4 = results.get('phase4_multitask_best.pt', {}).get('local', {})
    p5 = results.get('phase5_anatomical_best.pt', {}).get('local', {})
    if p4 and p5:
        v4 = p4['consistency_violation_rate']
        v5 = p5['consistency_violation_rate']
        print(f'  Phase 4 (no constraint)  violation rate : {v4:.4f}')
        print(f'  Phase 5 (with constraint) violation rate: {v5:.4f}')
        change = (v5 - v4) / v4 * 100 if v4 else float("nan")
        print(f'  Change                                  : {v5 - v4:+.4f} ({change:+.1f}%)')
        print(f'  Lesion Dice  Phase 4 -> Phase 5         : '
              f'{p4["lesion_dice"]:.4f} -> {p5["lesion_dice"]:.4f}')
        print(f'  Lesion Dice (positive only)             : '
              f'{p4["lesion_dice_positive_only"]:.4f} -> {p5["lesion_dice_positive_only"]:.4f}')
        verdict = ('CONSTRAINT HELPS' if v5 < v4 else 'CONSTRAINT DOES NOT REDUCE VIOLATIONS')
        print(f'  Verdict: {verdict}')
    return results


if __name__ == '__main__':
    with Tee() as tee:
        results = main()
    render_console_png(tee.text, os.path.join(FIGURES_DIR, 'checkpoint_verification.png'),
                       title='scripts/01_verify_checkpoints.py')
    save_phase_result('checkpoint_verification', results)
    print('\nSaved figure + results.')
