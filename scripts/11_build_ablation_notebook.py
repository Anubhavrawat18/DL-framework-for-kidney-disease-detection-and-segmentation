"""Generates Ablation_and_Encoder_Study.ipynb.

The notebook is kept deliberately thin: dataset, training loop and metrics come from src/, so
the notebook holds only the study design, the run loop and the reporting. Written with nbformat
so the JSON is guaranteed valid.
"""
import os
import sys

import nbformat as nbf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROJECT_ROOT  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, 'Ablation_and_Encoder_Study.ipynb')

MD_INTRO = """# Ablation and Encoder Study

Two questions, one protocol:

1. **Module ablation** — which parts of the multi-task model actually contribute? Each module is
   switched off in turn, leaving everything else fixed.
2. **Encoder comparison** — does a lighter backbone cost accuracy? ResNet-34 is compared against
   EfficientNet-B0, MobileNetV3-Large and ShuffleNetV2.

### Protocol, and why it looks like this

Earlier work on this project compared single training runs and drew conclusions from
differences of 0.01–0.03 AUC. That was a mistake: on this test split (298 images, only 68
healthy) the bootstrap 95% confidence interval for AUC is about 0.085 wide, and **retraining
the identical recipe with a different seed moves AUC by up to 0.074**. Differences smaller than
that carry no information.

So every configuration here is run with **3 seeds** and reported as mean ± standard deviation.
A configuration is only treated as different from the reference if the gap is large relative to
that spread.

To keep the study affordable it runs at **reduced fidelity**: 256×256 input and 12 epochs,
rather than the 512×512 / 15-epoch protocol used for the headline results. Absolute numbers are
therefore lower than the main results and **should not be quoted alongside them** — this study
is for comparing configurations against each other under identical conditions.

The model from the final epoch is evaluated (no checkpoint selection), because checkpoint
selection on a noisy validation metric was itself shown to shift AUC by ~0.037.
"""

CODE_SETUP = """import json, os, time, itertools
import numpy as np
import torch

from src.config import RESULTS_DIR, FIGURES_DIR
from src.data import URICADSMultiTaskDataset, load_manifest
from src.engine import evaluate, get_device, make_loader, set_seed, train_model, _forward
from src.metrics import best_threshold, classification_metrics
from src.model import MultiTaskNetFlex, ENCODER_NAMES

IMG_SIZE = (256, 256)   # reduced fidelity: this study compares configurations, not absolutes
EPOCHS   = 12
BATCH    = 8
LR       = 1e-4
WORKERS  = 6
SEEDS    = [42, 143, 244]

RESULT_PATH = os.path.join(RESULTS_DIR, 'ablation_encoder_study_results.json')

device = get_device()
manifest = load_manifest()
print('device :', torch.cuda.get_device_name(0))
print('splits :', manifest['split'].value_counts().to_dict())
print(f'protocol: {IMG_SIZE[0]}x{IMG_SIZE[1]}, {EPOCHS} epochs, batch {BATCH}, '
      f'{len(SEEDS)} seeds per configuration')
"""

MD_DESIGN = """## Study design

`MultiTaskNetFlex` lets each module be switched off independently. A disabled head returns
`None` rather than an untrained tensor, so it cannot contribute a meaningless score; the
evaluation path substitutes a constant logit, which reads as chance-level.

The anatomical consistency term is a loss, not a module, so it is removed from the active loss
terms rather than from the architecture.
"""

CODE_CONFIGS = """def terms_for(modules, use_consistency):
    \"\"\"Loss terms must match the heads that exist, or training references a missing output.\"\"\"
    terms = ['diagnosis', 'global_pathology']
    if modules.get('use_local_pathology', True):
        terms.append('local_pathology')
    if modules.get('use_kidney_decoder', True):
        terms.append('kidney_seg')
    if modules.get('use_lesion_decoder', True):
        terms.append('lesion_seg')
        if use_consistency:
            terms.append('consistency')   # penalizes lesion mass outside the true kidney
    return terms


MODULE_ABLATIONS = [
    ('full',                'Full model (reference)',            {},                                                        True),
    ('no_lesion_pooling',   'Without lesion-guided pooling',     {'lesion_guided_pooling': False},                          True),
    ('no_consistency',      'Without anatomical consistency',    {},                                                        False),
    ('no_lesion_decoder',   'Without lesion segmentation',       {'use_lesion_decoder': False},                             False),
    ('no_local_head',       'Without local-pathology head',      {'use_local_pathology': False},                            True),
    ('no_kidney_decoder',   'Without kidney segmentation',       {'use_kidney_decoder': False},                             True),
    ('classification_only', 'Classification heads only',         {'use_kidney_decoder': False, 'use_lesion_decoder': False}, False),
]

EXTRA_ENCODERS = ['efficientnet_b0', 'mobilenet_v3_large', 'shufflenet_v2_x1_0']

CONFIGS = []
for key, label, modules, use_cons in MODULE_ABLATIONS:
    CONFIGS.append(dict(key=key, label=label, study='module', encoder='resnet34',
                        modules=modules, terms=terms_for(modules, use_cons)))
for enc in EXTRA_ENCODERS:
    CONFIGS.append(dict(key=f'encoder_{enc}', label=enc, study='encoder', encoder=enc,
                        modules={}, terms=terms_for({}, True)))

print(f'{len(CONFIGS)} configurations x {len(SEEDS)} seeds = '
      f'{len(CONFIGS) * len(SEEDS)} training runs')
for c in CONFIGS:
    print(f\"  [{c['study']:7s}] {c['key']:20s} {c['encoder']:20s} \"
          f\"terms: {','.join(c['terms'])}\")
"""

CODE_RUNONE = """def run_one(cfg, seed):
    \"\"\"Trains one configuration with one seed and returns its test metrics.\"\"\"
    set_seed(seed)
    splits = {s: URICADSMultiTaskDataset(manifest[manifest['split'] == s], img_size=IMG_SIZE)
              for s in ('train', 'val', 'test')}

    model = MultiTaskNetFlex(encoder=cfg['encoder'], **cfg['modules']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())

    t0 = time.time()
    train_model(model, splits['train'], splits['val'], device, epochs=EPOCHS,
                batch_size=BATCH, lr=LR, active_terms=cfg['terms'], consistency='ratio',
                checkpoint_name=None, num_workers=WORKERS, verbose=False)
    train_s = time.time() - t0

    metrics = evaluate(model, make_loader(splits['test'], BATCH, False, WORKERS), device,
                       use_amp=True)

    # Operating point picked on validation, applied to test - never tuned on test.
    def probs(ds):
        p, t = [], []
        with torch.no_grad():
            for b in make_loader(ds, BATCH, False, WORKERS):
                with torch.autocast('cuda'):
                    out = _forward(model, b['image'].to(device))
                d = out['diagnosis']
                d = torch.full((b['image'].shape[0],), -20.0, device=device) if d is None else d
                p.extend(torch.sigmoid(d.float()).cpu().numpy())
                t.extend(b['global_diagnosis'].numpy())
        return np.asarray(t), np.asarray(p)

    vt, vp = probs(splits['val'])
    tt, tp = probs(splits['test'])
    thr = best_threshold(vt, vp, 'balanced_accuracy')
    tuned = classification_metrics(tt, tp, threshold=thr)

    del model
    torch.cuda.empty_cache()
    return {
        'diagnosis_auc': metrics['diagnosis_auc'],
        'local_pathology_mean_auc': metrics['local_pathology_mean_auc'],
        'kidney_dice': metrics['kidney_dice'],
        'lesion_dice_positive_only': metrics['lesion_dice_positive_only'],
        'balanced_accuracy': tuned['balanced_accuracy'],
        'sensitivity': tuned['sensitivity'],
        'specificity': tuned['specificity'],
        'threshold': thr,
        'params_total': n_params,
        'params_encoder': n_encoder,
        'train_seconds': train_s,
    }
"""

CODE_LOOP = """results = {}
if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH) as fh:
        results = json.load(fh)
    print(f'resuming: {len(results)} runs already stored')

total = len(CONFIGS) * len(SEEDS)
done = 0
for cfg in CONFIGS:
    for seed in SEEDS:
        run_id = f\"{cfg['key']}__seed{seed}\"
        done += 1
        if run_id in results:
            print(f'[{done:2d}/{total}] {run_id:34s} cached')
            continue
        out = run_one(cfg, seed)
        out.update(key=cfg['key'], label=cfg['label'], study=cfg['study'],
                   encoder=cfg['encoder'], seed=seed)
        results[run_id] = out
        with open(RESULT_PATH, 'w') as fh:      # written each run, so the study is resumable
            json.dump(results, fh, indent=1)
        print(f\"[{done:2d}/{total}] {run_id:34s} AUC={out['diagnosis_auc']:.4f} \"
              f\"kDice={out['kidney_dice']:.4f} \"
              f\"lAUC={out['local_pathology_mean_auc']:.4f} \"
              f\"({out['train_seconds']/60:.1f} min)\")
print('\\ndone')
"""

CODE_AGG = """def aggregate(study):
    rows = {}
    for r in results.values():
        if r['study'] != study:
            continue
        rows.setdefault(r['key'], {'label': r['label'], 'encoder': r['encoder'],
                                   'params': r['params_total'], 'runs': []})['runs'].append(r)
    out = []
    for key, info in rows.items():
        entry = {'key': key, 'label': info['label'], 'encoder': info['encoder'],
                 'params_M': info['params'] / 1e6, 'n': len(info['runs'])}
        for metric in ('diagnosis_auc', 'local_pathology_mean_auc', 'kidney_dice',
                       'lesion_dice_positive_only', 'balanced_accuracy'):
            vals = np.array([r[metric] for r in info['runs']], dtype=float)
            entry[metric] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        out.append(entry)
    return out


def show(study, title, ref_key=None):
    rows = aggregate(study)
    if ref_key:
        rows.sort(key=lambda r: (r['key'] != ref_key, -r['diagnosis_auc'][0]))
    ref = next((r for r in rows if r['key'] == ref_key), None)
    print(title)
    print('=' * 118)
    print(f\"{'configuration':34s} {'params':>8s} {'diag AUC':>16s} {'multi AUC':>16s} \"
          f\"{'kidney Dice':>16s} {'lesion Dice+':>16s} {'vs ref':>8s}\")
    print('-' * 118)
    for r in rows:
        delta = ('  ref' if ref is None or r['key'] == ref_key
                 else f\"{r['diagnosis_auc'][0] - ref['diagnosis_auc'][0]:+.4f}\")
        print(f\"{r['label'][:34]:34s} {r['params_M']:7.2f}M \"
              f\"{r['diagnosis_auc'][0]:.4f}±{r['diagnosis_auc'][1]:.4f}  \"
              f\"{r['local_pathology_mean_auc'][0]:.4f}±{r['local_pathology_mean_auc'][1]:.4f}  \"
              f\"{r['kidney_dice'][0]:.4f}±{r['kidney_dice'][1]:.4f}  \"
              f\"{r['lesion_dice_positive_only'][0]:.4f}±{r['lesion_dice_positive_only'][1]:.4f}  \"
              f\"{delta:>8s}\")
    print()
    return rows

module_rows = show('module', 'MODULE ABLATION  (ResNet-34 encoder, 3 seeds each)', 'full')
encoder_rows = show('encoder', 'ENCODER COMPARISON (full model, 3 seeds each)')
"""

CODE_FIGS = """import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY, TEAL, GRAY, RED, LINE = '#21295C', '#1C7293', '#A7B1BB', '#B5483B', '#DCE3E8'
SEED_NOISE = 0.0258   # SD of AUC from seed alone, measured on the full-fidelity protocol

def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color=LINE, linewidth=0.7)
    ax.set_axisbelow(True)

# --- module ablation -------------------------------------------------------
rows = sorted(module_rows, key=lambda r: r['key'] != 'full')
ref = rows[0]
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=130)
x = np.arange(len(rows))
ax.bar(x, [r['diagnosis_auc'][0] for r in rows], yerr=[r['diagnosis_auc'][1] for r in rows],
       capsize=4, width=0.6, color=[NAVY] + [GRAY] * (len(rows) - 1))
ax.axhspan(ref['diagnosis_auc'][0] - SEED_NOISE, ref['diagnosis_auc'][0] + SEED_NOISE,
           color=RED, alpha=0.10, label='±1 SD of seed-only noise around the reference')
ax.axhline(ref['diagnosis_auc'][0], color=RED, linestyle='--', linewidth=1.3)
ax.set_xticks(x)
ax.set_xticklabels([r['label'].replace('Without ', 'no ') for r in rows], rotation=18,
                   ha='right', fontsize=9)
# Truncate the axis to the range the data occupies, otherwise a 0-1 axis hides both the
# differences and the noise band.
lo = min(r['diagnosis_auc'][0] - r['diagnosis_auc'][1] for r in rows)
hi = max(r['diagnosis_auc'][0] + r['diagnosis_auc'][1] for r in rows)
ax.set_ylim(lo - 0.02, hi + 0.015)
ax.set_ylabel('Diagnosis AUC  (axis truncated)')
ax.set_title('Module ablation: effect of removing each component', fontsize=12.5)
ax.legend(fontsize=9, frameon=False)
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'ablation_modules.png'), facecolor='white')
plt.show()

# --- encoder comparison ----------------------------------------------------
enc_rows = encoder_rows + [r for r in module_rows if r['key'] == 'full']
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=130)
labels = [r['label'].replace('Full model (reference)', 'resnet34') for r in enc_rows]
x = np.arange(len(enc_rows))
axes[0].bar(x, [r['diagnosis_auc'][0] for r in enc_rows],
            yerr=[r['diagnosis_auc'][1] for r in enc_rows], capsize=4, width=0.6, color=TEAL)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=18, ha='right', fontsize=9)
elo = min(r['diagnosis_auc'][0] - r['diagnosis_auc'][1] for r in enc_rows)
ehi = max(r['diagnosis_auc'][0] + r['diagnosis_auc'][1] for r in enc_rows)
axes[0].set_ylim(elo - 0.02, ehi + 0.015)
axes[0].set_ylabel('Diagnosis AUC  (axis truncated)')
axes[0].set_title('Accuracy by encoder', fontsize=12)
_style(axes[0])

for r, lbl in zip(enc_rows, labels):
    axes[1].errorbar(r['params_M'], r['diagnosis_auc'][0], yerr=r['diagnosis_auc'][1],
                     fmt='o', color=NAVY, capsize=4, markersize=8)
    axes[1].annotate(lbl, (r['params_M'], r['diagnosis_auc'][0]),
                     textcoords='offset points', xytext=(8, 6), fontsize=9)
axes[1].set_xlabel('Total parameters (millions)')
axes[1].set_ylabel('Diagnosis AUC')
axes[1].set_title('Accuracy against model size', fontsize=12)
_style(axes[1])
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'ablation_encoders.png'), facecolor='white')
plt.show()
print('figures written to results/figures/')
"""

MD_READING = """## Reading these results

Compare each gap against the seed-only noise band drawn on the first figure. A module whose
removal moves AUC by less than that band has not been shown to contribute — with three seeds
this study can detect large effects, not small ones.

Two specific cautions:

- **Removing a head changes what the shared encoder is trained to represent.** A drop after
  removing the kidney decoder therefore measures the loss of that auxiliary supervision, not
  just the loss of the output.
- **Absolute values are lower than the headline results** because this runs at 256×256 for 12
  epochs. Quote them only against each other.
"""


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(MD_INTRO),
        nbf.v4.new_code_cell(CODE_SETUP),
        nbf.v4.new_markdown_cell(MD_DESIGN),
        nbf.v4.new_code_cell(CODE_CONFIGS),
        nbf.v4.new_code_cell(CODE_RUNONE),
        nbf.v4.new_markdown_cell('## Run the study\n\nEach run is written to disk as it '
                                 'finishes, so the notebook can be interrupted and restarted '
                                 'without losing work.'),
        nbf.v4.new_code_cell(CODE_LOOP),
        nbf.v4.new_markdown_cell('## Results'),
        nbf.v4.new_code_cell(CODE_AGG),
        nbf.v4.new_code_cell(CODE_FIGS),
        nbf.v4.new_markdown_cell(MD_READING),
    ]
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                                 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': '3.12'}
    with open(OUT, 'w', encoding='utf-8') as fh:
        nbf.write(nb, fh)
    return OUT


if __name__ == '__main__':
    print('Wrote', build())
