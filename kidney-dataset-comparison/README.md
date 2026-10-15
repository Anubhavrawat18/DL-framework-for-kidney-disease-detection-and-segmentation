# Kidney Ultrasound Encoder Comparison (leakage-controlled)

Self-contained study comparing five ImageNet-pretrained encoders on the
`Kidney Dataset/` folder (classes: `Cyst`, `Normal`, `stone`), with the dataset's
substantial duplicate/near-duplicate leakage identified and removed first.

Encoders: **DenseNet-201, ViT-Base/16, VGG19, InceptionV3, Xception**.

## Why this exists

A naive stratified random split over the supplied files reaches ~100% test
accuracy within two epochs. That is not model skill — it is memorisation.
Fingerprinting showed that **82.4% of test images in such a split have a
duplicate or near-duplicate sitting in the training set**, because the dataset
was augmented *before* it was split (there are 1,188 `aug_<n>_<name>.jpg` files
whose source images are all still present), and because consecutive frames from the same sweep
are near-identical.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `code/01_audit_leakage.py` | Fingerprints every image: raw-byte md5, decoded-pixel md5, 64-bit dHash, 64-bit DCT pHash, 32x32 normalised thumbnail. Writes `cache/inventory.csv` + `cache/fingerprints.npz`. |
| 2a | `code/02_probe_neardup.py` | Sweeps near-duplicate thresholds and reports resulting component sizes, so the grouping rule is data-driven rather than guessed. |
| 2b | `code/03_build_clean_split.py` | Union-find grouping over (exact dup / augmentation-to-source / near-duplicate), one representative per group, stratified split over groups, then **independent re-verification at a looser threshold**. Writes `cache/split_clean.csv`, `cache/split_leaky.csv`, `cache/leakage_report.json`. |
| 3 | `code/04_train_one.py` | Fine-tunes one encoder on one split, writes a metrics JSON. |
| 4 | `code/05_run_all.py` | Runs all five encoders, one subprocess each so VRAM is released between models. |
| 5 | `code/06_make_report.py` | Aggregates JSONs into `results/summary_*.csv` and `results/REPORT.md`. |

## Reproducing

From the parent directory (`URI-CADS dataset/`), with `.venv` active:

```bash
PY=.venv/Scripts/python.exe
D="kidney-dataset-comparison"
$PY -I $D/code/01_audit_leakage.py --data-root "Kidney Dataset" --out-dir $D/cache
$PY -I $D/code/02_probe_neardup.py --cache $D/cache
$PY -I $D/code/03_build_clean_split.py --cache $D/cache --out-dir $D/cache
$PY -I $D/code/05_run_all.py --python $PY --split-csv $D/cache/split_clean.csv --out-dir $D/results --tag clean --epochs 4
$PY -I $D/code/06_make_report.py --results-dir $D/results --cache $D/cache --out $D/results/REPORT.md
```

Requires `timm` and `tabulate` (installed into the existing `.venv`).

## Leakage controls, in order of how much they matter

1. **Augmentation-to-source linking.** `aug_<n>_<base>.jpg` is linked to `<base>`.
   All 1,188 augmented files had their source still in the folder. This alone
   would wreck any random split.
2. **Exact-duplicate collapse.** 635 redundant byte-identical copies.
3. **Near-duplicate collapse.** pHash hamming <= 6 **and** 32x32 thumbnail
   correlation >= 0.95. The two-gate rule matters: pHash alone over-merges
   visually similar ultrasound frames. This threshold sits on a stable plateau — varying
   hamming from 4 to 10 at corr 0.95 changes the group count by under 3%, while
   dropping corr to 0.90 chains distinct frames into 345-image blobs.
4. **Split over groups, not images**, so no group straddles train/val/test.
5. **Independent re-verification** at a deliberately looser threshold
   (hamming <= 10, corr >= 0.90). Residual cross-split neighbours after
   cleaning: **0**.

Net effect: 7,534 supplied files -> **3,105 genuinely independent images**
(58.8% of the dataset was duplicate or near-duplicate material).

## The bigger problem: acquisition shortcuts

Removing duplicates was necessary but **not sufficient**. The three classes come
from different acquisition sources, and the scanner artefacts nearly give away
the label on their own:

- `Cyst` is **only ever** 227x227 or 224x224; `Normal` and `stone` are **only
  ever** 512x512, so resolution alone identifies Cyst perfectly.
- `stone` frames carry visible scanner UI (depth rulers, `10 8.0 mm` captions,
  coloured markers) that `Normal` frames largely lack.

`code/10_shortcut_probe.py` quantifies this by destroying the anatomy and fitting
a *linear* model on what is left, using the same leakage-free clean split:

| Probe (anatomy destroyed) | Test Acc | Macro-F1 |
|---|---|---|
| A. width+height only (2 features) | 0.7652 | 0.5918 |
| B. 32x32 thumbnail | 0.8885 | 0.8789 |
| **C. border ring only, centre 60% removed** | **0.9452** | **0.9464** |
| D. 8x8 thumbnail | 0.7828 | 0.7597 |

Blanking the central 60% of every image still lets logistic regression reach
**94.5%**. The fine-tuned encoders sit only a few points above that.

**Mitigation:** centre-cropping suppresses it. Linear-probe accuracy by crop:
1.00 -> 0.9256, 0.70 -> 0.6830, 0.55 -> 0.6125, 0.45 -> 0.4423 (chance). The
`anat` arm re-runs all five encoders with a 0.55 centre crop; those numbers are
the ones that most reflect anatomy.

## Arms

| Tag | Split | Epochs | Purpose |
|---|---|---|---|
| `clean` | grouped, leakage-free | 4 | the headline comparison |
| `anat` | clean + 0.55 centre crop | 4 | shortcut-suppressed, most trustworthy |
| `strict` | over-merged (1,272 groups) | 4 | pessimistic residual-leakage bound |
| `leaky` | naive random over raw files | 2 | reproduces the ~100% for contrast |

## Honest limitations

- Patient identity is not encoded in the filenames, so grouping is content-based.
  It removes duplicate and near-duplicate leakage but cannot rule out residual
  same-patient leakage between visually dissimilar frames of one patient.
- Single-split numbers: encoder gaps under ~3 points are inside noise on a test
  set this size. Use paired k-fold over the same groups before ranking encoders.
- The class imbalance after dedup (`stone` shrank most) is why macro averages,
  not plain accuracy, are the figures to quote.

See `results/REPORT.md` for the metrics.
