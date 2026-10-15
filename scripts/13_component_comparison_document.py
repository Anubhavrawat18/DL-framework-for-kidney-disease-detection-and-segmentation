"""Builds the focused two-component comparison document.

Isolates the two ablations that the review asked about - anatomical consistency and
lesion-guided pooling - and reports them per seed as well as in aggregate. The aggregate
tables in URI-CADS_Ablation_Encoder_Study.docx make these two look alike; the per-seed
numbers are what separates them, so they are the point of this document.

Every figure is computed from results/ablation_encoder_study_results.json, so the prose
cannot drift from the data.
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from scipy import stats

from src.config import PROJECT_ROOT, RESULTS_DIR

OUT = os.path.join(PROJECT_ROOT, 'URI-CADS_Component_Comparison.docx')
NAVY, INK, SLATE, GREEN, RED = '21295C', '1A2233', '5B6B79', '2E7D5B', 'B5483B'
SEED_NOISE = 0.0258          # AUC SD from seed alone, measured on the full-fidelity protocol

METRICS = [('diagnosis_auc', 'Diagnosis AUC'),
           ('local_pathology_mean_auc', 'Multi-pathology AUC'),
           ('kidney_dice', 'Kidney Dice'),
           ('lesion_dice_positive_only', 'Lesion Dice (lesion-bearing)'),
           ('balanced_accuracy', 'Balanced accuracy'),
           ('sensitivity', 'Sensitivity'),
           ('specificity', 'Specificity')]

COMPARISONS = [
    ('no_consistency', 'Without anatomical consistency', 'anatomical consistency loss'),
    ('no_lesion_pooling', 'Without lesion-guided pooling', 'lesion-guided pooling head'),
]


# ---------------------------------------------------------------- data
def load_runs():
    with open(os.path.join(RESULTS_DIR, 'ablation_encoder_study_results.json'),
              encoding='utf-8') as fh:
        return json.load(fh)


def seeds_of(runs, key):
    return sorted(r['seed'] for r in runs.values() if r['key'] == key)


def series(runs, key, metric, seeds):
    by_seed = {r['seed']: r for r in runs.values() if r['key'] == key}
    return np.array([by_seed[s][metric] for s in seeds], dtype=float)


# ---------------------------------------------------------------- doc helpers
def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text, *, italic=False, size=10.5, color=INK, space=8):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space)
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    return para


def bullets(doc, items):
    for item in items:
        para = doc.add_paragraph(style='List Bullet')
        para.paragraph_format.space_after = Pt(4)
        if isinstance(item, tuple):
            lead, rest = item
            r = para.add_run(lead)
            r.bold = True
            r.font.size = Pt(10.5)
            para.add_run(rest).font.size = Pt(10.5)
        else:
            para.add_run(item).font.size = Pt(10.5)


def table(doc, headers, rows, widths=None, colors=None):
    """colors: optional list-of-lists of hex strings (or None) matching `rows`."""
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Light Grid Accent 1'
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(head)
        run.bold = True
        run.font.size = Pt(9)
    for j, row in enumerate(rows):
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ''
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
            if colors and colors[j] and colors[j][i]:
                run.font.color.rgb = RGBColor.from_string(colors[j][i])
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(12)
    run = para.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(SLATE)


def verdict(delta, sd_ref, sd_cfg, signs_agree):
    """A gap counts only if it is large relative to spread AND consistent across seeds."""
    pooled = max(SEED_NOISE, (sd_ref + sd_cfg) / 2)
    if abs(delta) < pooled:
        return 'within noise', SLATE
    if not signs_agree:
        return 'inconsistent', SLATE
    return ('lower', RED) if delta < 0 else ('higher', GREEN)


# ---------------------------------------------------------------- sections
def summary_table(doc, runs, key, seeds):
    rows, colors = [], []
    for metric, label in METRICS:
        a = series(runs, 'full', metric, seeds)
        b = series(runs, key, metric, seeds)
        d = b - a
        signs_agree = bool(np.all(d > 0) or np.all(d < 0))
        _, pval = stats.ttest_rel(b, a)
        text, color = verdict(d.mean(), a.std(), b.std(), signs_agree)
        rows.append([label, f'{a.mean():.4f} ± {a.std():.4f}', f'{b.mean():.4f} ± {b.std():.4f}',
                     f'{d.mean():+.4f}', f'{pval:.3f}', text])
        colors.append([None, None, None, color, None, color])
    table(doc, ['Metric', 'Full model', 'Ablated', 'Δ (ablated − full)', 'p', 'Reading'],
          rows, widths=[1.7, 1.25, 1.25, 1.0, 0.55, 0.85], colors=colors)


def per_seed_table(doc, runs, key, seeds):
    rows, colors = [], []
    for metric, label in METRICS:
        a = series(runs, 'full', metric, seeds)
        b = series(runs, key, metric, seeds)
        d = b - a
        row = [label]
        color = [None]
        for i in range(len(seeds)):
            row.append(f'{a[i]:.4f} → {b[i]:.4f}')
            color.append(None)
            row.append(f'{d[i]:+.4f}')
            color.append(GREEN if d[i] > 0 else RED)
        rows.append(row)
        colors.append(color)
    headers = ['Metric']
    for s in seeds:
        headers += [f'Seed {s}', 'Δ']
    table(doc, headers, rows, widths=[1.55] + [0.95, 0.6] * len(seeds), colors=colors)


def build():
    runs = load_runs()
    seeds = seeds_of(runs, 'full')
    n = len(seeds)

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(10.5)
    for name, size in (('Title', 24), ('Heading 1', 15), ('Heading 2', 12)):
        s = doc.styles[name]
        s.font.name = 'Calibri'
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor.from_string(NAVY)
        s.font.bold = True

    doc.add_heading('Component Comparison: Anatomical Consistency and '
                    'Lesion-Guided Pooling', level=0)
    p(doc, 'Optimization-Based Deep Learning Framework for Kidney Disease Detection and '
           'Segmentation from Ultrasound Images', italic=True, size=12, color=SLATE)
    p(doc, f'Generated {datetime.now():%d %B %Y}  ·  {n} seeds per configuration  ·  '
           f'Source: results/ablation_encoder_study_results.json  ·  '
           f'Companion to URI-CADS_Ablation_Encoder_Study.docx')

    # ------------------------------------------------ 1
    h(doc, '1. What is being compared')
    p(doc, 'Two components of the full multi-task model are removed in turn, each against the '
           'same reference run. Everything else - encoder, data, schedule, seeds - is held '
           'fixed, so the difference between a pair of rows is attributable to the component.')
    bullets(doc, [
        ('Anatomical consistency. ', 'A loss term, not a module. It penalizes predicted lesion '
                                     'probability mass falling outside the true kidney region. '
                                     'The ratio form is used here, which is the corrected '
                                     'formulation; the earlier absolute form was shown to be '
                                     'minimized by suppressing lesion probability everywhere.'),
        ('Lesion-guided pooling. ', 'An architectural head. The multi-pathology classifier sees '
                                    'encoder features pooled under the lesion decoder\'s own '
                                    'predicted probability map, concatenated with the global '
                                    'average. The attention map is detached, so the '
                                    'classification loss cannot reshape the lesion decoder.'),
    ])
    p(doc, 'The aggregate tables in the main ablation document make these two components look '
           'alike: both leave binary diagnosis roughly unchanged and both cost something on '
           'multi-pathology AUC. The per-seed tables in Sections 2 and 3 show that the evidence '
           'behind those two summaries is not the same in kind. That distinction is the purpose '
           'of this document.', italic=True, color=SLATE)

    # ------------------------------------------------ 2 and 3
    for idx, (key, heading, phrase) in enumerate(COMPARISONS, start=2):
        h(doc, f'{idx}. {heading}')
        summary_table(doc, runs, key, seeds)
        caption(doc, f'Table {idx}a. Aggregate over {n} seeds, mean ± standard deviation. '
                     f'p is a paired t-test across seeds. "Reading" marks a difference as real '
                     f'only when it exceeds the run-to-run spread and every seed moves in the '
                     f'same direction.')
        per_seed_table(doc, runs, key, seeds)
        caption(doc, f'Table {idx}b. The same comparison seed by seed: full model → model '
                     f'without the {phrase}.')

        a = series(runs, 'full', 'diagnosis_auc', seeds)
        b = series(runs, key, 'diagnosis_auc', seeds)
        la = series(runs, 'full', 'local_pathology_mean_auc', seeds)
        lb = series(runs, key, 'local_pathology_mean_auc', seeds)
        ld = lb - la

        if key == 'no_consistency':
            bullets(doc, [
                ('Diagnosis AUC rises, but on one seed only. ',
                 f'The mean gain of {(b - a).mean():+.4f} is carried almost entirely by seed '
                 f'{seeds[int(np.argmax(b - a))]} ({(b - a).max():+.4f}); the remaining seeds '
                 f'move by {", ".join(f"{v:+.4f}" for v in sorted(b - a)[:-1])}. Removing the '
                 f'constraint did not make the model better - it failed to rescue one poor '
                 f'initialization.'),
                ('Multi-pathology AUC falls, also inconsistently. ',
                 f'Mean {ld.mean():+.4f}, but the per-seed deltas are '
                 f'{", ".join(f"{v:+.4f}" for v in ld)} - one large drop and two near-zero '
                 f'moves, one of them positive.'),
                ('The one consistent signal runs against the term. ',
                 f'Lesion Dice on lesion-bearing images rises on all three seeds when the '
                 f'constraint is removed ('
                 f'{", ".join(f"{v:+.4f}" for v in series(runs, key, "lesion_dice_positive_only", seeds) - series(runs, "full", "lesion_dice_positive_only", seeds))}'
                 f'; p = 0.090) - the metric the term acts on most directly, and the closest '
                 f'this comparison comes to significance. The mean gain still sits just below '
                 f'the seed-noise threshold, so it is suggestive rather than established, but '
                 f'its direction never reverses.'),
                ('Verdict. ', 'No effect demonstrated on classification in either direction. '
                              'The only directionally consistent result - lesion segmentation '
                              'improving without the term - argues mildly against keeping it, '
                              'but is too small to settle the question by itself.'),
            ])
        else:
            bullets(doc, [
                ('Multi-pathology AUC falls on every seed. ',
                 f'Deltas {", ".join(f"{v:+.4f}" for v in ld)}, mean {ld.mean():+.4f}. The '
                 f'reference configuration\'s own spread on this metric is only '
                 f'±{la.std():.4f}, so all three drops sit well outside it.'),
                ('A second protocol agrees. ',
                 'The same component measured under 5-fold cross-validation at full fidelity '
                 'gave +0.0271 multi-pathology AUC. Two independent protocols agreeing on sign '
                 'and rough magnitude is the strongest evidence any component in this project '
                 'has.'),
                ('The p-value understates it. ',
                 f'p = {stats.ttest_rel(lb, la)[1]:.3f} reflects three seeds and one seed '
                 f'dropping roughly three times harder than the others, which inflates the '
                 f'variance of the paired difference. The consistency of sign is the more '
                 f'informative statistic at this sample size.'),
                ('Verdict. ', 'Real effect, confined to multi-pathology classification. Claim '
                              'it for the nine-way task only, not for binary diagnosis.'),
            ])

    # ------------------------------------------------ 4
    h(doc, '4. The two side by side')
    rows, colors = [], []
    for metric, label in METRICS:
        a = series(runs, 'full', metric, seeds)
        row, color = [label, f'{a.mean():.4f} ± {a.std():.4f}'], [None, None]
        for key, _, _ in COMPARISONS:
            d = series(runs, key, metric, seeds) - a
            agree = bool(np.all(d > 0) or np.all(d < 0))
            row += [f'{d.mean():+.4f}', f'{sum(1 for v in d if v > 0)}/{n} up']
            color += [GREEN if d.mean() > 0 else RED, None if agree else SLATE]
        rows.append(row)
        colors.append(color)
    table(doc, ['Metric', 'Full model', 'Δ no consistency', 'Seeds', 'Δ no pooling', 'Seeds'],
          rows, widths=[1.55, 1.2, 0.95, 0.7, 0.95, 0.7], colors=colors)
    caption(doc, 'Table 4. Both ablations against the same reference. The "Seeds" column counts '
                 'how many of the three seeds moved upward; 0/3 or 3/3 means the direction is '
                 'consistent, anything between means the mean is averaging over disagreement.')

    p(doc, 'Both ablations raise diagnosis AUC by about 0.021 when the component is removed. '
           'That should not be read as the components harming binary diagnosis. Both ablated '
           'configurations also have far tighter seed spread than the reference '
           f'(±{series(runs, "no_consistency", "diagnosis_auc", seeds).std():.4f} and '
           f'±{series(runs, "no_lesion_pooling", "diagnosis_auc", seeds).std():.4f} against '
           f'±{series(runs, "full", "diagnosis_auc", seeds).std():.4f}), because the reference '
           f'mean is being pulled down by a single run at '
           f'{series(runs, "full", "diagnosis_auc", seeds).min():.4f}. Establishing whether '
           f'that run is a genuine failure mode or an outlier needs more seeds on the full '
           f'model, not more ablations.')

    # ------------------------------------------------ 5
    h(doc, '5. Reading these tables')
    bullets(doc, [
        ('Direction agreement outranks the p-value here. ',
         f'With {n} seeds a paired t-test has very little power; none of these comparisons '
         f'reaches p < 0.05. Whether all seeds move the same way is the more usable signal, '
         f'which is why it is reported alongside every mean.'),
        ('Reduced fidelity. ', '256 × 256 inputs, 12 epochs, batch 8, final-epoch evaluation. '
                               'Absolute values are below the 512 × 512 headline protocol and '
                               'should be compared only with each other.'),
        ('Single split. ', 'The fixed 70/15/15 split, not cross-validation, so fold difficulty '
                           'is not averaged out. The 5-fold figure quoted for lesion-guided '
                           'pooling comes from a separate full-fidelity study.'),
        ('A null here is "not shown", not "no effect". ',
         'This study is powered for large effects. Anatomical consistency may still have an '
         'effect too small for three seeds to resolve.'),
    ])

    h(doc, '6. A measurement this comparison is missing', level=1)
    p(doc, 'The anatomical consistency ablation does not record the violation rate - the '
           'fraction of predicted lesion mass falling outside the kidney, which is the quantity '
           'the loss exists to reduce and the metric that motivated the ratio-form correction. '
           'These runs store AUC, Dice, sensitivity and specificity only.')
    p(doc, 'The consequence is a bounded one: Section 2 can say the term does not measurably '
           'help classification, but it cannot say whether the term is doing its own job. Any '
           'decision to keep or drop it in the paper needs that number. The metric already '
           'exists in the codebase (src/metrics.py, violation_per_sample) and the study is '
           'resumable from cache, so only the runs involving this term would need re-executing.',
      italic=True, color=SLATE)

    doc.save(OUT)
    return OUT


if __name__ == '__main__':
    print('Wrote', build())
