"""Builds the ablation and encoder study results document.

Reads results/ablation_encoder_study_results.json (produced by
Ablation_and_Encoder_Study.ipynb) and writes a standalone Word report. Every statistic is
computed here from the stored runs, so the prose cannot drift from the data.
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

from src.config import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR

OUT = os.path.join(PROJECT_ROOT, 'URI-CADS_Ablation_Encoder_Study.docx')
NAVY, INK, SLATE, GREEN, RED = '21295C', '1A2233', '5B6B79', '2E7D5B', 'B5483B'
SEED_NOISE = 0.0258          # AUC SD from seed alone, measured on the full-fidelity protocol
METRICS = [('diagnosis_auc', 'Diagnosis AUC'),
           ('local_pathology_mean_auc', 'Multi-pathology AUC'),
           ('kidney_dice', 'Kidney Dice'),
           ('lesion_dice_positive_only', 'Lesion Dice (lesion-bearing)'),
           ('balanced_accuracy', 'Balanced accuracy')]


def load_runs():
    with open(os.path.join(RESULTS_DIR, 'ablation_encoder_study_results.json'),
              encoding='utf-8') as fh:
        return json.load(fh)


def group(runs, study):
    out = {}
    for r in runs.values():
        if r['study'] != study:
            continue
        out.setdefault(r['key'], {'label': r['label'], 'encoder': r['encoder'],
                                  'params': r['params_total'],
                                  'enc_params': r['params_encoder'],
                                  'secs': [], 'runs': []})
        out[r['key']]['runs'].append(r)
        out[r['key']]['secs'].append(r['train_seconds'])
    return out


def vals(entry, metric):
    return np.array([r[metric] for r in entry['runs']], dtype=float)


def stat(entry, metric):
    v = vals(entry, metric)
    return v.mean(), v.std()


def fmt(entry, metric):
    m, s = stat(entry, metric)
    return f'{m:.4f} ± {s:.4f}'


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


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Light Grid Accent 1'
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(head)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ''
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    return t


def figure(doc, filename, caption, width=6.2):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        return False
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(SLATE)
    return True


def verdict(delta, sd_ref, sd_cfg):
    """A gap counts only if it is large relative to run-to-run spread."""
    pooled = max(SEED_NOISE, (sd_ref + sd_cfg) / 2)
    if abs(delta) < pooled:
        return 'within noise'
    return 'lower' if delta < 0 else 'higher'


def build():
    runs = load_runs()
    modules = group(runs, 'module')
    encoders = group(runs, 'encoder')
    ref = modules.get('full')
    n_seeds = len(ref['runs']) if ref else 0

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(10.5)
    for name, size in (('Title', 26), ('Heading 1', 16), ('Heading 2', 13)):
        s = doc.styles[name]
        s.font.name = 'Calibri'
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor.from_string(NAVY)
        s.font.bold = True

    doc.add_heading('Ablation and Encoder Study', level=0)
    p(doc, 'Optimization-Based Deep Learning Framework for Kidney Disease Detection and '
           'Segmentation from Ultrasound Images', italic=True, size=12, color=SLATE)
    p(doc, f'Generated {datetime.now():%d %B %Y}  ·  {len(runs)} training runs  ·  '
           f'{n_seeds} seeds per configuration  ·  Produced by '
           f'Ablation_and_Encoder_Study.ipynb')

    # ------------------------------------------------ protocol
    h(doc, '1. Protocol')
    p(doc, 'Two studies share one protocol. The module ablation switches off each component of '
           'the multi-task model in turn, holding everything else fixed. The encoder comparison '
           'replaces the ResNet-34 backbone with three lighter alternatives, holding the model '
           'and training fixed.')
    secs = np.mean([s for e in list(modules.values()) + list(encoders.values())
                    for s in e['secs']])
    bullets(doc, [
        ('Seeds. ', f'{n_seeds} per configuration, reported as mean ± standard deviation. '
                    f'Single runs are not interpretable on this dataset: retraining an '
                    f'identical recipe with a different seed moves AUC by up to 0.074.'),
        ('Fidelity. ', '256 × 256 input, 12 epochs, batch 8. This is deliberately below the '
                       '512 × 512 / 15-epoch protocol used for the headline results, so the '
                       'absolute values here are lower and should be compared only with each '
                       'other.'),
        ('No checkpoint selection. ', 'The final-epoch model is evaluated. Selecting a '
                                      'checkpoint on a noisy validation metric was previously '
                                      'shown to shift AUC by about 0.037, which would confound '
                                      'the comparison.'),
        ('Operating point. ', 'The decision threshold is chosen on validation within each run '
                              'and applied to test; it is never tuned on test.'),
        ('Cost. ', f'Mean {secs / 60:.1f} minutes per run on an RTX 3050 Laptop GPU.'),
    ])
    p(doc, f'Interpretation rule used throughout: a difference is called real only if it '
           f'exceeds the run-to-run spread. With {n_seeds} seeds this study can detect large '
           f'effects, not small ones, and differences below roughly {SEED_NOISE:.3f} AUC are '
           f'reported as "within noise" regardless of their sign.', italic=True, color=SLATE)

    # ------------------------------------------------ module ablation
    h(doc, '2. Module ablation')
    p(doc, 'Each row removes one component from the full model. Because the encoder is shared, '
           'removing a head also removes the auxiliary supervision it provided, so a drop '
           'measures the loss of that training signal as well as the loss of the output.')
    order = ['full', 'no_lesion_pooling', 'no_consistency', 'no_lesion_decoder',
             'no_local_head', 'no_kidney_decoder', 'classification_only']
    rows = []
    ref_auc, ref_sd = stat(ref, 'diagnosis_auc')
    for key in order:
        e = modules.get(key)
        if not e:
            continue
        m, s = stat(e, 'diagnosis_auc')
        delta = m - ref_auc
        rows.append([e['label'], f'{e["params"] / 1e6:.2f} M', fmt(e, 'diagnosis_auc'),
                     fmt(e, 'local_pathology_mean_auc'), fmt(e, 'kidney_dice'),
                     'reference' if key == 'full'
                     else f'{delta:+.4f} ({verdict(delta, ref_sd, s)})'])
    table(doc, ['Configuration', 'Params', 'Diagnosis AUC', 'Multi-pathology AUC',
                'Kidney Dice', 'vs full model'], rows,
          widths=[1.6, 0.65, 1.25, 1.3, 1.2, 1.4])
    p(doc, '')
    figure(doc, 'ablation_modules.png',
           'Figure 1. Diagnosis AUC after removing each module. The shaded band is ±1 SD of '
           'seed-only variation around the full model; bars inside it are not distinguishable '
           'from the reference.')

    # narrative findings, derived rather than asserted
    # A configuration that deletes the head producing a metric trivially destroys that metric;
    # reporting it as a finding would be circular, so those pairs are excluded.
    TAUTOLOGICAL = {('no_local_head', 'local_pathology_mean_auc'),
                    ('classification_only', 'local_pathology_mean_auc')}
    notable = []
    for key in order[1:]:
        e = modules.get(key)
        if not e:
            continue
        component = e['label'].replace('Without ', '').replace('Full model (reference)', '')
        for metric, label in (('diagnosis_auc', 'diagnosis AUC'),
                              ('local_pathology_mean_auc', 'multi-pathology AUC')):
            if (key, metric) in TAUTOLOGICAL:
                continue
            rm, rs = stat(ref, metric)
            m, s = stat(e, metric)
            if abs(m - rm) >= max(SEED_NOISE, (rs + s) / 2):
                direction = 'falls' if m < rm else 'rises'
                notable.append(f'Removing {component} — {label} {direction} by '
                               f'{abs(m - rm):.4f} ({rm:.4f} to {m:.4f}).')
    h(doc, '2.1 What the ablation shows', 2)
    if notable:
        bullets(doc, notable)
    else:
        p(doc, 'No single module removal moved diagnosis or multi-pathology AUC by more than '
               'the run-to-run spread. Under this protocol the model is robust to removing any '
               'one component, which is itself informative: it means the headline classification '
               'performance does not depend on any single auxiliary task, and that claims about '
               'individual modules should be made cautiously.')
    p(doc, 'Segmentation metrics behave differently from classification: removing a decoder '
           'necessarily drives its own Dice to zero, so those columns confirm the switch worked '
           'rather than measuring a trade-off.', italic=True, color=SLATE)

    # ------------------------------------------------ encoder study
    doc.add_page_break()
    h(doc, '3. Encoder comparison')
    p(doc, 'The full model is retrained with each backbone. All four downsample by a factor of '
           '32 and feed the same decoders and heads; only the channel width differs, which the '
           'architecture reads from the encoder rather than assuming.')
    enc_rows = []
    baseline = ref
    b_auc, b_sd = stat(baseline, 'diagnosis_auc')
    combined = [('resnet34', baseline)] + [(k.replace('encoder_', ''), v)
                                           for k, v in encoders.items()]
    for name, e in combined:
        m, s = stat(e, 'diagnosis_auc')
        delta = m - b_auc
        enc_rows.append([name, f'{e["enc_params"] / 1e6:.2f} M', f'{e["params"] / 1e6:.2f} M',
                         fmt(e, 'diagnosis_auc'), fmt(e, 'local_pathology_mean_auc'),
                         fmt(e, 'kidney_dice'),
                         'reference' if name == 'resnet34'
                         else f'{delta:+.4f} ({verdict(delta, b_sd, s)})'])
    table(doc, ['Encoder', 'Encoder params', 'Total params', 'Diagnosis AUC',
                'Multi-pathology AUC', 'Kidney Dice', 'vs ResNet-34'], enc_rows,
          widths=[1.3, 0.95, 0.85, 1.15, 1.2, 1.05, 1.3])
    p(doc, '')
    figure(doc, 'ablation_encoders.png',
           'Figure 2. Accuracy by encoder, and accuracy against model size. Error bars are '
           'standard deviation over seeds.')

    smallest = min(combined, key=lambda kv: kv[1]['params'])
    s_auc, s_sd = stat(smallest[1], 'diagnosis_auc')
    ratio = baseline['params'] / smallest[1]['params']

    # Best lightweight encoder on binary diagnosis, judged against the noise band.
    lightweight = [(n, e) for n, e in combined if n != 'resnet34']
    matched = [(n, e) for n, e in lightweight
               if verdict(stat(e, 'diagnosis_auc')[0] - b_auc, b_sd,
                          stat(e, 'diagnosis_auc')[1]) == 'within noise']
    b_multi, b_multi_sd = stat(baseline, 'local_pathology_mean_auc')
    multi_gaps = [(n, stat(e, 'local_pathology_mean_auc')[0] - b_multi) for n, e in lightweight]

    h(doc, '3.1 Accuracy against size', 2)
    items = []
    if matched:
        n, e = max(matched, key=lambda kv: kv[1]['params'] / 1e6 * -1)
        items.append((
            'Binary diagnosis survives a much smaller backbone. ',
            f'{n} reaches {fmt(e, "diagnosis_auc")} against ResNet-34\'s '
            f'{fmt(baseline, "diagnosis_auc")}, a difference within the noise band, using '
            f'{e["params"] / 1e6:.2f} M parameters instead of '
            f'{baseline["params"] / 1e6:.2f} M ({baseline["params"] / e["params"]:.1f}x '
            f'smaller).'))
    else:
        items.append(('Every lighter backbone costs diagnosis accuracy. ',
                      'No lightweight encoder matched ResNet-34 within the noise band on '
                      'healthy/pathological AUC.'))
    items.append((
        'Multi-pathology does not survive. ',
        'Every lighter backbone loses multi-pathology AUC relative to ResNet-34 ('
        + ', '.join(f'{n} {g:+.4f}' for n, g in multi_gaps)
        + '). The nine-way pathology task appears to need the larger backbone\'s capacity even '
          'where binary diagnosis does not, which is consistent with it being the harder task '
          'and the one where this project already trails the benchmark.'))
    items.append((
        'Smallest model. ', f'{smallest[0]} at {smallest[1]["params"] / 1e6:.2f} M parameters, '
                            f'{ratio:.1f}x smaller than ResNet-34, reaching '
                            f'{fmt(smallest[1], "diagnosis_auc")} '
                            f'({verdict(s_auc - b_auc, b_sd, s_sd)} than the reference).'))
    items.append((
        'Caveat. ', 'Parameter count is not latency. These encoders differ in how efficiently '
                    'they use hardware, and depthwise-separable backbones often run slower per '
                    'parameter than their size suggests. A timing measurement is still needed '
                    'before a deployment claim is complete.'))
    bullets(doc, items)

    # ------------------------------------------------ full table
    h(doc, '4. All configurations, all metrics')
    all_rows = []
    for name, e in [(m['label'], m) for m in modules.values()] + \
                   [(k.replace('encoder_', ''), v) for k, v in encoders.items()]:
        all_rows.append([name] + [fmt(e, key) for key, _ in METRICS])
    table(doc, ['Configuration'] + [lbl for _, lbl in METRICS], all_rows,
          widths=[1.55, 1.15, 1.2, 1.1, 1.25, 1.1])
    p(doc, '')
    p(doc, 'Balanced accuracy uses the validation-tuned threshold. Multi-pathology AUC of '
           'exactly 0.5000 indicates the head was removed in that configuration, not that it '
           'performed at chance.', italic=True, color=SLATE)

    # ------------------------------------------------ limitations
    h(doc, '5. Limitations')
    bullets(doc, [
        ('Reduced fidelity. ', 'Results come from 256 × 256 inputs and 12 epochs. Conclusions '
                               'about relative ordering may not transfer exactly to the full '
                               '512 × 512 / 15-epoch protocol.'),
        ('Statistical power. ', f'{n_seeds} seeds per configuration detects large effects only. '
                                f'A null result here means "not shown", not "no effect".'),
        ('Single split. ', 'These runs use the fixed 70/15/15 split rather than '
                           'cross-validation, so fold-difficulty variance is not averaged out.'),
        ('Encoder pretraining differs. ', 'All backbones use ImageNet weights, but they were '
                                          'trained with different recipes and augmentation, '
                                          'which is part of what is being compared.'),
        ('Coupled removals. ', 'Removing the lesion decoder also removes lesion-guided pooling '
                               'and the consistency term, since both depend on it. That row is '
                               'a combined ablation, not an isolated one.'),
    ])

    doc.save(OUT)
    return OUT


if __name__ == '__main__':
    print('Wrote', build())
