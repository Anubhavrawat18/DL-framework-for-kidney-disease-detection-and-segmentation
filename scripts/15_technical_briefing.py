"""Builds the technical briefing: architecture, components, training, evaluation, and a Q&A.

This is a study document for defending the work in person, so it is written to be read
start to finish rather than skimmed. Numbers are pulled from the results files and from the
live model definition wherever possible, so the briefing cannot quote a figure the project
does not actually produce.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.data import load_manifest, parse_local_lesions
from src.model import MultiTaskNetV2, MultiTaskNetV3

OUT = os.path.join(PROJECT_ROOT, 'URI-CADS_Technical_Briefing.docx')
NAVY, INK, SLATE, GREEN, RED = '21295C', '1A2233', '5B6B79', '2E7D5B', 'B5483B'

BENCH_DIAG, BENCH_MULTI = 0.8741, 0.8190     # URI-CADS published, 5-fold CV


# ---------------------------------------------------------------- facts
def model_facts():
    v2, v3 = MultiTaskNetV2(pretrained=False), MultiTaskNetV3(pretrained=False)
    n = lambda m: sum(p.numel() for p in m.parameters())
    enc = sum(p.numel() for p in v2.encoder.parameters())
    dec = sum(p.numel() for p in v2.kidney_decoder.parameters())
    with torch.no_grad():
        feat = v2.encoder(torch.zeros(1, 3, 512, 512))
    return {'v2': n(v2), 'v3': n(v3), 'encoder': enc, 'decoder': dec,
            'feat_shape': tuple(feat.shape[1:]),
            'heads': n(v2) - enc - 2 * dec}


def data_facts():
    m = load_manifest()
    counts = collections.Counter()
    n_lesion = 0
    for _, row in m.iterrows():
        lesions = parse_local_lesions(row['extra_lines'])
        if lesions:
            n_lesion += 1
        for les in lesions:
            counts[les['pathology_id']] += 1
    splits = {s: (int((m['split'] == s).sum()),
                  int(m[m['split'] == s]['global_diagnosis'].sum()))
              for s in ('train', 'val', 'test')}
    return {'total': len(m), 'splits': splits, 'n_lesion': n_lesion,
            'per_id': dict(sorted(counts.items())),
            'pathological': int(m['global_diagnosis'].sum())}


def results():
    out = {}
    for name in ('phase2', 'phase3', 'phase4', 'phase5', 'phase5b', 'phase6_final'):
        with open(os.path.join(RESULTS_DIR, f'{name}_results.json'), encoding='utf-8') as fh:
            out[name] = json.load(fh)['test_metrics']
    for tag in ('base', 'v3', 'v3_ens3'):
        with open(os.path.join(RESULTS_DIR, f'phase8_kfold_cv_{tag}_results.json'),
                  encoding='utf-8') as fh:
            out[f'cv_{tag}'] = json.load(fh)['summary']
    with open(os.path.join(RESULTS_DIR, 'phase6_search_results.json'), encoding='utf-8') as fh:
        out['search'] = json.load(fh)
    with open(os.path.join(RESULTS_DIR, 'phase7_ablations_results.json'),
              encoding='utf-8') as fh:
        out['phase7'] = json.load(fh)
    return out


def p7(res, needle):
    """phase7_ablations_results.json is keyed by run name; match on the human label."""
    for entry in res['phase7'].values():
        if isinstance(entry, dict) and needle in entry.get('label', ''):
            return entry['test_metrics']
    raise KeyError(f'no Phase 7 ablation whose label contains {needle!r}')


# ---------------------------------------------------------------- doc helpers
def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text, *, italic=False, bold=False, size=10.5, color=INK, space=8, align=None):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space)
    if align is not None:
        para.alignment = align
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    return para


def bullets(doc, items, style='List Bullet'):
    for item in items:
        para = doc.add_paragraph(style=style)
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
        c = t.rows[0].cells[i]
        c.text = ''
        r = c.paragraphs[0].add_run(head)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ''
            cells[i].paragraphs[0].add_run(str(val)).font.size = Pt(9)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def qa(doc, question, answer, *, followups=None):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(3)
    para.paragraph_format.space_before = Pt(9)
    r = para.add_run('Q.  ' + question)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = RGBColor.from_string(NAVY)
    body = doc.add_paragraph()
    body.paragraph_format.space_after = Pt(3)
    body.paragraph_format.left_indent = Inches(0.25)
    run = body.add_run(answer)
    run.font.size = Pt(10.5)
    if followups:
        f = doc.add_paragraph()
        f.paragraph_format.space_after = Pt(6)
        f.paragraph_format.left_indent = Inches(0.25)
        fr = f.add_run('If pressed: ' + followups)
        fr.italic = True
        fr.font.size = Pt(9.5)
        fr.font.color.rgb = RGBColor.from_string(SLATE)


# ---------------------------------------------------------------- build
def build():
    mf, df, res = model_facts(), data_facts(), results()

    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Calibri'
    st.font.size = Pt(10.5)
    for name, size in (('Title', 25), ('Heading 1', 16), ('Heading 2', 12.5),
                       ('Heading 3', 11)):
        s = doc.styles[name]
        s.font.name = 'Calibri'
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor.from_string(NAVY)
        s.font.bold = True

    doc.add_heading('Technical Briefing', level=0)
    p(doc, 'Optimization-Based Deep Learning Framework for Kidney Disease Detection and '
           'Segmentation from Ultrasound Images', italic=True, size=12, color=SLATE)
    p(doc, 'A study guide for defending the architecture, the training procedure, the '
           'evaluation, and the results. Read Sections 1-8 to understand the system; use '
           'Section 10 to rehearse.', color=SLATE)

    # ============================================================ 1
    h(doc, '1. The problem and the data')
    p(doc, 'The task is computer-aided diagnosis on renal ultrasound. Given a single B-mode '
           'ultrasound image of a kidney, the system must decide whether the kidney is healthy '
           'or pathological, identify which pathologies are present, outline the kidney, and '
           'outline any lesions inside it.')
    p(doc, 'The dataset is URI-CADS, a public renal ultrasound benchmark. It is the same data '
           'the reference publication uses, which is what makes a direct numerical comparison '
           'possible.')

    tr, va, te = df['splits']['train'], df['splits']['val'], df['splits']['test']
    table(doc, ['Split', 'Images', 'Pathological', 'Healthy', 'Share pathological'],
          [['Train', tr[0], tr[1], tr[0] - tr[1], f'{tr[1] / tr[0]:.1%}'],
           ['Validation', va[0], va[1], va[0] - va[1], f'{va[1] / va[0]:.1%}'],
           ['Test', te[0], te[1], te[0] - te[1], f'{te[1] / te[0]:.1%}'],
           ['Total', df['total'], df['pathological'],
            df['total'] - df['pathological'], f'{df["pathological"] / df["total"]:.1%}']],
          widths=[1.3, 1.0, 1.2, 1.0, 1.4])

    h(doc, 'What each image comes with', level=2)
    bullets(doc, [
        ('The image. ', 'A B-mode ultrasound frame, including the scanner\'s own interface '
                        'furniture (depth scale, machine settings). Resized to 512 x 512.'),
        ('A kidney mask. ', 'A MATLAB .mat file holding a hand-traced binary mask of the '
                            'kidney outline. This is a true pixel-level annotation.'),
        ('A global diagnosis. ', 'One bit: healthy (0) or pathological (1).'),
        ('Global pathologies. ', 'A 2-element binary vector for conditions affecting the whole '
                                 'organ rather than a specific spot.'),
        (f'Lesion bounding boxes. ', f'Zero or more boxes, each with a pathology ID from 1 to '
                                     f'9. {df["n_lesion"]} of {df["total"]} images '
                                     f'({df["n_lesion"] / df["total"]:.0%}) carry at least one '
                                     f'box.'),
    ])
    p(doc, 'The lesion annotation is the single most important data caveat in the project. The '
           'dataset gives rectangles, not traced lesion outlines. The code converts each box '
           'into a filled rectangular mask and takes the union, so what the lesion decoder is '
           'trained against is a blocky approximation of the true lesion. A model that predicts '
           'a correctly placed but organically shaped lesion is penalised against a rectangle. '
           'Reported lesion Dice is therefore a lower bound on real localisation quality.',
      italic=True, color=SLATE)

    h(doc, 'Class imbalance', level=2)
    p(doc, f'Only {df["total"] - df["pathological"]} of {df["total"]} images are healthy '
           f'({(df["total"] - df["pathological"]) / df["total"]:.0%}). Two consequences follow, '
           f'and both come up in questioning. First, plain accuracy is not a meaningful '
           f'headline: a model that answers "pathological" every time already scores '
           f'{df["pathological"] / df["total"]:.0%}. Balanced accuracy - the mean of '
           f'sensitivity and specificity - is used instead. Second, unweighted cross-entropy is '
           f'minimised by leaning toward the majority class, which is exactly what was observed '
           f'early on: sensitivity 0.96 against specificity 0.46.')

    # ============================================================ 2
    h(doc, '2. What the model predicts')
    p(doc, 'One forward pass produces five outputs from one shared image representation. This '
           'is the defining property of the design: it is a single network doing five jobs, not '
           'five networks.')
    table(doc, ['Output', 'Shape (per image)', 'Type', 'What it means'],
          [['Diagnosis', 'scalar logit', 'Binary', 'Healthy vs pathological.'],
           ['Global pathology', '2 logits', 'Multi-label',
            'Whole-organ conditions; each independently present or absent.'],
           ['Local pathology', '9 logits', 'Multi-label',
            'Which of the nine annotated lesion types appear anywhere in the image.'],
           ['Kidney segmentation', '512 x 512 logits', 'Dense',
            'Per-pixel probability that the pixel is kidney tissue.'],
           ['Lesion segmentation', '512 x 512 logits', 'Dense',
            'Per-pixel probability that the pixel belongs to a lesion region.']],
          widths=[1.35, 1.25, 0.9, 4.4])
    p(doc, 'All five are logits. Sigmoid converts them to probabilities; segmentation masks are '
           'thresholded at 0.5, and the diagnosis threshold is tuned on validation data rather '
           'than left at 0.5.')
    p(doc, 'Multi-label, not multi-class. The pathology heads use independent sigmoids rather '
           'than a softmax, because a kidney can have several pathologies at once. A softmax '
           'would force the probabilities to compete and sum to one, which is the wrong '
           'assumption here.', italic=True, color=SLATE)

    # ============================================================ 3
    h(doc, '3. The architecture')
    p(doc, 'One encoder feeds five heads. The encoder learns a single representation that has '
           'to serve every task at once, which is where the multi-task benefit comes from - and '
           'also where the tension between tasks comes from.')

    h(doc, 'Data flow, end to end', level=2)
    ch, fh, fw = mf['feat_shape']
    bullets(doc, [
        ('Input. ', 'A 512 x 512 x 3 image, normalised to roughly [-1, 1] by (x - 0.5) / 0.5.'),
        ('Encoder. ', f'ResNet-34 with the classifier and average-pool removed, pretrained on '
                      f'ImageNet. It downsamples by a factor of 32, producing a feature map of '
                      f'{ch} channels at {fh} x {fw}. Every spatial position in that grid '
                      f'summarises a patch of the original image.'),
        ('Global average pooling. ', f'Collapses the {ch} x {fh} x {fw} map to a single '
                                     f'{ch}-dimensional vector describing the whole image. This '
                                     f'vector feeds the three classification heads.'),
        ('Classification heads. ', f'Three independent linear layers on that vector: '
                                   f'{ch} to 1 for diagnosis, {ch} to 2 for global pathology, '
                                   f'{ch} to 9 for local pathology.'),
        ('Segmentation decoders. ', f'Two separate decoders, one for kidney and one for '
                                    f'lesion, each taking the full {ch} x {fh} x {fw} feature '
                                    f'map (not the pooled vector - spatial detail is needed) '
                                    f'and upsampling it back to 512 x 512.'),
    ])

    h(doc, 'Inside a decoder', level=2)
    p(doc, 'Each decoder is five identical upsampling blocks followed by a 1 x 1 convolution. '
           'A block is: 3 x 3 convolution, batch normalisation, ReLU, then bilinear upsampling '
           'by 2. The channel count halves at each step - 512, 256, 128, 64, 32, 16 - and the '
           'spatial size doubles each time. Five doublings is exactly the factor of 32 the '
           'encoder removed, so the output lands back at full resolution. The final 1 x 1 '
           'convolution maps 16 channels to the single logit channel.')
    p(doc, 'This is a plain decoder, not a U-Net: there are no skip connections carrying '
           'high-resolution detail across from the encoder. That is a deliberate simplification '
           'and a fair thing to be asked about - see Section 10.', italic=True, color=SLATE)

    h(doc, 'Parameter budget', level=2)
    table(doc, ['Part', 'Parameters', 'Share'],
          [['ResNet-34 encoder (shared)', f'{mf["encoder"]:,}',
            f'{mf["encoder"] / mf["v2"]:.0%}'],
           ['Kidney decoder', f'{mf["decoder"]:,}', f'{mf["decoder"] / mf["v2"]:.0%}'],
           ['Lesion decoder', f'{mf["decoder"]:,}', f'{mf["decoder"] / mf["v2"]:.0%}'],
           ['Three classification heads', f'{mf["heads"]:,}', f'{mf["heads"] / mf["v2"]:.2%}'],
           ['Total (V2)', f'{mf["v2"]:,}', '100%'],
           ['Total (V3, with lesion-guided pooling)', f'{mf["v3"]:,}',
            f'+{mf["v3"] - mf["v2"]:,}']],
          widths=[3.1, 1.4, 1.0])
    p(doc, 'The classification heads are almost free. Nearly all the capacity is the shared '
           'encoder plus the two decoders, which is why adding classification tasks to a '
           'segmentation network costs so little and why the multi-task argument is a strong '
           'one on efficiency grounds alone.', italic=True, color=SLATE)

    # ============================================================ 4
    h(doc, '4. Role of each component')
    p(doc, 'The right way to justify a component is to say what measurably breaks without it. '
           'The figures below come from the single-task ablations and the module ablation '
           'study, not from intuition.')

    diag_only = p7(res, 'diagnosis only')
    kid_only = p7(res, 'kidney segmentation only')
    les_only = p7(res, 'lesion segmentation only')
    ph4 = res['phase4']

    table(doc, ['Component', 'Role', 'Evidence it earns its place'],
          [['Shared ResNet-34 encoder',
            'Converts the ultrasound into features every task reads. Pretrained on ImageNet '
            'so it starts with generic edge and texture detectors.',
            'Swapping it for lighter backbones costs multi-pathology AUC in every case '
            '(-0.053 EfficientNet-B0, -0.089 MobileNetV3-L, -0.094 ShuffleNetV2).'],
           ['Diagnosis head',
            'The headline clinical output: healthy or pathological.',
            f'Trained alone it reaches {diag_only["diagnosis_auc"]:.4f} AUC. Inside the '
            f'multi-task model it reaches {ph4["diagnosis_auc"]:.4f} - the auxiliary tasks are '
            f'worth about {ph4["diagnosis_auc"] - diag_only["diagnosis_auc"]:+.3f} AUC.'],
           ['Global pathology head',
            'Whole-organ findings, as a multi-label output.',
            'Provides extra supervision on the pooled vector at almost no parameter cost.'],
           ['Local pathology head',
            'Which of the nine lesion types are present.',
            'The task where the model still trails the benchmark, and the one lesion-guided '
            'pooling was built to improve.'],
           ['Kidney decoder',
            'Pixel-level kidney outline; also the anatomical frame the consistency term '
            'refers to.',
            f'Trained alone it reaches {kid_only["kidney_dice"]:.4f} Dice; in the '
            f'multi-task model {ph4["kidney_dice"]:.4f}. Segmentation pays a small price for '
            f'sharing, unlike classification which gains.'],
           ['Lesion decoder',
            'Pixel-level lesion region, and the attention source for lesion-guided pooling.',
            f'Alone it reaches {les_only["lesion_dice_positive_only"]:.4f} '
            f'lesion-bearing Dice; multi-task {ph4["lesion_dice_positive_only"]:.4f}. Removing '
            f'it costs 0.028 multi-pathology AUC.'],
           ['Anatomical consistency term',
            'A loss, not a module. Penalises predicted lesion probability lying outside the '
            'true kidney.',
            f'Cuts the violation rate from {ph4["consistency_violation_rate"]:.4f} (Phase 4) '
            f'to {res["phase5b"]["consistency_violation_rate"]:.4f} in its corrected form. No '
            f'measurable effect on classification.'],
           ['Lesion-guided pooling',
            'Lets the local-pathology head read features weighted by the lesion decoder\'s own '
            'predicted map, instead of a single global average.',
            '+0.0271 multi-pathology AUC under 5-fold cross-validation, and -0.0428 when '
            'removed in the 3-seed ablation. Two protocols, same direction.']],
          widths=[1.4, 2.5, 3.0])

    # ============================================================ 5
    h(doc, '5. How training works')
    bullets(doc, [
        ('Optimiser. ', 'Adam, learning rate 1e-4, batch size 4, 15 epochs, seed 42.'),
        ('Mixed precision. ', 'Forward and backward passes run in float16 under a gradient '
                              'scaler, which is what makes 512 x 512 training fit on a laptop '
                              'RTX 3050.'),
        ('Checkpoint selection. ', 'After each epoch the model is scored on validation and the '
                                   'best one is kept, rather than simply taking the last epoch.'),
        ('Augmentation. ', 'Geometric only: horizontal flip, rotation up to 10 degrees, 10% '
                           'scale jitter, 5% translation. Applied identically to image and '
                           'masks.'),
    ])
    p(doc, 'Two augmentation decisions are worth being able to defend. There is no vertical '
           'flip, because depth increases downward in an ultrasound sweep and flipping '
           'vertically produces an anatomically impossible image. There is no brightness, '
           'contrast or gamma jitter, because echogenicity is itself diagnostic - two of the '
           'dataset labels concern tissue brightness, so perturbing intensity destroys the '
           'signal the classifier must read. This was measured, not assumed: the aggressive '
           'intensity profile dropped diagnosis AUC from 0.9014 to 0.8363.')

    h(doc, 'The loss function', level=2)
    p(doc, 'The total loss is a weighted sum of six terms. Phases 2 to 5 set every weight to '
           '1.0; Phase 6 replaces them with an optimised vector.')
    table(doc, ['Term', 'Applies to', 'Function'],
          [['Diagnosis', 'Binary head', 'Binary cross-entropy with logits.'],
           ['Global pathology', '2-way multi-label head', 'Binary cross-entropy with logits.'],
           ['Local pathology', '9-way multi-label head', 'Binary cross-entropy with logits.'],
           ['Kidney segmentation', 'Kidney decoder',
            'Soft Dice loss, 1 - (2|A n B| + e) / (|A| + |B| + e), computed on sigmoid '
            'probabilities rather than thresholded masks so it stays differentiable.'],
           ['Lesion segmentation', 'Lesion decoder', 'Soft Dice loss, same form.'],
           ['Anatomical consistency', 'Lesion decoder + kidney ground truth',
            'Fraction of predicted lesion probability mass falling outside the true kidney.']],
          widths=[1.3, 1.6, 4.0])
    p(doc, 'Dice rather than cross-entropy for segmentation because the foreground is a small '
           'fraction of the image. Pixel-wise cross-entropy is dominated by the easy background '
           'and can be driven low by a model that predicts almost nothing; Dice measures '
           'overlap directly and does not reward that.', italic=True, color=SLATE)

    # ============================================================ 6
    h(doc, '6. The anatomical consistency constraint, and the bug in it')
    p(doc, 'This is the part of the project most worth presenting confidently, because it is a '
           'diagnosed negative result followed by a verified fix - which is a stronger story '
           'than a component that simply worked.')
    p(doc, 'The idea: a lesion inside a kidney cannot be outside the kidney. If the lesion '
           'decoder puts probability mass beyond the kidney boundary, that is anatomically '
           'impossible and should be penalised. The quantity reported is the violation rate - '
           'the fraction of predicted lesion mass lying outside the true kidney.')

    h(doc, 'What went wrong', level=2)
    p(doc, 'The original (Phase 5) term penalised the absolute lesion probability mass outside '
           'the kidney: mean of sigmoid(lesion) x (1 - kidney). The reported metric, however, '
           'is a ratio - outside mass divided by total mass. Those are not the same objective. '
           'The absolute form can be driven to zero by suppressing lesion probability '
           'everywhere, which shrinks the numerator and the denominator together and leaves the '
           'ratio untouched. The model found that shortcut.')
    table(doc, ['Configuration', 'Violation rate', 'Reading'],
          [['Phase 4 - no constraint at all', f'{ph4["consistency_violation_rate"]:.4f}',
            'Baseline'],
           ['Phase 5 - absolute-mass constraint',
            f'{res["phase5"]["consistency_violation_rate"]:.4f}',
            'Worse than no constraint at all'],
           ['Phase 5b - ratio constraint',
            f'{res["phase5b"]["consistency_violation_rate"]:.4f}',
            'Roughly a fourfold reduction']],
          widths=[2.8, 1.3, 2.3])
    p(doc, 'The fix was to optimise the ratio directly. Because a global rescaling of lesion '
           'probability leaves a ratio unchanged, the degenerate solution disappears and the '
           'only way to reduce the loss is to actually move lesion mass inside the kidney.')
    p(doc, 'Say this as a finding, not an apology. It shows the metric was audited rather than '
           'trusted, and the general lesson - that a surrogate loss must optimise the quantity '
           'you report, not merely correlate with it - is exactly the kind of point a reviewer '
           'rewards.', italic=True, color=SLATE)

    # ============================================================ 7
    h(doc, '7. Lesion-guided pooling')
    p(doc, 'The reference system predicts each pathology from an explicitly localised region. '
           'The baseline here predicts nine pathology types from one globally averaged feature '
           'vector, so a small lesion is averaged together with the entire image before the '
           'classifier ever sees it. That is a structural disadvantage, and it showed: '
           'multi-pathology AUC was the one metric clearly below the benchmark.')
    p(doc, 'The mechanism, step by step:')
    bullets(doc, [
        'Take the lesion decoder\'s output and apply a sigmoid to get a probability map.',
        'Detach it from the computation graph, so no gradient flows back through it.',
        'Downsample it to the encoder feature grid and normalise it to sum to one.',
        'Use it as attention weights: multiply the encoder features by the map and sum '
        'spatially, producing a lesion-focused feature vector.',
        'Concatenate that with the ordinary global average vector and feed the pair to the '
        'local-pathology head, which is therefore twice as wide.',
    ], style='List Number')
    p(doc, 'Why detach. Without it, the local-pathology loss could reshape the lesion decoder '
           'to make classification easier - inventing convenient "lesions" that improve the '
           'classifier while degrading segmentation. Detaching means the lesion decoder stays '
           'supervised only by its own Dice term, and the attention is read-only. Expect to be '
           'asked this; it is the most interesting design decision in the model.', italic=True,
      color=SLATE)
    cvb, cv3 = res['cv_base'], res['cv_v3']
    p(doc, f'Measured effect under 5-fold cross-validation: multi-pathology AUC rises from '
           f'{cvb["local_pathology_mean_auc"]["mean"]:.4f} to '
           f'{cv3["local_pathology_mean_auc"]["mean"]:.4f}, an improvement of '
           f'{cv3["local_pathology_mean_auc"]["mean"] - cvb["local_pathology_mean_auc"]["mean"]:+.4f}. '
           f'An independent 3-seed ablation at reduced fidelity found -0.0428 when the '
           f'component was removed, with every seed agreeing in direction.')

    # ============================================================ 8
    h(doc, '8. The optimization layer')
    p(doc, 'Phases 2 to 5 set every loss weight to 1.0 and used fixed image-enhancement '
           'parameters. Both are arbitrary. Phase 6 searches for better values using '
           'population-based metaheuristics.')

    h(doc, 'The search space - 10 dimensions', level=2)
    table(doc, ['Parameter', 'Range', 'What it controls'],
          [['6 loss weights', '0.1 to 5.0 each',
            'Relative importance of diagnosis, global pathology, local pathology, kidney '
            'segmentation, lesion segmentation, and consistency.'],
           ['CLAHE clip limit', '1.0 to 5.0', 'Strength of local contrast enhancement.'],
           ['CLAHE tile grid', '4 to 16', 'How locally that contrast is equalised.'],
           ['Denoise strength', '0 to 20', 'Non-local-means speckle suppression.'],
           ['Gamma', '0.5 to 2.0', 'Brightness curve.']],
          widths=[1.5, 1.2, 4.2])

    h(doc, 'The fitness function', level=2)
    p(doc, 'A single scalar maximised on the validation split:')
    p(doc, 'F  =  0.30 x diagnosis AUC  +  0.15 x local-pathology AUC  +  0.25 x kidney Dice  '
           '+  0.25 x lesion-bearing Dice  +  0.05 x (1 - violation rate)',
      bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    p(doc, 'All five components lie in [0, 1] and are higher-is-better, so no rescaling is '
           'needed. Note the fourth term: lesion Dice enters restricted to lesion-bearing '
           'images. Using the dataset-wide mean would have been a serious error, because about '
           'a third of images contain no lesion and a correct empty prediction scores near 1.0 '
           'on those - so the optimiser could have collected roughly 0.32 for free by learning '
           'to predict nothing at all.')

    h(doc, 'The comparison', level=2)
    p(doc, 'Three optimisers were compared under an identical budget of 60 fitness evaluations '
           'each, where one evaluation means training a model from scratch. Equal budget is the '
           'point: comparing optimisers by iteration count would be unfair, since they use '
           'different population sizes.')
    table(doc, ['Optimiser', 'Best fitness', 'Evaluations', 'Wall clock'],
          [[name, f'{v["best_fitness"]:.4f}', v['n_evaluations'],
            f'{v["wall_clock_hours"]:.2f} h']
           for name, v in res['search']['results'].items()],
          widths=[1.4, 1.2, 1.2, 1.2])
    p(doc, f'The Coati Optimization Algorithm won, but by {0.6080 - 0.6006:.4f} fitness over '
           f'Grey Wolf - far smaller than the run-to-run noise of this pipeline. The honest '
           f'statement is that the three optimisers are indistinguishable at this budget.',
      italic=True, color=SLATE)
    p(doc, 'Searching at reduced fidelity. Each evaluation trains a model, so full-fidelity '
           'search would be impossible. The search runs at smaller resolution, fewer epochs, '
           'and a cheaper denoiser, and only the single best configuration per optimiser is '
           'retrained at full fidelity. This is the standard low-fidelity-proxy pattern; the '
           'assumption it rests on is that the proxy ranks configurations consistently with '
           'full fidelity, not that it reproduces absolute values.')

    # ============================================================ 9
    h(doc, '9. How the system is evaluated')
    h(doc, 'Metrics and why each one', level=2)
    table(doc, ['Metric', 'Why it is used'],
          [['AUC', 'Threshold-free, so it measures ranking quality independently of where the '
                   'decision boundary is placed. This is the number the reference publication '
                   'reports, which makes it the comparable one.'],
           ['Balanced accuracy', 'Mean of sensitivity and specificity. Plain accuracy is '
                                 'misleading at 77% prevalence; balanced accuracy is not.'],
           ['Sensitivity / specificity', 'What a clinical reader asks for. Missing disease and '
                                         'false alarms have very different costs.'],
           ['Dice', 'Overlap between predicted and true region, from 0 to 1.'],
           ['Lesion Dice, lesion-bearing only', 'The dataset-wide lesion Dice is close to '
                                                'meaningless: a model with no lesion head at '
                                                'all scores 0.3188, because empty predictions '
                                                'on lesion-free images score near 1.0. '
                                                'Restricting to the 203 lesion-bearing test '
                                                'images separates a real model (about 0.50) '
                                                'from nothing (0.0000).'],
           ['Violation rate', 'Fraction of predicted lesion mass outside the true kidney; the '
                              'anatomical plausibility check.']],
          widths=[1.7, 5.2])

    h(doc, 'Protocol', level=2)
    bullets(doc, [
        ('Threshold tuning. ', 'The decision threshold is chosen on the validation split to '
                               'maximise balanced accuracy, then applied unchanged to test. It '
                               'is never tuned on test data.'),
        ('Cross-validation. ', '5-fold, matching the reference publication, so the comparison '
                               'is like for like rather than one split against five.'),
        ('Seed variance. ', 'Retraining an identical recipe with a different random seed moves '
                            'diagnosis AUC by about 0.026 standard deviation, with a spread up '
                            'to 0.074. Any single-split difference smaller than that carries no '
                            'information.'),
        ('Ensembling. ', 'The final system averages the predicted probabilities of three models '
                         'trained with different seeds.'),
    ])
    p(doc, 'The seed-variance measurement is load-bearing for the whole results section. It is '
           'the reason improvements are reported with error bars and paired tests rather than '
           'as single numbers, and it is worth volunteering rather than waiting to be asked - '
           'it demonstrates the results were stress-tested.', italic=True, color=SLATE)

    # ============================================================ 10
    h(doc, '10. Results')
    h(doc, 'Single-split progression', level=2)
    # Phases 2 and 3 are the three-head architecture: they have no local-pathology head and no
    # lesion decoder at all. Their stored 0.5000 / 0.0000 entries are placeholders for an absent
    # head, not measured performance, so printing them as numbers would invite the reader to
    # compare a missing head against a trained one.
    rows, absent = [], {'phase2', 'phase3'}
    for key, label in [('phase2', 'Phase 2 - 3-head baseline'),
                       ('phase3', 'Phase 3 - fixed enhancement'),
                       ('phase4', 'Phase 4 - full 5-head multi-task'),
                       ('phase5', 'Phase 5 - consistency (absolute)'),
                       ('phase5b', 'Phase 5b - consistency (ratio)'),
                       ('phase6_final', 'Phase 6 - optimized weights + enhancement')]:
        m = res[key]
        na = key in absent
        rows.append([label, f'{m["diagnosis_auc"]:.4f}',
                     'no head' if na else f'{m["local_pathology_mean_auc"]:.4f}',
                     f'{m["kidney_dice"]:.4f}',
                     'no head' if na else f'{m["lesion_dice_positive_only"]:.4f}',
                     'no head' if na else f'{m["consistency_violation_rate"]:.4f}'])
    table(doc, ['Phase', 'Diagnosis AUC', 'Multi-path. AUC', 'Kidney Dice',
                'Lesion Dice+', 'Violation'], rows,
          widths=[2.5, 1.0, 1.0, 0.95, 0.9, 0.85])
    p(doc, 'Phases 2 and 3 use the three-head architecture, which has no local-pathology head '
           'and no lesion decoder, so those columns are marked "no head" rather than scored.',
      italic=True, size=9, color=SLATE)
    p(doc, 'Read this table with care, and say so before being challenged on it. These are '
           'single-split numbers, and the seed noise floor is about 0.026 AUC. Most of the '
           'movement in the diagnosis column is therefore not interpretable. The columns that '
           'do move meaningfully are kidney Dice and the violation rate.', italic=True,
      color=SLATE)

    h(doc, 'Cross-validated results - the defensible numbers', level=2)
    cve = res['cv_v3_ens3']
    table(doc, ['System', 'Diagnosis AUC', 'Multi-pathology AUC', 'Kidney Dice'],
          [['Baseline multi-task (V2)',
            f'{cvb["diagnosis_auc"]["mean"]:.4f} ± {cvb["diagnosis_auc"]["std"]:.4f}',
            f'{cvb["local_pathology_mean_auc"]["mean"]:.4f} ± '
            f'{cvb["local_pathology_mean_auc"]["std"]:.4f}',
            f'{cvb["kidney_dice"]["mean"]:.4f} ± {cvb["kidney_dice"]["std"]:.4f}'],
           ['+ lesion-guided pooling (V3)',
            f'{cv3["diagnosis_auc"]["mean"]:.4f} ± {cv3["diagnosis_auc"]["std"]:.4f}',
            f'{cv3["local_pathology_mean_auc"]["mean"]:.4f} ± '
            f'{cv3["local_pathology_mean_auc"]["std"]:.4f}',
            f'{cv3["kidney_dice"]["mean"]:.4f} ± {cv3["kidney_dice"]["std"]:.4f}'],
           ['+ 3-seed ensemble (final)',
            f'{cve["diagnosis_auc"]["mean"]:.4f} ± {cve["diagnosis_auc"]["std"]:.4f}',
            f'{cve["local_pathology_mean_auc"]["mean"]:.4f} ± '
            f'{cve["local_pathology_mean_auc"]["std"]:.4f}',
            f'{cve["kidney_dice"]["mean"]:.4f} ± {cve["kidney_dice"]["std"]:.4f}'],
           ['URI-CADS published benchmark', f'{BENCH_DIAG:.4f}', f'{BENCH_MULTI:.4f}', '-']],
          widths=[2.2, 1.6, 1.7, 1.4])
    gap_d = cve['diagnosis_auc']['mean'] - BENCH_DIAG
    gap_m = cve['local_pathology_mean_auc']['mean'] - BENCH_MULTI
    p(doc, f'Headline: the final system exceeds the published benchmark on healthy versus '
           f'pathological classification by {gap_d:+.4f} AUC '
           f'({cve["diagnosis_auc"]["mean"]:.4f} against {BENCH_DIAG:.4f}), and remains below '
           f'it on multi-pathology classification by {gap_m:+.4f} '
           f'({cve["local_pathology_mean_auc"]["mean"]:.4f} against {BENCH_MULTI:.4f}). Both '
           f'statements should be made together.', bold=True)

    # ============================================================ 11
    h(doc, '11. Anticipated questions')
    p(doc, 'Grouped by area. Answers are phrased the way they should be said aloud.',
      italic=True, color=SLATE)

    h(doc, 'Architecture', level=2)
    qa(doc, 'Why multi-task learning instead of separate models for each task?',
       f'Two reasons, one measured and one practical. Measured: the diagnosis head reaches '
       f'{diag_only["diagnosis_auc"]:.4f} AUC trained alone and '
       f'{ph4["diagnosis_auc"]:.4f} inside the multi-task model, so the segmentation tasks act '
       f'as a regulariser that improves classification by about 0.056 AUC. Practical: the three '
       f'classification heads are {mf["heads"]:,} parameters out of {mf["v2"] / 1e6:.1f} '
       f'million, under {mf["heads"] / mf["v2"]:.2%} of the model, so those '
       f'outputs are nearly free once the encoder and decoders exist. Five separate models '
       f'would cost roughly five times the memory and give a worse classifier.',
       followups='Segmentation does pay a small price for sharing - kidney Dice is '
                 f'{kid_only["kidney_dice"]:.4f} single-task against '
                 f'{ph4["kidney_dice"]:.4f} multi-task. The trade is net positive, not free.')
    qa(doc, 'Why ResNet-34 and not something deeper, or a transformer?',
       'ResNet-34 is deep enough to give strong features while fitting 512 x 512 training on a '
       '4 GB laptop GPU, and it is a standard, reproducible choice. The reference system uses '
       'ResNet-50. A transformer would need substantially more data than 1,985 images to train '
       'or fine-tune reliably.',
       followups='Four encoders were compared empirically. EfficientNet-B0 matched ResNet-34 on '
                 'binary diagnosis within noise at 2.3 times fewer parameters, but every '
                 'lighter backbone lost multi-pathology AUC.')
    qa(doc, 'Why is there no U-Net skip connection in the decoder?',
       'It is a plain decoder by design: the same feature map feeds the classification heads, '
       'so keeping the path simple keeps the shared representation the thing that is being '
       'tested. Skip connections would very likely improve boundary sharpness, and kidney Dice '
       'is already about 0.89 to 0.91 without them, so this is a clear avenue for improvement '
       'rather than a claim that skips would not help.',
       followups='Be direct that this is untested, not disproven.')
    qa(doc, 'Why two separate decoders rather than one with two output channels?',
       'Separate decoders let each task shape its own upsampling path, and they make the '
       'ablation clean - the lesion decoder can be removed without touching kidney '
       'segmentation. A shared decoder with two channels would be cheaper and is a reasonable '
       'alternative that was not tested.')
    qa(doc, 'Why global average pooling rather than attention for the classification heads?',
       'Global average pooling is the standard, parameter-free choice and it is what the '
       'baseline uses. The limitation is exactly the one lesion-guided pooling addresses: '
       'averaging dilutes a small lesion across the whole image. So the answer is that the '
       'model uses both - plain averaging for the two whole-organ heads, and averaging plus '
       'lesion-weighted pooling for the head that has to find small localised findings.')

    h(doc, 'Data and labels', level=2)
    qa(doc, 'What exactly are the nine local pathology classes?',
       'They are the lesion-type identifiers annotated in the URI-CADS dataset, encoded as '
       'integers 1 to 9 in the bounding-box annotations. The code treats them as opaque '
       'indices, so the clinical names must be quoted from the dataset publication rather than '
       'from this implementation.',
       followups='This is a genuine gap - look the names up in the URI-CADS paper before the '
                 'meeting and be ready to name them. It is the most likely question you cannot '
                 'currently answer from the code.')
    qa(doc, 'Your lesion ground truth is bounding boxes. Is that not a problem for segmentation?',
       'Yes, and it is stated openly. The boxes are converted to filled rectangles, so the '
       'lesion decoder is trained against a coarse approximation and is penalised for '
       'predicting an accurately shaped non-rectangular lesion. Reported lesion Dice is '
       'therefore pessimistic. What the lesion branch really provides is reliable localisation, '
       'which is what the attention mechanism consumes - and there the box is perfectly '
       'adequate.')
    qa(doc, 'How do you handle the class imbalance?',
       f'Three ways. The headline metric is balanced accuracy rather than accuracy, because '
       f'always answering "pathological" already scores {df["pathological"] / df["total"]:.0%}. '
       f'The decision threshold is tuned on validation rather than fixed at 0.5. And weighted '
       f'cross-entropy and focal loss were both implemented and tested as training-time '
       f'corrections.')
    qa(doc, 'Is there patient-level leakage between your splits?',
       'The splits come from the dataset manifest and are applied at image level. If the '
       'dataset contains multiple frames from the same patient, image-level splitting could '
       'place them on both sides. This has not been verified against patient identifiers and '
       'should be checked before publication.',
       followups='Do not claim it is clean. Say it is unverified and that you will confirm it - '
                 'an overclaim here is far more damaging than the admission.')

    h(doc, 'Training and losses', level=2)
    qa(doc, 'Why Dice loss for segmentation instead of cross-entropy?',
       'Because the foreground is a small fraction of the image. Pixel-wise cross-entropy is '
       'dominated by the easy background and can be made small by a model that predicts almost '
       'nothing. Dice measures overlap directly, so an empty prediction scores zero on a '
       'lesion-bearing image.')
    qa(doc, 'Why no intensity augmentation? Most ultrasound pipelines use it.',
       'Because echogenicity is diagnostic in renal ultrasound - two of the dataset\'s labels '
       'concern tissue brightness, so jittering brightness or gamma perturbs the exact feature '
       'the classifier must read. These images are also dark, mean intensity around 25 out of '
       '255, so a 15% brightness shift moves pixels by more than the image\'s own mean. It was '
       'tested rather than assumed: the aggressive intensity profile dropped diagnosis AUC from '
       '0.9014 to 0.8363.')
    qa(doc, 'How do you select the final model within a training run?',
       'By validation score after each epoch, keeping the best rather than the last. For the '
       'multi-task phases the selection score is the mean of kidney and lesion validation Dice.',
       followups='A weakness worth owning: that selection score uses the dataset-wide lesion '
                 'Dice, which is the same quantity criticised elsewhere for rewarding empty '
                 'predictions. Using the lesion-bearing figure for selection too would be more '
                 'consistent, and it has not been changed.')

    h(doc, 'Optimization', level=2)
    qa(doc, 'Why a metaheuristic instead of grid search, random search, or Bayesian optimization?',
       'The space is 10-dimensional and continuous, so grid search is infeasible - even three '
       'values per dimension would be 59,049 evaluations, and one evaluation means training a '
       'model. Population-based metaheuristics need no gradient and no smoothness assumption. '
       'Bayesian optimisation would be the stronger comparison and was not run; that is a fair '
       'criticism.',
       followups='Random search at matched budget is the baseline a reviewer will ask for. It '
                 'is cheap to add and would strengthen the paper considerably.')
    qa(doc, 'Did the optimization actually improve anything?',
       'Honestly: not on classification. The Coati algorithm won the three-way comparison by '
       '0.0074 fitness over Grey Wolf, which is well inside the noise of this pipeline, and the '
       'optimised configuration did not beat equal loss weights on diagnosis AUC. What it did '
       'improve is kidney Dice, up to 0.9081, and the violation rate, down about fivefold. The '
       'fitness function put half its weight on segmentation, so the search did exactly what it '
       'was told - it traded classification for segmentation.',
       followups='This is the project\'s weakest claim and the framing most likely to be '
                 'challenged. Say it plainly first; a reviewer who finds it themselves will '
                 'trust the rest of the work less.')
    qa(doc, 'Why compare three optimizers at all?',
       'To test whether the choice of metaheuristic matters, under an equal budget of 60 model '
       'trainings each. The answer is that it does not, at this budget - which is itself a '
       'reportable result and more useful than presenting only the winner.')

    h(doc, 'Evaluation and statistics', level=2)
    qa(doc, 'How do you know your improvements are real and not random variation?',
       'By measuring the noise first. Retraining an identical recipe under different seeds '
       'moves diagnosis AUC by roughly 0.026 standard deviation, with a spread up to 0.074. '
       'Anything smaller than that on a single split is not evidence. Comparisons that matter '
       'are run under matched 5-fold cross-validation with paired tests, and component claims '
       'are reported with seeds and error bars.')
    qa(doc, 'Why report AUC rather than accuracy?',
       'AUC is threshold-free, so it measures how well the model ranks cases independently of '
       'where the decision boundary sits, and it is the metric the reference publication '
       'reports - so it is the one that permits a fair comparison. Balanced accuracy, '
       'sensitivity and specificity are reported alongside it, because AUC alone cannot answer '
       '"what happens at the operating point".')
    qa(doc, 'Why is your lesion Dice only about 0.49? That seems low.',
       'Three reasons, in order of size. The ground truth is rectangular, so a correctly shaped '
       'prediction is penalised. Lesions are small and low-contrast in ultrasound. And the '
       'figure quoted is restricted to lesion-bearing images, which is the honest denominator - '
       'the dataset-wide number would be about 0.34 and would look better only because empty '
       'predictions on lesion-free images score near 1.0.')
    qa(doc, 'Is 5-fold cross-validation on 1,985 images enough?',
       'It is the same protocol as the reference publication, which is what makes the '
       'comparison valid. It is not a substitute for an external cohort, and no external '
       'validation has been done - the model has never been tested on data from a different '
       'scanner or hospital.')

    h(doc, 'Results and comparison', level=2)
    qa(doc, 'How does this compare to the published URI-CADS results?',
       f'Above it on binary classification, below it on multi-pathology. Final system '
       f'{cve["diagnosis_auc"]["mean"]:.4f} ± {cve["diagnosis_auc"]["std"]:.4f} AUC against '
       f'{BENCH_DIAG:.4f} published, and '
       f'{cve["local_pathology_mean_auc"]["mean"]:.4f} ± '
       f'{cve["local_pathology_mean_auc"]["std"]:.4f} against {BENCH_MULTI:.4f}. Both under '
       f'5-fold cross-validation on the same dataset.',
       followups='The reference uses a two-stage detector, Mask R-CNN plus Faster R-CNN with a '
                 'ResNet-50 backbone, which localises explicitly. That architectural advantage '
                 'is the most likely explanation for its stronger multi-pathology result.')
    qa(doc, 'What is the single biggest contribution of this work?',
       'Two things that survive scrutiny. First, lesion-guided pooling: a way to give a '
       'classification head localised evidence by reusing the segmentation branch it already '
       'has, worth about +0.027 multi-pathology AUC under cross-validation and confirmed by an '
       'independent ablation. Second, the diagnosed failure of the anatomical consistency loss '
       'and its correction, which is a transferable lesson about surrogate objectives.')
    qa(doc, 'What would you do next with more time?',
       'Validate on an external cohort; add random search at matched budget as an optimiser '
       'baseline; add skip connections to the decoder; and re-run the optimiser with a '
       'classification-weighted fitness function, since the current one was told to prioritise '
       'segmentation and duly did.')

    h(doc, 'Hard questions', level=2)
    qa(doc, 'Your title says "optimization-based". Does the optimization carry the paper?',
       'No, and it should not be presented as if it does. The metaheuristic layer improved '
       'segmentation and anatomical plausibility but did not improve classification over equal '
       'weights, and the three optimisers are indistinguishable at the budget tested. The '
       'results that hold up are the multi-task architecture, lesion-guided pooling, and the '
       'consistency-loss correction.',
       followups='The framing is under review for exactly this reason. Volunteering that is '
                 'much stronger than defending a title the evidence does not support.')
    qa(doc, 'Several of your components show no measurable effect. Why keep them?',
       'Because "no measurable effect" at three seeds means not shown, not disproven - the '
       'study is powered for large effects only. The components that are kept on evidence are '
       'lesion-guided pooling and the ratio-form consistency term, which improves the metric it '
       'was designed to improve. Components without evidence are reported as such rather than '
       'claimed.')
    qa(doc, 'Is this clinically usable?',
       'Not yet, and that claim is not made. It has been validated on one public dataset with '
       'no external cohort, no inference-latency measurement, no prospective evaluation and no '
       'reader study. At 77% prevalence the operating point also still needs to be chosen '
       'deliberately for whichever error - a missed pathology or a false alarm - matters more '
       'in the intended use.')

    # ============================================================ 12
    h(doc, '12. Weaknesses to state before you are asked')
    p(doc, 'Raising these yourself converts each one from a vulnerability into evidence that '
           'you understand your own system.')
    bullets(doc, [
        ('The optimization does not improve classification. ', 'It improves segmentation and '
         'anatomical consistency. The three metaheuristics are statistically indistinguishable '
         'at the budget tested.'),
        ('Multi-pathology AUC is below the benchmark. ', f'About {abs(gap_m):.3f} short. The '
         f'likely cause is structural - the reference localises explicitly with a detector.'),
        ('No external validation. ', 'One dataset, one scanner population.'),
        ('Lesion ground truth is boxes. ', 'Reported lesion Dice understates real performance, '
                                           'but cannot be corrected without better annotation.'),
        ('Patient-level split integrity is unverified. ', 'Splits are applied per image.'),
        ('No random-search baseline for the optimizer. ', 'The obvious comparison a reviewer '
                                                          'will ask for.'),
        ('Model selection uses a criticised metric. ', 'Checkpoints are chosen on dataset-wide '
         'lesion Dice, the same quantity argued elsewhere to be uninformative.'),
        ('No latency measurement. ', 'Parameter counts are known, but inference time on target '
                                     'hardware has not been measured, so no deployability claim '
                                     'is made.'),
    ])

    # ============================================================ 13
    h(doc, '13. One-minute summary')
    p(doc, 'If you are asked to describe the whole system in a sentence or two, say this:',
      italic=True, color=SLATE)
    p(doc, 'A single ResNet-34 encoder feeds five heads that jointly predict a healthy versus '
           'pathological diagnosis, two whole-organ pathology labels, nine lesion-type labels, '
           'a kidney segmentation mask and a lesion segmentation mask. The lesion branch is '
           'reused as an attention map so the lesion-type classifier reads localised evidence '
           'rather than a global average, and an anatomical consistency term keeps predicted '
           'lesions inside the predicted kidney. Loss weights and image-enhancement parameters '
           'are tuned by a population-based optimizer against a validation fitness score. Under '
           'matched 5-fold cross-validation the final ensemble reaches '
           f'{cve["diagnosis_auc"]["mean"]:.3f} AUC for healthy versus pathological, above the '
           f'{BENCH_DIAG:.3f} published for this dataset, and '
           f'{cve["local_pathology_mean_auc"]["mean"]:.3f} for multi-pathology, below the '
           f'{BENCH_MULTI:.3f} published.', bold=True)

    doc.save(OUT)
    return OUT


if __name__ == '__main__':
    print('Wrote', build())
