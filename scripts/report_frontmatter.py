"""Abstract, acknowledgement, contents and the three list pages."""
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

C = WD_ALIGN_PARAGRAPH.CENTER
J = WD_ALIGN_PARAGRAPH.JUSTIFY
L = WD_ALIGN_PARAGRAPH.LEFT
R = WD_ALIGN_PARAGRAPH.RIGHT

FONT = 'Times New Roman'
BODY_PT = 12
PH = '<<{}>>'


def abstract(rp, r, cv, p7, TITLE):
    auc, auc_sd = cv(r, 'v3_ens3', 'diagnosis_auc')
    loc, loc_sd = cv(r, 'v3_ens3', 'local_pathology_mean_auc')
    base_loc, _ = cv(r, 'base', 'local_pathology_mean_auc')
    v3_loc, _ = cv(r, 'v3', 'local_pathology_mean_auc')
    diag_only = p7(r, 'diagnosis only')['diagnosis_auc']
    mt = r['phase4']['test_metrics']['diagnosis_auc']

    rp.para('ABSTRACT', size=14, bold=True, underline=True, align=C,
            spacing=1.5, after=18)
    rp.para(
        'Renal ultrasound is the first-line imaging modality for suspected kidney '
        'disease because it is inexpensive, portable and free of ionising radiation. '
        'Its interpretation, however, is operator dependent, and the images are '
        'degraded by speckle noise and poor soft-tissue contrast, so subtle findings '
        'are easily missed. Existing computer-aided diagnosis systems for this '
        'modality either classify the whole image, giving no spatial evidence for the '
        'decision, or employ heavy region-proposal detectors that localise each '
        'finding at considerable computational cost.', align=J, after=10)
    rp.para(
        'This project develops a single-encoder multi-task convolutional network that '
        'performs five tasks in one forward pass: healthy-versus-pathological '
        'diagnosis, whole-organ pathology recognition, nine-way lesion-type '
        'recognition, kidney segmentation and lesion segmentation. The system is '
        'developed and evaluated on the public URI-CADS benchmark of 1,985 annotated '
        'renal ultrasound images.', align=J, after=10)
    rp.para(
        'Three principal findings are reported. First, multi-task training materially '
        'improves classification: the diagnosis head reaches {:.4f} AUC when trained '
        'alone and {:.4f} within the multi-task model, an improvement obtained at '
        'negligible parameter cost because the three classification heads account for '
        'under 0.03 per cent of the network. Second, a lesion-guided pooling mechanism '
        'is proposed, in which the lesion segmentation branch is reused as a detached '
        'spatial attention map so that the lesion-type classifier reads localised '
        'rather than globally averaged evidence; this raises multi-pathology AUC from '
        '{:.4f} to {:.4f} under matched five-fold cross-validation. Third, an '
        'anatomical consistency constraint expressed as absolute probability mass is '
        'shown to admit a degenerate minimiser that worsens the very metric it '
        'targets, and a corrected ratio formulation is derived which reduces the '
        'anatomical violation rate approximately fourfold.'.format(
            diag_only, mt, base_loc, v3_loc), align=J, after=10)
    rp.para(
        'Combined with seed ensembling, the final system attains {:.4f} ± {:.4f} '
        'AUC for healthy-versus-pathological classification under five-fold '
        'cross-validation, exceeding the 0.8741 published for the same benchmark by a '
        'dual-detector architecture built on a larger backbone. Multi-pathology AUC '
        'reaches {:.4f} ± {:.4f}. A matched-budget comparison of three '
        'population-based metaheuristics for loss weighting and image enhancement is '
        'additionally reported; the three algorithms are found to be statistically '
        'indistinguishable, which is presented as a negative result of practical '
        'value.'.format(auc, auc_sd, loc, loc_sd), align=J, after=14)
    rp.para(
        'Keywords: renal ultrasound, computer-aided diagnosis, multi-task learning, '
        'attention mechanism, medical image segmentation, deep learning.',
        align=J, italic=True)
    rp.page_break()


def acknowledgement(rp):
    rp.para('ACKNOWLEDGEMENT', size=14, bold=True, underline=True, align=C,
            spacing=1.5, after=18)
    rp.para('It is my pleasure to express with deep sense of gratitude to '
            '{}, {}, School of Computer Science and Engineering, Vellore Institute of '
            'Technology, Chennai, for the constant guidance, continual encouragement '
            'and understanding extended throughout the course of this project. I am '
            'deeply indebted for the valuable technical insight and for the freedom '
            'given to pursue the ideas reported here.'.format(
                PH.format('GUIDE NAME'), PH.format('DESIGNATION')), align=J, after=10)
    rp.para('It is with gratitude that I would like to extend my thanks to the '
            'visionary leader {}, Chancellor, Vellore Institute of Technology, and to '
            '{}, Vice President, and {}, Vice Chancellor, for providing an excellent '
            'academic environment.'.format(
                PH.format('CHANCELLOR NAME'), PH.format('VICE PRESIDENT NAME'),
                PH.format('VICE CHANCELLOR NAME')), align=J, after=10)
    rp.para('Special mention to {}, Dean, {}, Associate Dean, and {}, Head of '
            'Department, School of Computer Science and Engineering, for their '
            'encouragement and for the facilities made available for carrying out '
            'this work.'.format(PH.format('DEAN NAME'),
                                PH.format('ASSOCIATE DEAN NAME'),
                                PH.format('HoD NAME')), align=J, after=10)
    rp.para('My sincere thanks to all the faculty members and staff at Vellore '
            'Institute of Technology, Chennai, who extended their support during the '
            'course of this project. I also thank my family and friends for their '
            'constant encouragement.', align=J, after=24)
    rp.blank(2)
    rp.para('Place: Chennai', align=L, spacing=1.5, after=6)
    rp.para('Date: ' + PH.format('DATE'), align=L, spacing=1.5, after=18)
    rp.para(PH.format('NAME OF THE STUDENT'), size=12, bold=True, align=R,
            spacing=1.0)
    rp.page_break()


CONTENTS = [
    ('front', 'DECLARATION', 'ii'),
    ('front', 'CERTIFICATE', 'iii'),
    ('front', 'ABSTRACT', 'iv'),
    ('front', 'ACKNOWLEDGEMENT', 'v'),
    ('front', 'LIST OF FIGURES', 'vii'),
    ('front', 'LIST OF TABLES', 'viii'),
    ('front', 'LIST OF ACRONYMS', 'ix'),
    ('chap', 'CHAPTER 1  INTRODUCTION', '1'),
    ('sec', '1.1  Introduction', '1'),
    ('sec', '1.2  Overview of Renal Ultrasound Imaging', '2'),
    ('sec', '1.3  Challenges in Automated Renal Ultrasound Analysis', '3'),
    ('sec', '1.4  Problem Statement', '4'),
    ('sec', '1.5  Objectives', '5'),
    ('sec', '1.6  Scope of the Project', '6'),
    ('sec', '1.7  Organisation of the Report', '7'),
    ('chap', 'CHAPTER 2  BACKGROUND AND LITERATURE SURVEY', '8'),
    ('sec', '2.1  Introduction', '8'),
    ('sec', '2.2  Deep Learning for Renal Ultrasound Classification', '8'),
    ('sec', '2.3  Kidney Segmentation in Ultrasound', '10'),
    ('sec', '2.4  Multi-Task Learning in Medical Imaging', '11'),
    ('sec', '2.5  Attention and Weakly Supervised Localisation', '12'),
    ('sec', '2.6  Metaheuristic Optimisation in Medical Imaging', '13'),
    ('sec', '2.7  Research Gaps Identified', '14'),
    ('chap', 'CHAPTER 3  SYSTEM DESIGN AND METHODOLOGY', '16'),
    ('sec', '3.1  Introduction', '16'),
    ('sec', '3.2  Dataset Description', '16'),
    ('sec', '3.3  Preprocessing and Image Enhancement', '18'),
    ('sec', '3.4  Data Augmentation', '19'),
    ('sec', '3.5  Proposed Network Architecture', '20'),
    ('sec', '3.6  Lesion-Guided Pooling', '23'),
    ('sec', '3.7  Loss Functions', '25'),
    ('sec', '3.8  Anatomical Consistency Constraint', '27'),
    ('sec', '3.9  Metaheuristic Optimisation Framework', '29'),
    ('chap', 'CHAPTER 4  IMPLEMENTATION', '32'),
    ('sec', '4.1  Hardware and Software Environment', '32'),
    ('sec', '4.2  Training Configuration', '33'),
    ('sec', '4.3  Evaluation Metrics', '34'),
    ('sec', '4.4  Experimental Protocol', '36'),
    ('sec', '4.5  Software Module Organisation', '37'),
    ('chap', 'CHAPTER 5  RESULTS AND DISCUSSION', '39'),
    ('sec', '5.1  Measurement Noise Floor', '39'),
    ('sec', '5.2  Phase-wise Development Results', '40'),
    ('sec', '5.3  Benefit of Multi-Task Learning', '42'),
    ('sec', '5.4  Effect of Lesion-Guided Pooling', '43'),
    ('sec', '5.5  Comparison with the Benchmark System', '44'),
    ('sec', '5.6  Anatomical Consistency Constraint', '46'),
    ('sec', '5.7  Component Ablation Study', '47'),
    ('sec', '5.8  Encoder Comparison', '49'),
    ('sec', '5.9  Metaheuristic Optimiser Comparison', '50'),
    ('sec', '5.10  Qualitative Results', '52'),
    ('sec', '5.11  Discussion', '53'),
    ('chap', 'CHAPTER 6  CONCLUSION AND FUTURE WORK', '56'),
    ('sec', '6.1  Conclusion', '56'),
    ('sec', '6.2  Limitations', '57'),
    ('sec', '6.3  Future Work', '58'),
    ('chap', 'REFERENCES', '60'),
    ('chap', 'APPENDICES', '64'),
]


def _dotted(rp, text, page, bold=False, indent=0.0):
    p = rp.doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(0)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(13.5), 2, 2)
    r = p.add_run(text + '\t' + page)
    r.font.name = FONT
    r.font.size = Pt(BODY_PT)
    r.bold = bold


def contents(rp):
    rp.para('CONTENTS', size=14, bold=True, underline=True, align=C,
            spacing=1.5, after=14)
    rp.para('Page numbers below are indicative. Update them once the final layout is '
            'fixed, or replace this list with an automatic table of contents '
            '(References → Table of Contents in Word).',
            size=10, italic=True, align=J, color='FF0000', after=14)
    for kind, text, page in CONTENTS:
        _dotted(rp, text, page, bold=(kind == 'chap'),
                indent=1.0 if kind == 'sec' else 0.0)
    rp.page_break()


def list_of_figures(rp, figures):
    rp.para('LIST OF FIGURES', size=14, bold=True, underline=True, align=C,
            spacing=1.5, after=14)
    _dotted(rp, 'FIGURE NO.    TITLE', 'PAGE NO.', bold=True)
    rp.blank(1, size=6)
    for num, cap in figures:
        _dotted(rp, '{}    {}'.format(num, cap[:72]), PH.format('p'))
    rp.page_break()


def list_of_tables(rp, tables):
    rp.para('LIST OF TABLES', size=14, bold=True, underline=True, align=C,
            spacing=1.5, after=14)
    _dotted(rp, 'TABLE NO.    TITLE', 'PAGE NO.', bold=True)
    rp.blank(1, size=6)
    for num, cap in tables:
        _dotted(rp, '{}    {}'.format(num, cap[:72]), PH.format('p'))
    rp.page_break()


ACRONYMS = [
    ('AMP', 'Automatic Mixed Precision'),
    ('AUC', 'Area Under the Receiver Operating Characteristic Curve'),
    ('AUPRC', 'Area Under the Precision-Recall Curve'),
    ('BCE', 'Binary Cross-Entropy'),
    ('CAD', 'Computer-Aided Diagnosis'),
    ('CKD', 'Chronic Kidney Disease'),
    ('CLAHE', 'Contrast Limited Adaptive Histogram Equalisation'),
    ('CNN', 'Convolutional Neural Network'),
    ('COA', 'Coati Optimization Algorithm'),
    ('CV', 'Cross-Validation'),
    ('GAP', 'Global Average Pooling'),
    ('GPU', 'Graphics Processing Unit'),
    ('GWO', 'Grey Wolf Optimizer'),
    ('ReLU', 'Rectified Linear Unit'),
    ('RoI', 'Region of Interest'),
    ('RPN', 'Region Proposal Network'),
    ('SD', 'Standard Deviation'),
    ('URI-CADS', 'Ultrasound Renal Imaging Computer-Aided Diagnosis System'),
    ('WOA', 'Whale Optimization Algorithm'),
]


def list_of_acronyms(rp):
    rp.para('LIST OF ACRONYMS', size=14, bold=True, underline=True, align=C,
            spacing=1.5, after=14)
    t = rp.doc.add_table(rows=0, cols=2)
    t.style = 'Table Grid'
    for a, full in ACRONYMS:
        cells = t.add_row().cells
        cells[0].text = a
        cells[1].text = full
        cells[0].width = Cm(3.5)
        cells[1].width = Cm(11.0)
        for cell, bold in ((cells[0], True), (cells[1], False)):
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(2)
                for run in p.runs:
                    run.font.name = FONT
                    run.font.size = Pt(BODY_PT)
                    run.bold = bold
    rp.page_break()
