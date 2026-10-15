"""Assembles BCSE497J_Project_Report.docx from the pieces in report_*.py."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load('rpt_base', '16_project_report.py')
fm = load('rpt_front', 'report_frontmatter.py')
c12 = load('rpt_c12', 'report_chapters.py')
c3 = load('rpt_c3', 'report_chapters2.py')
c46 = load('rpt_c46', 'report_chapters3.py')


def main():
    r = base.load_results()
    rp = base.Report()

    # ---- front matter -----------------------------------------------------
    base.title_page(rp, cover=True)      # cover page
    base.title_page(rp)                  # title page
    base.declaration(rp)
    base.certificate(rp)
    fm.abstract(rp, r, base.cv, base.p7, base.TITLE)
    fm.acknowledgement(rp)
    fm.contents(rp)

    # the figure/table lists need the body built first, so remember where to
    # splice them in: build the body, then append the lists at the end with a
    # note. Simpler and honest: build body first into the same document is not
    # possible with python-docx insertion, so the lists are generated from the
    # known chapter content below and placed here.
    fm.list_of_figures(rp, FIGURES)
    fm.list_of_tables(rp, TABLES)
    fm.list_of_acronyms(rp)

    # ---- body -------------------------------------------------------------
    c12.chapter1(rp, r)
    c12.chapter2(rp)
    c3.chapter3(rp, r, base.p7)
    c46.chapter4(rp)
    c46.chapter5(rp, r, base.cv, base.p7, base.abl_stat)
    c46.chapter6(rp, r, base.cv)

    # ---- back matter ------------------------------------------------------
    c46.references(rp)
    c46.appendices(rp)

    out = rp.save()
    print('Wrote', out)
    print('  figures listed :', len(FIGURES))
    print('  tables listed  :', len(TABLES))
    print('  figures emitted:', len(rp.figures))
    print('  tables emitted :', len(rp.tables))
    if [f[0] for f in rp.figures] != [f[0] for f in FIGURES]:
        print('  WARNING: figure list does not match figures actually emitted')
        print('    listed :', [f[0] for f in FIGURES])
        print('    emitted:', [f[0] for f in rp.figures])
    if [t[0] for t in rp.tables] != [t[0] for t in TABLES]:
        print('  WARNING: table list does not match tables actually emitted')
        print('    listed :', [t[0] for t in TABLES])
        print('    emitted:', [t[0] for t in rp.tables])
    return out


# These must mirror exactly what the chapters emit; the check above enforces it.
FIGURES = [
    ('3.1', 'Overall architecture of the proposed multi-task network'),
    ('3.2', 'Structure of the ResNet-34 encoder and the BasicBlock residual unit'),
    ('3.3', 'The lesion-guided pooling mechanism'),
    ('5.1', 'Component ablation: effect of removing each module'),
    ('5.2', 'Encoder comparison across diagnosis and lesion-type AUC'),
    ('5.3', 'Convergence of the three metaheuristics against evaluation budget'),
    ('5.4', 'Qualitative predictions on a held-out test case'),
    ('5.5', 'Receiver operating characteristic curve for classification'),
]

TABLES = [
    ('3.1', 'Composition of the URI-CADS dataset and the splits used'),
    ('3.2', 'Parameter budget of the proposed network'),
    ('3.3', 'Loss terms used in training'),
    ('3.4', 'The ten-dimensional optimisation search space'),
    ('4.1', 'Hardware and software environment'),
    ('4.2', 'Training hyperparameters'),
    ('4.3', 'Evaluation metrics and their justification'),
    ('4.4', 'Organisation of the implementation'),
    ('5.1', 'Test-set results across the development phases'),
    ('5.2', 'Single-task versus multi-task training'),
    ('5.3', 'Five-fold cross-validated comparison with the benchmark'),
    ('5.4', 'Effect of the anatomical consistency formulation'),
    ('5.5', 'Component ablation, three seeds per configuration'),
    ('5.6', 'Comparison of encoder backbones'),
    ('5.7', 'Optimiser comparison at a matched budget of 60 evaluations'),
]


if __name__ == '__main__':
    main()
