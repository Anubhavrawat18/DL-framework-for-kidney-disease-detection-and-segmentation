"""Chapters 4-6, references and appendices."""
from docx.enum.text import WD_ALIGN_PARAGRAPH

C = WD_ALIGN_PARAGRAPH.CENTER
J = WD_ALIGN_PARAGRAPH.JUSTIFY
L = WD_ALIGN_PARAGRAPH.LEFT

PH = '<<{}>>'
BENCH_DIAG, BENCH_MULTI = 0.8741, 0.8190
SEED_NOISE = 0.0258


# =========================================================== CHAPTER 4
def chapter4(rp):
    rp.chapter(4, 'Implementation')

    rp.section('4.1', 'Hardware and Software Environment')
    rp.para('All models were implemented in PyTorch (Paszke et al., 2019) using '
            'torchvision backbones pretrained on ImageNet (Russakovsky et al., 2015). '
            'Training and evaluation were performed on a single NVIDIA RTX 3050 '
            'laptop GPU with 4 GB of video memory. The modest hardware is worth '
            'stating explicitly, because it constrained several design decisions: '
            'automatic mixed precision (Micikevicius et al., 2018) was required to '
            'fit 512 x 512 training within the available memory, and the batch size '
            'was limited to four.', after=8)
    rp.table('4.1', 'Hardware and software environment',
             ['Item', 'Specification'],
             [['GPU', 'NVIDIA RTX 3050 Laptop, 4 GB VRAM'],
              ['Framework', 'PyTorch with torchvision'],
              ['Precision', 'Automatic mixed precision (float16 with gradient '
                            'scaling)'],
              ['Operating system', PH.format('OPERATING SYSTEM AND VERSION')],
              ['Python version', PH.format('PYTHON VERSION')],
              ['CUDA version', PH.format('CUDA VERSION')]],
             widths=[5.0, 10.0])

    rp.section('4.2', 'Training Configuration')
    rp.para('Unless otherwise stated, models were trained at 512 x 512 resolution for '
            '15 epochs using the Adam optimiser (Kingma & Ba, 2015) with a learning '
            'rate of 0.0001 and a batch size of four. After each epoch the model was '
            'scored on the validation split and the best checkpoint retained, rather '
            'than simply taking the final epoch.', after=8)
    rp.table('4.2', 'Training hyperparameters',
             ['Hyperparameter', 'Value'],
             [['Input resolution', '512 x 512'],
              ['Epochs', '15'],
              ['Optimiser', 'Adam'],
              ['Learning rate', '0.0001'],
              ['Batch size', '4'],
              ['Random seed (primary)', '42'],
              ['Mixed precision', 'Enabled'],
              ['Checkpoint selection', 'Best validation combined Dice'],
              ['Augmentation', 'Geometric only (flip, rotate, scale, translate)']],
             widths=[7.0, 8.0])

    rp.section('4.3', 'Evaluation Metrics')
    rp.para('The choice of metric matters more than usual on this dataset, because '
            'the class imbalance makes the obvious choices misleading. The metrics '
            'used, and the reason for each, are set out in Table 4.3.', after=8)
    rp.table('4.3', 'Evaluation metrics and their justification',
             ['Metric', 'Justification'],
             [['AUC', 'Threshold-free, so it measures ranking quality independently '
                      'of where the decision boundary is placed. It is also the '
                      'metric the reference system reports, which makes comparison '
                      'possible.'],
              ['Balanced accuracy', 'The mean of sensitivity and specificity. Plain '
                                    'accuracy is misleading at 77 per cent '
                                    'prevalence; balanced accuracy is not.'],
              ['Sensitivity and specificity', 'What a clinical reader requires, since '
                                              'missed disease and false alarms carry '
                                              'different costs.'],
              ['Dice coefficient', 'Overlap between predicted and true region, '
                                   'ranging from zero to one.'],
              ['Lesion Dice (lesion-bearing only)',
               'The dataset-wide lesion Dice is uninformative because a model with no '
               'lesion head scores 0.3188 through correct empty predictions. '
               'Restricting to the 203 lesion-bearing test images separates a working '
               'model from a null one.'],
              ['Violation rate', 'Fraction of predicted lesion mass lying outside the '
                                 'true kidney; the anatomical plausibility check.']],
             widths=[4.5, 10.5], font=10.5)

    rp.section('4.4', 'Experimental Protocol')
    rp.para('The decision threshold is selected on the validation split to maximise '
            'balanced accuracy and then applied unchanged to the test split. It is '
            'never tuned on test data. Comparisons that support a claim in this '
            'report are conducted under five-fold cross-validation, matching the '
            'protocol used by the reference system, so that fold difficulty is '
            'averaged out rather than confounded with the comparison.', after=8)
    rp.para('Before any comparison was interpreted, the magnitude of random variation '
            'was measured by retraining one configuration with five different random '
            'seeds. This noise floor, reported in Section 5.1, governs the '
            'interpretation of every result that follows.', after=8)

    rp.section('4.5', 'Software Module Organisation')
    rp.para('The implementation is organised into reusable modules so that every '
            'experiment runs through the same data loading, training and evaluation '
            'code, which removes a common source of inconsistency between reported '
            'phases.', after=8)
    rp.table('4.4', 'Organisation of the implementation',
             ['Module', 'Responsibility'],
             [['config', 'Paths, image size, class counts and the global random seed'],
              ['data', 'Manifest loading, dataset class, enhancement and augmentation'],
              ['model', 'Network definitions and the configurable encoder factory'],
              ['losses', 'The six loss terms and the diagnosis loss variants'],
              ['metrics', 'Per-sample Dice, violation rate and classification metrics'],
              ['engine', 'Training and evaluation loops shared by every experiment'],
              ['phase6', 'Search space, fitness function and the resumable optimiser '
                         'driver'],
              ['reporting', 'Figure generation and result serialisation']],
             widths=[4.0, 11.0])


# =========================================================== CHAPTER 5
def chapter5(rp, r, cv, p7, abl_stat):
    rp.chapter(5, 'Results and Discussion')

    # ---- 5.1 noise floor
    rp.section('5.1', 'Measurement Noise Floor')
    rp.para('Before reporting any comparison, the magnitude of variation attributable '
            'to training stochasticity alone was quantified. Retraining an identical '
            'configuration with different random seeds moves diagnosis AUC by a '
            'standard deviation of {:.4f}, with an observed range of 0.074 across '
            'five seeds.'.format(SEED_NOISE), after=8)
    rp.para('On a dataset of this size, therefore, a single-split AUC difference '
            'below roughly 0.03 carries no information. This figure governs the '
            'interpretation of every result in this chapter, and it is reported '
            'first because several differences measured during this project fall '
            'below it and are consequently reported as null rather than as findings.',
            after=8)

    # ---- 5.2 phase progression
    rp.section('5.2', 'Phase-wise Development Results')
    rows = []
    for key, label in [('phase2', 'Phase 2: three-head baseline'),
                       ('phase3', 'Phase 3: fixed enhancement'),
                       ('phase4', 'Phase 4: full five-head multi-task'),
                       ('phase5', 'Phase 5: consistency (absolute form)'),
                       ('phase5b', 'Phase 5b: consistency (ratio form)'),
                       ('phase6_final', 'Phase 6: optimised weights and enhancement')]:
        m = r[key]['test_metrics']
        na = key in ('phase2', 'phase3')
        rows.append([label, '{:.4f}'.format(m['diagnosis_auc']),
                     'no head' if na else '{:.4f}'.format(m['local_pathology_mean_auc']),
                     '{:.4f}'.format(m['kidney_dice']),
                     'no head' if na else '{:.4f}'.format(m['lesion_dice_positive_only']),
                     'no head' if na else '{:.4f}'.format(m['consistency_violation_rate'])])
    rp.table('5.1', 'Test-set results across the development phases',
             ['Phase', 'Diagnosis AUC', 'Lesion-type AUC', 'Kidney Dice',
              'Lesion Dice', 'Violation'], rows,
             widths=[5.0, 2.3, 2.3, 2.0, 1.9, 1.8], font=10)
    rp.para('Phases 2 and 3 use the three-head architecture, which has no lesion-type '
            'head and no lesion decoder, so those columns are marked accordingly '
            'rather than scored. Table 5.1 should be read with the noise floor of '
            'Section 5.1 in mind: these are single-split results, and most of the '
            'movement in the diagnosis column is therefore not interpretable. The '
            'columns that do move meaningfully are kidney Dice and the violation '
            'rate.', after=8)

    # ---- 5.3 multi-task
    rp.section('5.3', 'Benefit of Multi-Task Learning')
    d_only = p7(r, 'diagnosis only')
    k_only = p7(r, 'kidney segmentation only')
    l_only = p7(r, 'lesion segmentation only')
    ph4 = r['phase4']['test_metrics']
    rp.table('5.2', 'Single-task versus multi-task training',
             ['Configuration', 'Diagnosis AUC', 'Lesion-type AUC', 'Kidney Dice',
              'Lesion Dice'],
             [['Diagnosis only', '{:.4f}'.format(d_only['diagnosis_auc']),
               '{:.4f}'.format(d_only['local_pathology_mean_auc']),
               '{:.4f}'.format(d_only['kidney_dice']),
               '{:.4f}'.format(d_only['lesion_dice_positive_only'])],
              ['Kidney segmentation only', '{:.4f}'.format(k_only['diagnosis_auc']),
               '{:.4f}'.format(k_only['local_pathology_mean_auc']),
               '{:.4f}'.format(k_only['kidney_dice']),
               '{:.4f}'.format(k_only['lesion_dice_positive_only'])],
              ['Lesion segmentation only', '{:.4f}'.format(l_only['diagnosis_auc']),
               '{:.4f}'.format(l_only['local_pathology_mean_auc']),
               '{:.4f}'.format(l_only['kidney_dice']),
               '{:.4f}'.format(l_only['lesion_dice_positive_only'])],
              ['Multi-task (all five heads)', '{:.4f}'.format(ph4['diagnosis_auc']),
               '{:.4f}'.format(ph4['local_pathology_mean_auc']),
               '{:.4f}'.format(ph4['kidney_dice']),
               '{:.4f}'.format(ph4['lesion_dice_positive_only'])]],
             widths=[5.2, 2.6, 2.6, 2.4, 2.2], font=10.5)
    gain = ph4['diagnosis_auc'] - d_only['diagnosis_auc']
    rp.para('The diagnosis head trained in isolation reaches {:.4f} AUC. The same '
            'head inside the multi-task model reaches {:.4f}, an improvement of '
            '{:.4f} which is more than twice the noise floor and therefore a real '
            'effect. The segmentation tasks act as a regulariser on the shared '
            'encoder, forcing it to retain spatial structure that a '
            'classification-only objective would discard.'.format(
                d_only['diagnosis_auc'], ph4['diagnosis_auc'], gain), after=8)
    rp.para('The benefit is not symmetric, and this should be stated plainly rather '
            'than concealed. Kidney segmentation trained alone reaches {:.4f} Dice '
            'against {:.4f} in the joint model, so segmentation pays a modest price '
            'for sharing capacity with the classification objectives. The trade is '
            'favourable overall, but it is a trade rather than a uniform '
            'gain.'.format(k_only['kidney_dice'], ph4['kidney_dice']), after=8)

    # ---- 5.4 lesion-guided pooling
    rp.section('5.4', 'Effect of Lesion-Guided Pooling')
    b_loc, b_sd = cv(r, 'base', 'local_pathology_mean_auc')
    v_loc, v_sd = cv(r, 'v3', 'local_pathology_mean_auc')
    rp.para('Under matched five-fold cross-validation, adding lesion-guided pooling '
            'raises lesion-type AUC from {:.4f} ± {:.4f} to {:.4f} ± '
            '{:.4f}, an improvement of {:.4f}.'.format(
                b_loc, b_sd, v_loc, v_sd, v_loc - b_loc), after=8)
    f_m, f_s = abl_stat(r, 'full', 'local_pathology_mean_auc')
    n_m, n_s = abl_stat(r, 'no_lesion_pooling', 'local_pathology_mean_auc')
    rp.para('An independent ablation at reduced fidelity with three random seeds '
            'found lesion-type AUC falling from {:.4f} ± {:.4f} to {:.4f} '
            '± {:.4f} when the component was removed, a decrease of {:.4f}, '
            'with every seed agreeing in direction. Two protocols differing in '
            'resolution, epoch count, split strategy and random seed therefore agree '
            'on both the sign and the approximate magnitude of the effect, which is '
            'the strongest evidence obtained for any component in this '
            'project.'.format(f_m, f_s, n_m, n_s, n_m - f_m), after=8)

    # ---- 5.5 benchmark
    rp.section('5.5', 'Comparison with the Benchmark System')
    rows = []
    for tag, label in [('base', 'Baseline multi-task'),
                       ('v3', 'With lesion-guided pooling'),
                       ('v3_ens3', 'With seed ensemble (final system)')]:
        a, asd = cv(r, tag, 'diagnosis_auc')
        ba, basd = cv(r, tag, 'diagnosis_balanced_accuracy')
        lo, losd = cv(r, tag, 'local_pathology_mean_auc')
        kd, kdsd = cv(r, tag, 'kidney_dice')
        rows.append([label, '{:.4f} ± {:.4f}'.format(a, asd),
                     '{:.4f} ± {:.4f}'.format(ba, basd),
                     '{:.4f} ± {:.4f}'.format(lo, losd),
                     '{:.4f} ± {:.4f}'.format(kd, kdsd)])
    rows.append(['Reference system (published)', '{:.4f}'.format(BENCH_DIAG), '—',
                 '{:.4f}'.format(BENCH_MULTI), '—'])
    rp.table('5.3', 'Five-fold cross-validated comparison with the benchmark',
             ['System', 'Diagnosis AUC', 'Balanced acc.', 'Lesion-type AUC',
              'Kidney Dice'], rows, widths=[4.6, 2.9, 2.6, 2.9, 2.5], font=10)
    fa, fsd = cv(r, 'v3_ens3', 'diagnosis_auc')
    fl, flsd = cv(r, 'v3_ens3', 'local_pathology_mean_auc')
    rp.para('The final system attains {:.4f} ± {:.4f} AUC for '
            'healthy-versus-pathological classification, exceeding the {:.4f} '
            'published for the same dataset by {:.4f}, while using a ResNet-34 '
            'encoder against the reference’s ResNet-50 and a single forward pass '
            'against a dual-detector pipeline. Lesion-type AUC reaches {:.4f} ± '
            '{:.4f} against the published {:.4f}.'.format(
                fa, fsd, BENCH_DIAG, fa - BENCH_DIAG, fl, flsd, BENCH_MULTI), after=8)
    rp.para('Two caveats attach to this comparison and are stated here rather than '
            'left for a reader to find. First, the binary gain comes principally '
            'from seed ensembling rather than from lesion-guided pooling, which is '
            'specific to the multi-label task. Second, the lesion-type figures are '
            'not measured on an identical label space: this system predicts the nine '
            'raw annotation identifiers, three of which have fewer than 40 positive '
            'images in the entire dataset, whereas the reference groups them into '
            'four consolidated classes before training. Averaging per-class AUC over '
            'very sparse classes depresses the mean, so the apparent deficit should '
            'be interpreted with caution.', after=8)

    # ---- 5.6 consistency
    rp.section('5.6', 'Anatomical Consistency Constraint')
    rp.table('5.4', 'Effect of the anatomical consistency formulation',
             ['Configuration', 'Violation rate', 'Diagnosis AUC', 'Kidney Dice',
              'Lesion Dice'],
             [['No constraint (Phase 4)',
               '{:.4f}'.format(ph4['consistency_violation_rate']),
               '{:.4f}'.format(ph4['diagnosis_auc']),
               '{:.4f}'.format(ph4['kidney_dice']),
               '{:.4f}'.format(ph4['lesion_dice_positive_only'])],
              ['Absolute-mass form (Phase 5)',
               '{:.4f}'.format(r['phase5']['test_metrics']['consistency_violation_rate']),
               '{:.4f}'.format(r['phase5']['test_metrics']['diagnosis_auc']),
               '{:.4f}'.format(r['phase5']['test_metrics']['kidney_dice']),
               '{:.4f}'.format(r['phase5']['test_metrics']['lesion_dice_positive_only'])],
              ['Ratio form (Phase 5b)',
               '{:.4f}'.format(r['phase5b']['test_metrics']['consistency_violation_rate']),
               '{:.4f}'.format(r['phase5b']['test_metrics']['diagnosis_auc']),
               '{:.4f}'.format(r['phase5b']['test_metrics']['kidney_dice']),
               '{:.4f}'.format(r['phase5b']['test_metrics']['lesion_dice_positive_only'])]],
             widths=[5.2, 2.8, 2.6, 2.4, 2.0], font=10.5)
    rp.para('The absolute-mass formulation made the violation rate worse than using '
            'no constraint at all, rising from {:.4f} to {:.4f}. The corrected ratio '
            'formulation reduces it to {:.4f}, roughly a fourfold improvement over '
            'the unconstrained baseline. Differences in the diagnosis column fall '
            'within the noise floor and are not interpreted.'.format(
                ph4['consistency_violation_rate'],
                r['phase5']['test_metrics']['consistency_violation_rate'],
                r['phase5b']['test_metrics']['consistency_violation_rate']), after=8)
    rp.para('This result is reported because the failure mode is general rather than '
            'specific to this project. Any auxiliary constraint written as a '
            'surrogate for a reported ratio metric is vulnerable to the same '
            'degeneracy, and the symptom, a constraint that worsens the quantity it '
            'targets, is visible only if that quantity is measured both with and '
            'without the constraint.', after=8)

    # ---- 5.7 ablation
    rp.section('5.7', 'Component Ablation Study')
    order = [('full', 'Full model (reference)'),
             ('no_lesion_pooling', 'Without lesion-guided pooling'),
             ('no_consistency', 'Without anatomical consistency'),
             ('no_lesion_decoder', 'Without lesion segmentation'),
             ('no_kidney_decoder', 'Without kidney segmentation'),
             ('classification_only', 'Classification heads only')]
    rows = []
    for key, label in order:
        d_m, d_s = abl_stat(r, key, 'diagnosis_auc')
        l_m, l_s = abl_stat(r, key, 'local_pathology_mean_auc')
        k_m, k_s = abl_stat(r, key, 'kidney_dice')
        rows.append([label, '{:.4f} ± {:.4f}'.format(d_m, d_s),
                     '{:.4f} ± {:.4f}'.format(l_m, l_s),
                     '{:.4f} ± {:.4f}'.format(k_m, k_s)
                     if k_m > 0 else '—'])
    rp.table('5.5', 'Component ablation, three seeds per configuration',
             ['Configuration', 'Diagnosis AUC', 'Lesion-type AUC', 'Kidney Dice'],
             rows, widths=[5.6, 3.2, 3.2, 3.0], font=10.5)
    rp.figure('5.1', 'Component ablation: effect of removing each module',
              'ablation_modules.png', width=14.0)
    rp.para('No removal moved diagnosis AUC beyond the noise band. The only component '
            'with a demonstrable effect is lesion-guided pooling, and only on the '
            'multi-label task. With three seeds this study detects large effects '
            'only; a null result indicates that an effect was not demonstrated, not '
            'that it is absent.', after=8)

    # ---- 5.8 encoders
    rp.section('5.8', 'Encoder Comparison')
    enc_rows = []
    for key, label, params in [('full', 'ResNet-34 (baseline)', '24.44 M'),
                               ('encoder_efficientnet_b0', 'EfficientNet-B0',
                                '10.72 M'),
                               ('encoder_mobilenet_v3_large', 'MobileNetV3-Large',
                                '8.20 M'),
                               ('encoder_shufflenet_v2_x1_0', 'ShuffleNetV2',
                                '6.78 M')]:
        d_m, d_s = abl_stat(r, key, 'diagnosis_auc')
        l_m, l_s = abl_stat(r, key, 'local_pathology_mean_auc')
        enc_rows.append([label, params, '{:.4f} ± {:.4f}'.format(d_m, d_s),
                         '{:.4f} ± {:.4f}'.format(l_m, l_s)])
    rp.table('5.6', 'Comparison of encoder backbones',
             ['Encoder', 'Parameters', 'Diagnosis AUC', 'Lesion-type AUC'],
             enc_rows, widths=[4.6, 3.0, 3.7, 3.7], font=10.5)
    rp.figure('5.2', 'Encoder comparison across diagnosis and lesion-type AUC',
              'ablation_encoders.png', width=14.0)
    rp.para('EfficientNet-B0 matches ResNet-34 on binary diagnosis within the noise '
            'band while using 2.3 times fewer parameters. All three lightweight '
            'backbones, however, lose lesion-type AUC. The practical reading is that '
            'a smaller backbone is defensible if binary triage is the deployment '
            'target, but not if multi-label pathology recognition is required.',
            after=8)

    # ---- 5.9 optimisers
    rp.section('5.9', 'Metaheuristic Optimiser Comparison')
    opt_rows = [[name, '{:.4f}'.format(v['best_fitness']), str(v['n_evaluations']),
                 '{:.2f} h'.format(v['wall_clock_hours'])]
                for name, v in r['search']['results'].items()]
    rp.table('5.7', 'Optimiser comparison at a matched budget of 60 evaluations',
             ['Optimiser', 'Best validation fitness', 'Evaluations', 'Wall clock'],
             opt_rows, widths=[4.0, 4.5, 3.2, 3.3], font=10.5)
    rp.figure('5.3', 'Convergence of the three metaheuristics against evaluation '
              'budget', 'phase6_convergence.png', width=14.5)
    fits = [v['best_fitness'] for v in r['search']['results'].values()]
    rp.para('The three optimisers finished within {:.4f} fitness of one another, far '
            'below this pipeline’s run-to-run variation. No meaningful '
            'difference between them is therefore claimed. More importantly, the '
            'optimised configuration did not improve classification over equal loss '
            'weights: it raised kidney Dice to {:.4f} and reduced the violation rate '
            'approximately fivefold, while diagnosis AUC fell to {:.4f}. This is '
            'consistent with the fitness function, which placed half its weight on '
            'the two segmentation terms; the search traded classification for '
            'segmentation exactly as instructed.'.format(
                max(fits) - min(fits),
                r['phase6_final']['test_metrics']['kidney_dice'],
                r['phase6_final']['test_metrics']['diagnosis_auc']), after=8)
    rp.para('This is reported as a negative result for the optimisation framing '
            'rather than presenting the winning optimiser as a contribution. It also '
            'bears on prior work: absent a matched-budget comparison, an improvement '
            'attributed to a particular metaheuristic is difficult to separate from '
            'the improvement any search of comparable budget would have produced.',
            after=8)

    # ---- 5.10 qualitative
    rp.section('5.10', 'Qualitative Results')
    rp.figure('5.4', 'Qualitative predictions on a held-out test case. Dashed '
              'contours denote expert annotation and filled regions denote model '
              'prediction', 'case_617.png', width=15.0, subdir='showcase')
    rp.para('Figure 5.4 shows predictions on a held-out test case across successive '
            'model versions. The kidney boundary is recovered closely. Lesion '
            'predictions are correctly located but organically shaped, whereas the '
            'ground truth is rectangular by construction, which is the annotation '
            'limitation discussed in Section 3.2 and which causes reported lesion '
            'Dice to understate true localisation quality.', after=8)
    rp.figure('5.5', 'Receiver operating characteristic curve for '
              'healthy-versus-pathological classification',
              'phase6_final_roc.png', width=11.0)

    # ---- 5.11 discussion
    rp.section('5.11', 'Discussion')
    rp.para('The central finding of this project is that the localisation advantage '
            'normally obtained by adopting a detection architecture can be partially '
            'recovered within a multi-task segmentation network at negligible '
            'parameter cost. Lesion-guided pooling adds 4,608 parameters and no new '
            'supervision, yet improves multi-label pathology recognition by {:.4f} '
            'AUC under cross-validation. The mechanism is available to any '
            'multi-task network that already segments the structure of '
            'interest.'.format(v_loc - b_loc), after=8)
    rp.para('A second observation deserves emphasis because it has implications '
            'beyond this project. The largest single improvement obtained was not '
            'architectural: seed ensembling lifted diagnosis AUC by {:.4f} over the '
            'single model, more than any design change tested. On a dataset of fewer '
            'than 2,000 images, variance reduction is a more effective lever than '
            'architectural refinement, and much of the published variation between '
            'methods at this scale may be seed variance rather than genuine method '
            'difference.'.format(
                cv(r, 'v3_ens3', 'diagnosis_auc')[0] - cv(r, 'v3', 'diagnosis_auc')[0]),
            after=8)
    rp.para('The multi-label task remains the weaker side of these results. '
            'Independently of the label-space caveat noted in Section 5.5, the '
            'reference architecture retains a genuine structural advantage: it '
            'classifies each detected region separately, whereas the pathology head '
            'here must represent an arbitrary number of findings of different types '
            'in a single pair of pooled vectors. Lesion-guided pooling narrows this '
            'gap but cannot close it, since one attention map cannot separate two '
            'lesions of different types in the same image.', after=8)


# =========================================================== CHAPTER 6
def chapter6(rp, r, cv):
    rp.chapter(6, 'Conclusion and Future Work')

    rp.section('6.1', 'Conclusion')
    fa, fsd = cv(r, 'v3_ens3', 'diagnosis_auc')
    rp.para('This project set out to obtain the classification accuracy associated '
            'with detection-based computer-aided diagnosis without adopting a '
            'detector architecture. A single-encoder multi-task network was '
            'developed that performs diagnosis, multi-label pathology recognition, '
            'kidney segmentation and lesion segmentation in one forward pass, and '
            'was evaluated on the public URI-CADS benchmark.', after=8)
    rp.para('The objectives stated in Section 1.5 were met as follows:', after=6)
    rp.bullets([
        'The multi-task benefit was quantified under matched conditions: diagnosis '
        'AUC improves from 0.8436 single-task to 0.8995 multi-task, more than twice '
        'the measured noise floor.',
        'Lesion-guided pooling was designed and validated, improving lesion-type AUC '
        'by 0.0271 under five-fold cross-validation and corroborated by an '
        'independent three-seed ablation.',
        'An anatomical consistency constraint was formulated, found to fail in its '
        'obvious form, diagnosed, and corrected; the ratio formulation reduces the '
        'violation rate approximately fourfold.',
        'Three metaheuristics were compared at matched budget and found '
        'indistinguishable, and the optimised configuration was shown to trade '
        'classification for segmentation in accordance with its fitness function.',
        'A measurement protocol was established, quantifying the seed-induced noise '
        'floor so that reported improvements can be separated from random variation.',
        'The final system reaches {:.4f} ± {:.4f} AUC for '
        'healthy-versus-pathological classification, exceeding the published '
        'benchmark of {:.4f} on the same dataset and protocol.'.format(
            fa, fsd, BENCH_DIAG),
    ])

    rp.section('6.2', 'Limitations')
    rp.para('The following limitations constrain the conclusions that may be drawn, '
            'and are stated explicitly rather than left implicit:', after=6)
    rp.bullets([
        'Coarse lesion annotation. The dataset provides bounding boxes rather than '
        'traced contours, so reported lesion Dice is a lower bound and cannot be '
        'improved without re-annotation.',
        'Label-space mismatch. The lesion-type comparison against the benchmark is '
        'not measured on an identical label space, so the size of that gap cannot '
        'presently be stated with confidence.',
        'No external validation. All results come from one public dataset. The '
        'system has not been evaluated on images from a different scanner, '
        'institution or patient population.',
        'Unverified split integrity. Splits are applied per image. If the dataset '
        'contains multiple frames from the same patient, correlated frames could '
        'appear on both sides of a split; this could not be verified against patient '
        'identifiers.',
        'Limited statistical power in ablations. Three seeds per configuration '
        'detects large effects only, so several components are reported as having no '
        'demonstrated effect rather than no effect.',
        'No latency measurement. Parameter counts are reported but inference time on '
        'target hardware was not measured, so efficiency claims rest on model size '
        'and architectural simplicity.',
    ])

    rp.section('6.3', 'Future Work')
    rp.para('Several directions follow directly from the limitations above.', after=6)
    rp.numbered([
        'Evaluate lesion-type performance under the benchmark’s grouped '
        'taxonomy. This would resolve the comparability caveat and may also improve '
        'learning, since three of the nine current classes have too few positive '
        'examples to be learnable.',
        'Validate on an external cohort. This is the prerequisite for any claim of '
        'clinical applicability.',
        'Extend lesion-guided pooling to one attention map per pathology class, '
        'addressing the inability of a single map to separate two lesions of '
        'different types within the same image.',
        'Replace the plain decoder with a skip-connected variant to sharpen '
        'segmentation boundaries.',
        'Add a matched-budget random search baseline, placing the metaheuristic '
        'comparison on firmer methodological ground.',
        'Measure inference latency on representative portable hardware to support '
        'the deployability argument with timing rather than parameter count.',
    ])


# =========================================================== BACK MATTER
REFERENCES = [
    'Al-Salman, O., & Cevik, M. (2026). A functionally guided U-Net for chronic '
    'kidney disease assessment: Joint structural segmentation and eGFR prediction '
    'with a structure-function consistency loss. Electronics, 15(1), 176. '
    'https://doi.org/10.3390/electronics15010176',

    'Anuar, D. A. S., Ibrahim, N., & Huong, A. (2026). MSE-guided hybrid U-Net '
    'framework for automatic kidney segmentation and spatial localization. '
    'International Journal of Advanced Computer Science and Applications, 17(6), '
    '247-254.',

    'Ates, G., Turk, F., Akcin, E. T., & Gungor, M. (2025). Decision-support for '
    'restorative dentistry: Hybrid optimization enhances detection on panoramic '
    'radiographs. Healthcare, 13(22), 2904. '
    'https://doi.org/10.3390/healthcare13222904',

    'Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter '
    'optimization. Journal of Machine Learning Research, 13, 281-305.',

    'Buades, A., Coll, B., & Morel, J.-M. (2005). A non-local algorithm for image '
    'denoising. Proceedings of the IEEE Conference on Computer Vision and Pattern '
    'Recognition, 60-65.',

    'Caruana, R. (1997). Multitask learning. Machine Learning, 28(1), 41-75.',

    'Dehghani, M., Montazeri, Z., Trojovska, E., & Trojovsky, P. (2023). Coati '
    'optimization algorithm: A new bio-inspired metaheuristic algorithm for solving '
    'optimization problems. Knowledge-Based Systems, 259, 110011. '
    'https://doi.org/10.1016/j.knosys.2022.110011',

    'He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017). Mask R-CNN. '
    'Proceedings of the IEEE International Conference on Computer Vision, 2961-2969.',

    'He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image '
    'recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern '
    'Recognition, 770-778.',

    'Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., '
    'Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for '
    'MobileNetV3. Proceedings of the IEEE/CVF International Conference on Computer '
    'Vision, 1314-1324.',

    'Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. '
    'Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, '
    '7132-7141.',

    'Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network '
    'training by reducing internal covariate shift. Proceedings of the International '
    'Conference on Machine Learning, 448-456.',

    'Kausar, F., & Ramamurthy, B. (2025). Optimized feature selection for kidney '
    'ultrasound image classification using binary coati weighted mean vector '
    'algorithm. Journal of Computer Science, 21(12), 2986-3004. '
    'https://doi.org/10.3844/jcssp.2025.2986.3004',

    'Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. '
    'International Conference on Learning Representations.',

    'Krishna, T. B., Poreddy, A. K. R., Sindhu, K. G., & Kokil, P. (2025). Automated '
    'maternal fetal ultrasound image identification using a hybrid vision '
    'transformer model. In Pattern Recognition (ICPR 2024), Lecture Notes in '
    'Computer Science (Vol. 15311, pp. 253-269). Springer Nature Switzerland. '
    'https://doi.org/10.1007/978-3-031-78195-7_17',

    'Kuo, C.-C., Chang, C.-M., Liu, K.-T., Lin, W.-K., Chiang, H.-Y., Chung, C.-W., '
    'Ho, M.-R., Sun, P.-R., Yang, R.-L., & Chen, K.-T. (2019). Automation of the '
    'kidney function prediction and classification through ultrasound-based kidney '
    'imaging using deep learning. npj Digital Medicine, 2, 29. '
    'https://doi.org/10.1038/s41746-019-0104-2',

    'Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable '
    'predictive uncertainty estimation using deep ensembles. Advances in Neural '
    'Information Processing Systems, 30.',

    'Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss '
    'for dense object detection. Proceedings of the IEEE International Conference on '
    'Computer Vision, 2980-2988.',

    'Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018). ShuffleNet V2: Practical '
    'guidelines for efficient CNN architecture design. Proceedings of the European '
    'Conference on Computer Vision, 116-131.',

    'Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., '
    'Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed '
    'precision training. International Conference on Learning Representations.',

    'Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully convolutional '
    'neural networks for volumetric medical image segmentation. Proceedings of the '
    'Fourth International Conference on 3D Vision, 565-571.',

    'Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. Advances '
    'in Engineering Software, 95, 51-67. '
    'https://doi.org/10.1016/j.advengsoft.2016.01.008',

    'Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. '
    'Advances in Engineering Software, 69, 46-61. '
    'https://doi.org/10.1016/j.advengsoft.2013.12.007',

    'Molina-Moreno, M., Gonzalez-Diaz, I., Rivera Gorrin, M., Burguera Vion, V., & '
    'Diaz-de-Maria, F. (2024). URI-CADS: A fully automated computer-aided diagnosis '
    'system for ultrasound renal imaging. Journal of Imaging Informatics in '
    'Medicine, 37(4), 1458-1474. https://doi.org/10.1007/s10278-024-01055-4',

    'Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, '
    'T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., '
    'DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., '
    'Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance '
    'deep learning library. Advances in Neural Information Processing Systems, 32.',

    'Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards '
    'real-time object detection with region proposal networks. Advances in Neural '
    'Information Processing Systems, 28.',

    'Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks '
    'for biomedical image segmentation. Medical Image Computing and '
    'Computer-Assisted Intervention, 234-241.',

    'Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., '
    'Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., & Fei-Fei, L. (2015). '
    'ImageNet large scale visual recognition challenge. International Journal of '
    'Computer Vision, 115(3), 211-252.',

    'Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. '
    '(2017). Grad-CAM: Visual explanations from deep networks via gradient-based '
    'localization. Proceedings of the IEEE International Conference on Computer '
    'Vision, 618-626.',

    'Seraj, A., Monazami, S. P., Davoodi, R., Seraj, J., Ghattan Kashani, H., '
    'Sajjadian Moosavi, A., & Shariat Panahi, M. (2025). Deep learning framework for '
    'automated frame selection in kidney ultrasound. Scientific Reports, 15, 41940. '
    'https://doi.org/10.1038/s41598-025-25755-1',

    'Shtaiyat, A., & Younes, H. A. (2024). Enhancing kidney outlines in 2D '
    'ultrasound images using deep learning. Journal of Medicinal and Chemical '
    'Sciences, 7(4), 909-921. https://doi.org/10.26655/JMCHEMSCI.2024.7.4',

    'Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for '
    'convolutional neural networks. Proceedings of the International Conference on '
    'Machine Learning, 6105-6114.',

    'Zhang, Y., Hou, Y., Qiu, T., Zhuang, Y., Chen, K., Ling, W., Luo, Y., & Lin, J. '
    '(2025). Deep learning in renal ultrasound: Applications, challenges, and future '
    'outlook. Frontiers in Oncology, 15, 1730628. '
    'https://doi.org/10.3389/fonc.2025.1730628',

    'Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning '
    'deep features for discriminative localization. Proceedings of the IEEE '
    'Conference on Computer Vision and Pattern Recognition, 2921-2929.',

    'Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In '
    'Graphics Gems IV (pp. 474-485). Academic Press.',

    'Zuo, Y., Li, J., & Tian, J. (2025). A segmentation network with two distinct '
    'attention modules for the segmentation of multiple renal structures in '
    'ultrasound images. Diagnostics, 15(15), 1978. '
    'https://doi.org/10.3390/diagnostics15151978',
]


def references(rp):
    rp.page_break()
    rp.unnumbered_heading('References')
    rp.para('All references are presented in APA format and are listed '
            'alphabetically by first author surname.', italic=True, size=11, after=12)
    from docx.shared import Cm, Pt
    for ref in REFERENCES:
        p = rp.doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(8)
        pf.left_indent = Cm(1.0)
        pf.first_line_indent = Cm(-1.0)        # APA hanging indent
        p.alignment = J
        run = p.add_run(ref)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)


def appendices(rp):
    rp.page_break()
    rp.unnumbered_heading('Appendices')
    rp.para('Appendix 1  Source Code Organisation', size=12, bold=True,
            align=L, spacing=1.5, after=6)
    rp.para('The complete implementation is organised into the modules listed in '
            'Table 4.4. ' + PH.format('INSERT REPOSITORY URL OR CODE LISTING HERE'),
            after=10)
    rp.para('Appendix 2  Additional Qualitative Results', size=12, bold=True,
            align=L, spacing=1.5, after=6)
    rp.para('Further held-out test cases are shown below. ' +
            PH.format('INSERT ADDITIONAL FIGURES IF REQUIRED'), after=10)
    rp.para('Appendix 3  Complete Per-Fold Results', size=12, bold=True,
            align=L, spacing=1.5, after=6)
    rp.para('Per-fold values underlying the cross-validated means reported in '
            'Table 5.3. ' + PH.format('INSERT PER-FOLD TABLE IF REQUIRED'), after=10)
