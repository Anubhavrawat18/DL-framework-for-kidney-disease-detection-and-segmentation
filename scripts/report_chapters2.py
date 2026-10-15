"""Chapters 3-6, references and appendices."""
from docx.enum.text import WD_ALIGN_PARAGRAPH

C = WD_ALIGN_PARAGRAPH.CENTER
J = WD_ALIGN_PARAGRAPH.JUSTIFY
L = WD_ALIGN_PARAGRAPH.LEFT

PH = '<<{}>>'
BENCH_DIAG, BENCH_MULTI = 0.8741, 0.8190
SEED_NOISE = 0.0258


# =========================================================== CHAPTER 3
def chapter3(rp, r, p7):
    rp.chapter(3, 'System Design and Methodology')

    rp.section('3.1', 'Introduction')
    rp.para('This chapter describes the dataset, the preprocessing pipeline and the '
            'proposed network. The design is driven by a single observation: a '
            'multi-task network that already segments the lesion computes, as an '
            'ordinary part of its forward pass, a dense map indicating where the '
            'abnormality lies. That map is normally consumed only by the segmentation '
            'loss. The architecture presented here reuses it to give the pathology '
            'classifier localised evidence.', after=8)

    rp.section('3.2', 'Dataset Description')
    rp.para('All experiments use the URI-CADS benchmark, a public collection of 1,985 '
            'B-mode renal ultrasound images released with the reference system of '
            'Molina-Moreno et al. (2024). Each image carries four kinds of '
            'annotation: a hand-traced binary kidney mask, a binary '
            'healthy-versus-pathological diagnosis, two whole-organ pathology labels, '
            'and zero or more lesion bounding boxes each tagged with one of nine '
            'lesion-type identifiers.', after=8)
    rp.table('3.1', 'Composition of the URI-CADS dataset and the splits used',
             ['Split', 'Images', 'Pathological', 'Healthy', 'Share pathological'],
             [['Train', '1,389', '1,074', '315', '77.3%'],
              ['Validation', '298', '231', '67', '77.5%'],
              ['Test', '298', '230', '68', '77.2%'],
              ['Total', '1,985', '1,535', '450', '77.3%']],
             widths=[3.6, 2.4, 3.0, 2.2, 3.8])
    rp.para('Two properties of the dataset shape the design. First, it is markedly '
            'imbalanced: 1,535 of 1,985 images are pathological, so a model that '
            'always answers "pathological" already achieves 77.3 per cent accuracy. '
            'Raw accuracy is therefore not a meaningful headline metric, and balanced '
            'accuracy is reported instead. Second, 1,323 of the 1,985 images contain '
            'at least one annotated lesion box; the held-out test split contains 203 '
            'such images.', after=8)
    rp.para('The lesion annotation deserves particular attention. The dataset '
            'supplies bounding boxes rather than traced lesion contours. These are '
            'converted into filled rectangular masks and unioned where an image '
            'carries several. The lesion decoder is therefore supervised against a '
            'blocky approximation of the true lesion, and is penalised for predicting '
            'an accurately shaped non-rectangular region. Reported lesion Dice is '
            'consequently a lower bound on true localisation quality.', after=8)

    rp.section('3.3', 'Preprocessing and Image Enhancement')
    rp.para('Images are resized to 512 x 512 pixels and normalised to approximately '
            'the range [-1, 1]. Kidney masks are resized with nearest-neighbour '
            'interpolation to preserve their binary character. Lesion boxes are '
            'scaled to the target resolution and rasterised.', after=8)
    rp.para('An optional enhancement pipeline addresses the two dominant image '
            'quality problems. Contrast limited adaptive histogram equalisation '
            '(Zuiderveld, 1994) is applied to the luminance channel in LAB colour '
            'space to improve local contrast without amplifying noise globally; '
            'non-local means denoising (Buades et al., 2005) suppresses speckle; and '
            'a gamma correction adjusts the overall brightness curve. The four '
            'parameters governing this pipeline are fixed in the baseline '
            'configuration and are among the variables optimised in Section 3.9.',
            after=8)

    rp.section('3.4', 'Data Augmentation')
    rp.para('Augmentation is restricted to geometric transformations: horizontal flip '
            'with probability 0.5, rotation of up to 10 degrees, scale jitter of 10 '
            'per cent and translation of 5 per cent of the image extent. Each is '
            'applied identically to the image and to both masks so that the '
            'correspondence between them is preserved.', after=8)
    rp.para('Two exclusions are deliberate and are defended here because they run '
            'against common practice. Vertical flipping is not used, because depth '
            'increases downward in an ultrasound sweep and a vertically flipped image '
            'is anatomically impossible. Intensity augmentation, meaning brightness, '
            'contrast and gamma jitter, is also excluded, because echogenicity is '
            'itself diagnostic in renal ultrasound: two of the dataset pathology '
            'labels concern tissue brightness, so perturbing intensity degrades the '
            'very feature the classifier must read. These images are additionally '
            'dark, with a mean intensity near 25 of 255, so a nominally modest 15 per '
            'cent brightness shift exceeds the image mean. This was verified rather '
            'than assumed: an aggressive intensity augmentation profile reduced '
            'diagnosis AUC from 0.9014 to 0.8363.', after=8)

    rp.section('3.5', 'Proposed Network Architecture')
    rp.para('The network consists of one shared encoder feeding five task-specific '
            'heads. A ResNet-34 backbone (He et al., 2016) pretrained on ImageNet, '
            'with its classifier and global pooling removed, maps the 512 x 512 x 3 '
            'input to a feature map of 512 channels at 16 x 16 resolution, a '
            'cumulative downsampling factor of 32.', after=8)
    rp.placeholder_figure('3.1', 'Overall architecture of the proposed multi-task '
                          'network', 'overall architecture diagram: input, shared '
                          'ResNet-34 encoder, global average pooling to three linear '
                          'heads, and the full feature map to two decoders')
    rp.para('That single feature map is consumed in two different ways. Global '
            'average pooling reduces it to a 512-dimensional vector that feeds three '
            'linear classification heads: one logit for the binary diagnosis, two for '
            'whole-organ pathologies and nine for lesion types. Separately, the full '
            'spatial feature map, without pooling, feeds two independent decoders '
            'that upsample it back to input resolution to produce the kidney and '
            'lesion masks. Each decoder comprises five blocks of 3 x 3 convolution, '
            'batch normalisation (Ioffe & Szegedy, 2015), ReLU activation and '
            'bilinear upsampling by a factor of two, halving the channel count at '
            'each stage, followed by a 1 x 1 convolution to a single output channel.',
            after=8)
    rp.placeholder_figure('3.2', 'Structure of the ResNet-34 encoder and the '
                          'BasicBlock residual unit',
                          'encoder diagram: stem, four residual stages with tensor '
                          'shapes and parameter counts, and the BasicBlock detail '
                          'showing identity and projection shortcuts')
    rp.table('3.2', 'Parameter budget of the proposed network',
             ['Component', 'Parameters', 'Share'],
             [['ResNet-34 encoder (shared)', '21,284,672', '87.1%'],
              ['Kidney decoder', '1,572,833', '6.4%'],
              ['Lesion decoder', '1,572,833', '6.4%'],
              ['Three classification heads', '6,156', '0.03%'],
              ['Total', '24,436,494', '100%']],
             widths=[7.5, 4.0, 3.5])
    rp.para('The parameter budget shown in Table 3.2 is the basis of the efficiency '
            'argument for this design. The three classification heads together '
            'account for 6,156 parameters out of 24.4 million, under 0.03 per cent of '
            'the network. Once the encoder and decoders exist for segmentation, the '
            'classification outputs are therefore effectively free, whereas five '
            'separate single-task models would cost approximately five times the '
            'memory.', after=8)
    rp.para('Capacity within the encoder is heavily concentrated at the coarsest '
            'scales: the final two residual stages hold 19.94 million of the '
            'encoder’s 21.28 million parameters. This is directly relevant to '
            'the mechanism described next, because it means the representation '
            'available to the classifier is dominated by features computed at 32 x 32 '
            'and 16 x 16 resolution, where a small lesion occupies very few spatial '
            'positions.', after=8)

    rp.section('3.6', 'Lesion-Guided Pooling')
    rp.para('Global average pooling summarises the entire feature map in one vector. '
            'For a binary healthy-versus-pathological decision this is adequate, '
            'because diffuse changes affect the whole organ. For multi-label lesion '
            'type recognition it is a structural disadvantage: a focal lesion '
            'occupying a small fraction of the field is averaged together with a '
            'large volume of normal tissue before the classifier ever sees it. This '
            'is the most plausible explanation for the one metric on which the '
            'baseline model clearly trailed the reference system, which localises '
            'each finding explicitly before classifying it.', after=8)
    rp.para('Lesion-guided pooling addresses this without introducing a detector. Let '
            'F denote the encoder feature map and L the lesion decoder output logits. '
            'The mechanism proceeds in five steps:', after=6)
    rp.numbered([
        'Apply the logistic function to L to obtain a predicted lesion probability '
        'map.',
        'Detach that map from the computation graph, so no gradient flows back '
        'through it.',
        'Resample the map to the spatial resolution of the encoder feature grid and '
        'normalise it to sum to one, producing an attention map A.',
        'Form a lesion-focused descriptor by multiplying the encoder features by A '
        'and summing over space.',
        'Concatenate that descriptor with the conventional globally averaged '
        'descriptor and pass the pair to the lesion-type classification head, which '
        'is consequently twice as wide.',
    ])
    rp.placeholder_figure('3.3', 'The lesion-guided pooling mechanism',
                          'pooling diagram: lesion decoder output, sigmoid, '
                          'stop-gradient, resample and normalise, weighted sum, '
                          'concatenation with the global vector')
    rp.para('The detachment in step two is the critical design decision. Were '
            'gradients permitted to flow from the lesion-type classification loss '
            'back into the lesion decoder, that loss could reshape the decoder to '
            'make classification easier, producing convenient pseudo-lesions that '
            'improve the classifier while degrading the segmentation output a '
            'clinician actually sees. In a diagnostic setting where the segmentation '
            'is itself part of the deliverable, that coupling is unacceptable. '
            'Detaching makes the attention read-only: the lesion decoder remains '
            'supervised solely by its own Dice term, and lesion-guided pooling can '
            'only exploit localisation the decoder has already learned for its own '
            'purposes. The mechanism adds 4,608 parameters, a widening of one linear '
            'layer, and requires no additional supervision.', after=8)

    rp.section('3.7', 'Loss Functions')
    rp.para('The total objective is a weighted sum of six terms covering the five '
            'heads plus the anatomical constraint. In the baseline configuration '
            'every weight is set to one; Section 3.9 replaces these with an optimised '
            'vector.', after=8)
    rp.table('3.3', 'Loss terms used in training',
             ['Term', 'Applies to', 'Function'],
             [['Diagnosis', 'Binary head', 'Binary cross-entropy with logits'],
              ['Global pathology', 'Two-way multi-label head',
               'Binary cross-entropy with logits'],
              ['Local pathology', 'Nine-way multi-label head',
               'Binary cross-entropy with logits'],
              ['Kidney segmentation', 'Kidney decoder', 'Soft Dice loss'],
              ['Lesion segmentation', 'Lesion decoder', 'Soft Dice loss'],
              ['Anatomical consistency', 'Lesion decoder and kidney mask',
               'Ratio of lesion mass outside the kidney']],
             widths=[4.2, 4.8, 6.0])
    rp.para('The three classification terms use binary cross-entropy applied '
            'independently per label. The pathology heads are multi-label rather than '
            'multi-class, since a kidney may exhibit several pathologies '
            'simultaneously and a softmax would impose a spurious competition between '
            'them. Weighted cross-entropy and focal loss (Lin et al., 2017) were '
            'additionally implemented and evaluated as training-time corrections for '
            'the class imbalance described in Section 3.2.', after=8)
    rp.para('The two segmentation terms use the soft Dice loss (Milletari et al., '
            '2016), computed on sigmoid probabilities rather than thresholded masks so '
            'that it remains differentiable. Dice is preferred to pixel-wise '
            'cross-entropy because the foreground occupies a small fraction of the '
            'image; a cross-entropy objective dominated by easy background can be '
            'driven low by a model that predicts almost nothing, whereas Dice measures '
            'overlap directly and an empty prediction on a lesion-bearing image scores '
            'zero.', after=8)

    rp.section('3.8', 'Anatomical Consistency Constraint')
    rp.para('A lesion within a kidney cannot lie outside the kidney. This anatomical '
            'fact can be encoded as an auxiliary loss penalising predicted lesion '
            'probability falling outside the annotated kidney region. The quantity '
            'reported is the violation rate, defined as the fraction of predicted '
            'lesion probability mass lying outside the true kidney.', after=8)
    rp.para('The constraint was initially formulated as the absolute outside mass, '
            'that is, the mean over pixels of the predicted lesion probability '
            'multiplied by the complement of the kidney mask. Measured against an '
            'unconstrained baseline, this formulation made the violation rate worse '
            'rather than better. The cause is that the loss and the reported metric '
            'are not the same objective: the absolute form is minimised by '
            'suppressing lesion probability everywhere, which shrinks the numerator '
            'and the denominator of the reported ratio together and leaves the ratio '
            'unchanged while reducing the loss. The optimiser found this degenerate '
            'solution.', after=8)
    rp.para('The corrected formulation optimises the ratio directly. Because a global '
            'rescaling of lesion probability leaves a ratio invariant, the degenerate '
            'minimiser disappears, and the only remaining way to reduce the loss is '
            'to move probability mass inside the kidney. Section 5.6 reports the '
            'measured effect of both formulations.', after=8)

    rp.section('3.9', 'Metaheuristic Optimisation Framework')
    rp.para('The six loss weights and the four enhancement parameters are free '
            'hyperparameters that the phases above set by convention. A '
            'ten-dimensional search space is defined over them, with the loss weights '
            'bounded in the interval 0.1 to 5.0 and the enhancement parameters '
            'bounded by their physically meaningful ranges.', after=8)
    rp.table('3.4', 'The ten-dimensional optimisation search space',
             ['Parameter', 'Range', 'Controls'],
             [['Six loss weights', '0.1 to 5.0', 'Relative importance of each term'],
              ['CLAHE clip limit', '1.0 to 5.0', 'Strength of local contrast '
                                                 'enhancement'],
              ['CLAHE tile grid', '4 to 16', 'Locality of contrast equalisation'],
              ['Denoising strength', '0 to 20', 'Speckle suppression'],
              ['Gamma', '0.5 to 2.0', 'Brightness curve']],
             widths=[4.5, 3.5, 7.0])
    rp.para('The search maximises a scalar fitness computed on the validation split, '
            'formed as a weighted sum of 0.30 times diagnosis AUC, 0.15 times '
            'lesion-type AUC, 0.25 times kidney Dice, 0.25 times lesion-bearing Dice '
            'and 0.05 times one minus the violation rate. All five components lie in '
            'the interval zero to one and are higher-is-better, so no rescaling is '
            'required.', after=8)
    rp.para('One detail of this fitness function is essential rather than cosmetic. '
            'The lesion term uses Dice computed over lesion-bearing images only, not '
            'the dataset-wide mean. Approximately one third of images contain no '
            'lesion, and on those a correct all-zero prediction scores close to one; a '
            'model with no lesion head at all achieves a dataset-wide mean lesion Dice '
            'of 0.3188. Optimising the dataset-wide mean would therefore have '
            'rewarded a degenerate model that predicts no lesions anywhere, '
            'collecting roughly 0.32 for free.', after=8)
    rp.para('Three optimisers are compared: the Coati Optimization Algorithm, the '
            'Whale Optimization Algorithm and the Grey Wolf Optimizer. Each is given '
            'an identical budget of 60 fitness evaluations, where one evaluation means '
            'training a model from scratch. Equal budget rather than equal iteration '
            'count is the fair basis for comparison, since the algorithms use '
            'different population sizes. Because each evaluation trains a model, the '
            'search runs at reduced fidelity, with smaller input resolution, fewer '
            'epochs and a cheaper denoiser, and only the best configuration per '
            'optimiser is retrained at full fidelity.', after=8)
