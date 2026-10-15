"""Chapters 1-6 of the project report. All figures come from results/*.json."""
from docx.enum.text import WD_ALIGN_PARAGRAPH

C = WD_ALIGN_PARAGRAPH.CENTER
J = WD_ALIGN_PARAGRAPH.JUSTIFY
L = WD_ALIGN_PARAGRAPH.LEFT

PH = '<<{}>>'
BENCH_DIAG, BENCH_MULTI = 0.8741, 0.8190
SEED_NOISE = 0.0258


# =========================================================== CHAPTER 1
def chapter1(rp, r):
    rp.chapter(1, 'Introduction')

    rp.section('1.1', 'Introduction')
    rp.para('Kidney disease is a substantial and growing global health burden, and it '
            'frequently progresses silently until renal function is already severely '
            'impaired. Early detection therefore has a disproportionate effect on '
            'outcome, which places considerable value on screening methods that are '
            'cheap enough and portable enough to be applied widely.', after=8)
    rp.para('Ultrasound is the first-line imaging modality for suspected renal '
            'disease. It is inexpensive, requires no ionising radiation, can be '
            'performed at the bedside, and provides real-time visualisation of renal '
            'structure. These properties make it the natural candidate for primary '
            'care screening. The difficulty is interpretation: reading a renal '
            'ultrasound reliably requires experience that is not uniformly available '
            'at the point of first contact, and the images themselves are among the '
            'hardest in diagnostic imaging to interpret.', after=8)
    rp.para('This project develops an automated system that reads a single B-mode '
            'renal ultrasound image and produces, in one pass, a diagnosis, a set of '
            'pathology labels, an outline of the kidney and an outline of any lesion '
            'within it. The emphasis throughout is on classification accuracy, since '
            'that is the output on which a screening decision rests.', after=8)

    rp.section('1.2', 'Overview of Renal Ultrasound Imaging')
    rp.para('B-mode ultrasound forms an image by transmitting high-frequency sound '
            'pulses into tissue and measuring the amplitude of the returning echoes. '
            'Tissue interfaces of differing acoustic impedance reflect more strongly '
            'and therefore appear brighter. In renal imaging this produces a '
            'characteristic appearance in which the renal cortex, the medullary '
            'pyramids and the central echogenic sinus can be distinguished when image '
            'quality permits.', after=8)
    rp.para('Several clinically important findings are expressed through this '
            'appearance. Simple and complicated cysts appear as anechoic or partly '
            'echogenic rounded regions. Hydronephrosis appears as dilatation of the '
            'collecting system. Loss of corticomedullary differentiation and '
            'increased cortical echogenicity are diffuse changes affecting the whole '
            'organ rather than a discrete location, and are associated with chronic '
            'parenchymal disease. The distinction between findings that are localised '
            'and findings that are diffuse is important for this project, because the '
            'two require different machine-learning treatments.', after=8)

    rp.section('1.3', 'Challenges in Automated Renal Ultrasound Analysis')
    rp.para('Automating this reading is difficult for reasons specific to the '
            'modality:', after=6)
    rp.bullets([
        'Speckle noise. Coherent interference produces a granular texture that '
        'obscures boundaries and varies between acquisitions of the same anatomy.',
        'Low soft-tissue contrast. The acoustic impedance difference between renal '
        'cortex and surrounding tissue is small, so the organ boundary is often '
        'ambiguous even to an expert.',
        'Operator dependence. Probe angle, depth setting, gain and focus all alter '
        'the appearance of the same kidney, introducing variation unrelated to '
        'pathology.',
        'Diagnostic echogenicity. Because two of the pathologies of interest are '
        'themselves defined by tissue brightness, routine intensity-based data '
        'augmentation risks destroying the signal the classifier must read.',
        'Class imbalance. Pathological cases dominate the available annotated data, '
        'so a model can score highly on raw accuracy while being of no clinical use.',
        'Coarse lesion annotation. Lesions are commonly annotated as bounding boxes '
        'rather than traced outlines, which limits what a segmentation model can be '
        'supervised against and penalises accurate non-rectangular predictions.',
    ])

    rp.section('1.4', 'Problem Statement')
    rp.para('Published computer-aided diagnosis systems for renal ultrasound fall '
            'broadly into two families, and each carries a structural disadvantage. '
            'Image-level classifiers are compact and fast but return no spatial '
            'evidence, which limits clinical trust and makes errors difficult to '
            'audit. Detection-based systems localise each finding explicitly and '
            'generally classify better, because a classifier reading a localised '
            'region is not forced to infer a small finding from a whole-image '
            'summary; the cost is a region proposal network, region-of-interest '
            'pooling and separate mask heads, which together are substantially '
            'heavier than a classification backbone.', after=8)
    rp.para('The problem addressed by this project is therefore stated as follows: to '
            'obtain the localisation advantage of the detection family, and the '
            'classification accuracy that follows from it, without adopting a '
            'detector architecture, so that the resulting system remains light enough '
            'for the portable hardware that makes ultrasound attractive in the first '
            'place.', after=8)

    rp.section('1.5', 'Objectives')
    rp.numbered([
        'To design a single-encoder multi-task convolutional network that jointly '
        'performs healthy-versus-pathological diagnosis, multi-label pathology '
        'recognition, kidney segmentation and lesion segmentation.',
        'To quantify, under matched experimental conditions, how much the '
        'classification task benefits from being trained alongside segmentation.',
        'To design and evaluate a mechanism that supplies localised evidence to the '
        'pathology classifier by reusing the segmentation branch, without introducing '
        'region proposals or additional supervision.',
        'To formulate an anatomical consistency constraint that penalises predicted '
        'lesion probability lying outside the kidney, and to verify empirically that '
        'it reduces the quantity it targets.',
        'To apply population-based metaheuristic optimisation to the multi-task loss '
        'weighting and image-enhancement parameters, and to compare optimisers under '
        'an equal evaluation budget.',
        'To establish a measurement protocol appropriate to the size of the dataset, '
        'so that reported improvements can be distinguished from random variation.',
        'To benchmark the final system against the published results for the same '
        'dataset under the same cross-validation protocol.',
    ])

    rp.section('1.6', 'Scope of the Project')
    rp.para('The work is confined to single-frame, two-dimensional B-mode renal '
            'ultrasound images drawn from one public benchmark dataset. Doppler, '
            'contrast-enhanced and three-dimensional ultrasound are outside scope, as '
            'are video sequences and any fusion with clinical or laboratory data.',
            after=8)
    rp.para('The system is developed as a research prototype. It is validated '
            'retrospectively on held-out data from a single cohort. No external '
            'cohort validation, prospective evaluation or reader study has been '
            'performed, and no claim of clinical readiness is made. Inference latency '
            'on target hardware has not been measured, so efficiency claims rest on '
            'parameter count and architectural simplicity rather than on measured '
            'throughput.', after=8)

    rp.section('1.7', 'Organisation of the Report')
    rp.para('The remainder of this report is organised as follows. Chapter 2 reviews '
            'prior work on deep learning for renal ultrasound, on kidney '
            'segmentation, on multi-task learning and attention mechanisms, and on '
            'metaheuristic optimisation in medical imaging, and identifies the gaps '
            'this project addresses. Chapter 3 presents the dataset, the '
            'preprocessing pipeline and the proposed architecture, including the '
            'lesion-guided pooling mechanism and the loss formulation. Chapter 4 '
            'describes the implementation, the training configuration and the '
            'evaluation protocol. Chapter 5 reports and discusses the experimental '
            'results. Chapter 6 concludes and identifies future work.', after=8)


# =========================================================== CHAPTER 2
def chapter2(rp):
    rp.chapter(2, 'Background and Literature Survey')

    rp.section('2.1', 'Introduction')
    rp.para('This chapter surveys the work on which this project builds. It covers '
            'four areas: classification of renal ultrasound images, segmentation of '
            'renal structures, multi-task learning and attention mechanisms in '
            'medical imaging, and the application of metaheuristic search to medical '
            'imaging pipelines. The chapter closes by stating the gaps that motivate '
            'the design presented in Chapter 3.', after=8)

    rp.section('2.2', 'Deep Learning for Renal Ultrasound Classification')
    rp.para('Kuo et al. (2019) demonstrated that renal function can be estimated '
            'directly from B-mode ultrasound. Using 4,505 kidney ultrasound images '
            'labelled with estimated glomerular filtration rate derived from serum '
            'creatinine, they fine-tuned an ImageNet-pretrained ResNet to predict '
            'kidney function and chronic kidney disease status. The study is relevant '
            'here for two reasons: it establishes transfer learning from ImageNet as '
            'an effective strategy in this modality despite the domain gap, and it '
            'provides a reference point for the quantity of data required.', after=8)
    rp.para('Subsequent work has applied convolutional classifiers to other '
            'image-level decisions in the same modality. Seraj et al. (2025) '
            'addressed the upstream problem of frame selection, training a classifier '
            'to identify the most diagnostically informative frame within an '
            'acquisition and thereby reduce the subjectivity introduced when an '
            'operator chooses a frame manually. Zhang et al. (2025) survey the field '
            'more broadly and identify interobserver variability as the limitation '
            'that most strongly motivates automation in renal ultrasound.', after=8)
    rp.para('The common limitation of this family is that a whole-image classifier '
            'returns a label without indicating which part of the image produced it. '
            'For a screening tool intended to support a non-specialist reader, that '
            'absence of spatial evidence is a material weakness.', after=8)

    rp.section('2.3', 'Kidney Segmentation in Ultrasound')
    rp.para('A parallel line of work treats the spatial problem directly. Shtaiyat '
            'and Younes (2024) review deep learning approaches to delineating the '
            'kidney outline in two-dimensional ultrasound, and Anuar et al. (2026) '
            'propose a hybrid U-Net framework for kidney segmentation together with '
            'an evaluation methodology designed to be robust to small positional '
            'errors, which is a recurring difficulty when assessing segmentation in a '
            'noisy modality.', after=8)
    rp.para('Zuo et al. (2025) extend this to multiple renal structures, introducing '
            'a segmentation network with two distinct attention modules to handle '
            'speckle noise and ambiguous boundaries. Their work is the closest prior '
            'use of attention within renal ultrasound, although the attention there '
            'serves to refine the segmentation output itself rather than to inform a '
            'diagnostic classifier.', after=8)
    rp.para('Segmentation systems of this kind produce localisation but not '
            'diagnosis. The two capabilities are generally developed separately, '
            'which leaves unexploited the possibility that each might improve the '
            'other.', after=8)

    rp.section('2.4', 'Multi-Task Learning in Medical Imaging')
    rp.para('Multi-task learning rests on the argument advanced by Caruana (1997) '
            'that training related tasks against a shared representation acts as an '
            'inductive bias: each task constrains the shared encoder, discouraging it '
            'from fitting features useful to only one objective. In medical imaging '
            'the argument is attractive because annotation is expensive, so extracting '
            'additional supervisory signal from annotations that already exist is '
            'valuable.', after=8)
    rp.para('Al-Salman and Cevik (2026) apply this directly to renal imaging with a '
            'dual-headed U-Net that performs structural segmentation jointly with '
            'estimated glomerular filtration rate prediction, coupling the two through '
            'an explicit structure-function consistency loss. Their work is the '
            'nearest precedent for the consistency constraint developed in '
            'Section 3.8 of this report, with the distinction that their coupling '
            'links two predictive heads whereas the constraint proposed here '
            'constrains a single head against an anatomical prior.', after=8)

    rp.section('2.5', 'Attention and Weakly Supervised Localisation')
    rp.para('The standard route from a classifier to a spatial explanation is '
            'attribution. Zhou et al. (2016) showed with class activation mapping '
            'that a globally pooled convolutional classifier implicitly encodes '
            'localisation, and Selvaraju et al. (2017) generalised this to arbitrary '
            'architectures with Grad-CAM. Both methods derive a map from an already '
            'trained classifier, so the map reflects whatever features the classifier '
            'happened to rely on, which may or may not correspond to the clinically '
            'relevant structure.', after=8)
    rp.para('A second route is to learn attention as part of the network. Hu et al. '
            '(2018) recalibrate channel responses through squeeze-and-excitation, and '
            'the renal segmentation network of Zuo et al. (2025) noted above applies '
            'spatial attention within the segmentation path. In both cases the '
            'attention is learned from the objective it serves.', after=8)
    rp.para('Neither route exploits a situation that arises naturally in a '
            'segmentation-capable multi-task network: a dense spatial map that has '
            'already been supervised against expert annotation and is therefore known '
            'to indicate the structure of interest. Using such a map to guide a '
            'classifier is the mechanism proposed in this project.', after=8)

    rp.section('2.6', 'Metaheuristic Optimisation in Medical Imaging')
    rp.para('Population-based metaheuristics are gradient-free search algorithms '
            'inspired by natural processes. The Grey Wolf Optimizer (Mirjalili et al., '
            '2014), the Whale Optimization Algorithm (Mirjalili & Lewis, 2016) and the '
            'more recent Coati Optimization Algorithm (Dehghani et al., 2023) are '
            'representative. They require no gradient and no smoothness assumption, '
            'which makes them applicable to hyperparameters that cannot be optimised '
            'by backpropagation, such as loss weights and preprocessing parameters.',
            after=8)
    rp.para('Within medical imaging, Kausar and Ramamurthy (2025) apply a binary Coati '
            'algorithm to feature selection for kidney ultrasound classification, and '
            'Ates et al. (2025) report a hybrid optimisation-assisted pipeline for '
            'classification of dental panoramic radiographs. The former is the closest '
            'prior work to the optimisation component of this project, sharing both '
            'the algorithm family and the imaging modality; it differs in target, '
            'optimising feature selection for a fixed classifier rather than the loss '
            'weighting of a multi-task network.', after=8)
    rp.para('A recurring methodological weakness in this literature is the absence of '
            'controls. Reported gains are frequently attributed to a particular '
            'metaheuristic without a matched-budget comparison against an alternative '
            'algorithm, or against the random search baseline established by Bergstra '
            'and Bengio (2012). This project addresses that gap explicitly.', after=8)

    rp.section('2.7', 'Research Gaps Identified')
    rp.para('The survey above identifies four gaps that this project addresses:',
            after=6)
    rp.numbered([
        'Classification and segmentation of renal ultrasound are usually developed '
        'separately, so the benefit each confers on the other is rarely quantified '
        'under matched conditions.',
        'The localisation advantage of detection architectures has not been pursued '
        'within a lightweight multi-task network, where a supervised spatial map is '
        'already available and unused by the classifier.',
        'Auxiliary constraints in medical imaging are commonly proposed without '
        'verifying that the quantity they target actually improves, which permits a '
        'surrogate loss with a degenerate minimiser to go undetected.',
        'Metaheuristic optimisation results in medical imaging are seldom reported '
        'with a matched-budget comparison between optimisers or against random '
        'search, so the contribution of the specific algorithm cannot be isolated.',
    ])
