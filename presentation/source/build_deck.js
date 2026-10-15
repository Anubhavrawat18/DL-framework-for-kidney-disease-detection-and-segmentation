const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const docx = require('docx');

const D = JSON.parse(fs.readFileSync(path.join(__dirname, 'deck_data.json'), 'utf8'));
const X = JSON.parse(fs.readFileSync(path.join(__dirname, 'deck_extra.json'), 'utf8'));
const PROJ = 'C:/StudyMaterialDoNotTouch/publications/URI-CADS dataset';
const OUT = PROJ + '/presentation';
const DECK = OUT + '/URI-CADS_Project_Review.pptx';
const SCRIPT = process.env.SCRIPT_OUT || (OUT + '/Presentation_Script_Two_Presenters.docx');

const C = {
  navy: '21295C', ink: '1A2233', slate: '5B6B79', line: 'DCE3E8', tint: 'F3F6F9',
  white: 'FFFFFF', teal: '1C7293', amber: 'D9912B', green: '2E7D5B', red: 'B5483B',
  gray: 'A7B1BB', ice: 'CADCFC', greenTint: 'E6F2EC', amberTint: 'FBF0DF',
};
const H = 'Cambria', B = 'Calibri';
const W = 13.333, M = 0.6, CW = W - 2 * M;
const A = 'Anubhav Rawat', K = 'Kaustubh';

const f2 = v => v.toFixed(2), f3 = v => v.toFixed(3), f4 = v => v.toFixed(4);
const pct = v => (v * 100).toFixed(1) + '%';
const T = ph => D[ph].test;
const ABL = k => D.ablations[k].test;
const S = D.search.results;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'Optimization-Based Deep Learning Framework for Kidney Disease Detection and Segmentation';
pres.author = 'Anubhav Rawat, Kaustubh';

let n = 0;
const scriptLog = [];

function newSlide({ kicker, title, lede, name, dark = false, speaker, script }) {
  n += 1;
  const s = pres.addSlide();
  s.background = { color: dark ? C.navy : C.white };
  if (!dark) {
    if (kicker) s.addText(kicker.toUpperCase(), { x: M, y: 0.42, w: CW, h: 0.3, fontFace: B, fontSize: 12, bold: true, color: C.teal, charSpacing: 2, margin: 0, isTextBox: true });
    if (title) s.addText(title, { x: M, y: 0.72, w: CW, h: 0.7, fontFace: H, fontSize: 30, bold: true, color: C.navy, margin: 0, valign: 'middle', isTextBox: true });
    if (lede) s.addText(lede, { x: M, y: 1.44, w: CW, h: 0.5, fontFace: B, fontSize: 15, color: C.slate, margin: 0, valign: 'top', isTextBox: true });
    s.addText('Kidney Ultrasound CAD  ·  Project Review', { x: M, y: 7.02, w: 6, h: 0.3, fontFace: B, fontSize: 10, color: C.slate, margin: 0, isTextBox: true });
    s.addText(String(n), { x: W - M - 1, y: 7.02, w: 1, h: 0.3, fontFace: B, fontSize: 10, color: C.slate, align: 'right', margin: 0, isTextBox: true });
  }
  if (script) {
    s.addNotes(`${speaker}:\n\n` + script.join('\n\n'));
    scriptLog.push({ n, title: name || title || kicker, speaker, script });
  }
  return s;
}

function card(s, x, y, w, h, fill = C.tint, line = C.line) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line, width: 0.75 }, rectRadius: 0.06 });
}

function text(s, t, o) {
  s.addText(t, Object.assign({ fontFace: B, fontSize: 14, color: C.ink, margin: 0, valign: 'top', isTextBox: true }, o));
}

function badge(s, x, y, num, fill = C.navy) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 0.46, h: 0.46, fill: { color: fill }, line: { type: 'none' }, rectRadius: 0.08 });
  text(s, num, { x, y, w: 0.46, h: 0.46, fontFace: H, fontSize: 13, bold: true, color: C.white, align: 'center', valign: 'middle' });
}

function numbered(s, x, y, w, num, head, body, { headSize = 15, bodySize = 12.5, bodyH = 0.62 } = {}) {
  badge(s, x, y, num);
  text(s, head, { x: x + 0.62, y: y - 0.02, w: w - 0.62, h: 0.36, fontSize: headSize, bold: true });
  text(s, body, { x: x + 0.62, y: y + 0.34, w: w - 0.62, h: bodyH, fontSize: bodySize, color: C.slate });
}

function bullets(items, { size = 13.5, color = C.ink, gap = 6 } = {}) {
  return items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: gap, fontSize: size, color } }));
}

function stat(s, x, y, w, value, label, { valueSize = 34, color = C.navy, labelColor = C.slate } = {}) {
  text(s, value, { x, y, w, h: 0.62, fontFace: H, fontSize: valueSize, bold: true, color, valign: 'bottom' });
  text(s, label, { x, y: y + 0.66, w, h: 0.55, fontSize: 12, color: labelColor });
}

function chip(s, x, y, label, done) {
  const w = 0.12 + label.length * 0.085;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.3, fill: { color: done ? C.greenTint : C.amberTint }, line: { type: 'none' }, rectRadius: 0.06 });
  text(s, label, { x, y, w, h: 0.3, fontSize: 10, bold: true, color: done ? C.green : '8A5A12', align: 'center', valign: 'middle' });
}

function chartOpts(o) {
  return Object.assign({
    catAxisLabelColor: C.slate, valAxisLabelColor: C.slate, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: B, valAxisLabelFontFace: B,
    valGridLine: { color: C.line, size: 0.75 }, catGridLine: { style: 'none' },
    catAxisLineColor: C.line, valAxisLineShow: false,
    showLegend: true, legendPos: 't', legendFontSize: 11, legendFontFace: B, legendColor: C.ink,
    showTitle: false, dataLabelFontSize: 10, dataLabelFontFace: B, dataLabelColor: C.ink,
  }, o);
}

function barChart(s, labels, series, o) {
  s.addChart(pres.charts.BAR, series.map(se => ({ name: se.name, labels, values: se.values })), chartOpts(Object.assign({
    barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 70, showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.000',
  }, o)));
}

function lineChart(s, labels, series, o) {
  s.addChart(pres.charts.LINE, series.map(se => ({ name: se.name, labels, values: se.values })), chartOpts(Object.assign({
    lineSize: 2.25, lineDataSymbol: 'none',
  }, o)));
}

function chartLabel(s, t, x, y, w) {
  text(s, t, { x, y, w, h: 0.32, fontSize: 13, bold: true, color: C.ink });
}

function table(s, rows, o) {
  s.addTable(rows, Object.assign({ fontFace: B, fontSize: 12, color: C.ink, border: { type: 'solid', pt: 0.75, color: C.line }, valign: 'middle' }, o));
}
const th = t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy } } });
const td = (t, extra = {}) => ({ text: t, options: extra });

// ---------------------------------------------------------------- 1. Title
{
  const s = newSlide({ dark: true, name: 'Title and introduction', speaker: A, script: [
    `Good morning, everyone. I'm Anubhav Rawat, and I'm presenting with Kaustubh. Our project, guided by Sudharson S, is an optimization-based deep learning framework for detecting and segmenting kidney disease in ultrasound images.`,
    `At our last review we had finished five of our nine phases. Since then we have retrained every phase under matched conditions, completed the full metaheuristic optimization study, and run most of our ablation experiments. We'll show you all of it, including the results that did not go the way we expected, because those turned out to be some of our most useful findings.`,
  ] });
  text(s, 'PROJECT REVIEW', { x: 0.8, y: 0.8, w: 8, h: 0.35, fontSize: 13, bold: true, color: C.ice, charSpacing: 3 });
  text(s, 'Optimization-Based Deep Learning Framework for Kidney Disease Detection and Segmentation from Ultrasound Images',
    { x: 0.8, y: 1.3, w: 11.7, h: 2.2, fontFace: H, fontSize: 38, bold: true, color: C.white, valign: 'middle' });
  text(s, 'Multi-task learning, an anatomical consistency constraint and metaheuristic optimization on the URI-CADS benchmark',
    { x: 0.8, y: 3.55, w: 11.2, h: 0.8, fontSize: 18, color: C.ice });
  const cols = [['PRESENTED BY', 'Anubhav Rawat (23BCE1471)\nKaustubh (23BCE1555)'], ['GUIDE', 'Sudharson S'], ['DATE', '16 September 2026']];
  const xs = [0.8, 5.6, 9.0];
  cols.forEach(([k, v], i) => {
    text(s, k, { x: xs[i], y: 5.35, w: 3.8, h: 0.3, fontSize: 11, bold: true, color: C.ice, charSpacing: 2 });
    text(s, v, { x: xs[i], y: 5.7, w: 4.2, h: 0.9, fontSize: 16, color: C.white });
  });
}

// ---------------------------------------------------------------- 2. Agenda
{
  const s = newSlide({ kicker: 'Overview', title: 'What We Will Cover', speaker: A, script: [
    `Here's our plan. I'll cover the problem, the gaps in existing research, our dataset and methodology, and how we designed the optimization study. Kaustubh will then take you through the results, from the baseline phases to the optimizer study and ablations, and close with our findings, current status and next steps.`,
    `On the right is what has changed since the last review. The whole pipeline now runs on our own GPU and was verified against our earlier checkpoints. Every phase was retrained under identical settings. The optimizer study is complete, with 180 model trainings. And four of the five ablation runs are done.`,
  ] });
  const items = [
    ['01', 'Problem and research gaps', A], ['02', 'Dataset and methodology', A], ['03', 'Optimization design', A],
    ['04', 'Results: Phases 2 to 5', K], ['05', 'Results: optimization and ablations', K], ['06', 'Findings, status and next steps', K],
  ];
  items.forEach(([num, t, who], i) => {
    const y = 2.15 + i * 0.75;
    badge(s, M, y, num);
    text(s, t, { x: M + 0.65, y: y + 0.04, w: 4.7, h: 0.4, fontSize: 17, bold: true, valign: 'middle' });
    text(s, who, { x: M + 5.35, y: y + 0.04, w: 1.7, h: 0.4, fontSize: 13, color: C.slate, align: 'right', valign: 'middle' });
  });
  card(s, 7.95, 2.1, 4.78, 4.45);
  text(s, 'Progress since the last review', { x: 8.25, y: 2.35, w: 4.2, h: 0.4, fontSize: 16, bold: true, color: C.navy });
  text(s, bullets([
    'Pipeline reimplemented on a local GPU and verified against earlier checkpoints (within 0.0024)',
    'All phases retrained under identical settings: same split, seed and 15 epochs',
    'Anatomical constraint diagnosed, corrected and re-tested',
    'Optimizer study complete: COA vs WOA vs GWO, 180 model trainings',
    'Ablations: 4 of 5 runs complete',
  ], { size: 13.5, gap: 9 }), { x: 8.25, y: 2.85, w: 4.25, h: 3.55 });
}

// ---------------------------------------------------------------- 3. Motivation
{
  const s = newSlide({ kicker: 'Motivation', title: 'Why Kidney Ultrasound Needs Computer-Aided Diagnosis',
    lede: 'Ultrasound is the first test for renal disease, but reading it depends heavily on who is holding the probe.', speaker: A, script: [
    `Why kidney ultrasound? Chronic kidney disease affects roughly one in ten people worldwide, and ultrasound is the first imaging test for the kidney because it is cheap, radiation-free and widely available. Its weakness is that interpretation depends heavily on the operator, so diagnoses vary between clinicians.`,
    `A useful computer-aided system has to answer three questions from a single image: is this kidney abnormal, what kind of pathology is present, and where are the kidney and its lesions. Those three needs are exactly why we built one multi-task model instead of three separate ones.`,
  ] });
  [['~10%', 'of the global population is affected by chronic kidney disease'],
   ['1st-line', 'imaging modality for renal evaluation is ultrasound'],
   ['High', 'inter-observer variability in manual interpretation']].forEach(([v, l], i) => {
    const x = M + i * 4.115;
    card(s, x, 2.15, 3.9, 1.6);
    stat(s, x + 0.3, 2.25, 3.4, v, l, { valueSize: 36 });
  });
  text(s, 'What a clinically useful system must do from one image', { x: M, y: 4.05, w: CW, h: 0.4, fontSize: 17, bold: true, color: C.navy });
  [['01', 'Detect', 'Decide whether the kidney is healthy or pathological.'],
   ['02', 'Characterize', 'Identify which global and local pathologies are present.'],
   ['03', 'Localize', 'Outline the kidney and mark where the lesions are.']].forEach(([num, h, b], i) => {
    numbered(s, M + i * 4.115, 4.6, 3.9, num, h, b, { bodyH: 0.75 });
  });
  text(s, 'Three questions from the same image make the case for a single multi-task network.', { x: M, y: 6.2, w: CW, h: 0.45, fontSize: 15, italic: true, color: C.teal });
}

// ---------------------------------------------------------------- 4. Benchmark and gaps
{
  const s = newSlide({ kicker: 'Background', title: 'Benchmark System and the Gaps We Target', speaker: A, script: [
    `Our reference point is URI-CADS, a public, peer-reviewed renal ultrasound benchmark, together with its published system. That system uses two heavy detection networks, a Mask R-CNN and a Faster R-CNN on a ResNet-50 backbone. It reports an AUC of 0.8741 for healthy versus pathological, and 0.8190 for multi-pathology, both using five-fold cross-validation.`,
    `Reading their methodology closely, we confirmed five gaps. Their task losses are simply added with equal weight. Preprocessing is never tuned. Lesions are annotated only as bounding boxes, with no pixel masks. Validation stays within their own dataset. And the cost of running two detection networks is not discussed. Each of our contributions targets one of these gaps.`,
  ] });
  card(s, M, 2.1, 4.6, 4.6, C.navy, C.navy);
  text(s, 'REFERENCE SYSTEM', { x: M + 0.35, y: 2.35, w: 4, h: 0.3, fontSize: 11, bold: true, color: C.ice, charSpacing: 2 });
  text(s, 'URI-CADS (Molina-Cabello et al.)', { x: M + 0.35, y: 2.7, w: 4, h: 0.45, fontFace: H, fontSize: 19, bold: true, color: C.white });
  text(s, bullets(['Dual Mask R-CNN + Faster R-CNN', 'ResNet-50 backbone', 'Five-fold cross-validation on its own data'], { size: 13.5, color: C.white, gap: 5 }),
    { x: M + 0.35, y: 3.25, w: 4, h: 1.2 });
  stat(s, M + 0.35, 4.6, 2.0, '0.8741', 'AUC, healthy vs pathological', { valueSize: 30, color: C.white, labelColor: C.ice });
  stat(s, M + 2.45, 4.6, 2.0, '0.8190', 'AUC, multi-pathological', { valueSize: 30, color: C.white, labelColor: C.ice });
  const gaps = [
    ['Unweighted loss sum', 'Task losses are added with equal weight; no tuned or learned weighting.'],
    ['No optimized preprocessing', 'Standard augmentation only; image enhancement is never tuned.'],
    ['No pixel-level lesion masks', 'Lesions exist only as bounding boxes, so lesion segmentation is not addressed.'],
    ['Single-dataset validation', 'Cross-validation within one dataset; no external or cross-cohort test.'],
    ['Compute cost not addressed', 'Two detection networks; deployability on point-of-care devices is not discussed.'],
  ];
  gaps.forEach(([h, b], i) => numbered(s, 5.6, 2.1 + i * 0.93, 7.13, String(i + 1).padStart(2, '0'), h, b, { bodyH: 0.45, bodySize: 12.5 }));
}

// ---------------------------------------------------------------- 5. Contributions
{
  const s = newSlide({ kicker: 'Contributions', title: 'Six Contributions, One Per Gap', speaker: A, script: [
    `These are our six contributions. First, instead of equal loss weights, we search for the weights. Second, we tune the image enhancement parameters jointly with those weights. Third, we compare three metaheuristics, the Coati, Whale and Grey Wolf optimizers, under an equal budget. Fourth, an anatomical constraint that keeps lesion predictions inside the kidney. Fifth, a lightweight single-encoder design. And sixth, validation beyond a single split.`,
    `The status labels are honest: the first five are implemented and evaluated, and the validation work is scripted but still to run.`,
  ] });
  const items = [
    ['Optimized loss weighting', 'Six task-loss weights searched instead of fixed at 1.', 'EVALUATED', true],
    ['Optimized enhancement', 'CLAHE, denoising and gamma tuned jointly with the weights.', 'EVALUATED', true],
    ['Optimizer comparison', 'COA vs WOA vs GWO at an equal budget of 60 trainings each.', 'COMPLETE', true],
    ['Anatomical consistency', 'Penalizes lesion predictions outside the kidney; corrected ratio form.', 'COMPLETE', true],
    ['Lightweight architecture', 'One ResNet-34 encoder, five heads, 24.4 M parameters.', 'COMPLETE', true],
    ['Beyond a single split', 'Five-fold cross-validation and an external cohort test.', 'PENDING', false],
  ];
  items.forEach(([h, b, st, done], i) => {
    const x = M + (i % 3) * 4.115, y = 2.1 + Math.floor(i / 3) * 2.35;
    card(s, x, y, 3.9, 2.15);
    badge(s, x + 0.3, y + 0.3, String(i + 1).padStart(2, '0'));
    chip(s, x + 3.9 - 0.3 - (0.12 + st.length * 0.085), y + 0.38, st, done);
    text(s, h, { x: x + 0.3, y: y + 0.92, w: 3.35, h: 0.4, fontSize: 16, bold: true });
    text(s, b, { x: x + 0.3, y: y + 1.32, w: 3.35, h: 0.75, fontSize: 12.5, color: C.slate });
  });
}

// ---------------------------------------------------------------- 6. Dataset
{
  const d = X.diag;
  const s = newSlide({ kicker: 'Data', title: 'URI-CADS Renal Ultrasound Dataset', lede: 'A public, peer-reviewed benchmark with pixel-level kidney masks and bounding-box lesion annotations.', speaker: A, script: [
    `The dataset contains ${X.n_images.toLocaleString('en-US')} ultrasound images: ${d['0']} healthy and ${d['1'].toLocaleString('en-US')} pathological. It is imbalanced toward disease, which is why our splits are stratified by diagnosis.`,
    `Every image has a pixel-level kidney mask, shown here as the teal outline. Lesions are annotated as bounding boxes, shown in orange, across nine local pathology categories. We split the data 70, 15 and 15 percent into 1,389 training, 298 validation and 298 test images, with a fixed seed, so every phase sees exactly the same data.`,
  ] });
  s.addImage({ path: OUT + '/assets/dataset_examples.png', x: M, y: 2.0, w: CW, h: CW / 3.7209 });
  text(s, 'Teal outline: pixel-level kidney mask     Orange boxes: annotated lesion regions', { x: M, y: 5.3, w: CW, h: 0.3, fontSize: 11.5, italic: true, color: C.slate });
  [[X.n_images.toLocaleString('en-US'), 'annotated ultrasound images'],
   [`${d['0']} / ${d['1'].toLocaleString('en-US')}`, 'healthy / pathological'],
   ['9', 'local pathology categories, plus 2 global labels'],
   ['1,389 / 298 / 298', 'train / validation / test, stratified']].forEach(([v, l], i) => {
    stat(s, M + i * 3.07, 5.62, 2.95, v, l, { valueSize: 26 });
  });
}

// ---------------------------------------------------------------- 7. Label distribution
{
  const lpi = X.lesions_per_image;
  const five = [5, 6, 7, 8, 9, 12].reduce((a, k) => a + (lpi[k] || 0), 0);
  const cat = X.boxes_per_category;
  const top2 = (cat['2'] + cat['7']) / X.total_boxes;
  const s = newSlide({ kicker: 'Data', title: 'Two Label Properties That Shape the Evaluation', speaker: A, script: [
    `Two properties of the labels shape everything that follows. On the left, ${X.n_no_lesion} images, about a third of the dataset, contain no lesion at all. Keep that in mind, because it becomes important when we evaluate lesion segmentation.`,
    `On the right, the lesion categories are very unbalanced. Categories 2 and 7 account for ${pct(top2)} of all ${X.total_boxes.toLocaleString('en-US')} lesion boxes, while categories 5 and 6 have only ${cat['5']} and ${cat['6']}. So results for the rare categories will be noisy, and we treat them with caution.`,
  ] });
  chartLabel(s, 'Images by number of annotated lesions', M, 2.05, 5.9);
  barChart(s, ['0', '1', '2', '3', '4', '5+'], [{ name: 'Images', values: [lpi['0'], lpi['1'], lpi['2'], lpi['3'], lpi['4'], five] }],
    { x: M, y: 2.4, w: 5.9, h: 3.3, chartColors: [C.navy], showLegend: false, dataLabelFormatCode: '#,##0', valAxisLabelFormatCode: '#,##0' });
  chartLabel(s, 'Lesion boxes per local pathology category', 6.83, 2.05, 5.9);
  const catLabels = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];
  barChart(s, catLabels, [{ name: 'Boxes', values: catLabels.map(c => cat[c]) }],
    { x: 6.83, y: 2.4, w: 5.9, h: 3.3, chartColors: [C.teal], showLegend: false, dataLabelFormatCode: '#,##0', valAxisLabelFormatCode: '#,##0' });
  card(s, M, 5.9, CW, 0.9);
  text(s, [
    { text: 'Reading the charts:  ', options: { bold: true, color: C.navy } },
    { text: `${pct(X.n_no_lesion / X.n_images)} of images have no lesion, which distorts naive lesion metrics.  Categories 2 and 7 hold ${pct(top2)} of boxes; categories 5 and 6 have only ${cat['5']} and ${cat['6']}, so their per-category scores are unreliable.`, options: { color: C.ink } },
  ], { x: M + 0.3, y: 5.98, w: CW - 0.6, h: 0.75, fontSize: 13.5, valign: 'middle' });
}

// ---------------------------------------------------------------- 8. Pipeline
{
  const s = newSlide({ kicker: 'Methodology', title: 'End-to-End Framework',
    lede: 'One shared encoder, five task heads, and an optimizer that tunes both the enhancement and the loss weights.', speaker: A, script: [
    `This is the whole pipeline. An image is resized to 512 by 512 pixels and passed through an enhancement stage: contrast-limited adaptive histogram equalization, or CLAHE, then denoising, then gamma correction. A single ResNet-34 encoder extracts features, and all five task heads read that same feature map. Their losses are combined into one weighted multi-task loss.`,
    `The bottom row is what makes the framework optimization-driven. A metaheuristic optimizer proposes the enhancement parameters and the six loss weights, a model is trained with them, and its validation performance is fed back to the optimizer as a fitness score.`,
  ] });
  const box = (x, y, w, h, t1, t2, dark = false) => {
    card(s, x, y, w, h, dark ? C.navy : C.tint, dark ? C.navy : C.line);
    text(s, [{ text: t1, options: { bold: true, fontSize: 14.5, color: dark ? C.white : C.navy, breakLine: true } },
             { text: t2, options: { fontSize: 11.5, color: dark ? C.ice : C.slate } }],
      { x: x + 0.12, y, w: w - 0.24, h, align: 'center', valign: 'middle' });
  };
  const arrow = (x1, y1, x2, y2) => {
    const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color: C.slate, width: 1.5, endArrowType: 'triangle' } };
    if ((y2 < y1 && x2 >= x1) || (y2 > y1 && x2 < x1)) o.flipV = true;
    s.addShape(pres.shapes.LINE, o);
  };
  const cy = 3.66;
  box(0.6, 3.1, 1.75, 1.12, 'Ultrasound image', 'resized to 512 x 512');
  arrow(2.35, cy, 2.62, cy);
  box(2.62, 3.1, 2.1, 1.12, 'Enhancement', 'CLAHE, denoise, gamma');
  arrow(4.72, cy, 4.99, cy);
  box(4.99, 3.1, 1.9, 1.12, 'Shared encoder', 'ResNet-34, pretrained');
  const heads = [['Diagnosis', 'healthy vs pathological'], ['Global pathology', 'PCD, HC'], ['Local pathology', '9 categories'],
                 ['Kidney segmentation', 'pixel mask'], ['Lesion segmentation', 'coarse mask']];
  heads.forEach(([h, sub], i) => {
    const y = 2.1 + i * 0.66, hc = y + 0.27;
    card(s, 7.35, y, 2.6, 0.54);
    text(s, [{ text: h + '  ', options: { bold: true, color: C.ink } }, { text: sub, options: { color: C.slate, fontSize: 10.5 } }],
      { x: 7.45, y, w: 2.45, h: 0.54, fontSize: 12, valign: 'middle' });
    arrow(6.89, cy, 7.35, hc);
    arrow(9.95, hc, 10.4, cy);
  });
  box(10.4, 3.1, 2.33, 1.12, 'Multi-task loss', 'L = sum of w_i x L_i, 6 terms');
  card(s, 2.62, 5.75, 10.11, 0.95, C.navy, C.navy);
  text(s, [{ text: 'Metaheuristic optimizer: COA, WOA or GWO', options: { bold: true, fontSize: 15, color: C.white, breakLine: true } },
           { text: 'proposes 10 parameters, trains a model, and receives its validation fitness in return', options: { fontSize: 12, color: C.ice } }],
    { x: 2.9, y: 5.75, w: 6.6, h: 0.95, valign: 'middle' });
  s.addShape(pres.shapes.LINE, { x: 3.67, y: 4.22, w: 0, h: 1.53, flipV: true, line: { color: C.amber, width: 2, endArrowType: 'triangle' } });
  text(s, 'enhancement parameters', { x: 3.8, y: 4.75, w: 2.3, h: 0.3, fontSize: 11, italic: true, color: C.slate });
  s.addShape(pres.shapes.LINE, { x: 11.56, y: 4.22, w: 0, h: 1.53, flipV: true, line: { color: C.amber, width: 2, endArrowType: 'triangle' } });
  text(s, 'loss weights', { x: 10.3, y: 4.75, w: 1.15, h: 0.3, fontSize: 11, italic: true, color: C.slate, align: 'right' });
}

// ---------------------------------------------------------------- 9. Architecture
{
  const setup = D.setup;
  const s = newSlide({ kicker: 'Methodology', title: 'Architecture: Five Heads on One Encoder', speaker: A, script: [
    `Here are the five heads in detail. Three are classifiers trained with binary cross-entropy: the diagnosis head, the head for the two global pathologies, and the head for the nine local pathology categories. Two are segmentation decoders trained with Dice loss: one for the kidney, and one for lesions, where the target is the union of the annotated lesion boxes. On top of those sits the anatomical consistency term.`,
    `We deliberately avoided Mask R-CNN and U-Net. The whole network has ${(setup.params_MultiTaskNetV2 / 1e6).toFixed(1)} million parameters and trained in ${setup.peak_vram_MultiTaskNetV2} gigabytes of GPU memory. That is the evidence behind our lightweight claim.`,
  ] });
  const rows = [
    [th('Head'), th('Output'), th('Loss'), th('Phase')],
    [td('Diagnosis', { bold: true }), td('1 logit: healthy vs pathological'), td('Binary cross-entropy'), td('2')],
    [td('Global pathology', { bold: true }), td('2 logits: PCD, HC (multi-label)'), td('Binary cross-entropy'), td('2')],
    [td('Local pathology', { bold: true }), td('9 logits, one per lesion category'), td('Binary cross-entropy'), td('4')],
    [td('Kidney segmentation', { bold: true }), td('512 x 512 mask, 5-stage upsampling decoder'), td('Dice loss'), td('2')],
    [td('Lesion segmentation', { bold: true }), td('512 x 512 mask of the lesion-box union'), td('Dice loss'), td('4')],
    [td('Anatomical consistency', { bold: true }), td('Lesion probability outside the kidney'), td('Ratio penalty'), td('5')],
  ];
  table(s, rows, { x: M, y: 2.1, w: 8.2, colW: [2.1, 3.4, 1.85, 0.85], rowH: 0.56, fontSize: 12.5 });
  text(s, 'No Mask R-CNN, Faster R-CNN or U-Net: every head reads the same encoder feature map.', { x: M, y: 6.25, w: 8.2, h: 0.45, fontSize: 13.5, italic: true, color: C.teal });
  [[`${(setup.params_MultiTaskNetV2 / 1e6).toFixed(1)} M`, 'trainable parameters in total'],
   [`${setup.peak_vram_MultiTaskNetV2} GB`, 'peak GPU memory in training (batch 4, 512 x 512, mixed precision)'],
   ['1 network', 'produces all five outputs in a single forward pass']].forEach(([v, l], i) => {
    card(s, 9.2, 2.1 + i * 1.55, 3.53, 1.38);
    stat(s, 9.45, 2.15 + i * 1.55, 3.1, v, l, { valueSize: 28 });
  });
}

// ---------------------------------------------------------------- 10. Losses
{
  const s = newSlide({ kicker: 'Methodology', title: 'Loss Design and the Anatomical Constraint', speaker: A, script: [
    `The training objective is a weighted sum of six losses. Through Phases 2 to 5 every weight is 1, which mirrors the URI-CADS design and gives us a fair baseline. In Phase 6, each weight is searched between 0.1 and 5.`,
    `The right side needs a little care, because it became one of our key findings. Our first consistency loss penalized the absolute amount of lesion probability lying outside the kidney. But the quantity we report is the fraction of lesion probability outside the kidney. The absolute version can be lowered just by predicting less lesion everywhere, which leaves that fraction unchanged. The corrected ratio version can only be lowered by moving predictions inside the kidney. Kaustubh will show you what that difference did in practice.`,
  ] });
  card(s, M, 2.1, 5.9, 4.6);
  text(s, 'Multi-task objective', { x: M + 0.3, y: 2.3, w: 5.3, h: 0.4, fontSize: 17, bold: true, color: C.navy });
  text(s, 'L = w1 L_diag + w2 L_global + w3 L_local\n     + w4 L_kidney + w5 L_lesion + w6 L_anat', { x: M + 0.3, y: 2.8, w: 5.4, h: 0.8, fontFace: H, fontSize: 15, color: C.ink });
  text(s, bullets([
    'Phases 2 to 5: every w = 1, the unweighted sum used by URI-CADS',
    'Phase 6: each w searched in [0.1, 5.0]',
    'Classification heads: binary cross-entropy',
    'Segmentation heads: Dice loss = 1 - 2|P ∩ G| / (|P| + |G|)',
  ], { size: 13, gap: 7 }), { x: M + 0.3, y: 3.75, w: 5.35, h: 2.8 });
  card(s, 6.83, 2.1, 5.9, 4.6);
  text(s, 'Anatomical consistency: two formulations', { x: 7.13, y: 2.3, w: 5.3, h: 0.4, fontSize: 17, bold: true, color: C.navy });
  text(s, [{ text: 'Original (Phase 5): absolute mass', options: { bold: true, color: C.red, breakLine: true } },
           { text: 'L = mean( p_lesion x (1 - kidney) )', options: { fontFace: H, color: C.ink } }],
    { x: 7.13, y: 2.85, w: 5.3, h: 0.75, fontSize: 14 });
  text(s, [{ text: 'Corrected (Phase 5b): fraction outside the kidney', options: { bold: true, color: C.green, breakLine: true } },
           { text: 'L = sum( p_lesion x (1 - kidney) ) / sum( p_lesion )', options: { fontFace: H, color: C.ink } }],
    { x: 7.13, y: 3.75, w: 5.3, h: 0.75, fontSize: 14 });
  text(s, 'Why it matters: the absolute form can be reduced by predicting less lesion everywhere, which leaves the reported fraction unchanged. The ratio form is unaffected by uniform scaling, so the only way to reduce it is to move predictions inside the kidney.',
    { x: 7.13, y: 4.75, w: 5.3, h: 1.8, fontSize: 13, color: C.slate });
}

// ---------------------------------------------------------------- 11. Enhancement
{
  const s = newSlide({ kicker: 'Methodology', title: 'Image Enhancement Layer', speaker: A, script: [
    `This slide shows the enhancement stage on two test images. The middle column uses the hand-picked settings from Phase 3, and the right column uses the settings the Coati optimizer found.`,
    `There are four parameters. The CLAHE clip limit caps how much local contrast is amplified. The tile grid sets the size of the neighbourhood that contrast is computed over. The denoising strength controls non-local-means filtering. And gamma reshapes the brightness curve. Non-local-means denoising costs about 0.6 seconds per image, so for each configuration we compute the enhanced images once, across 12 CPU workers, and reuse them for every epoch.`,
  ] });
  s.addImage({ path: OUT + '/assets/enhancement_comparison.png', x: M, y: 2.05, w: 4.65 * 1.4634, h: 4.65 });
  const params = [
    ['CLAHE clip limit', 'Caps local contrast amplification.  Search range 1.0 to 5.0'],
    ['CLAHE tile grid', 'Neighbourhood size for local contrast.  Search range 4 to 16'],
    ['Denoising strength h', 'Non-local-means filter strength.  Search range 0 to 20'],
    ['Gamma', 'Reshapes the brightness curve.  Search range 0.5 to 2.0'],
  ];
  params.forEach(([h, b], i) => numbered(s, 7.65, 2.1 + i * 0.93, 5.08, String(i + 1).padStart(2, '0'), h, b, { bodyH: 0.5, bodySize: 12.5 }));
  card(s, 7.65, 5.85, 5.08, 0.85);
  text(s, 'Enhancement is computed once per configuration across 12 CPU workers, because non-local-means denoising costs about 0.6 s per image.',
    { x: 7.9, y: 5.85, w: 4.6, h: 0.85, fontSize: 12.5, color: C.ink, valign: 'middle' });
}

// ---------------------------------------------------------------- 12. Optimization design
{
  const fw = D.search.fitness_weights;
  const s = newSlide({ kicker: 'Optimization design', title: 'How the Search Is Set Up', speaker: A, script: [
    `The optimizer searches ten parameters at once: six loss weights and four enhancement settings. Every candidate is scored with one fitness value on the validation set: 30 percent diagnosis AUC, 15 percent local pathology AUC, 25 percent each for kidney and lesion Dice, and 5 percent for anatomical consistency. We use lesion Dice on lesion-bearing images only, for a reason Kaustubh will explain shortly.`,
    `Every candidate means training a model, so we scored each one after 3 epochs at 256 by 256 pixels, which takes about two minutes. To keep the comparison fair, each optimizer received exactly 60 trainings, and all three started from the same six random candidates. Every evaluation was also cached, so an interrupted run could resume without repeating work.`,
  ] });
  card(s, M, 2.1, 6.3, 1.25);
  text(s, [{ text: 'Fitness, maximized on the validation split', options: { bold: true, color: C.navy, fontSize: 15, breakLine: true } },
           { text: 'F = 0.30 AUC_diag + 0.15 AUC_local + 0.25 Dice_kidney + 0.25 Dice_lesion+ + 0.05 (1 - violation)', options: { fontFace: H, fontSize: 13, color: C.ink } }],
    { x: M + 0.3, y: 2.15, w: 5.8, h: 1.15, valign: 'middle' });
  chartLabel(s, 'Share of fitness by objective', M, 3.55, 6.3);
  barChart(s, ['Diagnosis AUC', 'Kidney Dice', 'Lesion Dice (lesion images)', 'Local pathology AUC', 'Consistency'],
    [{ name: 'Weight', values: [fw.diagnosis_auc, fw.kidney_dice, fw.lesion_dice_positive_only, fw.local_pathology_mean_auc, fw.consistency] }],
    { x: M, y: 3.85, w: 6.3, h: 2.85, barDir: 'bar', chartColors: [C.navy], showLegend: false, dataLabelFormatCode: '0.00', valAxisMaxVal: 0.35, valAxisLabelFormatCode: '0.00', catAxisLabelFontSize: 11.5 });
  const cards = [
    ['10 parameters', '6 loss weights in [0.1, 5.0] and 4 enhancement settings, searched jointly'],
    ['Equal budget', '60 model trainings per optimizer, 180 in total, capped on trainings rather than iterations'],
    ['Cheap evaluation', 'Each candidate trained 3 epochs at 256 x 256, about 2 minutes, then scored on validation'],
    ['Common start', 'Same random seed, so all optimizers begin from the same 6 candidates; every result is cached'],
  ];
  cards.forEach(([h, b], i) => {
    const y = 2.1 + i * 1.17;
    card(s, 7.25, y, 5.48, 1.03);
    text(s, [{ text: h, options: { bold: true, color: C.navy, fontSize: 14.5, breakLine: true } }, { text: b, options: { color: C.slate, fontSize: 12.5 } }],
      { x: 7.5, y, w: 5.05, h: 1.03, valign: 'middle' });
  });
}

// ---------------------------------------------------------------- 13. Optimizers
{
  const s = newSlide({ kicker: 'Optimization design', title: 'Three Metaheuristics Compared', speaker: A, script: [
    `The three optimizers search in different ways. The Coati algorithm splits its population: half moves toward the best solution and half toward a random point, and then it runs a local search whose radius shrinks every iteration. The Whale algorithm encircles the best solution and uses a spiral update. The Grey Wolf optimizer moves every candidate toward the three best solutions found so far.`,
    `Before trusting them on the real problem, we checked all three on two standard test functions, with 1,000 evaluations each. Grey Wolf was best on the smooth sphere function, and Coati was best on Rastrigin, which is full of local optima. With that, I'll hand over to Kaustubh for the results.`,
  ] });
  const algs = [
    ['COA', 'Coati Optimization Algorithm', 'Dehghani et al., 2023', 'Half the population moves toward the best solution, half toward a random point; then a local search with a radius that shrinks each iteration.'],
    ['WOA', 'Whale Optimization Algorithm', 'Mirjalili and Lewis, 2016', 'Encircles the best solution, explores via random agents, and exploits with a spiral update; step size shrinks over time.'],
    ['GWO', 'Grey Wolf Optimizer', 'Mirjalili et al., 2014', 'Moves every candidate toward the three best solutions so far (alpha, beta, delta); exploration shrinks linearly.'],
  ];
  algs.forEach(([ab, name, ref, desc], i) => {
    const x = M + i * 4.115;
    card(s, x, 2.1, 3.9, 2.45);
    text(s, ab, { x: x + 0.3, y: 2.25, w: 1.5, h: 0.55, fontFace: H, fontSize: 26, bold: true, color: C.navy });
    text(s, [{ text: name, options: { bold: true, color: C.ink, breakLine: true } }, { text: ref, options: { color: C.slate, italic: true, fontSize: 11 } }],
      { x: x + 0.3, y: 2.8, w: 3.4, h: 0.6, fontSize: 13 });
    text(s, desc, { x: x + 0.3, y: 3.45, w: 3.35, h: 1.05, fontSize: 12, color: C.slate });
  });
  chartLabel(s, 'Sanity check on standard test functions (10 dimensions, 1,000 evaluations, mean of 5 seeds; lower is better)', M, 4.75, CW);
  table(s, [
    [th('Function'), th('Landscape'), th('COA'), th('WOA'), th('GWO')],
    [td('Sphere', { bold: true }), td('Smooth, one optimum'), td('6.9e-7'), td('3.6e-4'), td('1.3e-9', { bold: true, color: C.green })],
    [td('Rastrigin', { bold: true }), td('Many local optima'), td('25.0', { bold: true, color: C.green }), td('44.0'), td('37.6')],
  ], { x: M, y: 5.1, w: CW, colW: [2.2, 3.4, 2.18, 2.18, 2.17], rowH: 0.4, fontSize: 12.5 });
  text(s, 'GWO is strongest on smooth landscapes; COA is strongest on multimodal ones, which is closer to what a noisy hyperparameter search looks like.',
    { x: M, y: 6.45, w: CW, h: 0.45, fontSize: 13, italic: true, color: C.teal });
}

// ---------------------------------------------------------------- 14. Protocol & verification
{
  const v = D.verify;
  const s = newSlide({ kicker: 'Experimental setup', title: 'Protocol and Reproducibility Check', speaker: K, script: [
    `Thank you, Anubhav. Before we show any results, a word on how trustworthy they are. Every phase uses the same data split, the same seed, 15 epochs, the same batch size and learning rate, on the same GPU. We also moved the whole pipeline from Colab to local hardware, so the first thing we did was re-evaluate our earlier checkpoints with the new code. As the table shows, every metric reproduces within 0.0024.`,
    `A quick note on the metrics we'll use. AUC measures how well the model ranks cases, where 0.5 is chance. Dice measures overlap with the ground truth, where 1 is perfect. And the violation rate is the share of predicted lesion probability that falls outside the kidney, so lower is better.`,
  ] });
  card(s, M, 2.1, 5.6, 3.25);
  text(s, 'Training protocol, identical for every phase', { x: M + 0.3, y: 2.3, w: 5.1, h: 0.4, fontSize: 16, bold: true, color: C.navy });
  text(s, bullets([
    'Split: 70 / 15 / 15, stratified by diagnosis, seed 42',
    'Training: 15 epochs, batch 4, Adam, learning rate 1e-4',
    'Mixed precision; best checkpoint chosen on validation Dice',
    'Input: 512 x 512, normalized to [-1, 1]',
    `Hardware: ${D.setup.gpu.replace('NVIDIA GeForce ', '')}, 4 GB`,
  ], { size: 13, gap: 7 }), { x: M + 0.3, y: 2.8, w: 5.1, h: 2.5 });
  card(s, M, 5.5, 5.6, 1.25);
  text(s, [{ text: 'Metrics.  ', options: { bold: true, color: C.navy } },
           { text: 'AUC: ranking quality, 0.5 is chance.  Dice: overlap, 1 is perfect.  Violation rate: share of lesion probability outside the kidney, lower is better.', options: { color: C.ink } }],
    { x: M + 0.3, y: 5.5, w: 5.1, h: 1.25, fontSize: 12.5, valign: 'middle' });
  chartLabel(s, 'Earlier checkpoints re-evaluated with the local implementation', 6.55, 2.1, 6.18);
  const rows = [[th('Checkpoint'), th('Metric'), th('Original'), th('Reproduced')]];
  [['phase2_baseline_best.pt', 'Phase 2', 'diagnosis_auc', 'Diagnosis AUC'],
   ['phase2_baseline_best.pt', 'Phase 2', 'kidney_dice', 'Kidney Dice'],
   ['phase3_enhanced_best.pt', 'Phase 3', 'diagnosis_auc', 'Diagnosis AUC'],
   ['phase4_multitask_best.pt', 'Phase 4', 'local_pathology_mean_auc', 'Local path. AUC'],
   ['phase5_anatomical_best.pt', 'Phase 5', 'consistency_violation_rate', 'Violation rate']].forEach(([ck, ph, key, label]) => {
    rows.push([td(ph, { bold: true }), td(label), td(f4(v[ck].colab_reported[key])), td(f4(v[ck].local[key]))]);
  });
  table(s, rows, { x: 6.55, y: 2.5, w: 6.18, colW: [1.35, 2.13, 1.35, 1.35], rowH: 0.46, fontSize: 12.5 });
  card(s, 6.55, 5.5, 6.18, 1.25, C.greenTint, C.greenTint);
  text(s, [{ text: 'Largest difference: 0.0024', options: { bold: true, color: C.green, fontSize: 18, breakLine: true } },
           { text: 'The reimplementation matches the original, so earlier and new results are comparable.', options: { color: C.ink, fontSize: 12.5 } }],
    { x: 6.85, y: 5.5, w: 5.6, h: 1.25, valign: 'middle' });
}

// ---------------------------------------------------------------- 15. Phase 2 vs 3
{
  const p2 = T('phase2'), p3 = T('phase3');
  const s = newSlide({ kicker: 'Results  ·  Phases 2 and 3', title: 'Hand-Picked Enhancement Is Roughly Neutral', speaker: K, script: [
    `Phases 2 and 3 compare the baseline with hand-picked enhancement. At our last review this comparison was confounded, because Phase 3 had trained for 10 epochs against Phase 2's 15. We reran it at 15 epochs. Note that the vertical axis starts at 0.70, so the differences are visible.`,
    `Enhancement raised kidney Dice by only ${f3(p3.kidney_dice - p2.kidney_dice)}, while diagnosis AUC fell by ${f3(p2.diagnosis_auc - p3.diagnosis_auc)} and global pathology AUC by ${f3(p2.global_pathology_mean_auc - p3.global_pathology_mean_auc)}. So hand-picked enhancement is roughly neutral. That is a useful negative result, because it is exactly the motivation for optimizing the enhancement parameters rather than guessing them.`,
  ] });
  chartLabel(s, 'Test-set performance at matched 15 epochs (axis starts at 0.70)', M, 2.05, 7.4);
  barChart(s, ['Diagnosis AUC', 'Global pathology AUC', 'Kidney Dice'], [
    { name: 'Phase 2: no enhancement', values: [p2.diagnosis_auc, p2.global_pathology_mean_auc, p2.kidney_dice] },
    { name: 'Phase 3: hand-picked enhancement', values: [p3.diagnosis_auc, p3.global_pathology_mean_auc, p3.kidney_dice] },
  ], { x: M, y: 2.4, w: 7.4, h: 4.35, chartColors: [C.gray, C.teal], valAxisMinVal: 0.7, valAxisMaxVal: 0.95, valAxisLabelFormatCode: '0.00', catAxisLabelFontSize: 12.5 });
  card(s, 8.35, 2.1, 4.38, 4.65);
  text(s, 'What changed with enhancement', { x: 8.65, y: 2.3, w: 3.9, h: 0.4, fontSize: 16, bold: true, color: C.navy });
  [[`+${f3(p3.kidney_dice - p2.kidney_dice)}`, 'kidney Dice', C.green],
   [`-${f3(p2.diagnosis_auc - p3.diagnosis_auc)}`, 'diagnosis AUC', C.red],
   [`-${f3(p2.global_pathology_mean_auc - p3.global_pathology_mean_auc)}`, 'global pathology AUC', C.red]].forEach(([v, l, col], i) => {
    text(s, [{ text: v + '  ', options: { fontFace: H, fontSize: 24, bold: true, color: col } }, { text: l, options: { fontSize: 13.5, color: C.ink } }],
      { x: 8.65, y: 2.85 + i * 0.62, w: 3.9, h: 0.55, valign: 'middle' });
  });
  text(s, 'The earlier comparison used 10 vs 15 epochs and could not be interpreted. At matched budgets the gain is small and classification drops, which is why Phase 6 optimizes these parameters instead of fixing them by hand.',
    { x: 8.65, y: 4.8, w: 3.85, h: 1.85, fontSize: 12.5, color: C.slate });
}

// ---------------------------------------------------------------- 16. Lesion metric
{
  const p4 = T('phase4'), p2 = T('phase2');
  const s = newSlide({ kicker: 'Results  ·  Evaluation', title: 'The Standard Lesion Dice Was Hiding Real Performance', speaker: K, script: [
    `This finding changed how we evaluate lesions. Our earlier lesion Dice was about 0.34, which looked weak. But a third of the test images contain no lesion, and on those images a correct empty prediction scores almost 1.`,
    `To measure the effect, we evaluated a model with no lesion head at all, which predicts nothing anywhere. It scores ${f3(p2.lesion_dice)}, barely below our Phase 4 model's ${f3(p4.lesion_dice)}. Restricting to the ${p4.n_lesion_positive} test images that actually contain a lesion separates them clearly: zero for the empty model, and ${f3(p4.lesion_dice_positive_only)} for ours. So the lesion decoder was working all along; the metric was hiding it. We now report both numbers, and the optimizer only ever sees the second one.`,
  ] });
  chartLabel(s, 'Lesion Dice on the test set', M, 2.05, 6.7);
  barChart(s, ['Model with no lesion head', 'Phase 4 multi-task model'], [
    { name: `All ${p4.n_samples} test images`, values: [p2.lesion_dice, p4.lesion_dice] },
    { name: `Only the ${p4.n_lesion_positive} lesion-bearing images`, values: [p2.lesion_dice_positive_only, p4.lesion_dice_positive_only] },
  ], { x: M, y: 2.4, w: 6.7, h: 4.35, chartColors: [C.gray, C.navy], valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisLabelFormatCode: '0.0', catAxisLabelFontSize: 12.5 });
  const nums = [['0.319 vs 0.344', 'all images: an empty prediction nearly matches the model'],
                ['0.000 vs 0.506', 'lesion-bearing images: the gap is unmistakable']];
  nums.forEach(([v, l], i) => {
    card(s, 7.65, 2.1 + i * 1.45, 5.08, 1.3);
    stat(s, 7.95, 2.1 + i * 1.45, 4.6, v, l, { valueSize: 26 });
  });
  text(s, bullets([
    `${pct(1 - p4.n_lesion_positive / p4.n_samples)} of test images contain no lesion; an empty prediction scores about 1 on each of them`,
    'We now report both versions of lesion Dice',
    'The optimizer uses the lesion-bearing version, so it cannot be rewarded for predicting nothing',
  ], { size: 13, gap: 8 }), { x: 7.65, y: 5.1, w: 5.08, h: 1.7 });
}

// ---------------------------------------------------------------- 17. Anatomical consistency
{
  const p4 = T('phase4'), p5 = T('phase5'), p5b = T('phase5b');
  const red = (p4.consistency_violation_rate - p5b.consistency_violation_rate) / p4.consistency_violation_rate;
  const s = newSlide({ kicker: 'Results  ·  Phases 4 and 5', title: 'The Anatomical Constraint Works Only in Ratio Form', speaker: K, script: [
    `Here is what the change in formulation that Anubhav described actually did. Without any constraint, ${pct(p4.consistency_violation_rate)} of predicted lesion probability fell outside the kidney. Adding the original absolute-mass loss made it worse, ${pct(p5.consistency_violation_rate)}, the opposite of its purpose, and our earlier checkpoints showed the same thing.`,
    `The corrected ratio loss cut it to ${pct(p5b.consistency_violation_rate)}, a ${Math.round(red * 100)} percent reduction, while giving the best diagnosis AUC of any model, ${f4(p5b.diagnosis_auc)}. The cost is small: lesion Dice on lesion-bearing images went from ${f3(p4.lesion_dice_positive_only)} to ${f3(p5b.lesion_dice_positive_only)}, which is expected when annotated lesion boxes extend slightly past the kidney outline.`,
  ] });
  chartLabel(s, 'Violation rate on the test set (lower is better)', M, 2.05, 5.8);
  barChart(s, ['Phase 4\nno constraint', 'Phase 5\nabsolute form', 'Phase 5b\nratio form'], [
    { name: 'Violation rate', values: [p4.consistency_violation_rate, p5.consistency_violation_rate, p5b.consistency_violation_rate] },
  ], { x: M, y: 2.4, w: 5.8, h: 4.35, chartColors: [C.navy], showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 0.35, valAxisLabelFormatCode: '0.00', catAxisLabelFontSize: 12.5 });
  card(s, 6.75, 2.1, 5.98, 1.35, C.greenTint, C.greenTint);
  text(s, [{ text: `-${Math.round(red * 100)}%  `, options: { fontFace: H, fontSize: 34, bold: true, color: C.green } },
           { text: 'out-of-kidney lesion probability, Phase 4 to Phase 5b', options: { fontSize: 13.5, color: C.ink } }],
    { x: 7.05, y: 2.1, w: 5.5, h: 1.35, valign: 'middle' });
  const row = (label, k, fmt = f4, bestFn) => {
    const vals = [p4[k], p5[k], p5b[k]];
    const best = bestFn(...vals);
    return [td(label, { bold: true }), ...vals.map(v => td(fmt(v), v === best ? { bold: true, color: C.green } : {}))];
  };
  table(s, [
    [th('Test metric'), th('Phase 4'), th('Phase 5'), th('Phase 5b')],
    row('Violation rate', 'consistency_violation_rate', f4, Math.min),
    row('Diagnosis AUC', 'diagnosis_auc', f4, Math.max),
    row('Kidney Dice', 'kidney_dice', f4, Math.max),
    row('Lesion Dice (lesion images)', 'lesion_dice_positive_only', f4, Math.max),
  ], { x: 6.75, y: 3.65, w: 5.98, colW: [2.48, 1.17, 1.17, 1.16], rowH: 0.42, fontSize: 12.5 });
  text(s, 'The absolute form also raised violations on the original checkpoints (0.2572 to 0.3786), so the effect is reproducible.',
    { x: 6.75, y: 5.95, w: 5.98, h: 0.8, fontSize: 12.5, italic: true, color: C.slate });
}

// ---------------------------------------------------------------- 18. Training dynamics
{
  const ep = D.phase4.history.map(h => String(h.epoch));
  const hist = (ph, k) => D[ph].history.map(h => h[k]);
  const s = newSlide({ kicker: 'Results  ·  Phases 4 and 5', title: 'How the Constraint Acts During Training', speaker: K, script: [
    `These curves show the effect is systematic across training rather than a lucky final epoch. On the left is the validation violation rate at every epoch. The original absolute form, in orange, spikes to about 0.70 at epoch 4 and only recovers to the unconstrained level by the end. The ratio form, in navy, holds around 0.25 for the first eight epochs and then falls steadily, reaching 0.058 at epoch 15, and it is still falling, which suggests longer training could reduce it further.`,
    `On the right, lesion Dice on lesion-bearing images climbs along a similar path for all three models, which fits the small cost we saw on the previous slide.`,
  ] });
  chartLabel(s, 'Validation violation rate by epoch (lower is better)', M, 2.05, 5.95);
  lineChart(s, ep, [
    { name: 'Phase 4: none', values: hist('phase4', 'val_consistency_violation_rate') },
    { name: 'Phase 5: absolute', values: hist('phase5', 'val_consistency_violation_rate') },
    { name: 'Phase 5b: ratio', values: hist('phase5b', 'val_consistency_violation_rate') },
  ], { x: M, y: 2.4, w: 5.95, h: 3.7, chartColors: [C.gray, C.amber, C.navy], valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisLabelFormatCode: '0.0', valAxisTitle: 'violation rate', showValAxisTitle: false });
  chartLabel(s, 'Validation lesion Dice on lesion-bearing images by epoch', 6.78, 2.05, 5.95);
  lineChart(s, ep, [
    { name: 'Phase 4: none', values: hist('phase4', 'val_lesion_dice_positive_only') },
    { name: 'Phase 5: absolute', values: hist('phase5', 'val_lesion_dice_positive_only') },
    { name: 'Phase 5b: ratio', values: hist('phase5b', 'val_lesion_dice_positive_only') },
  ], { x: 6.78, y: 2.4, w: 5.95, h: 3.7, chartColors: [C.gray, C.amber, C.navy], valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisLabelFormatCode: '0.0' });
  card(s, M, 6.22, CW, 0.62);
  text(s, [{ text: 'Reading the curves:  ', options: { bold: true, color: C.navy } },
           { text: 'the ratio form drives a steady decline in the second half of training and is still improving at epoch 15; lesion Dice rises similarly for all three.', options: { color: C.ink } }],
    { x: M + 0.3, y: 6.22, w: CW - 0.6, h: 0.62, fontSize: 13, valign: 'middle' });
}

// ---------------------------------------------------------------- 19. Consolidated table
{
  const phases = [['2', 'phase2', 'Equal weights, no enhancement'], ['3', 'phase3', 'Hand-picked enhancement'],
                  ['4', 'phase4', 'Five heads, no constraint'], ['5', 'phase5', 'Constraint, absolute form'],
                  ['5b', 'phase5b', 'Constraint, ratio form'], ['6', 'phase6_final', 'COA-optimized weights + enhancement']];
  const cols = [['diagnosis_auc', 'max', 'Diag.\nAUC'], ['global_pathology_mean_auc', 'max', 'Global\nAUC'], ['local_pathology_mean_auc', 'max', 'Local\nAUC'],
                ['kidney_dice', 'max', 'Kidney\nDice'], ['lesion_dice', 'max', 'Lesion Dice\n(all)'], ['lesion_dice_positive_only', 'max', 'Lesion Dice\n(lesion imgs)'],
                ['consistency_violation_rate', 'min', 'Violation\nrate']];
  const noLesion = new Set(['phase2', 'phase3']);
  const best = {};
  cols.forEach(([k, dir]) => {
    const vals = phases.filter(([, ph]) => !(noLesion.has(ph) && ['local_pathology_mean_auc', 'lesion_dice', 'lesion_dice_positive_only', 'consistency_violation_rate'].includes(k))).map(([, ph]) => T(ph)[k]);
    best[k] = dir === 'max' ? Math.max(...vals) : Math.min(...vals);
  });
  const s = newSlide({ kicker: 'Results  ·  Summary', title: 'All Configurations on the Same Test Set', speaker: K, script: [
    `This table brings all six configurations together on the same 298 test images, with the best value in each column in bold green. The corrected anatomical model, Phase 5b, has the lowest violation rate, and its diagnosis AUC of 0.9014 is effectively tied with the baseline's 0.9013. Phase 4 gives the best lesion Dice on lesion-bearing images, and the COA-optimized model gives the best kidney Dice.`,
    `One important caveat. URI-CADS reports 0.8741 AUC under five-fold cross-validation, while these numbers come from a single split. Our 0.90 looks stronger, but the like-for-like comparison is Phase 8, which is next on our list.`,
  ] });
  const rows = [[th('Phase'), th('Configuration'), ...cols.map(([, , l]) => th(l))]];
  phases.forEach(([p, ph, label]) => {
    const t = T(ph);
    rows.push([td(p, { bold: true }), td(label), ...cols.map(([k]) => {
      if (noLesion.has(ph) && ['local_pathology_mean_auc', 'lesion_dice', 'lesion_dice_positive_only', 'consistency_violation_rate'].includes(k)) return td('—', { color: C.slate, align: 'center' });
      const v = t[k];
      return td(f4(v), Object.assign({ align: 'center' }, v === best[k] ? { bold: true, color: C.green } : {}));
    })]);
  });
  table(s, rows, { x: M, y: 2.1, w: CW, colW: [0.75, 3.43, 1.1, 1.1, 1.1, 1.1, 1.2, 1.25, 1.1], rowH: [0.62, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], fontSize: 12.5 });
  card(s, M, 5.9, CW, 0.85);
  text(s, [{ text: 'Caveat:  ', options: { bold: true, color: C.navy } },
           { text: 'single held-out split of 298 images, one seed per configuration. URI-CADS reports 0.8741 AUC under five-fold cross-validation, so the like-for-like comparison is Phase 8. Phases 2 and 3 have no lesion head.', options: { color: C.ink } }],
    { x: M + 0.3, y: 5.9, w: CW - 0.6, h: 0.85, fontSize: 13, valign: 'middle' });
}

// ---------------------------------------------------------------- 20. Convergence
{
  const labels = Array.from({ length: 60 }, (_, i) => String(i + 1));
  const curve = k => S[k].curve.map(c => -c);
  const s = newSlide({ kicker: 'Results  ·  Phase 6', title: 'Optimizer Study: 180 Model Trainings', speaker: K, script: [
    `Now the optimizer study: 180 model trainings in 5.6 hours of GPU time. Each line tracks the best fitness found so far against the number of trainings used. All three reach 0.592 within their first six trainings, because they share the same starting candidates.`,
    `Grey Wolf improved first, reaching 0.601 by its 18th training, and then stopped improving. Coati took longer but found the best configuration overall, 0.608, at its 36th training. Whale stalled for 27 trainings and finished at 0.598. None of the three improved in its last eight trainings. The budget was capped on trainings rather than iterations, so this is a fair head-to-head. Without that cap, Coati would have used about two and a half times as many trainings as the others. One honest note: the best search result is only about 0.016 above the best of the six random starting candidates.`,
  ] });
  chartLabel(s, 'Best validation fitness found so far (search fidelity: 256 x 256, 3 epochs)', M, 2.05, 7.7);
  lineChart(s, labels, [
    { name: 'COA', values: curve('COA') }, { name: 'WOA', values: curve('WOA') }, { name: 'GWO', values: curve('GWO') },
  ], { x: M, y: 2.4, w: 7.7, h: 4.0, chartColors: [C.navy, C.amber, C.teal], valAxisMinVal: 0.54, valAxisMaxVal: 0.62, valAxisLabelFormatCode: '0.00', catAxisLabelFrequency: 10, lineSize: 2.5 });
  text(s, 'Model trainings used', { x: M, y: 6.4, w: 7.7, h: 0.3, fontSize: 11.5, color: C.slate, align: 'center' });
  table(s, [
    [th('Optimizer'), th('Best fitness'), th('Found at'), th('Hours')],
    [td('COA', { bold: true }), td('0.6080', { bold: true, color: C.green }), td('training 36'), td(f2(S.COA.hours))],
    [td('GWO', { bold: true }), td(f4(S.GWO.best_fitness)), td('training 18'), td(f2(S.GWO.hours))],
    [td('WOA', { bold: true }), td(f4(S.WOA.best_fitness)), td('training 52'), td(f2(S.WOA.hours))],
  ], { x: 8.6, y: 2.1, w: 4.13, colW: [1.0, 1.13, 1.15, 0.85], rowH: 0.42, fontSize: 12.5 });
  text(s, bullets([
    'All three start at 0.592: the shared initial candidates',
    'GWO improves earliest; COA finds the highest fitness',
    'No optimizer improved in its last 8 trainings',
    'Best result is only +0.016 over the best random start',
  ], { size: 13, gap: 8 }), { x: 8.6, y: 4.0, w: 4.13, h: 2.75 });
}

// ---------------------------------------------------------------- 21. Chosen configurations
{
  const terms = ['diagnosis', 'global_pathology', 'local_pathology', 'kidney_seg', 'lesion_seg', 'consistency'];
  const s = newSlide({ kicker: 'Results  ·  Phase 6', title: 'What the Optimizers Chose', speaker: K, script: [
    `What did the optimizers choose? All three raised the local pathology and kidney segmentation weights well above the baseline value of 1. Coati and Whale cut the diagnosis and global pathology weights to near the minimum of 0.1, while Grey Wolf kept diagnosis high, at 2.9.`,
    `For enhancement, all three chose a gamma below 1 and a larger CLAHE tile grid than our hand-picked settings, and Coati and Whale almost switched denoising off. When three different algorithms converge on similar enhancement settings, that suggests those settings reflect a real signal rather than noise.`,
  ] });
  chartLabel(s, 'Loss weights found by each optimizer (baseline: all weights = 1.0)', M, 2.05, 7.1);
  barChart(s, ['Diagnosis', 'Global path.', 'Local path.', 'Kidney seg.', 'Lesion seg.', 'Consistency'],
    ['COA', 'WOA', 'GWO'].map(k => ({ name: k, values: terms.map(t => S[k].loss_weights[t]) })),
    { x: M, y: 2.4, w: 7.1, h: 3.8, chartColors: [C.navy, C.amber, C.teal], valAxisMinVal: 0, valAxisMaxVal: 5.5, dataLabelFormatCode: '0.0', dataLabelFontSize: 8.5, valAxisLabelFormatCode: '0', barGapWidthPct: 50 });
  chartLabel(s, 'Enhancement parameters', 8.05, 2.05, 4.68);
  const e = k => S[k].enhancement;
  table(s, [
    [th(''), th('Clip'), th('Tile'), th('Denoise'), th('Gamma')],
    [td('Hand-picked', { bold: true, color: C.slate }), td('2.00', { color: C.slate }), td('8', { color: C.slate }), td('10.0', { color: C.slate }), td('1.20', { color: C.slate })],
    ...['COA', 'WOA', 'GWO'].map(k => [td(k, { bold: true }), td(f2(e(k).clahe_clip_limit)), td(String(e(k).clahe_tile_grid_size)), td(e(k).denoise_h.toFixed(1)), td(f2(e(k).gamma))]),
  ], { x: 8.05, y: 2.4, w: 4.68, colW: [1.28, 0.8, 0.7, 1.0, 0.9], rowH: 0.42, fontSize: 12.5 });
  text(s, bullets([
    'All three: gamma below 1 and a larger tile grid than hand-picked',
    'COA and WOA: denoising nearly off',
    'COA and WOA: diagnosis weight near the 0.1 floor; GWO keeps it at 2.9',
  ], { size: 12.5, gap: 7 }), { x: 8.05, y: 4.65, w: 4.68, h: 1.6 });
  card(s, M, 6.3, CW, 0.5);
  text(s, [{ text: 'Pattern:  ', options: { bold: true, color: C.navy } }, { text: 'the cheap search consistently favoured segmentation-heavy weightings, which matters for the next slide.', options: { color: C.ink } }],
    { x: M + 0.3, y: 6.3, w: CW - 0.6, h: 0.5, fontSize: 13, valign: 'middle' });
}

// ---------------------------------------------------------------- 22. Full fidelity outcome
{
  const b = T('phase5b'), o = T('phase6_final'), wo = ABL('weights_only');
  const s = newSlide({ kicker: 'Results  ·  Phase 6', title: 'Does the Optimized Configuration Hold Up?', speaker: K, script: [
    `The most important question is whether the optimized configuration holds up after full training. We retrained Coati's winner for 15 epochs at full resolution. Kidney Dice improved, from ${f3(b.kidney_dice)} to ${f3(o.kidney_dice)}. But diagnosis AUC fell from ${f3(b.diagnosis_auc)} to ${f3(o.diagnosis_auc)}, and lesion Dice on lesion-bearing images fell from ${f3(b.lesion_dice_positive_only)} to ${f3(o.lesion_dice_positive_only)}. Keeping Coati's weights but removing enhancement shows the same lesion drop, so the weights are driving it, not the enhancement.`,
    `Our explanation is that the search judged candidates after only 3 epochs, when segmentation improves fastest, so it learned to starve the classification heads. We report this as it is, because it tells us precisely how to fix the search.`,
  ] });
  chartLabel(s, 'Full-fidelity test results (15 epochs, 512 x 512)', M, 2.05, 7.3);
  barChart(s, ['Diagnosis AUC', 'Kidney Dice', 'Lesion Dice\n(lesion images)', '1 - violation rate'], [
    { name: 'Equal weights (Phase 5b)', values: [b.diagnosis_auc, b.kidney_dice, b.lesion_dice_positive_only, 1 - b.consistency_violation_rate] },
    { name: 'COA-optimized (Phase 6)', values: [o.diagnosis_auc, o.kidney_dice, o.lesion_dice_positive_only, 1 - o.consistency_violation_rate] },
    { name: 'COA weights, no enhancement', values: [wo.diagnosis_auc, wo.kidney_dice, wo.lesion_dice_positive_only, 1 - wo.consistency_violation_rate] },
  ], { x: M, y: 2.4, w: 7.3, h: 4.35, chartColors: [C.gray, C.navy, C.teal], valAxisMinVal: 0, valAxisMaxVal: 1.05, valAxisLabelFormatCode: '0.0', dataLabelFontSize: 9, catAxisLabelFontSize: 12 });
  card(s, 8.25, 2.1, 4.48, 4.65);
  text(s, 'COA-optimized vs equal weights', { x: 8.55, y: 2.3, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: C.navy });
  [[`+${f3(o.kidney_dice - b.kidney_dice)}`, 'kidney Dice', C.green],
   [`-${f3(b.diagnosis_auc - o.diagnosis_auc)}`, 'diagnosis AUC', C.red],
   [`-${f3(b.lesion_dice_positive_only - o.lesion_dice_positive_only)}`, 'lesion Dice (lesion images)', C.red]].forEach(([v, l, col], i) => {
    text(s, [{ text: v + '  ', options: { fontFace: H, fontSize: 22, bold: true, color: col } }, { text: l, options: { fontSize: 13, color: C.ink } }],
      { x: 8.55, y: 2.8 + i * 0.55, w: 4.0, h: 0.5, valign: 'middle' });
  });
  text(s, [{ text: 'Likely cause.  ', options: { bold: true, color: C.navy } },
           { text: 'Candidates were scored after 3 epochs at 256 x 256, when segmentation gains come fastest. COA pushed the diagnosis weight to the 0.1 floor, a choice that does not survive full training.', options: { color: C.ink } }],
    { x: 8.55, y: 4.55, w: 3.95, h: 2.1, fontSize: 12.5 });
}

// ---------------------------------------------------------------- 23. Single vs multi-task
{
  const b = T('phase5b');
  const d = ABL('diagnosis_only'), k = ABL('kidney_seg_only'), l = ABL('lesion_seg_only');
  const s = newSlide({ kicker: 'Results  ·  Phase 7 ablations', title: 'Does Sharing One Encoder Help?', speaker: K, script: [
    `The single-task ablation asks whether sharing one encoder actually helps. Each single-task model was trained with exactly one loss term active, under identical conditions. For diagnosis the answer is clearly yes: trained alone it reaches ${f3(d.diagnosis_auc)} AUC, and in the multi-task model ${f3(b.diagnosis_auc)}.`,
    `For segmentation, not really. The kidney-only model reaches ${f3(k.kidney_dice)} Dice against ${f3(b.kidney_dice)}, and the lesion-only model ${f3(l.lesion_dice_positive_only)} against ${f3(b.lesion_dice_positive_only)}. So the multi-task design mainly strengthens classification, and one network replaces three at a small segmentation cost. These are single runs, so the small segmentation gaps still need to be confirmed with cross-validation.`,
  ] });
  chartLabel(s, 'Each task trained alone vs inside the multi-task model (test set)', M, 2.05, 7.3);
  barChart(s, ['Diagnosis AUC', 'Kidney Dice', 'Lesion Dice\n(lesion images)'], [
    { name: 'Single-task model', values: [d.diagnosis_auc, k.kidney_dice, l.lesion_dice_positive_only] },
    { name: 'Multi-task model (Phase 5b)', values: [b.diagnosis_auc, b.kidney_dice, b.lesion_dice_positive_only] },
  ], { x: M, y: 2.4, w: 7.3, h: 4.35, chartColors: [C.gray, C.navy], valAxisMinVal: 0, valAxisMaxVal: 1.05, valAxisLabelFormatCode: '0.0', catAxisLabelFontSize: 12.5 });
  const rows = [[`+${f3(b.diagnosis_auc - d.diagnosis_auc)}`, 'diagnosis AUC with sharing', C.green],
                [`-${f3(k.kidney_dice - b.kidney_dice)}`, 'kidney Dice with sharing', C.red],
                [`-${f3(l.lesion_dice_positive_only - b.lesion_dice_positive_only)}`, 'lesion Dice with sharing', C.red]];
  card(s, 8.25, 2.1, 4.48, 4.65);
  text(s, 'Effect of multi-task sharing', { x: 8.55, y: 2.3, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: C.navy });
  rows.forEach(([v, lab, col], i) => text(s, [{ text: v + '  ', options: { fontFace: H, fontSize: 22, bold: true, color: col } }, { text: lab, options: { fontSize: 13, color: C.ink } }],
    { x: 8.55, y: 2.8 + i * 0.55, w: 4.0, h: 0.5, valign: 'middle' }));
  text(s, bullets([
    'Sharing mainly strengthens classification',
    'Segmentation is marginally better trained alone',
    'One network replaces three single-task models',
    'Single seed: gaps of about 0.01 need Phase 8 confirmation',
  ], { size: 12.5, gap: 7 }), { x: 8.55, y: 4.55, w: 3.95, h: 2.1 });
}

// ---------------------------------------------------------------- 24. Per-category AUC
{
  const cats = ['1', '2', '3', '4', '5', '6', '7', '8', '9'];
  const tc = X.test_images_per_category;
  const la = ph => cats.map((c, i) => D[ph].local_aucs[String(i)]);
  const s = newSlide({ kicker: 'Results  ·  Local pathology', title: 'Per-Category Performance and Its Limits', speaker: K, script: [
    `Breaking local pathology down by category shows where the model is reliable, and where the numbers cannot be trusted. The two most common categories, 2 and 7, with ${tc['2']} and ${tc['7']} test images, reach AUCs of about 0.80 to 0.95. Categories 8 and 9 are the weakest, with category 9 near or below chance.`,
    `The counts under each category label matter. Categories 5 and 6 have only ${tc['5']} and ${tc['6']} test images, so their very high AUCs are not meaningful. For the paper, we will report per-category results with these counts shown, rather than relying on the mean alone.`,
  ] });
  chartLabel(s, 'Local pathology AUC by category (number of test images with that category in brackets)', M, 2.05, 8.5);
  barChart(s, cats.map(c => `${c} (${tc[c]})`), [
    { name: 'Phase 4', values: la('phase4') }, { name: 'Phase 5b', values: la('phase5b') },
  ], { x: M, y: 2.4, w: 8.5, h: 4.35, chartColors: [C.gray, C.navy], valAxisMinVal: 0.3, valAxisMaxVal: 1.0, valAxisLabelFormatCode: '0.0', dataLabelFormatCode: '0.00', dataLabelFontSize: 8.5, catAxisLabelFontSize: 11.5, barGapWidthPct: 45 });
  card(s, 9.4, 2.1, 3.33, 4.65);
  text(s, 'How to read this', { x: 9.65, y: 2.3, w: 2.9, h: 0.4, fontSize: 16, bold: true, color: C.navy });
  text(s, bullets([
    'AUC 0.5 is chance level',
    `Common categories 2 and 7 (${tc['2']} and ${tc['7']} test images) are strong`,
    'Categories 8 and 9 are weakest',
    `Categories 5 and 6 have ${tc['5']} and ${tc['6']} test images: their AUCs are not reliable`,
    'Report per-category counts alongside the mean',
  ], { size: 12.5, gap: 8 }), { x: 9.65, y: 2.8, w: 2.9, h: 3.85 });
}

// ---------------------------------------------------------------- 25. Findings
{
  const p4 = T('phase4'), p5 = T('phase5'), p5b = T('phase5b');
  const s = newSlide({ kicker: 'Findings', title: 'Five Findings From This Stage', speaker: K, script: [
    `Pulling this together, five findings. First, the anatomical constraint works, but only in its ratio form. Second, the standard lesion Dice was misleading, and we now report a version that is not. Third, hand-picked enhancement is essentially neutral once training budgets are matched.`,
    `Fourth, Coati wins the optimizer study at an equal budget, but its configuration does not beat the equal-weight model after full training. And fifth, multi-task sharing mainly benefits classification. The bottom line is that careful formulation and fair evaluation have mattered more so far than the choice of optimizer.`,
  ] });
  const f = [
    ['Constraint works only as a ratio', `Violation rate ${f3(p4.consistency_violation_rate)} to ${f3(p5b.consistency_violation_rate)} (-77%); the absolute form raised it to ${f3(p5.consistency_violation_rate)}.`],
    ['Lesion Dice needed correcting', `Empty model 0.319 vs ours 0.344 on all images; 0.000 vs 0.506 on lesion-bearing images.`],
    ['Hand-picked enhancement is neutral', 'At matched epochs: kidney Dice +0.003, diagnosis AUC -0.011.'],
    ['COA wins the search, not the final model', 'Search fitness 0.608 vs 0.601 (GWO) and 0.598 (WOA); after full training, AUC and lesion Dice fall.'],
    ['Sharing helps classification', 'Diagnosis AUC 0.844 alone vs 0.901 in the multi-task model; segmentation is marginally better alone.'],
  ];
  f.forEach(([h, b], i) => {
    const col = i < 3 ? 0 : 1, row = i < 3 ? i : i - 3;
    numbered(s, M + col * 6.23, 2.15 + row * 1.5, 5.9, String(i + 1).padStart(2, '0'), h, b, { headSize: 16, bodySize: 13, bodyH: 0.9 });
  });
  card(s, 6.83, 5.2, 5.9, 1.55, C.navy, C.navy);
  text(s, [{ text: 'BOTTOM LINE', options: { bold: true, color: C.ice, fontSize: 11, charSpacing: 2, breakLine: true } },
           { text: 'Careful formulation and fair evaluation have mattered more so far than the choice of optimizer.', options: { color: C.white, fontSize: 15, fontFace: H } }],
    { x: 7.13, y: 5.2, w: 5.35, h: 1.55, valign: 'middle' });
}

// ---------------------------------------------------------------- 26. Challenges
{
  const s = newSlide({ kicker: 'Reflection', title: 'Challenges and Lessons', speaker: K, script: [
    `A few lessons from getting here. Fair comparisons need matched budgets: epochs across phases, and model trainings, not iterations, across optimizers. Metrics and losses can quietly reward the wrong behaviour, as both the lesion Dice and consistency findings showed.`,
    `The cost of the search forces a trade-off with fidelity, and a cheap proxy can mis-rank configurations. And on a 4-gigabyte laptop GPU, engineering mattered: caching the enhanced images and loading data in parallel roughly halved our epoch time.`,
  ] });
  const items = [
    ['Fair comparisons need matched budgets', 'Phase 3 was first trained for 10 epochs against 15. Across optimizers, COA uses about 2.5x more evaluations per iteration, so budgets are capped on trainings.'],
    ['Metrics can reward the wrong behaviour', 'Empty images inflated lesion Dice, and the absolute consistency loss could be minimized without satisfying the constraint.'],
    ['Search cost vs fidelity', '180 trainings took 5.6 hours even at 256 x 256 and 3 epochs. Full fidelity would take about 70 hours, but the cheap proxy mis-judged the final model.'],
    ['Compute limits on a 4 GB GPU', 'Non-local-means costs 0.6 s per image, so enhanced images are cached. Parallel data loading cut epoch time from 77 s to 39 s.'],
  ];
  items.forEach(([h, b], i) => {
    const x = M + (i % 2) * 6.23, y = 2.1 + Math.floor(i / 2) * 2.35;
    card(s, x, y, 5.9, 2.15);
    badge(s, x + 0.3, y + 0.3, String(i + 1).padStart(2, '0'));
    text(s, h, { x: x + 0.95, y: y + 0.3, w: 4.7, h: 0.46, fontSize: 16, bold: true, valign: 'middle' });
    text(s, b, { x: x + 0.3, y: y + 0.95, w: 5.3, h: 1.1, fontSize: 13, color: C.slate });
  });
}

// ---------------------------------------------------------------- 27. Status
{
  const s = newSlide({ kicker: 'Status', title: 'Where the Project Stands', speaker: K, script: [
    `In terms of status, six of our nine phases are complete. Phase 7 is in progress, with four of its five ablation runs done. Phase 8, which is cross-validation and external validation, is fully scripted and ready to run. And Phase 9 is the write-up.`,
  ] });
  const ph = [
    ['1', 'Data pipeline', 'Ingestion, masks, lesion parsing, stratified split', 'COMPLETE', 1],
    ['2', 'Fixed-weight baseline', 'Diagnosis AUC 0.901, kidney Dice 0.899', 'COMPLETE', 1],
    ['3', 'Hand-picked enhancement', 'Rerun at matched 15 epochs', 'COMPLETE', 1],
    ['4', 'Five-head multi-task model', 'Lesion and local pathology heads added', 'COMPLETE', 1],
    ['5', 'Anatomical consistency', 'Absolute form diagnosed; ratio form validated', 'COMPLETE', 1],
    ['6', 'Metaheuristic optimization', 'COA, WOA, GWO; 180 trainings; full retrain', 'COMPLETE', 1],
    ['7', 'Ablations', '4 of 5 runs done; enhancement-only remaining', 'IN PROGRESS', 0],
    ['8', 'Cross-validation and external test', 'Five-fold scripts ready; proxy check ready', 'PLANNED', 0],
    ['9', 'Write-up', 'Paper draft from results document', 'PLANNED', 0],
  ];
  ph.forEach(([num, h, b, st, done], i) => {
    const x = M + (i % 3) * 4.115, y = 2.05 + Math.floor(i / 3) * 1.5;
    card(s, x, y, 3.9, 1.32, done ? C.tint : C.white);
    text(s, `PHASE ${num}`, { x: x + 0.25, y: y + 0.15, w: 1.5, h: 0.3, fontSize: 11, bold: true, color: C.teal, charSpacing: 2 });
    chip(s, x + 3.9 - 0.25 - (0.12 + st.length * 0.085), y + 0.14, st, done);
    text(s, h, { x: x + 0.25, y: y + 0.47, w: 3.45, h: 0.36, fontSize: 15, bold: true });
    text(s, b, { x: x + 0.25, y: y + 0.83, w: 3.45, h: 0.4, fontSize: 11.5, color: C.slate });
  });
  text(s, '6 of 9 phases complete  ·  Phase 7 in progress  ·  2 phases remaining', { x: M, y: 6.5, w: CW, h: 0.35, fontSize: 14, bold: true, color: C.navy, align: 'center' });
}

// ---------------------------------------------------------------- 28. Next steps
{
  const s = newSlide({ kicker: 'Looking ahead', title: 'Next Steps', speaker: K, script: [
    `Our next steps, in order. First, finish the last ablation, which isolates the contribution of enhancement on its own. Second, run five-fold cross-validation for both the optimized and equal-weight configurations; that gives the like-for-like comparison with URI-CADS and error bars on every number.`,
    `Third, retrain the Whale and Grey Wolf winners at full fidelity, to test directly whether our search ranks configurations correctly. Fourth, rerun the optimizer study with longer evaluations and a fitness that prevents any task from regressing. And finally, external validation and the paper. I'll hand back to Anubhav to conclude.`,
  ] });
  const steps = [
    ['Finish Phase 7', 'Run the last ablation, optimized enhancement with equal weights, to isolate the contribution of enhancement on its own.'],
    ['Phase 8: five-fold cross-validation', 'Evaluate optimized and equal-weight configurations under the URI-CADS protocol, for a direct comparison with 0.8741 AUC and error bars.'],
    ['Validate the search proxy', 'Retrain the WOA and GWO winners at full fidelity and check whether search fitness predicts final performance.'],
    ['Improve the optimizer study', 'Longer evaluations, a fitness that blocks regression on any task, and a narrower weight range so no head can be switched off.'],
    ['External validation and write-up', 'Test on a second cohort, then draft the paper from the accumulated results document (Phase 9).'],
  ];
  steps.forEach(([h, b], i) => numbered(s, M, 2.1 + i * 0.93, CW, String(i + 1).padStart(2, '0'), h, b, { headSize: 16, bodySize: 13, bodyH: 0.45 }));
}

// ---------------------------------------------------------------- 29. Conclusion
{
  const b = T('phase5b');
  const s = newSlide({ dark: true, name: 'Conclusion', speaker: A, script: [
    `To conclude. One lightweight network gives strong diagnosis, accurate kidney segmentation and real lesion localization. A correctly formulated anatomical constraint cuts out-of-kidney lesion predictions by 77 percent without hurting diagnosis. The optimizer study is complete and fair, and it showed us that the search setup matters as much as the algorithm. And the remaining work is already scripted and ready to run.`,
    `Thank you. We're happy to take your questions.`,
  ] });
  text(s, 'CONCLUSION', { x: 0.8, y: 0.8, w: 8, h: 0.35, fontSize: 13, bold: true, color: C.ice, charSpacing: 3 });
  text(s, 'Where We Stand', { x: 0.8, y: 1.2, w: 11.7, h: 0.9, fontFace: H, fontSize: 40, bold: true, color: C.white });
  text(s, bullets([
    `One lightweight network (24.4 M parameters) reaches diagnosis AUC ${f3(b.diagnosis_auc)}, kidney Dice ${f3(b.kidney_dice)} and lesion Dice ${f3(b.lesion_dice_positive_only)} on lesion-bearing images`,
    'A correctly formulated anatomical constraint cuts out-of-kidney lesion predictions by 77% without hurting diagnosis',
    'The optimizer study is complete and fair: COA leads, but the search setup matters as much as the algorithm',
    'Cross-validation and proxy validation are scripted and ready to run',
  ], { size: 18, color: C.white, gap: 14 }), { x: 0.8, y: 2.35, w: 11.4, h: 3.4 });
  text(s, 'Thank you. Questions are welcome.', { x: 0.8, y: 6.05, w: 11.7, h: 0.6, fontFace: H, fontSize: 24, bold: true, color: C.ice });
}

// ================================================================= Script document
function buildScriptDoc() {
  const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, BorderStyle, LevelFormat } = docx;
  const words = arr => arr.join(' ').split(/\s+/).length;
  const secs = arr => Math.round(words(arr) / 145 * 60 / 5) * 5;
  const fmt = sTot => `${Math.floor(sTot / 60)}:${String(sTot % 60).padStart(2, '0')}`;
  const totals = { [A]: 0, [K]: 0 };
  scriptLog.forEach(x => { x.secs = secs(x.script); totals[x.speaker] += x.secs; });
  const total = totals[A] + totals[K];

  const border = { style: BorderStyle.SINGLE, size: 4, color: 'DCE3E8' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const colW = [900, 5200, 2000, 1260];
  const cell = (t, w, { bold = false, fill, color = '1A2233' } = {}) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: t, bold, color, font: 'Calibri', size: 20 })] })],
  });
  const P = (t, o = {}, keepNext = false) => new Paragraph({ keepNext, keepLines: true, spacing: { after: 140, line: 300 }, children: [new TextRun(Object.assign({ text: t, font: 'Calibri', size: 22 }, o))] });

  const children = [];
  children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun('Presentation Script')] }));
  children.push(P('Optimization-Based Deep Learning Framework for Kidney Disease Detection and Segmentation from Ultrasound Images', { italics: true, color: '5B6B79' }));
  children.push(P(`Presenters: Anubhav Rawat (23BCE1471) and Kaustubh (23BCE1555)   ·   Guide: Sudharson S   ·   Review: 16 September 2026`));
  children.push(P(`Estimated speaking time: ${fmt(total)} in total (Anubhav ${fmt(totals[A])}, Kaustubh ${fmt(totals[K])}), at about 145 words per minute. The same text is in the speaker notes of each slide.`));

  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun('Running order')] }));
  const rows = [new TableRow({ tableHeader: true, children: ['Slide', 'Title', 'Speaker', 'Time'].map((t, i) => cell(t, colW[i], { bold: true, fill: '21295C', color: 'FFFFFF' })) })];
  scriptLog.forEach(x => rows.push(new TableRow({ children: [cell(String(x.n), colW[0]), cell(x.title, colW[1]), cell(x.speaker, colW[2]), cell(fmt(x.secs), colW[3])] })));
  children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: colW, rows }));

  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun('Delivery notes')] }));
  [
    'Whoever is speaking advances their own slides. The handovers are at slide 13 to 14 (Anubhav to Kaustubh) and slide 28 to 29 (Kaustubh to Anubhav).',
    'Slow down on the numbers that carry the argument: the 77% violation reduction, lesion Dice 0.000 vs 0.506, and the Phase 6 drop in diagnosis AUC.',
    'On results that did not work, state them plainly and move straight to what they tell us. Reviewers reward an honest diagnosis more than a weak result explained away.',
    'On chart slides, point at the chart before reading the numbers, so the audience knows where to look.',
    'If you are short of time, slides 13, 18 and 24 can be shortened to one sentence each without losing the argument.',
  ].forEach(t => children.push(new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 100 }, children: [new TextRun({ text: t, font: 'Calibri', size: 22 })] })));

  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun('Script')] }));
  let lastSpeaker = null;
  scriptLog.forEach(x => {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun(`Slide ${x.n}  ·  ${x.title}`)] }));
    children.push(new Paragraph({ spacing: { after: 120 }, keepNext: true, children: [
      new TextRun({ text: x.speaker.toUpperCase(), bold: true, color: x.speaker === A ? '21295C' : '1C7293', font: 'Calibri', size: 20 }),
      new TextRun({ text: `   ·   about ${fmt(x.secs)}`, color: '5B6B79', font: 'Calibri', size: 20 }),
      ...(lastSpeaker && lastSpeaker !== x.speaker ? [new TextRun({ text: '   ·   HANDOVER', bold: true, color: 'D9912B', font: 'Calibri', size: 20 })] : []),
    ] }));
    x.script.forEach(p => children.push(P(p, {}, true)));
    children.push(new Paragraph({ spacing: { after: 260 }, children: [new TextRun({ text: x.n < scriptLog.length ? '[Next slide]' : '[Open for questions]', italics: true, color: '5B6B79', font: 'Calibri', size: 20 })] }));
    lastSpeaker = x.speaker;
  });

  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun('Likely questions and suggested answers')] }));
  const qa = [
    [K, 'Your AUC is higher than URI-CADS. Is your model better?', 'Not yet demonstrated. Our 0.90 comes from one held-out split, while their 0.8741 comes from five-fold cross-validation, so the protocols differ. Phase 8 runs their protocol on our model, and only that comparison supports a claim either way.'],
    [K, 'Why did the optimized model perform worse after full training?', 'The search scored candidates after 3 epochs at 256 x 256. At that point segmentation improves fastest, so the search favoured weightings that starve the classification heads; COA drove the diagnosis weight to the 0.1 floor. The COA-weights-without-enhancement run shows the same lesion drop, which points to the weights. The fix is a higher-fidelity evaluation, a fitness that blocks regression on any task, and a narrower weight range.'],
    [A, 'Why not evaluate candidates at full fidelity?', 'One full-fidelity training takes about 23 minutes here, so 180 of them would take roughly 70 hours on our GPU. The cheap version took 5.6 hours. That trade-off is exactly what our proxy-validation experiment will measure.'],
    [A, 'Why the Coati algorithm specifically?', 'It is recent, it has not been applied to jointly tuning loss weights and preprocessing, and on our benchmark check it was strongest on the multimodal Rastrigin function. We compare it against two established algorithms, WOA and GWO, under an identical budget so the choice can be judged on evidence.'],
    [A, 'Is 60 trainings per optimizer enough for a 10-dimensional search?', 'It is a modest budget, and we say so. Notably, none of the three optimizers improved in its final eight trainings, so extra budget at this fidelity would likely add little. A higher-fidelity rerun matters more than a larger budget.'],
    [A, 'How do you get a lesion segmentation target when there are only bounding boxes?', 'We rescale every annotated box to 512 x 512 and take their union as a coarse mask. It is a deliberate approximation, since boxes include some non-lesion pixels, and it is what lets us train a lesion decoder at all where the benchmark has no pixel masks.'],
    [K, 'Why does the ratio constraint lower lesion Dice slightly?', 'Some annotated lesion boxes extend past the kidney outline. The constraint penalizes predictions there, so the model gives up a little overlap with those boxes. The drop is from 0.506 to 0.488, while out-of-kidney predictions fall by 77%.'],
    [K, 'Does the anatomical constraint need the kidney mask at inference?', 'No. The ground-truth kidney mask is used only inside the training loss. At inference the model takes only the image.'],
    [K, 'How do you know these differences are not random variation?', 'Each configuration is currently one seed on one split. Large effects, such as the 77% violation reduction or the 0.058 AUC gain from sharing, are well beyond plausible noise. Small gaps of around 0.01 are not, and Phase 8 cross-validation will provide the standard deviations to judge them.'],
    [A, 'What are the nine local pathology categories?', 'They follow the URI-CADS local taxonomy, which covers cysts, hydronephrosis, pyramids, lithiasis, angiomyolipoma, solid masses and cortical changes. We report them by their annotated number while we confirm the exact number-to-name mapping against the benchmark repository.'],
    [A, 'Is the model really edge-deployable?', 'The evidence so far is size: 24.4 million parameters, a single network, and 0.81 GB of GPU memory even during training. We have not yet measured inference latency on a low-power device, and that measurement is needed before the claim is complete.'],
    [A, 'Why ResNet-34 rather than a newer backbone?', 'It is lightweight, pretrained and well understood, and it sits below the ResNet-50 used by URI-CADS, which suits the lightweight comparison. A backbone swap is a reasonable future ablation.'],
  ];
  qa.forEach(([who, q, ans], i) => {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, keepNext: true, children: [new TextRun(`Q${i + 1}. ${q}`)] }));
    children.push(new Paragraph({ spacing: { after: 80 }, keepNext: true, children: [new TextRun({ text: `Suggested answerer: ${who}`, bold: true, color: '1C7293', font: 'Calibri', size: 20 })] }));
    children.push(P(ans));
  });

  const doc = new Document({
    creator: 'Anubhav Rawat, Kaustubh', title: 'Presentation Script',
    styles: {
      default: { document: { run: { font: 'Calibri', size: 22 } } },
      paragraphStyles: [
        { id: 'Title', name: 'Title', basedOn: 'Normal', next: 'Normal', run: { font: 'Cambria', size: 48, bold: true, color: '21295C' }, paragraph: { spacing: { after: 160 } } },
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: 'Cambria', size: 32, bold: true, color: '21295C' }, paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: 'Cambria', size: 26, bold: true, color: '21295C' }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
        { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: 'Calibri', size: 23, bold: true, color: '1A2233' }, paragraph: { spacing: { before: 200, after: 60 }, outlineLevel: 2 } },
      ],
    },
    numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
    sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children }],
  });
  return Packer.toBuffer(doc).then(buf => { fs.writeFileSync(SCRIPT, buf); return { total, totals }; });
}

fs.mkdirSync(OUT, { recursive: true });
pres.writeFile({ fileName: DECK }).then(() => buildScriptDoc()).then(({ total, totals }) => {
  console.log(`Wrote ${DECK} (${n} slides)`);
  console.log(`Wrote ${SCRIPT}`);
  console.log(`Speaking time: total ${Math.floor(total / 60)}m${total % 60}s | ${A} ${Math.floor(totals[A] / 60)}m${totals[A] % 60}s | ${K} ${Math.floor(totals[K] / 60)}m${totals[K] % 60}s`);
});
