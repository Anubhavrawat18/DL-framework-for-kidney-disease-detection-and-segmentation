"""Emits a standalone TikZ figure for each segmentation decoder.

The kidney and lesion decoders are structurally identical -- both come from the
same _decoder(512) factory and hold 1,572,833 parameters. What differs is the
supervision they receive and what consumes their output, so the two figures
share a body and differ in the right-hand panel. Those differences are read
from the source:

  kidney decoder
      training  : dice_loss(kidney_seg, kidney_mask_GT)
      evaluation: sigmoid -> threshold 0.5 -> Dice against the traced mask
      note      : the kidney PREDICTION feeds nothing else; the consistency
                  term uses the kidney GROUND TRUTH, not this output

  lesion decoder
      training  : dice_loss(lesion_seg, lesion_mask_GT)  (box-derived target)
                  + anatomical consistency term against the kidney GROUND TRUTH
      evaluation: sigmoid -> threshold 0.5 -> Dice, reported dataset-wide and
                  restricted to lesion-bearing images; plus the violation rate
      extra     : sigmoid -> DETACH -> pool -> normalise -> attention map that
                  drives lesion-guided pooling

Writes four files into paper/:
    fig_kidney_decoder_tikz.tex        figure block for the manuscript
    fig_lesion_decoder_tikz.tex
    test_fig_kidney_decoder.tex        self-contained, compiles on its own
    test_fig_lesion_decoder.tex
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import PROJECT_ROOT
from src.model import _decoder

PAPER = os.path.join(PROJECT_ROOT, 'paper')

PREAMBLE = r"""%% ===========================================================================
%% STANDALONE TEST FILE -- compile on its own:   pdflatex %(out)s
%%
%% Page geometry gives a ~12 cm text width, matching the Springer Nature single
%% column, so \resizebox{\textwidth} scales the figure by the same factor it
%% will in the manuscript.
%%
%% To move this into a document, copy only the \begin{figure}...\end{figure}
%% block and add to the preamble:
%%     \usepackage{tikz}
%%     \usetikzlibrary{calc,arrows.meta,decorations.pathreplacing}
%% ===========================================================================
\documentclass[11pt]{article}
\usepackage[a4paper,left=4.5cm,right=4.5cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc,arrows.meta,decorations.pathreplacing}
\begin{document}
"""

FOOTER = '\n\\end{document}\n'


def measure():
    """Shapes and parameter counts straight from the instantiated module."""
    d = _decoder(512)
    h = torch.zeros(1, 512, 16, 16)
    rows = []
    for blk in d:
        h = blk(h)
        rows.append((tuple(h.shape[1:]), sum(p.numel() for p in blk.parameters())))
    total = sum(p.numel() for p in d.parameters())
    return rows, total


STYLE = r"""  font=\sffamily\footnotesize,
  blk/.style   = {draw, rounded corners=2pt, minimum width=6.0cm,
                  minimum height=1.05cm, align=center, inner sep=2pt},
  io/.style    = {blk, fill=orange!14, draw=orange!75!black},
  up/.style    = {blk, fill=%(tint)s!12, draw=%(tint)s!65!black},
  fin/.style   = {blk, fill=violet!13, draw=violet!65!black},
  post/.style  = {blk, fill=black!6, draw=black!55},
  cons/.style  = {draw, rounded corners=2pt, minimum width=4.4cm,
                  minimum height=1.25cm, align=center, inner sep=3pt,
                  font=\sffamily\scriptsize},
  loss/.style  = {cons, fill=red!8, draw=red!55!black},
  metric/.style= {cons, fill=blue!9, draw=blue!55},
  attn/.style  = {cons, fill=green!12, draw=green!45!black},
  gt/.style    = {cons, fill=black!6, draw=black!55, minimum height=1.0cm},
  shape/.style = {font=\sffamily\scriptsize, text=black!70, anchor=west},
  hdr/.style   = {font=\sffamily\small\bfseries, text=black!80},
  note/.style  = {font=\sffamily\scriptsize, text=black!60, align=left},
  ar/.style    = {-{Stealth[length=2.4mm]}, semithick, draw=black!65},
  busar/.style = {-{Stealth[length=2.4mm]}, semithick, draw=black!60},
  gtar/.style  = {-{Stealth[length=2.2mm]}, semithick, draw=black!55, dashed},
  sk/.style    = {-{Stealth[length=2.4mm]}, semithick, draw=green!40!black}"""

ROW_Y = [0.00, -1.45, -2.90, -4.35, -5.80, -7.25, -8.70, -10.15, -11.60]


def body(name, tint, rows, total, consumers, notes, caption, label):
    """Assemble one figure. `consumers` is a list of (style, y, text, arrowstyle)."""
    shapes, params = zip(*rows)
    fmt = lambda n: '{:,}'.format(n).replace(',', '{,}')
    L = []
    A = L.append
    A(r'\begin{figure}[t]')
    A(r'\centering')
    A(r'\resizebox{\textwidth}{!}{%')
    A(r'\begin{tikzpicture}[')
    A(STYLE % {'tint': tint})
    A(r']')
    A('')
    A(r'%% ---------------- main column: the decoder itself')
    A(r'\node[hdr]   at (3.2, 1.15) {%s};' % name)
    A(r'\node[shape] at (6.4, 1.15) {\bfseries output shape};')
    A('')
    A(r'\node[io]  (d0) at (3.2, %6.2f) {Shared encoder feature map};' % ROW_Y[0])
    chans = [512, 256, 128, 64, 32, 16]
    for i in range(5):
        A(r'\node[up]  (d%d) at (3.2, %6.2f) {Up-block %d \ \ $%d \rightarrow %d$'
          r'\\[-1pt]{\scriptsize %s parameters}};'
          % (i + 1, ROW_Y[i + 1], i + 1, chans[i], chans[i + 1], fmt(params[i])))
    A(r'\node[fin] (d6) at (3.2, %6.2f) {Conv $1{\times}1$, \ $16 \rightarrow 1$'
      r'\\[-1pt]{\scriptsize %s parameters \ $\cdot$ \ no normalisation, '
      r'no activation}};' % (ROW_Y[6], fmt(params[5])))
    A(r'\node[io]  (d7) at (3.2, %6.2f) {Dense logit map \ \ {\scriptsize(raw '
      r'logits, squeezed to one channel)}};' % ROW_Y[7])
    A(r'\node[post](d8) at (3.2, %6.2f) {$\sigma(\cdot)$, threshold $0.5$ '
      r'$\Rightarrow$ predicted binary mask};' % ROW_Y[8])
    A('')
    for i in range(8):
        A(r'\draw[ar] (d%d) -- (d%d);' % (i, i + 1))
    A('')
    for i, s in enumerate(list(shapes[:1]) + list(shapes)):
        pass
    disp = [(512, 16, 16)] + list(shapes) + [(512, 512)]
    for i, sh in enumerate(disp):
        txt = ' \\times '.join(str(v) for v in sh)
        A(r'\node[shape] at (6.4, %6.2f) {$%s$};' % (ROW_Y[i], txt))
    A('')
    A(r'\draw[decorate, decoration={brace, amplitude=5pt, mirror}, black!55]')
    A(r'      (0.00, -0.925) -- (0.00, -7.775);')
    A(r'\node[font=\sffamily\scriptsize, text=black!65, rotate=90]')
    A(r'      at (-0.40, -4.35) {five $2\times$ upsamplings, total $\times 32$};')
    A('')
    A(r'\node[note, anchor=north west, text width=6.2cm] at (0.20, -12.35)')
    A(r'      {Total %s parameters. The first up-block alone holds 75\%% of them, '
      r'being the only one that operates on 512 input channels.};' % fmt(total))
    A('')
    A(r'%% ---------------- right panel: supervision and consumers')
    A(r'\node[hdr] at (11.8, 1.15) {Supervision and consumers};')
    for style, y, text, arrow in consumers:
        A(r'\node[%s] (%s) at (11.8, %6.2f) {%s};'
          % (style[0], style[1], y, text))
    A('')
    A(r'%% bus from the logit map up and down to each consumer')
    for style, y, text, arrow in consumers:
        A(r'\draw[%s] (6.20, %6.2f) -- (8.90, %6.2f) -- (8.90, %6.2f) -- (%s.west);'
          % (arrow, ROW_Y[7], ROW_Y[7], y, style[1]))
    A('')
    for n in notes:
        A(n)
    A('')
    A(r'\end{tikzpicture}}')
    A(r'\caption{%s}\label{%s}' % (caption, label))
    A(r'\end{figure}')
    return '\n'.join(L) + '\n'


def main():
    rows, total = measure()
    os.makedirs(PAPER, exist_ok=True)

    # ------------------------------------------------------- kidney decoder
    kidney_consumers = [
        (('loss', 'kloss'), -5.60,
         r'\textbf{Soft Dice loss} (training)\\[-1pt]against the traced kidney mask',
         'busar'),
        (('metric', 'kmet'), -8.00,
         r'\textbf{Kidney Dice} (reporting)\\[-1pt]thresholded at $0.5$', 'busar'),
        (('gt', 'kgt'), -10.80,
         r'Kidney mask, ground truth\\[-1pt]{\itshape hand-traced, pixel level}',
         'gtar'),
    ]
    kidney_notes = [
        r'\node[note, anchor=north west, text width=4.6cm] at (9.60, -12.35)',
        r'      {This output feeds no other module. The anatomical consistency '
        r'term constrains the \emph{lesion} decoder and refers to the kidney '
        r'\emph{ground truth}, not to this prediction.};',
    ]
    kidney_caption = (
        r'The kidney segmentation decoder. Five up-blocks restore the '
        r'$512{\times}16{\times}16$ encoder feature map to full resolution, '
        r'halving channels and doubling each spatial dimension at every stage, '
        r'after which a $1{\times}1$ convolution produces a single logit '
        r'channel. The logits are supervised during training by a soft Dice '
        r'loss computed on sigmoid probabilities against the hand-traced kidney '
        r'mask, and are thresholded at $0.5$ only at evaluation time. This '
        r'branch is a terminal output: no other part of the network consumes '
        r'its prediction.')
    k_fig = body('(a) Kidney decoder', 'teal', rows, total, kidney_consumers,
                 kidney_notes, kidney_caption, 'fig:kidney-decoder')

    # ------------------------------------------------------- lesion decoder
    lesion_consumers = [
        (('loss', 'lloss'), -3.20,
         r'\textbf{Soft Dice loss} (training)\\[-1pt]against the rasterised '
         r'lesion boxes', 'busar'),
        (('loss', 'lcons'), -5.60,
         r'\textbf{Anatomical consistency}\\[-1pt]lesion mass outside the kidney '
         r'ground truth', 'busar'),
        (('attn', 'latt'), -8.00,
         r'\textbf{Lesion-guided pooling}\\[-1pt]$\sigma(\cdot)$, '
         r'\textsc{detach}, pool, normalise', 'sk'),
        (('metric', 'lmet'), -10.40,
         r'\textbf{Lesion Dice and violation rate}\\[-1pt]reported overall and on '
         r'lesion-bearing images', 'busar'),
    ]
    lesion_notes = [
        r'\node[note, anchor=north west, text width=4.6cm] at (9.60, -12.35)',
        r'      {The green path is detached, so the pathology classification '
        r'loss can read this map but cannot reshape it. The Dice target is a '
        r'union of rasterised bounding boxes, not a traced contour, so reported '
        r'lesion Dice understates true localisation.};',
    ]
    lesion_caption = (
        r'The lesion segmentation decoder. Structurally this head is identical '
        r'to the kidney decoder and holds the same 1,572,833 parameters; what '
        r'differs is its supervision and its downstream use. Its logits are '
        r'supervised by a soft Dice loss against a union of rasterised lesion '
        r'bounding boxes, and additionally by the anatomical consistency term, '
        r'which penalises predicted lesion probability mass falling outside the '
        r'ground-truth kidney region. Beyond the loss, the same logits are '
        r'passed through a sigmoid and \emph{detached} to form the spatial '
        r'attention map used by lesion-guided pooling, so the multi-label '
        r'pathology classifier reads localised evidence without being able to '
        r'alter the segmentation it reads.')
    l_fig = body('(b) Lesion decoder', 'blue', rows, total, lesion_consumers,
                 lesion_notes, lesion_caption, 'fig:lesion-decoder')

    written = []
    for fig, stem in ((k_fig, 'kidney_decoder'), (l_fig, 'lesion_decoder')):
        fp = os.path.join(PAPER, 'fig_%s_tikz.tex' % stem)
        with open(fp, 'w', encoding='utf-8') as fh:
            fh.write(fig)
        written.append(fp)
        tp = os.path.join(PAPER, 'test_fig_%s.tex' % stem)
        standalone = fig.replace(r'\begin{figure}[t]', r'\begin{figure}[htbp]', 1)
        with open(tp, 'w', encoding='utf-8') as fh:
            fh.write(PREAMBLE % {'out': os.path.basename(tp)} + standalone + FOOTER)
        written.append(tp)

    for w in written:
        print('wrote', os.path.relpath(w, PROJECT_ROOT))
    print()
    print('measured from the module: %s parameters per decoder' % format(total, ','))
    for i, (sh, p) in enumerate(rows):
        print('  block %d -> %-18s %s params' % (i + 1, sh, format(p, ',')))


if __name__ == '__main__':
    main()
