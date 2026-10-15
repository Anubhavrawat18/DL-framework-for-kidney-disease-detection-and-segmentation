"""Result capture for the running results document.

Two things get captured per phase:
  * a terminal-style PNG of the phase's console output (the "screenshot of the output")
  * matplotlib figures (training curves, ROC, qualitative overlays, convergence plots)

Everything lands in results/ and is assembled into a Word document by scripts/build_report.py.
"""
import io
import json
import os
import sys
import textwrap
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, LOGS_DIR, RESULTS_DIR

TERMINAL_BG = '#0f1117'
TERMINAL_FG = '#d7dae0'
ACCENT = '#6ea8fe'


class Tee:
    """Mirrors stdout to an in-memory buffer so a phase's console output can be rendered."""

    def __init__(self):
        self.buffer = io.StringIO()
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self.buffer.write(data)

    def flush(self):
        self._stdout.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        return False

    @property
    def text(self):
        return self.buffer.getvalue()


def render_console_png(text, out_path, title='console', width_chars=104, max_lines=None):
    """Render captured console text as a terminal-styled PNG."""
    lines = []
    for raw in text.splitlines():
        if len(raw) <= width_chars:
            lines.append(raw)
        else:
            lines.extend(textwrap.wrap(raw, width_chars, subsequent_indent='    ') or [''])
    if max_lines and len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines // 2):]
        lines = head + [f'... [{len(lines) - max_lines} lines omitted] ...'] + tail

    line_h = 0.165
    fig_h = max(1.4, len(lines) * line_h + 0.75)
    fig_w = width_chars * 0.083 + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(TERMINAL_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(TERMINAL_BG)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((0, 1 - 0.42 / fig_h), 1, 0.42 / fig_h,
                               transform=ax.transAxes, color='#1b1f2a', zorder=1))
    for i, color in enumerate(('#ff5f57', '#febc2e', '#28c840')):
        ax.add_patch(plt.Circle((0.012 + i * 0.018, 1 - 0.21 / fig_h), 0.0055,
                                transform=ax.transAxes, color=color, zorder=2))
    ax.text(0.5, 1 - 0.21 / fig_h, title, transform=ax.transAxes, color='#8b93a7',
            fontsize=8, ha='center', va='center', family='monospace', zorder=2)

    y = 1 - (0.62 / fig_h)
    for line in lines:
        color = TERMINAL_FG
        stripped = line.strip()
        if stripped.startswith('=') or stripped.startswith('---'):
            color = '#4b5263'
        elif '->' in line or stripped.startswith('PHASE'):
            color = '#98c379'
        elif any(tok in line for tok in ('AUC', 'Dice', 'dice', 'fitness')):
            color = ACCENT
        elif stripped.startswith('WARNING') or stripped.startswith('Traceback'):
            color = '#e06c75'
        ax.text(0.012, y, line, transform=ax.transAxes, color=color, fontsize=7.4,
                family='monospace', va='top', ha='left')
        y -= line_h / fig_h

    fig.savefig(out_path, facecolor=TERMINAL_BG, bbox_inches=None)
    plt.close(fig)
    return out_path


def plot_training_curves(history, out_path, title):
    """Loss + the three headline validation metrics over epochs."""
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)

    axes[0].plot(epochs, [h['train_loss'] for h in history], marker='o', color='#e06c75')
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')

    axes[1].plot(epochs, [h['val_kidney_dice'] for h in history], marker='o',
                 label='kidney Dice', color='#61afef')
    if any('val_lesion_dice' in h for h in history):
        axes[1].plot(epochs, [h.get('val_lesion_dice', np.nan) for h in history], marker='s',
                     label='lesion Dice', color='#c678dd')
        axes[1].plot(epochs, [h.get('val_lesion_dice_positive_only', np.nan) for h in history],
                     marker='^', label='lesion Dice (positive only)', color='#e5c07b')
    axes[1].set_title('Validation Dice')
    axes[1].set_xlabel('epoch')
    axes[1].legend(fontsize=8)

    axes[2].plot(epochs, [h['val_diagnosis_auc'] for h in history], marker='o', color='#98c379')
    axes[2].set_title('Validation diagnosis AUC')
    axes[2].set_xlabel('epoch')
    axes[2].set_ylim(0.4, 1.0)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_roc(targets, probs, out_path, title):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(targets, probs)
    auc = roc_auc_score(targets, probs)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    ax.plot(fpr, tpr, color='#61afef', lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], '--', color='#999', lw=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_qualitative(model, dataset, device, out_path, n=4, title='Qualitative predictions',
                     indices=None):
    """Image | GT kidney | predicted kidney | GT lesion | predicted lesion."""
    import torch
    model.eval()
    if indices is None:
        indices = list(range(min(n, len(dataset))))
    rows = len(indices)
    fig, axes = plt.subplots(rows, 5, figsize=(16, 3.2 * rows), dpi=120)
    if rows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                out = model(image)
            if not isinstance(out, dict):
                out = {'kidney_seg': out[2], 'lesion_seg': None}

            display = (sample['image'].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            kidney_pred = (torch.sigmoid(out['kidney_seg'].float())[0] > 0.5).cpu().numpy()
            lesion_pred = (np.zeros_like(kidney_pred) if out['lesion_seg'] is None else
                           (torch.sigmoid(out['lesion_seg'].float())[0] > 0.5).cpu().numpy())

            panels = [(display, f'{sample["case_id"]} (input)', None),
                      (sample['kidney_mask'].numpy(), 'kidney GT', 'gray'),
                      (kidney_pred, 'kidney predicted', 'gray'),
                      (sample['lesion_mask'].numpy(), 'lesion GT (box union)', 'gray'),
                      (lesion_pred, 'lesion predicted', 'gray')]
            for c, (img, subtitle, cmap) in enumerate(panels):
                axes[r, c].imshow(img, cmap=cmap)
                axes[r, c].set_title(subtitle, fontsize=9)
                axes[r, c].axis('off')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_convergence(traces, out_path, title='Optimizer convergence'):
    """traces: {'COA': [trace entries], ...}. Plots best fitness vs iteration and vs evaluations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)
    colors = {'COA': '#e06c75', 'WOA': '#61afef', 'GWO': '#98c379'}
    for name, trace in traces.items():
        iters = [t['iteration'] for t in trace]
        best = np.minimum.accumulate([t['best_cost'] for t in trace])
        axes[0].plot(iters, -best, marker='o', label=name, color=colors.get(name))
        axes[1].plot([t['n_evals'] for t in trace], -best, marker='o', label=name,
                     color=colors.get(name))
    axes[0].set_xlabel('iteration')
    axes[1].set_xlabel('cumulative model trainings (fitness evaluations)')
    for ax in axes:
        ax.set_ylabel('best validation fitness')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_phase_result(phase, payload):
    """Persist a phase's structured results for the report builder."""
    path = os.path.join(RESULTS_DIR, f'{phase}_results.json')
    payload = dict(payload)
    payload['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def load_phase_result(phase):
    path = os.path.join(RESULTS_DIR, f'{phase}_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def figure_path(name):
    return os.path.join(FIGURES_DIR, name)
