"""Figures for the result-analysis document.

These are interpretive rather than diagnostic: each one exists to make a single argument that
the analysis document needs to defend.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, RESULTS_DIR

NAVY, TEAL, AMBER, RED, GREEN, GRAY, LINE = (
    '#21295C', '#1C7293', '#D9912B', '#B5483B', '#2E7D5B', '#A7B1BB', '#DCE3E8')
PUB = {'diagnosis_auc': 0.8741, 'local_pathology_mean_auc': 0.8190}


def load(name):
    with open(os.path.join(RESULTS_DIR, f'{name}_results.json'), encoding='utf-8') as fh:
        return json.load(fh)


def folds(payload, key):
    src = payload['summary'] if key in payload['summary'] else payload['tuned_summary']
    return np.array(src[key]['folds'], dtype=float)


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(LINE)
    ax.spines['bottom'].set_color(LINE)
    ax.tick_params(colors='#5B6B79', labelsize=10)
    ax.grid(axis='y', color=LINE, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def fig_cv_comparison():
    """The headline: cross-validated performance of each configuration vs the benchmark."""
    cfgs = [('cv_base', 'V2 baseline'), ('cv_balanced', '+ balanced BCE'),
            ('cv_v3', 'V3 pooling'), ('cv_v3_ens3', 'V3 + ensemble')]
    data = {k: load(f'phase8_kfold_{k}') for k, _ in cfgs}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    for ax, key, title in (
            (axes[0], 'diagnosis_auc', 'Healthy vs pathological AUC'),
            (axes[1], 'local_pathology_mean_auc', 'Multi-pathology AUC')):
        means = [folds(data[k], key).mean() for k, _ in cfgs]
        stds = [folds(data[k], key).std() for k, _ in cfgs]
        colors = [GRAY, GRAY, TEAL, NAVY]
        x = np.arange(len(cfgs))
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.6,
               error_kw={'ecolor': '#5B6B79', 'elinewidth': 1.2})
        ax.axhline(PUB[key], color=RED, linestyle='--', linewidth=1.6,
                   label=f'URI-CADS published ({PUB[key]:.4f})')
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.006, f'{m:.4f}', ha='center', fontsize=9.5, color='#1A2233')
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in cfgs], fontsize=10)
        ax.set_ylim(0.65, 0.96)
        ax.set_title(title, fontsize=12, color='#1A2233', pad=10)
        ax.legend(fontsize=9, frameon=False, loc='lower right')
        style(ax)
    axes[0].set_ylabel('AUC (mean ± SD over 5 folds)', fontsize=10, color='#5B6B79')
    fig.suptitle('Cross-validated performance under the URI-CADS protocol', fontsize=13.5)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'analysis_cv_comparison.png')
    fig.savefig(path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return path


def fig_noise_floor():
    """Why single-split comparisons were abandoned: effect sizes against the noise floor."""
    seed = load('seed_variance_ensemble')
    spread = seed['seed_auc_spread']
    std = seed['seed_auc_std']
    effects = [
        ('Phase 5b vs Phase 2\n(single split)', 0.0001, GRAY),
        ('Hand-picked enhancement\n(single split)', -0.0110, GRAY),
        ('COA optimization\n(single split)', -0.0261, GRAY),
        ('Geometric augmentation\n(single split)', -0.0325, GRAY),
        ('Intensity augmentation\n(single split)', -0.0651, AMBER),
        ('Final system vs baseline\n(5-fold, paired)', 0.0350, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    y = np.arange(len(effects))
    ax.axvspan(-spread / 2, spread / 2, color=RED, alpha=0.07, zorder=0,
               label=f'seed-only spread ({spread:.3f} wide)')
    ax.axvspan(-std, std, color=RED, alpha=0.13, zorder=0,
               label=f'±1 SD of seed-only variation (±{std:.3f})')
    ax.barh(y, [e[1] for e in effects], color=[e[2] for e in effects], height=0.55,
            zorder=3)
    ax.axvline(0, color='#5B6B79', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in effects], fontsize=9.5)
    ax.set_xlabel('Change in diagnosis AUC', fontsize=10, color='#5B6B79')
    ax.set_title('Reported effects against the measurement noise floor', fontsize=13, pad=12)
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=LINE, linewidth=0.7)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'analysis_noise_floor.png')
    fig.savefig(path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return path


def fig_paired_differences():
    """Per-fold paired differences: consistency of direction is the real evidence."""
    base, v3, ens = (load('phase8_kfold_cv_base'), load('phase8_kfold_cv_v3'),
                     load('phase8_kfold_cv_v3_ens3'))
    pairs = [('V3 pooling vs baseline\n(multi-pathology AUC)',
              folds(v3, 'local_pathology_mean_auc') - folds(base, 'local_pathology_mean_auc'),
              TEAL),
             ('Final system vs baseline\n(diagnosis AUC)',
              folds(ens, 'diagnosis_auc') - folds(base, 'diagnosis_auc'), NAVY)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    for ax, (title, diffs, color) in zip(axes, pairs):
        x = np.arange(1, 6)
        ax.bar(x, diffs, color=color, width=0.55)
        ax.axhline(0, color='#5B6B79', linewidth=1)
        ax.axhline(diffs.mean(), color=RED, linestyle='--', linewidth=1.4,
                   label=f'mean {diffs.mean():+.4f}')
        for xi, d in zip(x, diffs):
            ax.text(xi, d + (0.002 if d >= 0 else -0.004), f'{d:+.3f}',
                    ha='center', fontsize=9, color='#1A2233')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in x], fontsize=10)
        ax.set_title(title, fontsize=11.5, pad=10)
        ax.legend(fontsize=9, frameon=False)
        style(ax)
    axes[0].set_ylabel('Paired difference', fontsize=10, color='#5B6B79')
    fig.suptitle('Per-fold paired differences on identical folds', fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'analysis_paired_differences.png')
    fig.savefig(path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return path


def fig_operating_point():
    """Threshold choice moves specificity far more than any training intervention did."""
    ens = load('phase8_kfold_cv_v3_ens3')
    default = {k: folds(ens, f'diagnosis_{k}').mean()
               for k in ('sensitivity', 'specificity', 'balanced_accuracy')}
    tuned = {k: folds(ens, k).mean() for k in ('sensitivity', 'specificity',
                                               'balanced_accuracy')}
    labels = ['Sensitivity', 'Specificity', 'Balanced accuracy']
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    ax.bar(x - 0.2, [default['sensitivity'], default['specificity'],
                     default['balanced_accuracy']], width=0.38, color=GRAY,
           label='Default threshold 0.5')
    ax.bar(x + 0.2, [tuned['sensitivity'], tuned['specificity'],
                     tuned['balanced_accuracy']], width=0.38, color=NAVY,
           label='Threshold tuned on validation')
    for xi, (a, b) in enumerate(zip(
            [default['sensitivity'], default['specificity'], default['balanced_accuracy']],
            [tuned['sensitivity'], tuned['specificity'], tuned['balanced_accuracy']])):
        ax.text(xi - 0.2, a + 0.012, f'{a:.3f}', ha='center', fontsize=9.5)
        ax.text(xi + 0.2, b + 0.012, f'{b:.3f}', ha='center', fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.set_title('Effect of the decision threshold (final system, 5-fold means)',
                 fontsize=12.5, pad=26)
    ax.legend(fontsize=9.5, frameon=False, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.10))
    style(ax)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'analysis_operating_point.png')
    fig.savefig(path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return path


if __name__ == '__main__':
    for fn in (fig_cv_comparison, fig_noise_floor, fig_paired_differences,
               fig_operating_point):
        print('wrote', os.path.basename(fn()))
