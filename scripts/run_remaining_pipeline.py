"""Run the remaining long-running stages back to back, rebuilding the report after each.

Designed to survive being left alone overnight. Each stage is a separate subprocess, so a
failure in one is reported and the run continues to the next rather than losing everything.

The Phase 6 search is resumable: every fitness evaluation is cached to disk by parameter
vector, and the optimizers are seeded, so restarting after an interruption replays the
already-computed candidates instantly from cache and then carries on from where it stopped.

Usage:  python scripts/run_remaining_pipeline.py
        python scripts/run_remaining_pipeline.py --skip phase6_search
"""
import argparse
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

STAGES = [
    ('phase6_search', ['scripts/03_phase6_search.py', '--pop', '6', '--budget', '60'],
     'Phase 6 - metaheuristic search (COA vs WOA vs GWO)'),
    ('phase6_final', ['scripts/04_phase6_final.py'],
     'Phase 6 - retrain winning configuration at full fidelity'),
    ('phase7', ['scripts/05_phase7_ablations.py'],
     'Phase 7 - ablation study'),
    ('phase8_optimized', ['scripts/06_phase8_kfold.py', '--config', 'optimized'],
     'Phase 8 - 5-fold cross-validation, optimized configuration'),
    ('phase8_baseline', ['scripts/06_phase8_kfold.py', '--config', 'baseline'],
     'Phase 8 - 5-fold cross-validation, baseline configuration'),
]


def run(stage_args, label):
    print('\n' + '#' * 78)
    print(f'# {label}')
    print(f'# started {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('#' * 78, flush=True)
    t0 = time.time()
    result = subprocess.run([PYTHON] + stage_args, cwd=PROJECT_ROOT)
    elapsed = (time.time() - t0) / 3600
    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'\n# {label}: {status} after {elapsed:.2f} h', flush=True)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip', default='', help='comma-separated stage names to skip')
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}

    summary = []
    for name, stage_args, label in STAGES:
        if name in skip:
            print(f'skipping {name}')
            continue
        ok = run(stage_args, label)
        summary.append((name, ok))
        # Rebuild the results document after every stage so an interrupted run still leaves a
        # document reflecting everything finished up to that point.
        subprocess.run([PYTHON, 'scripts/build_report.py'], cwd=PROJECT_ROOT)

    print('\n' + '=' * 78)
    print('PIPELINE SUMMARY')
    print('=' * 78)
    for name, ok in summary:
        print(f'  {name:22s} {"OK" if ok else "FAILED"}')


if __name__ == '__main__':
    main()
