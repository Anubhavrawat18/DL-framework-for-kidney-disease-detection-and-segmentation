"""Phase 6: jointly optimize the six loss weights and the four enhancement parameters.

This is the project's core novelty - URI-CADS uses a plain unweighted loss sum and hand-picked
preprocessing; here both are searched by a metaheuristic. COA is the primary optimizer, with
WOA and GWO as benchmarks under an identical evaluation budget.

Each fitness evaluation trains a model from scratch at reduced fidelity (see src/phase6.py for
why, and for the fitness function's definition). Every evaluation is cached to disk, so an
interrupted run resumes where it stopped rather than repeating GPU work.

Usage:  python scripts/03_phase6_search.py --pop 6 --iters 6
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import FIGURES_DIR, LOGS_DIR
from src.data import load_manifest
from src.engine import get_device
from src.metaheuristics import OPTIMIZERS
from src.phase6 import (FITNESS_WEIGHTS, LOWER_BOUNDS, PARAM_NAMES, UPPER_BOUNDS,
                        Phase6Evaluator, decode)
from src.reporting import (Tee, plot_convergence, render_console_png, save_phase_result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=6)
    ap.add_argument('--iters', type=int, default=60,
                    help='iteration cap; the evaluation budget usually binds first')
    ap.add_argument('--budget', type=int, default=60,
                    help='model trainings per optimizer - the budget the comparison is made on')
    ap.add_argument('--epochs', type=int, default=3, help='epochs per fitness evaluation')
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--optimizers', default='COA,WOA,GWO')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--run-name', default='phase6',
                    help='namespaces the evaluation cache; use a different one for smoke tests')
    ap.add_argument('--save-as', default='phase6_search')
    ap.add_argument('--profile', default='balanced',
                    choices=('balanced', 'classification'))
    ap.add_argument('--guard', action='store_true',
                    help='penalize candidates that regress any task below its baseline')
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--scheduler', default='none', choices=('none', 'cosine'))
    ap.add_argument('--diagnosis-loss', default='bce',
                    choices=('bce', 'balanced', 'focal'))
    args = ap.parse_args()

    diag_cfg = {'bce': None,
                'balanced': {'type': 'weighted_bce', 'pos_weight': 'balanced'},
                'focal': {'type': 'focal', 'alpha': 0.25, 'gamma': 2.0}}[args.diagnosis_loss]

    device = get_device()
    manifest = load_manifest()

    print('=' * 78)
    print('PHASE 6 - METAHEURISTIC JOINT OPTIMIZATION')
    print('=' * 78)
    print(f'Search space ({len(PARAM_NAMES)} dimensions):')
    for name, lo, hi in zip(PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS):
        print(f'  {name:24s} [{lo:6.2f}, {hi:6.2f}]')
    from src.phase6 import FITNESS_PROFILES
    active_weights = FITNESS_PROFILES[args.profile]
    print(f'\nFitness profile {args.profile!r} (maximized, validation split):')
    for k, w in active_weights.items():
        print(f'  {w:.3f} * {k}')
    if args.guard:
        print('  regression guard: candidates dropping a task below baseline are penalized')
    print(f'Training per candidate: augment={args.augment} scheduler={args.scheduler} '
          f'diagnosis_loss={args.diagnosis_loss}')
    print(f'\nEvaluation fidelity: {args.img_size}x{args.img_size}, {args.epochs} epochs, '
          f'batch {args.batch_size}')
    print(f'Budget per optimizer: {args.budget} model trainings (population {args.pop}).')
    print('  The three algorithms consume evaluations per iteration at different rates - COA')
    print('  spends roughly 2.5x what WOA or GWO do - so the budget is capped on evaluations')
    print('  rather than iterations, which is what makes the comparison fair.')
    print(f'Optimizers: {args.optimizers}\n', flush=True)

    evaluator = Phase6Evaluator(
        manifest, device, img_size=(args.img_size, args.img_size), epochs=args.epochs,
        batch_size=args.batch_size, run_name=args.run_name, seed=args.seed,
        fitness_profile=args.profile, guard=args.guard, augment=args.augment,
        scheduler=args.scheduler, diagnosis_loss_cfg=diag_cfg)

    traces, results = {}, {}
    for opt_name in args.optimizers.split(','):
        opt_name = opt_name.strip()
        print('=' * 78)
        print(f'{opt_name}')
        print('=' * 78, flush=True)
        evaluator.current_optimizer = opt_name

        def callback(name, entry):
            print(f'  [{name}] iter {entry["iteration"]:2d} | '
                  f'best fitness={-entry["best_cost"]:.4f} | '
                  f'mean={-entry["mean_cost"]:.4f} | evals={entry["n_evals"]}', flush=True)

        t0 = time.time()
        optimizer = OPTIMIZERS[opt_name](
            evaluator, LOWER_BOUNDS, UPPER_BOUNDS, pop_size=args.pop,
            max_iter=args.iters, seed=args.seed, callback=callback,
            max_evals=args.budget)
        best_x, best_cost = optimizer.optimize()
        elapsed = time.time() - t0

        loss_weights, enhancement = decode(best_x)
        traces[opt_name] = optimizer.trace
        results[opt_name] = {
            'best_vector': np.asarray(best_x).tolist(),
            'best_fitness': -best_cost,
            'loss_weights': loss_weights,
            'enhancement': enhancement,
            'n_evaluations': optimizer.n_evals,
            'wall_clock_hours': round(elapsed / 3600, 3),
            'trace': optimizer.trace,
            'eval_history': optimizer.eval_history,
        }

        print(f'\n{opt_name} finished in {elapsed/3600:.2f} h over {optimizer.n_evals} '
              f'model trainings')
        print(f'  best fitness   : {-best_cost:.4f}')
        print('  loss weights   :')
        for k, v in loss_weights.items():
            print(f'      {k:20s} {v:.3f}')
        print('  enhancement    :')
        for k, v in enhancement.items():
            print(f'      {k:20s} {v}')
        print(flush=True)

        with open(os.path.join(LOGS_DIR, 'phase6_search_partial.json'), 'w') as fh:
            json.dump(results, fh, indent=2)

    print('=' * 78)
    print('OPTIMIZER COMPARISON')
    print('=' * 78)
    print(f'{"optimizer":10s} {"best fitness":>14s} {"evals":>8s} {"hours":>8s}')
    print('-' * 44)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]['best_fitness']):
        print(f'{name:10s} {r["best_fitness"]:14.4f} {r["n_evaluations"]:8d} '
              f'{r["wall_clock_hours"]:8.2f}')
    winner = max(results.items(), key=lambda kv: kv[1]['best_fitness'])
    print(f'\nBest optimizer: {winner[0]} (fitness {winner[1]["best_fitness"]:.4f})')

    plot_convergence(traces, os.path.join(FIGURES_DIR, 'phase6_convergence.png'),
                     'Phase 6 - COA vs WOA vs GWO convergence')
    save_phase_result(args.save_as, {
        'search_space': {n: [float(lo), float(hi)]
                         for n, lo, hi in zip(PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS)},
        'fitness_weights': active_weights,
        'fitness_profile': args.profile,
        'guard': args.guard,
        'fidelity': {'img_size': args.img_size, 'epochs': args.epochs,
                     'batch_size': args.batch_size},
        'budget': {'pop_size': args.pop, 'iterations': args.iters},
        'results': results,
        'winner': winner[0],
    })
    return results


if __name__ == '__main__':
    with Tee() as tee:
        main()
    render_console_png(tee.text, os.path.join(FIGURES_DIR, 'phase6_search_console.png'),
                       title='python scripts/03_phase6_search.py', max_lines=120)
