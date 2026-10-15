"""Population-based optimizers for Phase 6: COA (primary), WOA and GWO (benchmarks).

All three are written as MINIMIZERS of `objective(x) -> float`. The Phase 6 driver passes
`-fitness`, so lower is better here throughout.

Every optimizer reports the same trace format so convergence curves are directly comparable:
    {'iteration', 'best_cost', 'mean_cost', 'best_position', 'n_evals'}

References:
  COA - Dehghani et al., "Coati Optimization Algorithm", Knowledge-Based Systems, 2023.
  WOA - Mirjalili & Lewis, "The Whale Optimization Algorithm", Adv. Eng. Software, 2016.
  GWO - Mirjalili et al., "Grey Wolf Optimizer", Adv. Eng. Software, 2014.
"""
import numpy as np


class BudgetExhausted(Exception):
    """Raised internally once the evaluation budget is spent."""


class _Base:
    """Common machinery. Note the three algorithms spend evaluations per iteration at very
    different rates - COA performs roughly 2.5x as many as WOA or GWO for the same population
    and iteration count - so comparing them at equal ITERATIONS would be misleading. `max_evals`
    caps every optimizer at the same number of objective evaluations, which for Phase 6 means
    the same number of model trainings, and is the budget the comparison should be made on.
    """

    def __init__(self, objective, lb, ub, pop_size=8, max_iter=10, seed=42, callback=None,
                 max_evals=None):
        self.objective = objective
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)
        self.callback = callback
        self.n_evals = 0
        self.trace = []
        self.eval_history = []      # best-so-far after every single evaluation
        self.best_x = None
        self.best_cost = float('inf')

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def _evaluate(self, x):
        if self.max_evals is not None and self.n_evals >= self.max_evals:
            raise BudgetExhausted
        x = self._clip(x)
        self.n_evals += 1
        cost = self.objective(x)
        if cost < self.best_cost:
            self.best_cost, self.best_x = cost, np.asarray(x).copy()
        self.eval_history.append({'n_evals': self.n_evals, 'cost': float(cost),
                                  'best_cost': float(self.best_cost)})
        return cost

    def optimize(self):
        """Runs until the iteration limit or the evaluation budget, whichever comes first."""
        try:
            self.run()
        except BudgetExhausted:
            pass
        return self.best_x, self.best_cost

    def _init_population(self):
        pop = self.lb + self.rng.random((self.pop_size, self.dim)) * (self.ub - self.lb)
        costs = np.array([self._evaluate(p) for p in pop])
        return pop, costs

    def _record(self, iteration, pop, costs):
        best_idx = int(np.argmin(costs))
        entry = {
            'iteration': iteration,
            'best_cost': float(costs[best_idx]),
            'mean_cost': float(np.mean(costs)),
            'best_position': self._clip(pop[best_idx]).tolist(),
            'n_evals': self.n_evals,
        }
        self.trace.append(entry)
        if self.callback:
            self.callback(self.name, entry)
        return entry

    def run(self):
        raise NotImplementedError


class CoatiOptimizationAlgorithm(_Base):
    """COA: phase 1 hunts an iguana (exploration), phase 2 escapes predators (exploitation)."""

    name = 'COA'

    def run(self):
        pop, costs = self._init_population()
        best_idx = int(np.argmin(costs))
        best_x, best_cost = pop[best_idx].copy(), costs[best_idx]
        self._record(0, pop, costs)

        half = self.pop_size // 2
        for t in range(1, self.max_iter + 1):
            # --- Phase 1: hunting and attacking strategy on the iguana ---
            for i in range(self.pop_size):
                if i < half:
                    # Coatis in the tree move toward the iguana (current best).
                    r = self.rng.random(self.dim)
                    I = self.rng.integers(1, 3)
                    cand = pop[i] + r * (best_x - I * pop[i])
                else:
                    # The iguana falls to the ground at a random position.
                    iguana_g = self.lb + self.rng.random(self.dim) * (self.ub - self.lb)
                    cost_g = self._evaluate(iguana_g)
                    r = self.rng.random(self.dim)
                    I = self.rng.integers(1, 3)
                    if cost_g < costs[i]:
                        cand = pop[i] + r * (iguana_g - I * pop[i])
                    else:
                        cand = pop[i] + r * (pop[i] - iguana_g)

                cand = self._clip(cand)
                cand_cost = self._evaluate(cand)
                if cand_cost < costs[i]:
                    pop[i], costs[i] = cand, cand_cost

            # --- Phase 2: escaping from predators (local search, radius shrinks with t) ---
            lb_local = self.lb / t
            ub_local = self.ub / t
            for i in range(self.pop_size):
                r = self.rng.random(self.dim)
                cand = pop[i] + (1 - 2 * r) * (lb_local + r * (ub_local - lb_local))
                cand = self._clip(cand)
                cand_cost = self._evaluate(cand)
                if cand_cost < costs[i]:
                    pop[i], costs[i] = cand, cand_cost

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_x, best_cost = pop[best_idx].copy(), costs[best_idx]
            self._record(t, pop, costs)

        return best_x, best_cost


class WhaleOptimizationAlgorithm(_Base):
    name = 'WOA'

    def run(self):
        pop, costs = self._init_population()
        best_idx = int(np.argmin(costs))
        best_x, best_cost = pop[best_idx].copy(), costs[best_idx]
        self._record(0, pop, costs)

        for t in range(1, self.max_iter + 1):
            a = 2 - 2 * t / self.max_iter           # shrinks 2 -> 0
            a2 = -1 + t * (-1 / self.max_iter)       # for the spiral parameter l

            for i in range(self.pop_size):
                r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = self.rng.random()
                l = (a2 - 1) * self.rng.random() + 1

                if p < 0.5:
                    if np.mean(np.abs(A)) < 1:
                        D = np.abs(C * best_x - pop[i])          # encircle the best whale
                        cand = best_x - A * D
                    else:
                        rand_idx = self.rng.integers(self.pop_size)  # explore toward a random one
                        D = np.abs(C * pop[rand_idx] - pop[i])
                        cand = pop[rand_idx] - A * D
                else:
                    D = np.abs(best_x - pop[i])                   # spiral bubble-net attack
                    cand = D * np.exp(l) * np.cos(2 * np.pi * l) + best_x

                cand = self._clip(cand)
                cand_cost = self._evaluate(cand)
                if cand_cost < costs[i]:
                    pop[i], costs[i] = cand, cand_cost

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_x, best_cost = pop[best_idx].copy(), costs[best_idx]
            self._record(t, pop, costs)

        return best_x, best_cost


class GreyWolfOptimizer(_Base):
    name = 'GWO'

    def run(self):
        pop, costs = self._init_population()
        self._record(0, pop, costs)

        for t in range(1, self.max_iter + 1):
            order = np.argsort(costs)
            alpha, beta, delta = pop[order[0]].copy(), pop[order[1]].copy(), pop[order[2]].copy()
            a = 2 - 2 * t / self.max_iter

            for i in range(self.pop_size):
                new = np.zeros(self.dim)
                for leader in (alpha, beta, delta):
                    r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = np.abs(C * leader - pop[i])
                    new += leader - A * D
                cand = self._clip(new / 3.0)
                cand_cost = self._evaluate(cand)
                if cand_cost < costs[i]:
                    pop[i], costs[i] = cand, cand_cost

            self._record(t, pop, costs)

        best_idx = int(np.argmin(costs))
        return pop[best_idx].copy(), float(costs[best_idx])


OPTIMIZERS = {
    'COA': CoatiOptimizationAlgorithm,
    'WOA': WhaleOptimizationAlgorithm,
    'GWO': GreyWolfOptimizer,
}
