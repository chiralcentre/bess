"""Perfect-hindsight optimum, as a linear program.

Given the whole price series up front, this program finds the schedule no other schedule can
beat. There are three unknowns per interval -- energy bought, energy sold, resulting state of charge.

Nothing here forbids buying and selling in the same interval; the degradation cost prices it. 
Formulation and reasoning are in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from bess.asset.battery import BatterySpec
from bess.strategies.base import Schedule


class InfeasibleScheduleError(RuntimeError):
    """The LP had no feasible solution, usually an unreachable terminal SoC."""


class ForesightLP:
    """Revenue-maximising schedule with full knowledge of future prices."""

    name = "ForesightLP"

    def __init__(
        self,
        terminal_soc_mwh: float | None = None,
        tradable: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            terminal_soc_mwh: state of charge to finish on. Defaults to the
                spec's starting level, so the battery ends as it began and
                cannot book opening inventory as profit.
            tradable: optional per-interval mask. Intervals marked False are
                forced to do nothing, which is how price gaps are handled.
        """
        self.terminal_soc_mwh = terminal_soc_mwh
        self.tradable = tradable

    def plan(self, spec: BatterySpec, prices: pd.Series) -> Schedule:
        n = len(prices)
        if n == 0:
            raise ValueError("cannot optimise an empty price series")

        p = prices.to_numpy(dtype=float)
        delta = spec.degradation_cost_per_mwh
        terminal = (
            spec.initial_soc_mwh if self.terminal_soc_mwh is None else self.terminal_soc_mwh
        )

        # A solver takes one flat vector of unknowns, so the three quantities per interval are laid end to end:
        #
        #   index:  0 .. n-1    |   n .. 2n-1   |  2n .. 3n-1
        #   holds:  c_0 .. c_n-1|  d_0 .. d_n-1 |  s_0 .. s_n-1
        #           (buy)          (sell)          (state of charge)
        #
        # 864 unknowns for a 288-interval day. Every array below is indexed the
        # same way: one value per unknown, in that order.
        c_at, d_at, s_at = 0, n, 2 * n
        n_vars = 3 * n

        # Objective: one coefficient per unknown
        # The solver minimises `sum(obj[i] * x[i])`, so we feed it the negative of revenue
        # Selling gets a negative coefficient but buying gets a positive coefficient.
        # The SoC block is all zeros: holding energy is free, and s exists purely as bookkeeping to link intervals.
        obj = np.zeros(n_vars)
        obj[c_at : c_at + n] = p + delta
        obj[d_at : d_at + n] = -p + delta

        # Equalities: state of charge bookkeeping
        # s_t - s_{t-1} - eta_c * c_t + d_t / eta_d  =  0
        # 
        # Conceptually n-by-3n, but only three or four entries per row are
        # non-zero, so it is built as (row, column, value) triples: entry k
        # says row `rows[k]`, column `cols[k]`, holds `vals[k]`. Everything
        # unmentioned is zero.
        #
        # Interval 0 has no predecessor, so its opening level is a known number
        # and moves to the right-hand side instead.
        rows, cols, vals = [], [], []
        for t in range(n):
            rows.extend([t, t, t])
            cols.extend([s_at + t, c_at + t, d_at + t])
            vals.extend([1.0, -spec.charge_efficiency, 1.0 / spec.discharge_efficiency])
            if t > 0:
                rows.append(t)
                cols.append(s_at + t - 1)
                vals.append(-1.0)

        # Use CSR as there are many zeroes and the 2D array is sparse
        # For a real day the matrix is 288 × 864 = 248,832 entries, of which 1,151 are non-zero (~0.46%)
        a_eq = sparse.csr_array((vals, (rows, cols)), shape=(n, n_vars))
        b_eq = np.zeros(n)
        b_eq[0] = spec.initial_soc_mwh

        # Inequalities: shared connection rating
        # c_t + d_t <= connection limit. Two entries per row, both 1.0, so the triples are built with array arithmetic.
        connection_limit = max(spec.max_charge_mwh, spec.max_discharge_mwh)
        idx = np.arange(n)
        a_ub = sparse.csr_array(
            (
                np.ones(2 * n),
                (np.concatenate([idx, idx]), np.concatenate([c_at + idx, d_at + idx])),
            ),
            shape=(n, n_vars),
        )
        b_ub = np.full(n, connection_limit)

        # bounds: a floor and a ceiling for each unknown
        # Indexed exactly like `obj`
        lower = np.zeros(n_vars)
        upper = np.empty(n_vars)
        upper[c_at : c_at + n] = spec.max_charge_mwh
        upper[d_at : d_at + n] = spec.max_discharge_mwh
        upper[s_at : s_at + n] = spec.capacity_mwh

        if self.tradable is not None:
            if len(self.tradable) != n:
                raise ValueError("tradable mask must match the price series length")
            blocked = ~np.asarray(self.tradable, dtype=bool)
            upper[c_at : c_at + n][blocked] = 0.0
            upper[d_at : d_at + n][blocked] = 0.0

        # Pin the final state of charge so the battery cannot finish by liquidating energy it never paid for.
        lower[s_at + n - 1] = terminal
        upper[s_at + n - 1] = terminal

        result = linprog(
            c=obj,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=np.column_stack([lower, upper]),
            method="highs",
        )
        if not result.success:
            raise InfeasibleScheduleError(
                f"LP did not solve: {result.message}. A terminal state of charge of "
                f"{terminal:.1f} MWh may be unreachable within {n} intervals."
            )

        x = result.x
        return Schedule(
            name=self.name,
            charge_mwh=x[c_at : c_at + n],
            discharge_mwh=x[d_at : d_at + n],
            # We minimised the negative of revenue, so the sign flips back to $/stored MWh.
            water_value_aud_mwh=-np.asarray(result.eqlin.marginals, dtype=float),
            claimed_net_revenue_aud=-float(result.fun),
        )
