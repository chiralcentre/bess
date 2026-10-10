"""Perfect-hindsight optimum, as a linear program.

Given the whole price series up front, this finds the schedule that no other
schedule can beat. That makes it two things at once: an answer to the brief,
and the ceiling any live strategy is measured against.

The formulation is the standard one for storage arbitrage. For each interval
``t`` there are three unknowns -- energy bought ``c_t``, energy sold ``d_t``
(both grid-side), and the resulting state of charge ``s_t`` -- and the solver
chooses all of them at once to maximise::

    sum over t of   p_t * (d_t - c_t)  -  delta * (c_t + d_t)

subject to four groups of constraints:

1. **Bookkeeping** ``s_t = s_{t-1} + eta_c * c_t - d_t / eta_d``.
   Not a choice but an identity: buying stores only ``eta_c`` of what you pay
   for, and selling drains more than it delivers. Losses always come out of
   inventory. These are equalities, and their shadow prices are the water
   value this module reports.

2. **Tank** ``0 <= s_t <= capacity``. The lower bound is what forbids selling
   energy that was never bought, so every sale is preceded in time by a buy.

3. **Connection** ``c_t + d_t <= connection limit``. One grid connection with
   one rating, shared by both directions.

4. **Endpoints**. Start at the given state of charge and finish at the target,
   so leftover energy cannot be reported as profit.

Note on simultaneous flow: nothing above forbids ``c_t`` and ``d_t`` from both
being positive. That is deliberate. A battery can reverse direction many times
inside a 300-second interval, so the behaviour is physically reachable, and at
sufficiently negative prices it is genuinely profitable -- the round-trip
losses let the battery be paid to consume without filling up. Forbidding it
would need a binary variable and would suppress a real effect; instead it is
priced by ``degradation_cost_per_mwh`` and reported. See the README.
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

    name = "Perfect foresight (LP)"

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
                forced to do nothing, which is how price gaps are handled --
                a missing price is not a reason to trade on a stale one.
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

        # Variables are laid out as one flat vector:
        #   [ c_0 .. c_n-1 | d_0 .. d_n-1 | s_0 .. s_n-1 ]
        c_at, d_at, s_at = 0, n, 2 * n
        n_vars = 3 * n

        # -- objective ------------------------------------------------------
        # linprog minimises, so we minimise the negative of revenue. Buying
        # costs the price plus wear; selling earns the price less wear. A
        # negative price flips the sign of the buying term on its own, which is
        # how being paid to charge falls out without a special case.
        obj = np.zeros(n_vars)
        obj[c_at : c_at + n] = p + delta
        obj[d_at : d_at + n] = -p + delta

        # -- equality constraints: state of charge bookkeeping --------------
        # Row t:  s_t - s_{t-1} - eta_c * c_t + d_t / eta_d  =  0
        # Row 0 has no predecessor, so the starting level moves to the RHS.
        rows, cols, vals = [], [], []
        for t in range(n):
            rows += [t, t, t]
            cols += [s_at + t, c_at + t, d_at + t]
            vals += [1.0, -spec.charge_efficiency, 1.0 / spec.discharge_efficiency]
            if t > 0:
                rows.append(t)
                cols.append(s_at + t - 1)
                vals.append(-1.0)

        a_eq = sparse.csr_array((vals, (rows, cols)), shape=(n, n_vars))
        b_eq = np.zeros(n)
        b_eq[0] = spec.initial_soc_mwh

        # -- inequality constraints: shared connection rating ---------------
        # Row t:  c_t + d_t <= connection limit
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

        # -- bounds ---------------------------------------------------------
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

        # Pin the final state of charge so the battery cannot finish by
        # liquidating energy it never paid for.
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
            # Shadow prices of the bookkeeping rows. We minimised the negative
            # of revenue, so flipping the sign puts the water value back into
            # revenue terms: dollars per extra MWh sitting in the tank.
            water_value_aud_mwh=-np.asarray(result.eqlin.marginals, dtype=float),
            claimed_net_revenue_aud=-float(result.fun),
        )
