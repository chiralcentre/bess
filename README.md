# BESS arbitrage — NSW1

Decides when a 200 MWh battery should buy and sell energy in the NSW1 region of
the Australian National Electricity Market, using real five-minute dispatch
prices from AEMO, and explains every decision it makes.

> **Status:** market data, battery model and optimiser are complete with 47
> tests. Reporting and the dashboard are in progress; the Results section below
> will be filled in when they land.

## Running it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest
```

A 30-day parsed price series is committed under `data/sample/`, so **nothing
needs the network to run**. Re-fetching from AEMO takes about 51 seconds and
168 MB; loading the committed cache takes 0.24 seconds.

## Assumptions

The brief fixes the capacity at 200 MWh and the region as NSW1, and leaves
everything else to the modeller. All of it lives in
[`bess/config.py`](bess/config.py).

| Assumption | Value | Why |
|---|---|---|
| Power rating | **50 MW** grid-side | Not given by the brief, and the problem is unsolvable without it. 200 MWh / 50 MW is a **four-hour battery** — the duration built for energy arbitrage, long enough to cover a full evening peak. |
| Efficiency | **90% each leg → 81% round-trip** | Conservative; modern lithium-ion is 85–90%. Erring low keeps the revenue estimate cautious. |
| Starting SoC | **0 MWh** | Starting empty makes "cannot sell energy it never bought" true by construction, with no terminal condition needed to enforce it. |
| Terminal SoC | **returns to 0** each trading day | Stops the battery finishing by liquidating inventory it never paid for. |
| Degradation | **$5/MWh** of throughput | Cells have finite cycle life. Without this the optimiser runs many shallow cycles no warranty permits. The true figure depends on capex and warranty terms that are not public — treat it as an order of magnitude. |
| Trading day | **04:00–04:00 AEST** | Not midnight. In this window 00:00–04:00 is *expensive*, so a midnight boundary would cut the evening discharge off halfway through. |
| Window | **30 days**, 2026-09-09 to 2026-10-08 | One archive download per day. Long enough to show a daily revenue distribution; small enough to cache and commit. |
| Price taking | assumed | Unavoidable in a backtest, but **it biases revenue upward** — 50 MW discharging into a tight evening peak would suppress the price, exactly in the intervals that matter most. |
| Perfect fills | assumed | Real BESS submits offer curves and may not be dispatched. Also biases upward. |
| Scope | energy arbitrage only | No FCAS, no capacity contracts. This **understates** the real revenue stack, partly offsetting the two biases above. |

Not modelled, and named rather than hidden: marginal loss factors, network
charges, auxiliary load, temperature derating, outages, tax and capex. The
output is **gross trading revenue**, not an investment case.

### Seasonality caveat

The window is Australian spring: strong solar, soft demand, so a deep midday
trough but fewer summer scarcity spikes. It likely **understates** annualised
revenue.

## Architecture

```
bess/
├── marketdata/   fetch, parse and clean AEMO files   -> PriceSeries
├── asset/        battery limits and physics          -> SimulationResult
├── strategies/   decide when to buy and sell         -> Schedule
├── reporting/    trade log and KPIs
└── dashboard/    Plotly figures
```

Dependencies point one way. `marketdata` knows nothing about batteries,
`asset` knows nothing about linear programs, `strategies` knows nothing about
Streamlit.

Two ideas carry the design:

**The simulator is the only place money is computed.** Strategies propose a
schedule; [`asset/simulator.py`](bess/asset/simulator.py) replays it to produce
state of charge and revenue. So every strategy is scored by identical
arithmetic, and a hindsight optimiser can be compared fairly against a live one.
It also gives a free correctness check: the LP's own objective and the
simulator's independent recomputation must agree, and they do, to 3.6e-12.

**Infeasible requests are clipped, not rejected.** The simulator enforces
physics and records how much energy it had to remove. A broken strategy shows
up as non-zero `clipped_mwh` rather than as impossible profit.

## Method: a linear program

Finding the revenue-maximising schedule with full knowledge of prices is a
textbook linear program, and it solves a 288-interval day in **7 ms**.

Three unknowns per interval — energy bought `c`, energy sold `d`, both
grid-side, and the resulting state of charge `s`:

$$\max \sum_t \; p_t(d_t - c_t) \;-\; \delta(c_t + d_t)$$

| Constraint | | Why |
|---|---|---|
| Bookkeeping | `s_t = s_{t-1} + η_c·c_t − d_t/η_d` | Not a choice but an identity. Buying stores only `η_c` of what you pay for; selling drains more than it delivers. Losses always come out of inventory. |
| Tank | `0 ≤ s_t ≤ 200` | The **lower** bound is what forbids selling energy that was never bought, so every sale is preceded in time by a purchase. There is no shorting electricity. |
| Connection | `c_t + d_t ≤ 4.167 MWh` | One grid connection, one rating, shared by both directions. 50 MW for 1/12 of an hour. |
| Endpoints | `s_0 = s_T = 0` | Start and finish empty. |

Why an LP rather than a heuristic: it is exactly optimal, it is *less* code than
the greedy pairing alternative (no headroom bookkeeping, no SoC-path maximum),
and new requirements — cycle caps, time-varying limits, network charges — are
added as extra rows rather than by re-deriving a rule.

### Explainability: the water value

The bookkeeping constraints are equalities, so their shadow prices come out of
the solver directly. Each one is **the marginal value of a stored MWh at that
moment** — the battery's internal "water value", in $/MWh. The optimal policy
then collapses to a comparison a trader already thinks in:

- charge when `p_t/η_c < λ_t` — energy is cheaper than it is worth inside
- discharge when `p_t·η_d > λ_t` — it earns more than it is worth inside
- hold otherwise

Plotted against the price, the battery explains itself: above the line it
sells, below it buys. On 8 October the water value ranged $85.20–$159.20,
sitting between the $72.35 average buy and the $202.69 average sell, which is
the check that the sign convention is right.

### The minimum spread

At 81% round-trip you buy 1/0.81 = 1.235 MWh for every MWh you sell, so a
spread narrower than **23.5%** destroys value before degradation is counted.
This is why the optimiser idles on flat days rather than trading noise.

### Simultaneous charge and discharge

The formulation does not forbid `c_t` and `d_t` from both being positive, and
that is deliberate. A battery reverses direction in well under a second, so
across a 300-second interval it can do both, and at a negative price it is then
paid for the energy its own round-trip losses consume.

Holding state of charge flat means `d = rte·c`, so net grid energy is
`c(1 − rte)` while energy through the cells is `c(1 + rte)`. The first earns,
the second wears. It pays when

$$|p| \;>\; \delta\,\frac{1+\text{rte}}{1-\text{rte}} \;=\; 5 \times \frac{1.81}{0.19} \;=\; \$47.63$$

Forbidding it would need a binary variable and would suppress real behaviour.
Pricing it is enough: **the 30-day window bottoms out at −$12.42, so it never
pays there** and the LP never does it. The count is reported rather than
assumed away.

## Market data

Source: [`nemweb.com.au`](https://nemweb.com.au/Reports/) `DISPATCHIS` archive
files — one zip per day, each containing 288 zips, each containing one CSV.
The daily archive is used rather than the Current directory because 30 days is
30 downloads there against roughly 8,640.

Four interpretation rules, each a silent bug if taken the other way:

**Columns come from the `I` row, never from fixed positions.** An MMS file
holds several tables, each with its own header, and their versions differ — a
single DispatchIS file carries `CASE_SOLUTION` v2, `PRICE` v5 and `REGIONSUM`
v9 side by side. A fixed offset works until AEMO adds a column, then silently
reads the wrong field.

**Market time is AEST, fixed at UTC+10, all year.** The NEM does not observe
daylight saving. `Australia/Sydney` would inject one-hour shifts the market
never had, twice a year, and the schedule would look almost right.

**`SETTLEMENTDATE` labels the END of its interval.** A row stamped 14:05
describes 14:00–14:05. Confirmed empirically: the file for 8 October spans
00:05 to next-day 00:00. Reading it as interval-starting shifts the whole
schedule five minutes away from the prices that justified it.

**Gaps are never filled.** A stale price produces a trade that never happened.
Missing intervals stay in the index, are marked untradable, and are counted.

Other handling: intervention runs duplicate a timestamp, so `INTERVENTION = 0`
is preferred with a fallback where only an intervention run exists;
republished intervals keep the highest `RUNNO`; prices outside
[−$1,000, $20,000] are rejected as parsing errors rather than market events.

Downloads are cached by filename and written atomically, so an interrupted
fetch cannot masquerade as a complete cache entry.

### What the window contains

```
NSW1  2026-09-09 00:05 to 2026-10-09 00:00 AEST
  intervals          8,640 of 8,640 (100.00%)
  missing            0
  intervention rows  0 dropped
  duplicate rows     0 dropped
  negative prices    575 intervals
  RRP                $-12.42 to $384.09  (mean $77.29)
```

The data is clean: no gaps, no interventions, no republished intervals. The
handling for all three is covered by fixture tests rather than by live data —
worth stating, rather than implying the real window exercised it.

The shape is textbook: overnight $92–105, a solar trough of $38–46 from 09:00
to 14:00, an evening peak of $106 at 17:00–18:00, staying high to midnight.

## Results

*To be completed once reporting and the dashboard land.* Single-day figure for
8 October 2026: **$23,971.56** net revenue, 1.30 equivalent full cycles,
average buy $72.35 against average sell $202.69.

## Testing

```bash
.venv/Scripts/python -m pytest
```

47 tests. The ones that matter:

- **LP objective equals simulator revenue.** Two independent routes to the same
  number; disagreement means a sign error in the constraint matrix.
- **A four-interval case worked by hand** before the code existed: buy 8.333
  MWh across two negative intervals, sell 6.750 MWh, earn $1,433.33.
- **State of charge stays in bounds for any request**, including deliberately
  impossible schedules.
- **A missing interval is marked untradable, not forward-filled.**
- **The burn threshold fires below −$47.63 and not above it**, confirming the
  closed form in the solver rather than only on paper.

## Known weaknesses

- **Perfect hindsight.** The base result is a ceiling, not a forecast. A live
  strategy knowing only the past would capture meaningfully less; the
  `strategies` package is shaped so one drops in against the same simulator.
- **Price taking is the biggest single bias**, and it is worst in exactly the
  high-price intervals that contribute most of the revenue.
- **Revenue is concentrated.** The NEM is energy-only, so spikes are a design
  feature and a few intervals carry a large share of the value. A daily
  distribution matters more than a total.
- **One region, one asset, no network constraints.**
