# BESS arbitrage — NSW1

This repository works out when a 200 MWh battery should buy and sell energy in the NSW1 region
of Australia's National Electricity Market, from real five-minute dispatch
prices published by AEMO, and explains every decision it makes.

The optimiser assumes perfect hindsight, so its output is an upper bound on the
trading revenue achievable for a given day. That bound is **\$348,221** across 29 complete NEM trading days
in September–October 2026.

## Quick start

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

| | |
|---|---|
| `python run.py` | summary report in the terminal |
| `python -m streamlit run app.py` | interactive dashboard |
| `python -m pytest` | run tests |

A parsed 30-calendar-day price series is committed under `data/sample/`, so
**nothing needs the network**. Re-fetching from AEMO takes 51 s and 168 MB; the committed
cache loads in 0.24 s.

More views: `run.py --daily`, `--trades 20`, `--day 2026-10-04`, `--export out/`,
`--refresh`, `--horizon continuous`, and `--power-mw` / `--degradation` for
sensitivity runs.

## Architecture

```
bess/
├── marketdata/   AEMO files        -> PriceSeries
├── asset/        battery physics   -> SimulationResult
├── strategies/   decision rules    -> Schedule
├── reporting/    trade log, KPIs
├── dashboard/    Plotly figures
│   backtest.py   runs a strategy across trading days
│   config.py     every assumption, in one place
run.py            CLI
app.py            Streamlit layout
```

### Why these boundaries

Each package has **exactly one reason to change**, which is what earns the
split its overhead.

| Package | Changes when | Knows nothing about |
|---|---|---|
| `marketdata` | AEMO alters its file layout or format | batteries, optimisation |
| `asset` | the battery's physics or costs change | where prices came from, linear programs |
| `strategies` | you want a different decision rule | HTTP, zip files, Streamlit |
| `reporting` | someone wants a different number | how the schedule was produced |
| `dashboard` | the presentation changes | everything but reporting's output |

**Dependencies point one way.** `marketdata` imports nothing from the project.
`strategies` and `reporting` each depend on `asset`, and stay independent of
each other and of the UI. Delete `dashboard/` and the backtest still runs.

### Three types are the seams

Three frozen dataclasses, each crossing one boundary, which is what lets either
side be replaced:

- **`PriceSeries`** — prices, a tradable mask, and a data-quality report.
  Everything upstream is about files; everything downstream is about energy.
- **`Schedule`** — how much to buy and sell per interval, and optionally the
  water value. A strategy's entire output.
- **`SimulationResult`** — what actually happened, per interval, with the
  reason.

### Decisions worth defending

**The simulator is the only place money is computed.** Strategies propose a
schedule; [`asset/simulator.py`](bess/asset/simulator.py) replays it into state
of charge and revenue. Every strategy is therefore scored by identical
arithmetic, which makes a hindsight optimiser and a live one directly
comparable. It also gives the LP a free correctness check: its own objective
and the simulator's independent recomputation agree to **3.6e-12**, and a sign
error in the constraint matrix would break that.

**Infeasible requests are clipped to what the battery could really do**, with
the removed energy recorded. A broken strategy announces itself through
`clipped_mwh`. Across the full backtest the LP is clipped by 1.3e-13 MWh.

**`marketdata` is three modules for three reasons to change.**
[`nemweb.py`](bess/marketdata/nemweb.py) knows where files live and how they
nest. [`mms.py`](bess/marketdata/mms.py) knows AEMO's CSV grammar and would
parse any of their reports. [`prices.py`](bess/marketdata/prices.py) knows what
the fields *mean*, and holds all the judgment.

**`Strategy` is a Protocol.** Conformance is structural, so a strategy
satisfies it without importing or subclassing it and a three-line stub works in
a test. It is a static contract, unenforced at runtime until a type checker is
added.

**`BatterySpec` has no defaults.** It describes the *shape* of a battery;
[`config.py`](bess/config.py) supplies the *values*. Validation in
`__post_init__` carries the weight: values arrive from CLI flags and dashboard
sliders, and an efficiency above 1 is a perpetual motion machine the optimiser
would exploit.

**The result frame is a contract.** A DataFrame carries no schema, so
`SimulationResult` checks `REQUIRED_COLUMNS` on construction and names what is
missing. `action` and `binding_constraint` are `StrEnum`s stored as
categoricals: the column holds plain strings, code says `Action.CHARGE`, and a
typo raises `AttributeError`.

**Figures are built apart from Streamlit.**
[`dashboard/charts.py`](bess/dashboard/charts.py) returns Plotly objects and
imports no Streamlit, so charts are tested without a browser. A renamed column
would otherwise leave an empty panel and no error, so six tests assert the
traces carry data.

**Caching has two layers** — raw zips by filename in `data/raw/`, parsed rows
as parquet in `data/sample/`. The second skips the download *and* the
8,640-file parse, 51 s down to 0.24 s, which is what lets the dashboard start
offline.

**`backtest.py` sits at the package root.** Orchestration belongs to no single
package, and it is the only module that touches both strategies and market data.

## The optimiser

Finding the revenue-maximising schedule with known prices is a linear program.
Three unknowns per interval — energy bought $c$, energy sold $d$, both
grid-side, and the resulting state of charge $s$:

$$\max \sum_t p_t(d_t - c_t) - \delta(c_t + d_t)$$

| Constraint | Why |
|---|---|
| $s_t = s_{t-1} + \eta_c c_t - d_t / \eta_d$ | Pure accounting. Buying stores only $\eta_c$ of what you pay for; selling drains more than it delivers. Losses come out of inventory. |
| $0 \le s_t \le 200$ | The **lower** bound forbids selling energy never bought, so every sale follows a purchase in time. There is no shorting electricity. |
| $c_t + d_t \le 4.167$ MWh | One connection, one rating, shared by both directions. 50 MW for 1/12 h. |
|  $s_0 = s_T = 0$ | Start and finish empty. |

864 unknowns per day, solved by HiGHS in **7 ms**; all 29 days take 0.2 s.

An LP beats a heuristic on three counts. It is exactly optimal. It takes *less*
code than greedy spread-pairing, which needs headroom bookkeeping and a maximum
over the SoC path between every candidate pair. And new requirements — cycle
caps, time-varying limits, network charges — go in as extra rows.

### Explainability

The bookkeeping constraints are equalities, so their shadow prices fall out of
the solver. Each is **what a stored MWh is worth at that moment**: a
reservation price in \$/MWh, borrowed from hydro scheduling, where a dam
operator asks the same question about water behind the dam.

The policy is one comparison: charge when $p_t/\eta_c < \lambda_t$, discharge
when $p_t \eta_d > \lambda_t$, otherwise hold. $\lambda$ holds **flat while nothing binds** and jumps
when a limit bites, which is what makes it read as a clean line — on 4 October
it sat at \$112.19 for hours. The dashboard plots it over the price, and the
trade log quotes it in words:

> *Sold 162.5 MWh over 200 minutes at a volume-weighted \$165.36/MWh, above the
> \$124.11 the battery valued stored energy at.*

### Minimum spread

At 81% round-trip you buy 1.235 MWh for every MWh sold, so a spread narrower
than **23.5%** destroys value before degradation enters. That hurdle is why the
battery sits idle 62% of the time.

### Simultaneous charge and discharge

Nothing forbids $c_t$ and $d_t$ from both being positive, deliberately. A
battery reverses direction in well under a second, so across a 300-second
interval it can do both, and at a negative price it is paid for the energy its
own round-trip losses consume. Holding SoC flat means $d = \text{rte} \cdot c$, so it pays
when

$$|p| > \delta \frac{1 + \text{rte}}{1 - \text{rte}}$$

Suppressing it needs a binary variable and hides real behaviour. The
degradation cost prices it instead, and the effect is measurable:

| $\delta$ (AUD/MWh) | Pays below | Intervals observed |
|---:|---:|---:|
| 0.00 | −\$0.00 | **40** |
| 0.50 | −\$4.76 | 29 |
| 1.00 | −\$9.53 | 0 |
| 5.00 | −\$47.63 | 0 |

The threshold is necessary but not sufficient: the battery must also be full at
that moment. Any realistic degradation cost rules the behaviour out, so the
configured case shows zero. The count is reported either way.

## Market data

Source: `DISPATCHIS` daily archives from
[nemweb.com.au](https://nemweb.com.au/Reports/) — one zip per day, each holding
288 zips, each one CSV. The daily archive takes 30 downloads for a 30-day window, where `Current`
would take roughly 8,640.

Four interpretation rules, each a silent bug if taken the other way:

| Rule | What goes wrong otherwise |
|---|---|
| **Read columns from the `I` row** | One file carries `CASE_SOLUTION` v2, `PRICE` v5 and `REGIONSUM` v9 side by side. A fixed offset works until AEMO adds a column, then reads the wrong one. |
| **Market time is AEST, fixed UTC+10**, all year | The NEM has no daylight saving. `Australia/Sydney` injects hour shifts the market never had, and the schedule looks *almost* right. |
| **`SETTLEMENTDATE` labels the END** of its interval | Verified: the 8 Oct file spans 00:05 to next-day 00:00. Read as interval-starting, the whole schedule shifts five minutes away from the prices that justified it. |
| **Gaps are never filled** | A stale price produces a trade that never happened. Missing intervals stay in the index, marked untradable, and counted. |

Also handled: `INTERVENTION = 0` wins where an intervention run duplicates a
timestamp, with a fallback where only the intervention run exists; republished
intervals keep the highest `RUNNO`; a price outside [−\$1,000, \$20,000] raises
as a parsing error; and downloads are written atomically, so an interrupted
fetch can never masquerade as a complete cache entry.

```
NSW1  2026-09-09 00:05 to 2026-10-09 00:00 AEST
  intervals          8,640 of 8,640 (100.00%)
  missing            0
  intervention rows  0 dropped
  duplicate rows     0 dropped
  negative prices    575 intervals
  RRP                $-12.42 to $384.09  (mean $77.29)
```

This window is clean: no gaps, no interventions, no republished intervals.
**All three paths are exercised by fixture tests, and the live data never
triggered any of them.** The market runs 24/7/365, so all 288 intervals a day
carry a price.

**30 calendar days of data give 29 trading days of results.** The NEM trading
day starts at 04:00, four hours offset from the calendar day, so the window
straddles 31 trading days: 29 complete ones plus a 48-interval stub at the
start and a 240-interval stub at the end. The two stubs are dropped, because a
part-day's revenue cannot be compared with a whole one. 48 + (29 × 288) + 240 =
8,640.

## Results

Every figure below reproduces exactly. The window is fixed dates, nothing reads
the wall clock, the price series is committed, dependencies are pinned and the
solver is deterministic. A fresh clone with no network and no `data/raw/`
produces byte-identical output, down to the individual schedule.
`run.py --refresh` re-fetches from AEMO to confirm the cache still matches.

```
Perfect foresight (LP) over 29 trading days

  net revenue         $    348,221.37
    gross             $    412,334.85
    degradation       $    -64,113.48
  per MWh capacity    $      1,741.11

  bought / sold           7,084 / 5,738 MWh
  full cycles              31.9  (1.10/day)
  active intervals        37.6%

  average buy         $         31.28
  average sell        $        110.47
  captured spread     $         79.19   (3.53x vs 1.23x needed)

  best day            $     24,268.81
  median day          $     12,006.27
  worst day           $      1,192.00
```

### Sensitivity

**Power rating** (200 MWh fixed):

| MW | Duration | Net revenue | Cycles/day | Avg buy | Avg sell |
|---:|---:|---:|---:|---:|---:|
| 25 | 8.0 h | \$246,010 | 0.84 | \$32.20 | \$107.02 |
| **50** | **4.0 h** | **\$348,221** | **1.10** | **\$31.28** | **\$110.47** |
| 100 | 2.0 h | \$443,952 | 1.43 | \$30.86 | \$108.78 |
| 200 | 1.0 h | \$563,533 | 2.05 | \$32.62 | \$104.16 |

More power earns more, and the mechanism is in the last two columns: average
buy and sell prices **barely move** across the range. The gain comes from
fitting more cycles through the same spread. These are revenue figures — a
200 MW connection costs far more than a 50 MW one, and the model carries no
capex.

**Efficiency** (50 MW), and **degradation**, which has the least evidence
behind it and is among the largest levers:

| One-way $\eta$ | Round trip | Hurdle | Net revenue | | $\delta$ (AUD/MWh) | Net revenue | Cycles/day |
|---:|---:|---:|---:|:--:|---:|---:|---:|
| 0.85 | 72% | 38.4% | \$300,916 | | 0.0 | \$417,993 | 1.37 |
| **0.90** | **81%** | **23.5%** | **\$348,221** | | **5.0** | **\$348,221** | **1.10** |
| 0.95 | 90% | 10.8% | \$400,171 | | 10.0 | \$287,150 | 1.00 |
| 1.00 | 100% | 0.0% | \$461,374 | | 20.0 | \$183,422 | 0.76 |

The plausible \$0–20 degradation range moves revenue by a factor of 2.3, so the
headline is best read as a band.

## Assumptions

All in [`bess/config.py`](bess/config.py). Capacity and region are fixed by the
problem; everything else is a choice.

| | Value | Why |
|---|---|---|
| Power rating | **50 MW** grid-side | 200 MWh / 50 MW is a four-hour battery, the duration built for arbitrage — long enough to cover a full evening peak. |
| Efficiency | **90% per leg $\to$ 81% round-trip** | Conservative; modern lithium-ion reaches 85–90%. Erring low keeps the estimate cautious. |
| Starting SoC | **0 MWh** | Starting empty makes "cannot sell energy it never bought" true by construction. |
| Terminal SoC | **returns to 0** daily | Stops the battery finishing by liquidating inventory it never paid for. |
| Degradation | **\$5/MWh** throughput | Cells have finite cycle life. Omitting it lets the optimiser run shallow cycles no warranty permits. |
| Trading day | **04:00–03:59 AEST** | The NEM's own trading day, which is how settlement is defined. |
| Window | 2026-09-09 to 2026-10-08 | One archive download per day; long enough for a daily distribution, small enough to commit. |
| Price taking | assumed | Unavoidable in a backtest, and **the largest upward bias** — 50 MW discharging into a tight peak would suppress the price. |
| Perfect fills | assumed | A real BESS offers into dispatch and may not be taken. Also biases upward. |
| Scope | energy arbitrage only | No FCAS or capacity contracts, which **understates** the revenue stack and partly offsets the two biases above. |

Named here and left unmodelled: marginal loss factors, network charges,
auxiliary load, temperature derating, outages, tax and capex. The output is
gross trading revenue.

### What splitting into days costs

Days are optimised independently, each opening and closing empty. That matches
how the asset is run and how cycle limits are counted, and it makes daily
revenue comparable across the window. The cut does cost something:

| Boundary | Per day |
|---|---:|
| 00:00 | \$12,320 |
| **04:00** | **\$12,008** |
| 08:00 | \$12,549 |
| 12:00 | \$9,740 |
| 16:00 | \$3,810 |

A factor of three between best and worst, so the simplification is a real
constraint on the result. 08:00 scores 4% above the trading day here, an edge
that would evaporate in a different month; tuning the boundary on revenue fits
the sample and reports "days" no settlement statement agrees with.

**Both horizons are available**, over exactly the same 8,352 intervals:

| | Net revenue | Cycles/day | Days close empty |
|---|---:|---:|---:|
| `--horizon daily` (default) | \$348,221 | 1.10 | 29 of 29 |
| `--horizon continuous` | **\$376,190** | 1.07 | 4 of 29 |

Continuous earns **8.0% more** by carrying charge over the boundary, holding up
to 190.7 MWh overnight. It does so with *fewer* cycles, 1.07 against 1.10:
freed from emptying every day, it skips cycles that barely clear the hurdle and
saves the charge for better moments.

The daily reset is bookkeeping, so continuous is arguably the truer ceiling --
a real battery does not empty itself at 04:00. Daily stays the default because
it keeps each day a self-contained round trip, which is what makes per-day
figures comparable and auditable. Under a continuous horizon a day may sell
energy bought earlier, so its revenue becomes a cashflow.

## Testing

70 tests. The ones that matter:

- **LP objective equals simulator revenue** — two independent routes to one
  number; disagreement means a sign error in the constraint matrix.
- **A four-interval case worked by hand** before the code existed: buy 8.333
  MWh across two negative intervals, sell 6.750 MWh, earn \$1,433.33.
- **State of charge stays in bounds for any request**, including deliberately
  impossible schedules.
- **A missing interval is marked untradable** and keeps its own price out of
  the schedule.
- **The burn threshold fires below −\$47.63 and stays dormant above it**,
  confirming the closed form inside the solver.
- **The trade log reconciles with interval totals exactly**, which makes it a
  view of the same money.

## Limitations

- **Perfect hindsight.** The full price series is known in advance, so this is
  a ceiling. `strategies/` is shaped so a live rule drops in against the same
  simulator and is scored identically.
- **Price taking is the largest single bias**, and it bites hardest in exactly
  the high-price intervals that contribute most of the revenue.
- **Optimising days independently** costs 8.0% — see the horizon comparison
  above. It remains the default because it keeps per-day figures auditable.
- **Nothing is charged for standing still.** Degradation is priced per MWh of
  throughput, and state of charge is perfectly preserved while idle: 150 MWh
  parked for a week ends at exactly 150 MWh. Self-discharge, auxiliary load and
  calendar ageing are all absent, so holding a full battery is free in the
  model in a way it is not in reality. The daily reset capped this at 7.1 hours;
  a continuous horizon holds for up to 14.6.
- **Per-trade cashflow is one-sided.** A sell row books the full sale while the
  purchase sits in an earlier buy row. Totals reconcile exactly, and round-trip
  profit becomes unambiguous at day level, where each day opens and closes
  empty.
- **One region, one asset, no network constraints.**

## Next steps

1. **Operate without hindsight** — a trailing-percentile or forecast-driven
   rule against the same simulator, reporting what share of this ceiling it
   captures and where its worst decisions fall.
2. **Day-level round-trip P&L** in the trade log, which closes the one-sided
   reading above.
3. **Live ingestion** from `Reports/Current`, plus vintage-correct predispatch.
   The forecast has to be the one published *at time t*; reaching for the
   latest revision is lookahead leakage in disguise.
4. **Bid curves.** A real BESS offers into dispatch and waits to be taken, so
   "charge X MWh now" stops short of what an operator actually submits.
