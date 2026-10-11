# NSW1 BESS Revenue Maximisation

This repository works out when a 200 MWh battery should buy and sell energy in
the NSW1 region of Australia's National Electricity Market, from real
five-minute dispatch prices published by AEMO, and explains every decision it
makes.

The optimiser assumes perfect hindsight, so its output is an upper bound on the
trading revenue achievable for a given day.

## Quick start

Python version: Python 3.11 or newer.

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
**no network call is required**. Re-fetching from AEMO takes 51 s and 168 MB;
the committed cache loads in 0.24 s.

## Running a backtest

### The two modes of operation

The battery can be optimised one trading day at a time, or across the whole
window in a single pass. Both run over exactly the same 8,352 intervals.

**Daily** — each trading day opens and closes empty, so a day's revenue is a
self-contained round trip:

```bash
python run.py
```

```
  net revenue         $    348,221.37
  cycles per day           1.10
```

**Continuous** — one optimisation across all 29 days, with charge free to carry
over midnight:

```bash
python run.py --horizon continuous
```

```
  net revenue         $    376,190.50
  cycles per day           1.07
```

Continuous earns **8.0% more**, holding up to 190.7 MWh overnight, and does it
with *fewer* cycles. Freed from emptying every day, it skips marginal cycles
and saves the charge for better moments.

Daily is the default. [What splitting into days
costs](#what-splitting-into-days-costs) weighs that choice against the 8%.

### Every option

| Option | What it does |
|---|---|
| *(none)* | Data-quality report, then headline metrics for the window |
| `--horizon daily` / `continuous` | Optimise per trading day (default) or in one pass |
| `--daily` | One line per trading day: revenue, energy, price range |
| `--trades N` | The N largest trades by cashflow, each with its reason |
| `--day YYYY-MM-DD` | One trading day, interval by interval, active intervals only |
| `--export DIR` | Write `intervals.csv`, `trades.csv` and `daily.csv` |
| `--start YYYY-MM-DD` | First calendar day of the window |
| `--end YYYY-MM-DD` | Last calendar day of the window |
| `--power-mw MW` | Override the power rating for a sensitivity run |
| `--degradation AUD_PER_MWH` | Override the degradation cost |
| `--refresh` | Re-fetch from AEMO instead of reading the committed cache |

Flags combine:

```bash
python run.py --horizon continuous --daily    # which days carry charge over
python run.py --power-mw 100 --degradation 10 # how the sensitivity tables were made
python run.py --day 2026-10-04 --trades 5     # the best day, interval by interval
python run.py --refresh --start 2026-10-01 --end 2026-10-08  # ignore the cache
```

The last one re-fetches from AEMO, which is how to confirm the committed
parquet still matches the source.

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

Single responsibility package, one reason to change per package.

| Package | Gets edited when | Never needs to know |
|---|---|---|
| `marketdata` | AEMO changes its file layout or format | that a battery exists |
| `asset` | the battery's physics or costs change | where the prices came from |
| `strategies` | you want a different decision rule | HTTP, zip files, Streamlit |
| `reporting` | someone asks for a different number | how the schedule was produced |
| `dashboard` | the presentation changes | anything except reporting's output |

A new AEMO column touches `marketdata` and nothing else. A second strategy is
one new file, scored by the same simulator as the first. Imports run one way:
`marketdata` imports nothing else in the project, `strategies` and `reporting`
each use `asset` and nothing of each other, and the dashboard sits on top
reading reporting's output. Delete `dashboard/` and the backtest still runs.

### Noteworthy Design Decisions

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

**`Strategy` is a Protocol.** A Protocol is used instead of an ABC because
strategies only need to satisfy this interface; they share no base class or
implementation. Any compatible strategy or test stub works without inheriting
from anything. It is a static contract, unenforced at runtime until a type
checker is added.

**`BatterySpec` has no defaults.** It describes the *shape* of a battery;
[`config.py`](bess/config.py) supplies the *values*. Validation in
`__post_init__` carries the weight: values arrive from CLI flags and dashboard
sliders.

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
8,640-file parse, 51 s down to 0.24 s, and it is why the dashboard starts
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

That is 864 unknowns for a day, which solves in a median of 4 ms. The whole
29-day window takes 0.12 s.

Turning that into the arguments `scipy.optimize.linprog` wants is mechanical,
and [docs/lp-walkthrough.md](docs/lp-walkthrough.md) works it through on a
four-interval battery with every matrix printed in full.

### Explainability

The LP returns a second number per interval alongside the schedule: what the
energy already in the battery is worth, in \$/MWh. Hand it one more MWh at
interval $t$ and the day earns another $\lambda_t$. Stored energy needs its own
valuation because the price paid for it is history, and what counts is the best
remaining use. Hydro schedulers solve the same problem for water behind a dam,
so the name **water value** carries over.
[The walkthrough](docs/lp-walkthrough.md#what-the-shadow-prices-mean) has the
derivation and a check that it is right.

That one number turns a global answer into a local one. Instead of all 288
prices interacting, each interval becomes a comparison:

$$\text{charge while } \frac{p_t + \delta}{\eta_c} \le \lambda_t,
\qquad \text{discharge while } (p_t - \delta)\,\eta_d \ge \lambda_t,
\qquad \text{otherwise hold}$$

The efficiencies put both sides into $\lambda_t$'s units, the cost of a MWh
landing in the tank against the revenue from one leaving it, and $\delta$
charges the wear either way.

These are the conditions the solver satisfied, not an approximation of them.
Across 1,729 charge and 1,411 discharge intervals the worst-case slack is zero
to floating-point precision, and 94% of the 5,212 idle intervals fall in the
band between the two thresholds.

$\lambda_t$ holds flat while nothing binds and steps when a limit bites. On
4 October it took 17 values across the day, sitting at \$112.19 from 17:15 to
03:55, a staircase that reads clearly plotted over the price. The trade log
puts it in words:

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
own round-trip losses consume. Holding SoC flat means $d = \text{rte} \cdot c$,
so the net energy taken is $c(1 - \text{rte})$ while the wear is charged on
$c(1 + \text{rte})$. Set against doing nothing, it pays when

$$|p| > \delta \frac{1 + \text{rte}}{1 - \text{rte}}$$

Doing nothing is rarely the alternative, though. While the tank has room,
charging alone earns $|p| \cdot c$ against this action's
$|p| \cdot c(1 - \text{rte})$ **and** stores energy worth selling later, so it
wins at any price. The threshold only decides anything once charging is
blocked, which in practice means a full battery.

Suppressing it needs a binary variable and hides real behaviour. The
degradation cost prices it instead, and the effect is measurable:

| $\delta$ (AUD/MWh) | Pays below | Intervals observed |
|---:|---:|---:|
| 0.00 | −\$0.00 | **45** |
| 0.50 | −\$4.76 | 31 |
| 1.00 | −\$9.53 | 0 |
| **5.00** | **−\$47.63** | **0** |

The threshold is necessary but not sufficient: the battery must also be full at
that moment. Any realistic wear cost rules the behaviour out, so the configured
case shows zero, but the count is reported either way.

## Market Data

Source: `DISPATCHIS` daily archives from
[nemweb.com.au](https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/) —
one zip per day, each holding 288 zips, each one CSV. The daily archive takes
30 downloads for a 30-day window, where `Current` would take roughly 8,640.

Four interpretation rules, each a silent bug if taken the other way:

| Rule | What goes wrong otherwise |
|---|---|
| **Read columns from the `I` row** | One file carries `CASE_SOLUTION` v2, `PRICE` v5 and `REGIONSUM` v9 side by side. A fixed offset works until AEMO adds a column, then reads the wrong one. |
| **Market time is AEST, fixed UTC+10**, all year | The NEM has no daylight saving. `Australia/Sydney` injects hour shifts the market never had, and the schedule looks *almost* right. |
| **`SETTLEMENTDATE` labels the END** of its interval | Verified: the 8 Oct file spans 00:05 to next-day 00:00. Read as interval-starting, the whole schedule shifts five minutes away from the prices that justified it. |
| **Gaps are never filled** | A stale price produces a trade that never happened. Missing intervals stay in the index, marked untradable, and counted. |

Also handled: `INTERVENTION = 0` wins where an intervention run duplicates a
timestamp, with a fallback where only the intervention run exists; republished
intervals keep the highest `RUNNO`; a price below the −\$1,000 market floor
raises as a parsing error, with no matching ceiling because the Market Price
Cap is indexed annually and a hard-coded one would reject genuine scarcity
prices; and downloads are written atomically, so an interrupted fetch can never
masquerade as a complete cache entry.

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
ForesightLP over 29 trading days

  net revenue         $    348,221.37
    gross             $    412,334.85
    degradation       $    -64,113.48
  per MWh capacity    $      1,741.11

  bought / sold           7,084 / 5,738 MWh
  cycles per day           1.10
  active intervals        37.6%

  average buy         $         31.28
  average sell        $        110.47
  captured spread     $         79.19   (3.53x vs 1.23x needed)

  best day            $     24,268.81
  median day          $     12,006.27
  worst day           $      1,192.00
  top 3 days              19.5% of total

  2 partial day(s) dropped at the window edges; a part-day's revenue is not comparable.
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
| 0.85 | 72% | 38.4% | \$300,916 | | 0.0 | \$418,773 | 1.38 |
| **0.90** | **81%** | **23.5%** | **\$348,221** | | **5.0** | **\$348,221** | **1.10** |
| 0.95 | 90% | 10.8% | \$400,171 | | 10.0 | \$287,150 | 1.00 |
| 1.00 | 100% | 0.0% | \$461,374 | | 20.0 | \$183,422 | 0.76 |

The plausible \$0–20 degradation range moves revenue by a factor of 2.3, so the
headline is best read as a band.

## Assumptions

| | Value | Why |
|---|---|---|
| Power rating | **50 MW** grid-side | 200 MWh at 50 MW is a four-hour battery, the duration arbitrage plants are built to and long enough to cover a full evening peak. |
| Efficiency | **90% per leg, 81% round trip** | Modern lithium-ion reaches 85–90% round trip, so this sits at the cautious end. |
| Starting SoC | **0 MWh** | An empty battery cannot sell what it never bought, so that rule holds by construction. |
| Terminal SoC | **back to 0** each day | Otherwise, a day can finish by selling inventory nobody paid for. |
| Degradation | **\$5/MWh** of throughput | Cells have a finite cycle life. Leave it out and the optimiser chases shallow cycles no warranty would cover. |
| Trading day | **04:00–03:59 AEST** | The NEM's own definition, and how settlement is counted. |
| Window | 2026-09-09 to 2026-10-08 | One archive download per day: long enough to see a distribution of daily revenue, small enough to commit. |

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

## Testing

98 tests. The ones that matter:

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

- **\$348,221 is a ceiling, not a forecast.** The whole price series is known
  in advance, so the schedule is the best one hindsight could have picked.
  `strategies/` is shaped so a forecast-driven rule drops in against the same
  simulator and is scored the same way.
- **The battery is a price taker.** 50 MW discharging into a tight peak is
  assumed never to move the price it receives, which in practice it would.
- **Every trade fills.** A real plant offers into dispatch and may not be
  taken.
- **Energy arbitrage only**, with no FCAS or capacity contracts. That
  understates the result where the two points above overstate it, by an amount
  nothing here measures.
- **Splitting the window into days costs 8%**, in exchange for per-day figures
  that can be compared with each other.
- **Standing still is free.** Degradation is charged per MWh of throughput and
  nothing else, so 150 MWh parked for a week comes out as 150 MWh.
  Self-discharge, auxiliary load and calendar ageing are all absent. The
  battery never sits on charge longer than 6.2 hours here, so the effect is
  small, but a real operator still carries it.
- **One region, one asset, no network constraints.** Marginal loss factors,
  network charges, temperature derating, outages, tax and capex are absent too,
  so the figure is gross trading revenue.

## Next steps

1. **Operate without hindsight** — a forecast-driven rule against the same
   simulator, reporting what share of this ceiling it captures and where its
   worst decisions fall. AEMO publishes a forward price curve in
   `PredispatchIS_Reports`, republished every 30 minutes and reaching 16 to
   40 hours ahead, a median of 28. 
2. **Live ingestion** from `Reports/Current`, plus vintage-correct predispatch.
   The forecast has to be the one published *at time t*; reaching for the
   latest revision is lookahead leakage in disguise.
3. **Bid curves.** A real BESS offers into dispatch and waits to be taken, so
   "charge X MWh now" stops short of what an operator actually submits.
