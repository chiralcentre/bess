# LP walkthrough

A four-interval instance of the optimiser, small enough to print every argument
`scipy.optimize.linprog` receives and every number it returns. The formulation
and the reasoning behind it are in the [README](../README.md#the-optimiser);
this page is the mechanics.

## The instance

| | |
|---|---|
| Capacity | 10 MWh |
| Power rating | 60 MW, so 5 MWh per interval each way |
| Efficiency | $\eta_c = \eta_d = 0.9$ |
| Degradation $\delta$ | \$0 |
| Start / end SoC | 0 / 0 |
| Prices | \$10, \$20, \$90, \$40 |

**1. The indexing convention.** Fixed before anything is built. Every argument
is written against it, and `result.x` comes back in it.

```
  column:    0    1    2    3    4    5    6    7    8    9   10   11
   holds:   c0   c1   c2   c3   d0   d1   d2   d3   s0   s1   s2   s3
         |------------------||------------------||------------------|
               buy (c)             sell (d)      state of charge (s)

  c_at = 0        d_at = n = 4        s_at = 2n = 8        n_vars = 3n = 12
```

**2. $c$, the objective vector.** Buy carries $p_t + \delta$, sell carries
$-p_t + \delta$, and state of charge carries nothing. Why $\delta$ lands on
both is worked through [below](#why-delta-appears-on-both-blocks).

```
         c0    c1    c2    c3     d0     d1     d2     d3    s0   s1   s2   s3
  c  : 10.0  20.0  90.0  40.0  -10.0  -20.0  -90.0  -40.0   0.0  0.0  0.0  0.0
```

**3. $A_{eq}$ and $b_{eq}$.** The balance statement
$s_t = s_{t-1} + \eta_c c_t - d_t/\eta_d$ rearranged into
$-0.9\,c_t + 1.111\,d_t - s_{t-1} + s_t = 0$, one row per interval. Row 0 has
no $s_{-1}$, so the opening level becomes `b_eq[0]`. The triple-list assembly
is unpacked [below](#how-a_eq-is-assembled).

```
           c0      c1      c2      c3      d0      d1      d2      d3      s0      s1      s2      s3   =  b_eq
  row0 -0.900       .       .       .   1.111       .       .       .   1.000       .       .       .   =  0.000
  row1      .  -0.900       .       .       .   1.111       .       .  -1.000   1.000       .       .   =  0.000
  row2      .       .  -0.900       .       .       .   1.111       .       .  -1.000   1.000       .   =  0.000
  row3      .       .       .  -0.900       .       .       .   1.111       .       .  -1.000   1.000   =  0.000
```

**4. $A_{ub}$ and $b_{ub}$.** The shared connection, $c_t + d_t \le 5$. The
vectorised assembly, and why the row is needed at all, are
[below](#how-a_ub-is-assembled).

```
           c0      c1      c2      c3      d0      d1      d2      d3      s0      s1      s2      s3  <=  b_ub
  row0  1.000       .       .       .   1.000       .       .       .       .       .       .       .  <=  5.000
  row1      .   1.000       .       .       .   1.000       .       .       .       .       .       .  <=  5.000
  row2      .       .   1.000       .       .       .   1.000       .       .       .       .       .  <=  5.000
  row3      .       .       .   1.000       .       .       .   1.000       .       .       .       .  <=  5.000
```

**5. $l$ and $u$.** Power ratings, tank limits, and the terminal condition
written as a bound with $l = u$.

```
    var    lower    upper   source
  c0..c3   0.000    5.000   power rating
  d0..d3   0.000    5.000   power rating
  s0..s2   0.000   10.000   capacity
      s3   0.000    0.000   terminal SoC, lower = upper
```

**6. Solve.**

```python
result = linprog(c=obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                 bounds=np.column_stack([lower, upper]), method="highs")
```

```
  status: Optimization terminated successfully. (HiGHS Status 7: Optimal)
  result.fun = -424.00        revenue = -result.fun = $424.00

           c0     c1     c2     c3     d0     d1     d2     d3     s0     s1     s2     s3
result.x: 5.000  5.000  0.000  0.000  0.000  0.000  5.000  3.100  4.500  9.000  3.444  0.000

  A_eq @ x  vs  b_eq          A_ub @ x  vs  b_ub          bounds
    row0  0.0 = 0.0             row0  5.000 <= 5.000        c0 c1 at upper, c2 c3 at lower
    row1  0.0 = 0.0             row1  5.000 <= 5.000        d0 d1 at lower, d2 at upper
    row2  0.0 = 0.0             row2  5.000 <= 5.000        d3 s0 s1 s2 interior
    row3  0.0 = 0.0             row3  3.100 <= 5.000        s3 at lower

  result.eqlin.marginals = [-36, -36, -36, -36]  ->  water value $36.00/MWh
```

Reading the schedule: buy 5 MWh at \$10 and 5 at \$20, taking the tank to
9.0; sell 5 at \$90 and 3.1 at \$40, returning it to 0. Revenue is
\$450 + \$124 - \$150 = \$424. What the \$36.00 refers to is
[below](#what-the-shadow-prices-mean).

## Why delta appears on both blocks

The objective being maximised separates cashflow from wear:

$$\max \sum_t \underbrace{p_t(d_t - c_t)}_{\text{cashflow}} -
\underbrace{\delta(c_t + d_t)}_{\text{wear}}$$

Negating it for a minimiser and distributing gives the two coefficients:

$$\min \sum_t (p_t + \delta)\,c_t + (-p_t + \delta)\,d_t$$

The price term flips sign between the blocks because buying costs and selling
earns. The $\delta$ term keeps its sign in both, because it is a cost either
way: $-\delta(c + d)$ becomes $+\delta(c + d)$ under negation, and $c$ and $d$
sit inside that bracket together.

Physically, degradation is charged on throughput and the cells are indifferent
to direction. Every MWh pushed in and every MWh pulled out cycles the
chemistry. Since $c$ and $d$ are grid-side, a round trip of 1 MWh bought and
0.81 MWh sold is charged $\delta(1 + 0.81) = 1.81\delta$ rather than $2\delta$;
the losses are never sold, so they are never charged wear on the way out.

$\delta$ then widens the no-trade band from both sides, raising the price below
which buying pays and lowering the price above which selling pays:

$$\text{charge while } \frac{p_t + \delta}{\eta_c} \le \lambda_t,
\qquad \text{discharge while } (p_t - \delta)\,\eta_d \ge \lambda_t$$

It does one more thing that is easy to miss. Here is a full battery, pinned so
it cannot drain, at three prices with and without wear:

```
  p=  50.0  delta= 0.0  simultaneous=0  buy= 0.000  sell= 0.000  net $   0.00
  p=  50.0  delta= 5.0  simultaneous=0  buy= 0.000  sell= 0.000  net $   0.00
  p=   0.0  delta= 0.0  simultaneous=0  buy= 6.173  sell= 5.000  net $   0.00
  p=   0.0  delta= 5.0  simultaneous=0  buy= 0.000  sell= 0.000  net $   0.00
  p= -50.0  delta= 0.0  simultaneous=2  buy= 8.287  sell= 6.713  net $  78.73
  p= -50.0  delta= 5.0  simultaneous=2  buy= 8.287  sell= 6.713  net $   3.73
```

At a zero price with no wear the battery moves 6.17 MWh for \$0. The objective
is flat there, so churning ties with idling and the solver picks arbitrarily.
Adding $\delta$ breaks the tie and the battery correctly does nothing, which
keeps the output stable across solver versions.

At -\$50 it burns energy either way, because being paid \$50 to consume beats
the loss. $\delta$ prices that rather than forbidding it, cutting the take from
\$78.73 to \$3.73.

## How A_eq is assembled

`sparse.csr_array((vals, (rows, cols)), shape=...)` takes three parallel lists.
Entry $k$ says: row `rows[k]`, column `cols[k]`, holds `vals[k]`. Anything
unmentioned is zero, so the matrix is described by listing only its non-zeros.

Each interval contributes up to four terms, one per unknown in

$$s_t - s_{t-1} - \eta_c c_t + \frac{d_t}{\eta_d} = 0$$

The loop emits three unconditionally and a fourth where a predecessor exists:

```
 k  row  col  val      meaning
 0    0    8   1.000      row 0 gets +1.000 in column s0
 1    0    0  -0.900      row 0 gets -0.900 in column c0
 2    0    4   1.111      row 0 gets +1.111 in column d0

 3    1    9   1.000      row 1 gets +1.000 in column s1
 4    1    1  -0.900      row 1 gets -0.900 in column c1
 5    1    5   1.111      row 1 gets +1.111 in column d1
 6    1    8  -1.000      row 1 gets -1.000 in column s0     <- the t > 0 entry

 7    2   10   1.000      row 2 gets +1.000 in column s2
 8    2    2  -0.900      row 2 gets -0.900 in column c2
 9    2    6   1.111      row 2 gets +1.111 in column d2
10    2    9  -1.000      row 2 gets -1.000 in column s1

11    3   11   1.000      row 3 gets +1.000 in column s3
12    3    3  -0.900      row 3 gets -0.900 in column c3
13    3    7   1.111      row 3 gets +1.111 in column d3
14    3   10  -1.000      row 3 gets -1.000 in column s2
```

The lists stay in lockstep. `cols` holds `[s_at+t, c_at+t, d_at+t]` while
`vals` holds `[1.0, -eta_c, 1/eta_d]`, so position 0 pairs $s_t$ with $+1$,
position 1 pairs $c_t$ with $-\eta_c$, and position 2 pairs $d_t$ with
$+1/\eta_d$. The offsets are what turn "the $c$ of interval $t$" into a column
number.

Row 0 is short because $s_{-1}$ is not a variable. The opening level is a known
constant, so it cannot sit on the left with the unknowns and moves to the
right:

$$s_0 - \eta_c c_0 + \frac{d_0}{\eta_d} = s_{\text{init}}$$

which is `b_eq[0] = spec.initial_soc_mwh`. Every other entry of `b_eq` stays
zero, and starting the battery part-full changes only that one number.

The triples earn their place at full size. A day's $A_{eq}$ is 288 x 864 =
248,832 entries of which 1,151 are non-zero, or 0.46%. Building it densely
would store a quarter of a million numbers to express a thousand facts, and
every factorisation inside the solver would walk the zeros.

Structurally the $-1$ sub-diagonal is the important part: it is the only thing
linking one interval to the next. Without it each row would stand alone and
there would be no inventory to carry forward.

## How A_ub is assembled

Same triple format as $A_{eq}$, but the pattern is uniform: every row holds
exactly two entries, both $1.0$, at predictable columns, with no special case
for the first interval. That is what lets it be written as array arithmetic
rather than a loop.

```
idx                        = [0 1 2 3]
np.concatenate([idx, idx]) = [0 1 2 3 0 1 2 3]    <- row indices
c_at + idx                 = [0 1 2 3]
d_at + idx                 = [4 5 6 7]
concatenated               = [0 1 2 3 4 5 6 7]    <- column indices
np.ones(2*n)               = [1 1 1 1 1 1 1 1]
```

Read down in lockstep:

```
 k  row  col  val   meaning
 0    0    0   1.0   row 0 gets 1.0 in column c0
 1    1    1   1.0   row 1 gets 1.0 in column c1
 2    2    2   1.0   row 2 gets 1.0 in column c2
 3    3    3   1.0   row 3 gets 1.0 in column c3
 4    0    4   1.0   row 0 gets 1.0 in column d0     <- second pass, same rows
 5    1    5   1.0   row 1 gets 1.0 in column d1
 6    2    6   1.0   row 2 gets 1.0 in column d2
 7    3    7   1.0   row 3 gets 1.0 in column d3
```

The rows repeat because each needs two entries; the columns run straight
through because `c_at + idx` and `d_at + idx` are the two blocks offset.
`c_at` is zero so that term changes nothing, and it is kept for symmetry with
`d_at + idx`, naming which block each half addresses instead of leaving it
implied.

```
          c0    c1    c2    c3    d0    d1    d2    d3    s0    s1    s2    s3
 row0   1.0     .     .     .   1.0     .     .     .     .     .     .     .
 row1     .   1.0     .     .     .   1.0     .     .     .     .     .     .
 row2     .     .   1.0     .     .     .   1.0     .     .     .     .     .
 row3     .     .     .   1.0     .     .     .   1.0     .     .     .     .
```

Row $t$ reads $c_t + d_t \le b_{ub}[t]$, and the state-of-charge block is
empty: this constraint says nothing about the tank. The limit itself is
`max(max_charge_mwh, max_discharge_mwh)`, since one physical connection carries
flow both ways and has to be sized for the larger rating. It is the same number
every interval, hence `np.full`.

### Why the row exists

$c_t$ and $d_t$ already carry individual upper bounds of 4.167 MWh, so a row
capping their sum at 4.167 looks redundant. Bounds are independent, though:
nothing in them stops $c_t$ and $d_t$ both reaching 4.167 at once, which is
8.33 MWh through a 4.167 MWh connection.

Here is a full battery at -\$200/MWh, where burning energy pays, solved with
and without the row:

```
WITH    A_ub:  revenue $   299.92
    t=0  c=2.762  d=2.238  c+d= 5.000  within rating
    t=1  c=0.525  d=4.475  c+d= 5.000  within rating
    t=2  c=5.000  d=0.000  c+d= 5.000  within rating

WITHOUT A_ub:  revenue $   542.85
    t=0  c=5.000  d=4.050  c+d= 9.050  OVER the 5.0 MWh connection
    t=1  c=5.000  d=4.050  c+d= 9.050  OVER the 5.0 MWh connection
    t=2  c=5.000  d=4.050  c+d= 9.050  OVER the 5.0 MWh connection
```

Without it the LP pushes 9.05 MWh through a 5 MWh connection and books 81% more
revenue than physically exists. The simulator would clip the schedule back and
the reported figure would be unreachable.

That also explains the shape of the constraint: it only does work when both
legs are positive. While the battery is purely buying, $c_t + d_t \le 4.167$ is
already implied by $c_t \le 4.167$. The row exists to govern the
simultaneous-flow case, which the model permits rather than forbids and prices
with $\delta$.

### Why an inequality, in its own matrix

An inequality, because the battery is under no obligation to fill the
connection, and on most intervals it uses none of it.

A separate matrix, because `linprog` keeps the two apart and so do their duals.
$A_{eq}$ rows produce `result.eqlin.marginals`, which become the water value;
$A_{ub}$ rows produce `result.ineqlin.marginals`, which this code does not
read.

## What the shadow prices mean

The solver returns one extra number per equality row, and it is easy to read
past without knowing what it counts.

Start with the row itself. Written out, row $t$ says

$$s_t = s_{t-1} + \eta_c c_t - \frac{d_t}{\eta_d} + b_{eq}[t]$$

so `b_eq[t]` is a quantity of energy added to the tank at interval $t$ without
anyone buying it. It is normally zero, because energy only arrives through the
meter.

The shadow price is what happens to the day's profit if you raise that number.
In plain terms:

> **Give the battery one free MWh of stored energy at interval $t$. The day
> earns $\lambda_t$ more dollars.**

That is the whole definition. The units are dollars per MWh because the profit
is in dollars and the gift is in MWh. Row 0 is the same statement in more
familiar clothing, since `b_eq[0]` is the opening state of charge: raising it
means starting the day with more energy in the tank.

### Checking it

Take a case where the number varies. A 6 MWh tank over six intervals fills up
at $t = 1$, which splits the day in two:

```
baseline revenue $854.32
lambda = [22.22 22.22 61.11 61.11 61.11 61.11]

  t  lambda_t  gain for +0.001   gain/eps   match
  0    22.222          0.02222     22.222     yes
  1    22.222          0.02222     22.222     yes
  2    61.111          0.06111     61.111     yes
  3    61.111          0.06111     61.111     yes
  4    61.111          0.06111     61.111     yes
  5    61.111          0.06111     61.111     yes
```

Each row predicts its own gain exactly. A free MWh before the bottleneck is
worth \$22.22 and the same MWh after it is worth \$61.11, because energy on the
early side of a full tank cannot reach the later peak.

### Why the sign is flipped in the code

The objective handed to the solver is the negative of revenue, so the raw
marginals are the rate of change of *that*: -36 rather than +36. Negating them
gives the rate of change of revenue, which is the number a trader wants.

A positive value means stored energy is an asset. It can come back negative,
and across the real backtest it ranges from -\$3.74 to \$265.49. A negative
value means the battery is full while prices are below zero, so the energy it
is holding has become a liability.

### Why the equality rows in particular

`result.eqlin.marginals` carries one entry per row of $A_{eq}$, in row order,
so it lines up with the intervals without any slicing. The parallel
`result.ineqlin.marginals` holds the duals of the $A_{ub}$ rows, which this
code does not read.

This is also why the balance constraint is written as an equality. Only
equality rows produce this dual, and it is the whole basis of the
explainability layer.

### It is a rate, not a total

Push the gift far enough and the answer stops scaling:

```
  eps=  0.1  gain     2.222  gain/eps   22.222   linear
  eps=  1.0  gain    22.222  gain/eps   22.222   linear
  eps=  3.0  gain    50.000  gain/eps   16.667   basis changed
  eps=  6.0  gain    83.333  gain/eps   13.889   basis changed
  eps= 10.0  gain   119.333  gain/eps   11.933   basis changed
```

Up to about 1 MWh the prediction is exact. Beyond that the battery runs into
other limits and the marginal value falls away, which is ordinary diminishing
returns. $\lambda_t$ answers what the *next* MWh is worth, not what ten more
would be worth.

### What it is used for

A marginal value of stored energy is a reservation price, so the whole schedule
collapses to one comparison per interval: buy below it, sell above it, hold in
between. That is what the trade log quotes.

> *Bought 4.2 MWh over 5 minutes at a volume-weighted \$11.32/MWh, below the
> \$47.15 the battery valued stored energy at.*
