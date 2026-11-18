# Electric vehicle routing for smart waste collection in Brescia

**Fleet Size and Mix Vehicle Routing Problem (FSM-VRP) with electric trucks and smart bins**, built on real 2024 waste-generation data for the city of Brescia (Italy).

Joint project with Alessia Lecchi and Giacomo Pagani for the course *Optimization Methods for Business Analytics*, MSc in Analytics and Data Science for Economics and Management, University of Brescia (2026). All authors contributed to every stage.

![Baseline week](results/baseline_week.png)

## The problem

Traditional waste collection follows a fixed schedule: trucks visit every bin, full or not. With **smart bins** that report their fill level each night, a bin can be collected only when it actually needs to be.

- **17 central districts** of Brescia (≈124,000 inhabitants), one smart bin per stream in each: **51 bins** for paper, plastic and glass.
- A bin is collected the next morning only if its fill level reaches the **activation threshold α = 50%**.
- **12 electric trucks** of two types, each dedicated to one stream (paper: 3 large + 1 small; plastic and glass: 2 large + 2 small).
- All routes start and end at the real depot (Aprica facility, Villaggio Sereno) within a **2-hour shift** (05:00–07:00).
- **Objective:** minimise total energy consumption (kWh).

| | Small truck | Large truck |
|---|---|---|
| Payload | 1.0 t | 1.5 t |
| Battery | 15 kWh | 40 kWh |
| Consumption | 1.5 kWh/km | 3.0 kWh/km |

## The model

A mixed-integer linear program, written in **Pyomo** and solved with **HiGHS**. Because every truck serves a single stream, each day is **decomposed into three independent MIPs** (paper, plastic, glass), which keeps the problem tractable.

**Decision variables:** $x_{ijk} \in \{0,1\}$ if truck $k$ travels arc $(i,j)$; $y_{bk} \in \{0,1\}$ if truck $k$ empties bin $b$; $u_{ik} \ge 0$ visit order (subtour elimination).

**Objective:**

$$\min \sum_{(i,j) \in A} \sum_{k \in K} e_{ijk}\, x_{ijk}, \qquad e_{ijk} = c_{m(k)} \, d_{ij}$$

**Constraints:**

| | Constraint |
|---|---|
| C1 | Flow conservation at every bin |
| C2 | Each truck leaves and returns to the depot at most once (it may stay idle) |
| C3 | A truck can empty a bin only if it visits it |
| C4 | Every activated bin is emptied by exactly one truck |
| C5 | Payload capacity: $\sum_b q_b\, y_{bk} \le C_{m(k)}$ |
| C6 | Battery capacity: $\sum_{(i,j)} e_{ijk}\, x_{ijk} \le B_{m(k)}$ |
| C7 | Route duration (travel + 5 min service per bin) ≤ 120 min |
| C8 | Subtour elimination (Miller–Tucker–Zemlin) |

The full formulation is in the [project presentation](docs/presentation.pdf).

## Data

| File | Content | Source |
|---|---|---|
| [`data/distances.csv`](data/distances.csv) | Road distances (km) between the depot and all 33 districts | Real road network (collected by the authors) |
| [`data/fill_levels_alpha050.csv`](data/fill_levels_alpha050.csv) | Fill level (t) of every bin, by day and stream, with α = 50% | Daily waste generation per district from ISPRA 2024 and Comune di Brescia population data |
| [`data/fill_levels_alpha070.csv`](data/fill_levels_alpha070.csv) | Same, with α = 70% | As above |

Fill levels accumulate day by day and drop when a bin is emptied, so they depend on the threshold: that is why each α has its own file.

## Results

**Baseline week (α = 50%): 747 kWh.** Monday is the most demanding day (27% of the weekly total, all three streams active), Friday the lightest (4%). Paper accounts for 45% of the weekly energy, plastic for 30% and glass for 25%, which supports giving the paper fleet three large trucks instead of two.

**Monday in detail:** 11 trucks serve 17 bin collections across 11 districts, for **204.4 kWh**.

| Stream | Truck | Route | Load (t) | Energy (kWh) | Time (min) |
|---|---|---|---|---|---|
| Paper | Large | Depot → Crocifisso di Rosa → Centro storico nord → Depot | 1.18 | 24.5 | 26 |
| Paper | Large | Depot → Porta Cremona → Lamarmora → Depot | 1.49 | 12.4 | 18 |
| Paper | Small | Depot → Porta Venezia → Depot | 0.71 | 14.9 | 25 |
| Plastic | Large | Depot → Fiumicello → Porta Milano → Depot | 1.15 | 23.3 | 26 |
| Plastic | Large | Depot → S. Bartolomeo → Porta Venezia → Depot | 1.06 | 38.6 | 36 |
| Plastic | Small | Depot → S. Polo Parco → Depot | 0.55 | 8.8 | 17 |
| Plastic | Small | Depot → Crocifisso di Rosa → Depot | 0.51 | 12.1 | 21 |
| Glass | Large | Depot → Fiumicello → Porta Milano → Depot | 1.19 | 23.3 | 26 |
| Glass | Large | Depot → Porta Venezia → S. Rocchino → Depot | 1.06 | 29.8 | 30 |
| Glass | Small | Depot → Centro storico sud → Depot | 0.58 | 7.8 | 15 |
| Glass | Small | Depot → S. Polo Parco → Depot | 0.53 | 8.8 | 17 |

All routes for all days and scenarios are in [`results/routes.csv`](results/routes.csv).

### Sensitivity analysis

| Scenario | Infeasible sub-problems (out of 21) | Finding |
|---|---|---|
| Baseline | 0 | 747 kWh per week |
| Threshold α = 70% | 3 | Fewer, fuller collections on most days, but bins pile up: paper becomes infeasible on Thursday, Saturday and Monday (6–11 bins at once) |
| Congestion (20 km/h instead of 30) | 0 | Same routes and energy: the 2-hour window is never binding (longest route 49 min) |
| One large truck fewer per stream | 6 | Infeasible on the busiest days (Thursday paper, Saturday paper and plastic, Sunday glass, Monday plastic and glass) |

A higher threshold is therefore not unconditionally better, and fleet size, not shift length, is the binding resource on peak days.

### Limitations

- Energy consumption does not depend on the truck's load, and speed is constant on all roads.
- Each truck makes a single trip per shift, with no recharging.
- The analysis covers 17 central districts out of 33 and uses a 10% relative MIP gap to limit computation time.

## How to run

```bash
pip install -r requirements.txt

python fsm_vrp.py --day monday                       # one day, baseline
python fsm_vrp.py --day saturday --alpha 0.70        # one day, other threshold
python fsm_vrp.py --day monday --fleet reduced       # one day, reduced fleet
python run_experiments.py                            # full week, all scenarios (a few minutes)
```

`run_experiments.py` writes `results/daily_summary.csv`, `results/routes.csv`, `results/scenario_comparison.csv` and the figure above. HiGHS is installed with `highspy`, so no external solver is needed.

## Repository structure

```
├── fsm_vrp.py            # data loading, Pyomo model, solver, route extraction
├── run_experiments.py    # baseline week + sensitivity scenarios
├── data/                 # distances and fill levels (CSV)
├── results/              # outputs and figure
├── docs/presentation.pdf # project presentation with the full formulation
└── requirements.txt
```
