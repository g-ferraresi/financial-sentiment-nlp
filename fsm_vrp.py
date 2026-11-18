"""
Electric Fleet Size and Mix VRP for smart waste collection in Brescia
=====================================================================

Smart bins in 17 central districts of Brescia report their fill level every
night. A bin is collected the next morning only if its fill level reaches the
activation threshold alpha. A heterogeneous fleet of electric trucks (large and
small, each dedicated to one waste stream) leaves the depot and must return
within a 2-hour shift. The objective is to minimise total energy consumption.

The problem is decomposed by waste stream: since every vehicle serves exactly
one stream, paper, plastic and glass are three independent MIPs.

Model: Pyomo. Solver: HiGHS.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

DATA_DIR = Path(__file__).parent / "data"

DEPOT = "Depot"
WASTE_TYPES = ["paper", "plastic", "glass"]
VEHICLE_TYPES = ["small", "large"]
DAYS = ["tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "monday"]

# 17 central districts, one smart bin per waste stream in each
SELECTED_BINS = [
    "Borgo trento", "S. Eustacchio", "S. Bartolomeo", "S. Rocchino",
    "Brescia Antica", "Centro storico nord", "Centro storico sud",
    "Crocifisso di Rosa", "Porta Milano", "Porta Venezia",
    "Don Bosco", "Lamarmora", "Porta Cremona", "S. Polo Cimabue", "S. Polo Parco",
    "Fiumicello", "Primo Maggio",
]

# Vehicle-type parameters
PAYLOAD_T = {"small": 1.0, "large": 1.5}        # payload capacity (t)
BATTERY_KWH = {"small": 15.0, "large": 40.0}    # battery capacity (kWh)
CONSUMPTION_KWH_KM = {"small": 1.5, "large": 3.0}  # energy per km (kWh/km)

# Fleet: vehicle id -> (type, waste stream)
FLEETS = {
    "baseline": {
        1: ("large", "paper"), 2: ("large", "paper"), 3: ("large", "paper"), 4: ("small", "paper"),
        5: ("large", "glass"), 6: ("large", "glass"), 7: ("small", "glass"), 8: ("small", "glass"),
        9: ("large", "plastic"), 10: ("large", "plastic"), 11: ("small", "plastic"), 12: ("small", "plastic"),
    },
    # Sensitivity analysis: one large vehicle removed from each stream
    "reduced": {
        2: ("large", "paper"), 3: ("large", "paper"), 4: ("small", "paper"),
        6: ("large", "glass"), 7: ("small", "glass"), 8: ("small", "glass"),
        10: ("large", "plastic"), 11: ("small", "plastic"), 12: ("small", "plastic"),
    },
}


@dataclass
class Scenario:
    day: str = "monday"
    alpha: float = 0.50          # activation threshold (share of bin capacity)
    bin_capacity_t: float = 1.0  # capacity of each smart bin (t)
    speed_kmh: float = 30.0      # average urban speed
    service_min: float = 5.0     # service time per bin (min)
    max_route_min: float = 120.0 # shift length, 05:00-07:00 (min)
    fleet: str = "baseline"
    mip_gap: float = 0.10
    time_limit_s: int = 600
    fill_file: Path = field(default=None)

    def __post_init__(self):
        if self.fill_file is None:
            # Fill levels depend on past collections, hence on the threshold
            self.fill_file = DATA_DIR / f"fill_levels_alpha{round(self.alpha * 100):03d}.csv"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_distances(path=DATA_DIR / "distances.csv"):
    """Symmetric road distances (km) between the depot and all 33 districts."""
    df = pd.read_csv(path)
    d = {}
    for a, b, km in zip(df["from"], df["to"], df["distance_km"]):
        dist = max(float(km), 0.001)
        d[(a, b)] = dist
        d[(b, a)] = dist
    return d


def load_fill_levels(path, day):
    """Fill level (t) of each bin and waste stream on the evening before `day`."""
    df = pd.read_csv(path)
    df = df[df["day"] == day].set_index("neighbourhood")
    missing = [b for b in SELECTED_BINS if b not in df.index]
    if missing:
        raise ValueError(f"No fill data for: {missing}")
    return {(b, w): float(df.loc[b, w]) for b in SELECTED_BINS for w in WASTE_TYPES}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(w, bins, vehicles, fleet, dist, fill, sc):
    """Pyomo MIP for one waste stream w, restricted to its activated bins."""
    nodes = [DEPOT] + bins
    arcs = [(i, j) for (i, j) in dist if i in nodes and j in nodes]
    vtype = {k: fleet[k][0] for k in vehicles}

    m = pyo.ConcreteModel(name=f"FSM_VRP_{w}")
    m.N = pyo.Set(initialize=nodes)
    m.B = pyo.Set(initialize=bins)
    m.K = pyo.Set(initialize=vehicles)
    m.A = pyo.Set(initialize=arcs, dimen=2)

    travel_min = {(i, j): dist[i, j] / sc.speed_kmh * 60 for (i, j) in arcs}
    energy = {(i, j, k): CONSUMPTION_KWH_KM[vtype[k]] * dist[i, j]
              for (i, j) in arcs for k in vehicles}

    # Decision variables
    m.x = pyo.Var(m.A, m.K, domain=pyo.Binary)            # vehicle k travels arc (i, j)
    m.y = pyo.Var(m.B, m.K, domain=pyo.Binary)            # vehicle k empties bin b
    m.u = pyo.Var(m.N, m.K, domain=pyo.NonNegativeReals)  # MTZ visit order

    # Objective: total energy consumption (kWh)
    m.obj = pyo.Objective(
        expr=sum(energy[i, j, k] * m.x[i, j, k] for (i, j) in arcs for k in vehicles),
        sense=pyo.minimize)

    # C1 - Flow conservation at every bin
    m.c1_flow = pyo.Constraint(m.B, m.K, rule=lambda m, b, k:
        sum(m.x[i, b, k] for i in nodes if (i, b) in m.A)
        == sum(m.x[b, j, k] for j in nodes if (b, j) in m.A))

    # C2 - Each vehicle leaves and returns to the depot at most once (may stay idle)
    m.c2_depart = pyo.Constraint(m.K, rule=lambda m, k:
        sum(m.x[DEPOT, j, k] for j in bins) <= 1)
    m.c2_return = pyo.Constraint(m.K, rule=lambda m, k:
        sum(m.x[i, DEPOT, k] for i in bins) <= 1)

    # C3 - A vehicle can empty a bin only if it visits it
    m.c3_visit = pyo.Constraint(m.B, m.K, rule=lambda m, b, k:
        m.y[b, k] <= sum(m.x[i, b, k] for i in nodes if (i, b) in m.A))

    # C4 - Every activated bin is emptied by exactly one vehicle
    m.c4_mandatory = pyo.Constraint(m.B, rule=lambda m, b:
        sum(m.y[b, k] for k in vehicles) == 1)

    # C5 - Payload capacity
    m.c5_payload = pyo.Constraint(m.K, rule=lambda m, k:
        sum(fill[b, w] * m.y[b, k] for b in bins) <= PAYLOAD_T[vtype[k]])

    # C6 - Battery capacity
    m.c6_battery = pyo.Constraint(m.K, rule=lambda m, k:
        sum(energy[i, j, k] * m.x[i, j, k] for (i, j) in arcs) <= BATTERY_KWH[vtype[k]])

    # C7 - Route duration (travel + service) within the shift
    m.c7_duration = pyo.Constraint(m.K, rule=lambda m, k:
        sum(travel_min[i, j] * m.x[i, j, k] for (i, j) in arcs)
        + sc.service_min * sum(m.y[b, k] for b in bins) <= sc.max_route_min)

    # C8 - Subtour elimination (Miller-Tucker-Zemlin)
    n = len(bins)
    m.c8_mtz = pyo.Constraint(m.B, m.B, m.K, rule=lambda m, i, j, k:
        pyo.Constraint.Skip if i == j else
        m.u[i, k] - m.u[j, k] + n * m.x[i, j, k] <= n - 1)
    m.c8_lo = pyo.Constraint(m.B, m.K, rule=lambda m, b, k: m.u[b, k] >= 1)
    m.c8_hi = pyo.Constraint(m.B, m.K, rule=lambda m, b, k: m.u[b, k] <= n)

    m._arcs, m._energy, m._travel = arcs, energy, travel_min
    return m


def extract_routes(m, w, bins, vehicles, fleet, fill, sc):
    routes = []
    for k in vehicles:
        used = [(i, j) for (i, j) in m._arcs if pyo.value(m.x[i, j, k]) > 0.5]
        if not used:
            continue
        route, current, remaining = [DEPOT], DEPOT, list(used)
        while remaining:
            nxt = next(((i, j) for (i, j) in remaining if i == current), None)
            if nxt is None:
                break
            remaining.remove(nxt)
            current = nxt[1]
            route.append(current)
        emptied = [b for b in bins if pyo.value(m.y[b, k]) > 0.5]
        routes.append({
            "vehicle": k, "type": fleet[k][0], "waste": w,
            "route": " -> ".join(route),
            "bins": len(emptied),
            "load_t": sum(fill[b, w] for b in emptied),
            "energy_kwh": sum(m._energy[i, j, k] for (i, j) in used),
            "duration_min": sum(m._travel[i, j] for (i, j) in used) + sc.service_min * len(emptied),
        })
    return routes


def solve_day(sc: Scenario, verbose=True):
    """Solve one day: one MIP per active waste stream. Returns (summary, routes)."""
    dist = load_distances()
    fill = load_fill_levels(sc.fill_file, sc.day)
    fleet = FLEETS[sc.fleet]
    threshold = sc.alpha * sc.bin_capacity_t

    summary, routes = [], []
    for w in WASTE_TYPES:
        bins = [b for b in SELECTED_BINS if fill[b, w] >= threshold]
        vehicles = [k for k, (_, ww) in fleet.items() if ww == w]
        row = {"day": sc.day, "waste": w, "active_bins": len(bins),
               "vehicles_available": len(vehicles)}
        if not bins:
            summary.append({**row, "status": "no active bins", "energy_kwh": 0.0, "vehicles_used": 0})
            continue

        m = build_model(w, bins, vehicles, fleet, dist, fill, sc)
        solver = pyo.SolverFactory("appsi_highs")
        solver.options["time_limit"] = sc.time_limit_s
        solver.options["mip_rel_gap"] = sc.mip_gap
        res = solver.solve(m, load_solutions=False)
        cond = res.solver.termination_condition

        if cond not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
            summary.append({**row, "status": "infeasible", "energy_kwh": None, "vehicles_used": None})
            if verbose:
                print(f"[{sc.day} | {w}] {len(bins)} bins -> infeasible")
            continue

        m.solutions.load_from(res)
        r = extract_routes(m, w, bins, vehicles, fleet, fill, sc)
        for x in r:
            x["day"] = sc.day
        routes += r
        summary.append({**row, "status": str(cond), "energy_kwh": pyo.value(m.obj),
                        "vehicles_used": len(r)})
        if verbose:
            print(f"[{sc.day} | {w}] {len(bins)} bins, {len(r)} vehicles -> {pyo.value(m.obj):.2f} kWh")
    return summary, routes


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Solve the FSM-VRP for one day.")
    p.add_argument("--day", default="monday", choices=DAYS)
    p.add_argument("--alpha", type=float, default=0.50, choices=[0.50, 0.70])
    p.add_argument("--speed", type=float, default=30.0, help="average speed (km/h)")
    p.add_argument("--fleet", default="baseline", choices=list(FLEETS))
    a = p.parse_args()

    summary, routes = solve_day(Scenario(day=a.day, alpha=a.alpha, speed_kmh=a.speed, fleet=a.fleet))
    print()
    print(pd.DataFrame(routes)[["waste", "vehicle", "type", "route", "load_t", "energy_kwh", "duration_min"]]
          .round(2).to_string(index=False))
    total = sum(s["energy_kwh"] or 0 for s in summary)
    infeasible = [s["waste"] for s in summary if s["status"] == "infeasible"]
    print(f"\nTotal energy on {a.day}: {total:.2f} kWh")
    if infeasible:
        print(f"Infeasible (active bins left uncollected): {', '.join(infeasible)}")
