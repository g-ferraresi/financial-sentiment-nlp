"""
Runs the weekly baseline and the sensitivity scenarios, and writes
results/*.csv and results/*.png.

    python run_experiments.py              # baseline + all sensitivity scenarios
    python run_experiments.py --baseline   # baseline week only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fsm_vrp import DAYS, WASTE_TYPES, Scenario, solve_day

OUT = Path(__file__).parent / "results"

SCENARIOS = {
    "baseline":       dict(),
    "alpha_0.70":     dict(alpha=0.70),
    "speed_20kmh":    dict(speed_kmh=20.0),
    "reduced_fleet":  dict(fleet="reduced"),
}


def run_week(name, **params):
    summary, routes = [], []
    for day in DAYS:
        s, r = solve_day(Scenario(day=day, **params))
        summary += s
        routes += r
    summary = pd.DataFrame(summary).assign(scenario=name)
    routes = pd.DataFrame(routes).assign(scenario=name)
    return summary, routes


def plot_baseline(summary, path):
    df = (summary.pivot(index="day", columns="waste", values="energy_kwh")
          .reindex(DAYS)[WASTE_TYPES].fillna(0))
    colors = {"paper": "#2a78d6", "plastic": "#eda100", "glass": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    bottom = None
    for w in WASTE_TYPES:
        ax.bar(df.index.str.capitalize(), df[w], bottom=bottom, label=w.capitalize(),
               color=colors[w], width=0.6, edgecolor="white", linewidth=1.5)
        bottom = df[w] if bottom is None else bottom + df[w]
    for i, tot in enumerate(df.sum(axis=1)):
        ax.text(i, tot + 3, f"{tot:.0f}", ha="center", fontsize=9, color="#52514e")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(f"Daily energy consumption by waste stream - baseline week "
                 f"({df.values.sum():.0f} kWh)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e8e7e3"); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true", help="run the baseline week only")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)

    names = ["baseline"] if args.baseline else list(SCENARIOS)
    all_summary, all_routes = [], []
    for name in names:
        print(f"\n=== Scenario: {name} ===")
        s, r = run_week(name, **SCENARIOS[name])
        all_summary.append(s)
        all_routes.append(r)

    summary = pd.concat(all_summary, ignore_index=True)
    routes = pd.concat(all_routes, ignore_index=True)
    summary.round(3).to_csv(OUT / "daily_summary.csv", index=False)
    routes.round(3).to_csv(OUT / "routes.csv", index=False)

    base = summary[summary.scenario == "baseline"]
    plot_baseline(base, OUT / "baseline_week.png")

    # Scenario comparison. Energy is summed over feasible sub-problems only:
    # with infeasible days it is not comparable with the baseline.
    comp = (summary.groupby("scenario", sort=False)
            .agg(energy_feasible_kwh=("energy_kwh", "sum"),
                 infeasible_subproblems=("status", lambda s: (s == "infeasible").sum()))
            .round(2))
    comp.to_csv(OUT / "scenario_comparison.csv")
    print("\n", comp.to_string())
