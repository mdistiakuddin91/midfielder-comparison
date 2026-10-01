#!/usr/bin/env python3
"""Check image and counts for the "Ball recoveries" metric: Pedri vs Busquets.

Draws every ball recovery by Pedri and by Sergio Busquets at the 2022 World Cup as one dot on a pitch
(output/check_recoveries.png, two maps side by side) and prints how many happened in each third of the
pitch and how far from their own goal they were on average.

A recovery is the pizza charts' definition: a Ball Recovery event, periods 1-4, leaving out the ones
StatsBomb flags as a recovery failure. The counts are checked against the totals behind the charts.
This script does not touch the charts or midfielder_comparison.py.

Usage:  python check_recoveries.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

import midfielder_comparison as mc  # same data loading, player lookup and colours as the pizza charts

PLAYERS = ["Pedri", "Sergio Busquets"]
OUTPUT = mc.DEFAULT_OUTPUT_DIR / "check_recoveries.png"
DOT_COLOR = mc.GROUP_COLORS["Defending"]  # ball recoveries are a defending metric: same aqua as the charts
# Thirds of the 120-yard StatsBomb pitch; every team attacks towards x = 120, own goal is at x = 0.
THIRDS = [("Defensive third", 0, 40), ("Middle third", 40, 80), ("Attacking third", 80, 120.01)]
OWN_GOAL = (0, 40)  # centre of the goal the player's team defends


def recoveries_of(events: pd.DataFrame, player_id: int) -> pd.DataFrame:
    """Ball Recovery events by the player, as in the charts: periods 1-4, no flagged recovery failures."""
    ev = events[(events["type"] == "Ball Recovery") & (events["player_id"] == player_id)
                & (events["period"] <= 4) & ~events["ball_recovery_recovery_failure"].eq(True)]
    xy = np.array([loc[:2] for loc in ev["location"]], dtype=float)
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})


def draw_maps(data: dict) -> plt.Figure:
    fig_w, fig_h = 17.0, 7.4
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=mc.SURFACE)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=mc.SURFACE, line_color=mc.BASELINE,
                  linewidth=1.4, line_zorder=1, goal_type="box")
    for i, (name, (team, rec)) in enumerate(data.items()):
        ax = fig.add_axes([i * 0.5, 1.5 / fig_h, 0.5, 4.6 / fig_h])
        pitch.draw(ax=ax)
        for _, lo, _hi in THIRDS[1:]:  # faint third lines
            ax.plot([lo, lo], [0, 80], color=mc.GRID, lw=1.2, ls="-", zorder=1.5)
        # 2 px surface ring so overlapping dots stay countable
        pitch.scatter(rec["x"], rec["y"], ax=ax, s=70, color=DOT_COLOR, alpha=0.8,
                      edgecolors=mc.SURFACE, linewidth=1.0, zorder=3)
        centre = (i + 0.5) / 2
        fig.text(centre, 6.95 / fig_h, name, ha="center", va="center", fontsize=19, fontweight="bold", color=mc.INK)
        counts = [int(((rec["x"] >= lo) & (rec["x"] < hi)).sum()) for _, lo, hi in THIRDS]
        fig.text(centre, 6.55 / fig_h, f"{team}  ·  {len(rec)} ball recoveries  ·  "
                 f"{counts[0]} defensive / {counts[1]} middle / {counts[2]} attacking third",
                 ha="center", va="center", fontsize=11.5, color=mc.INK_2)
    fig.text(0.5, 1.25 / fig_h, "Attacking direction  →", ha="center", va="center", fontsize=10.5, color=mc.INK_2)
    fig.text(0.5, 0.8 / fig_h, "One dot per Ball Recovery event (failed recoveries left out), 2022 World Cup, "
             "extra time included. Faint lines mark the thirds.", ha="center", va="center", fontsize=10, color=mc.INK_2)
    fig.text(0.5, 0.4 / fig_h, "Data: StatsBomb", ha="center", va="center", fontsize=10, color=mc.INK_2)
    return fig


def main():
    matches, events, lineups = mc.load_tournament()
    squad = lineups.drop_duplicates("player_id").set_index("player_id")[["team", "player_name", "player_nickname"]]
    chart_totals = mc.compute_event_totals(events)["ball_recoveries"]

    data, ids = {}, {}
    for label, key, team in mc.TARGET_PLAYERS:
        if label in PLAYERS:
            ids[label] = mc.find_player(squad, key, team)
            data[label] = (team, recoveries_of(events, ids[label]))

    print("Ball recoveries by third of the pitch, 2022 World Cup")
    print(f"  {'Player':<17}{'Total':>6}" + "".join(f"{name:>18}" for name, _, _ in THIRDS)
          + f"{'Avg dist. from own goal':>26}{'Chart total':>13}{'Agrees':>8}")
    for name, (team, rec) in data.items():
        counts = [int(((rec["x"] >= lo) & (rec["x"] < hi)).sum()) for _, lo, hi in THIRDS]
        dist = np.hypot(rec["x"] - OWN_GOAL[0], rec["y"] - OWN_GOAL[1])
        cells = "".join(f"{f'{c} ({c / len(rec):.0%})':>18}" for c in counts)
        agrees = "yes" if len(rec) == chart_totals[ids[name]] else "NO"
        print(f"  {name:<17}{len(rec):>6}{cells}{dist.mean():>20.1f} yards{int(chart_totals[ids[name]]):>13}{agrees:>8}")
    print("  (Thirds: defensive x < 40, middle 40-80, attacking x >= 80 on the 120-yard pitch. Distance is to the "
          "centre of the player's own goal.)")

    fig = draw_maps(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor=mc.SURFACE)
    plt.close(fig)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
