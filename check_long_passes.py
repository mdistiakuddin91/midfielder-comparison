#!/usr/bin/env python3
"""Check image and counts for the "Long passes completed" metric.

Draws every completed pass longer than 30 yards by Pedri at the 2022 World Cup as an arrow on a pitch
(output/check_pedri_long_passes.png) and prints, for Pedri, Frenkie de Jong and Sergio Busquets, how many
there are, their average length and how many are switches of play (StatsBomb's pass_switch flag).

The filter is the one the pizza charts use for "Long passes completed" (completed pass, longer than
30 yards, periods 1-4, set pieces included), so the counts should equal the totals behind the charts;
that is checked at the end. This script does not touch the charts or midfielder_comparison.py.

Usage:  python check_long_passes.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mplsoccer import Pitch
from statsbombpy import sb

import midfielder_comparison as mc  # same data loading, player lookup and colours as the pizza charts

LONG_PASS_YARDS = 30  # StatsBomb pass_length is in yards; same threshold as the charts
MAP_PLAYER = "Pedri"  # whose passes are drawn
OUTPUT = mc.DEFAULT_OUTPUT_DIR / "check_pedri_long_passes.png"
LONG_COLOR = mc.GROUP_COLORS["Passing"]  # long passes are a passing metric: same blue as in the charts
SWITCH_COLOR = "#eb6834"                 # orange: the next colour in the same palette, clearly different
PASS_COLUMNS = ["period", "player_id", "location", "pass_end_location", "pass_length",
                "pass_outcome", "pass_type", "pass_switch"]


def fetch_passes(match_ids, player_ids) -> pd.DataFrame:
    """Every Pass event by `player_ids` in `match_ids`, straight from StatsBomb.

    The pizza script's cache does not keep pass_switch, so these few matches are downloaded in full.
    """
    def one(match_id):
        events = sb.events(match_id=match_id)
        passes = events[(events["type"] == "Pass") & events["player_id"].isin(player_ids)]
        return passes.reindex(columns=PASS_COLUMNS).assign(match_id=match_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return pd.concat(pool.map(one, match_ids), ignore_index=True)


def draw_map(mine: pd.DataFrame, name: str, team: str):
    """Pitch map: one arrow per pass, orange for switches of play, blue for the other long passes."""
    n_switch = int(mine["switch"].sum())
    fig_w, fig_h = 9.0, 8.1
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=mc.SURFACE)
    ax = fig.add_axes([0.0, 1.6 / fig_h, 1.0, 5.4 / fig_h])  # the pitch is centred in this box
    pitch = Pitch(pitch_type="statsbomb", pitch_color=mc.SURFACE, line_color=mc.BASELINE,
                  linewidth=1.4, line_zorder=1, goal_type="box")
    pitch.draw(ax=ax)

    # other long passes first, switches on top so they are never hidden
    for is_switch, color, z in [(False, LONG_COLOR, 3), (True, SWITCH_COLOR, 4)]:
        part = mine[mine["switch"] == is_switch]
        if part.empty:
            continue
        start = np.array(part["location"].tolist(), dtype=float)[:, :2]
        end = np.array(part["pass_end_location"].tolist(), dtype=float)[:, :2]
        pitch.arrows(start[:, 0], start[:, 1], end[:, 0], end[:, 1], ax=ax, width=2.2, headwidth=3.6,
                     headlength=4.2, headaxislength=3.8, color=color, alpha=0.85, zorder=z)
        pitch.scatter(start[:, 0], start[:, 1], ax=ax, s=22, color=color, edgecolors=mc.SURFACE,
                      linewidth=0.8, zorder=z)  # a dot where each pass started

    fig.text(0.5, 7.62 / fig_h, f"{name}: completed passes longer than {LONG_PASS_YARDS} yards",
             ha="center", va="center", fontsize=19, fontweight="bold", color=mc.INK)
    fig.text(0.5, 7.22 / fig_h,
             f"{team}  ·  2022 World Cup  ·  {len(mine)} passes  ·  average length "
             f"{mine['pass_length'].mean():.1f} yards  ·  {n_switch} switches of play",
             ha="center", va="center", fontsize=11.5, color=mc.INK_2)
    fig.text(0.5, 1.37 / fig_h, "Attacking direction  →", ha="center", va="center", fontsize=10.5,
             color=mc.INK_2)
    handles = [Line2D([0], [0], color=LONG_COLOR, lw=3), Line2D([0], [0], color=SWITCH_COLOR, lw=3)]
    labels = [f"Long pass ({len(mine) - n_switch})", f"Switch of play, StatsBomb pass_switch flag ({n_switch})"]
    fig.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=11, labelcolor=mc.INK,
               handlelength=2.4, columnspacing=2.4, bbox_to_anchor=(0.5, 0.92 / fig_h))
    fig.text(0.5, 0.4 / fig_h, "Data: StatsBomb  ·  dots mark where each pass started  ·  "
             "set pieces included", ha="center", va="center", fontsize=10, color=mc.INK_2)
    return fig


def main():
    matches, events, lineups = mc.load_tournament()
    squad = lineups.drop_duplicates("player_id").set_index("player_id")[["team", "player_name", "player_nickname"]]
    targets = [(label, mc.find_player(squad, key, team), team) for label, key, team in mc.TARGET_PLAYERS]

    teams = {team for _, _, team in targets}
    match_ids = matches.loc[matches["home_team"].isin(teams) | matches["away_team"].isin(teams), "match_id"].tolist()
    print(f"Downloading {len(match_ids)} {' / '.join(sorted(teams))} matches (pass_switch is not in the cache) ...")
    passes = fetch_passes(match_ids, [player_id for _, player_id, _ in targets])

    # the same test as "Long passes completed" in the charts: completed (no outcome), longer than 30 yards
    long_passes = passes[(passes["period"] <= 4) & passes["pass_outcome"].isna()
                         & (passes["pass_length"] > LONG_PASS_YARDS)].copy()
    long_passes["switch"] = long_passes["pass_switch"].eq(True)

    chart_totals = mc.compute_event_totals(events)["long_passes_completed"]
    print(f"\nCompleted passes longer than {LONG_PASS_YARDS} yards, 2022 World Cup")
    print(f"  {'Player':<17}{'Passes':>7}{'Avg length (yards)':>20}{'Switches of play':>19}"
          f"{'Chart total':>13}{'Agrees':>8}")
    for label, player_id, team in targets:
        mine = long_passes[long_passes["player_id"] == player_id]
        n_switch = int(mine["switch"].sum())
        agrees = "yes" if len(mine) == chart_totals[player_id] else "NO"
        print(f"  {label:<17}{len(mine):>7}{mine['pass_length'].mean():>20.1f}"
              f"{f'{n_switch} ({n_switch / len(mine):.0%})':>19}{int(chart_totals[player_id]):>13}{agrees:>8}")
    print("  (Chart total = 'Long passes completed' total behind the pizza charts, counted separately.)")

    player_id = next(pid for label, pid, _ in targets if label == MAP_PLAYER)
    team = next(team for label, _, team in targets if label == MAP_PLAYER)
    fig = draw_map(long_passes[long_passes["player_id"] == player_id], MAP_PLAYER, team)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor=mc.SURFACE)
    plt.close(fig)
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()