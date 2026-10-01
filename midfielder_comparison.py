#!/usr/bin/env python3
"""Compare Pedri, Frenkie de Jong and Sergio Busquets with pizza charts.

Data: StatsBomb open data for the 2022 FIFA World Cup (statsbombpy). Charts: mplsoccer PyPizza.

Steps
    1. Download every match (events + lineups); cached in .cache/ so later runs are instant.
    2. Work out minutes played per player and position from the lineups (extra time included).
    3. Keep players whose main position was central midfield and who played 270+ minutes.
    4. Calculate 12 per-90 metrics for that group and turn each into a percentile rank.
    5. Save one pizza chart per player and one side-by-side image, then print a check table.

Usage:  python midfielder_comparison.py [--refresh] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # we only write image files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from mplsoccer import PyPizza
from scipy.stats import percentileofscore
from statsbombpy import sb

warnings.filterwarnings("ignore", message="credentials were not supplied")  # open data needs none

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMPETITION_ID, SEASON_ID = 43, 106  # FIFA World Cup, 2022
MIN_MINUTES = 270

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "output"
CACHE_FILE = HERE / ".cache" / "wc2022_statsbomb.pkl"
CACHE_VERSION = 1  # bump when EVENT_TYPES / EVENT_COLUMNS change

# (name shown on the charts, text to look for in the lineup name, team)
TARGET_PLAYERS = [
    ("Pedri", "pedri", "Spain"),
    ("Frenkie de Jong", "frenkie de jong", "Netherlands"),
    ("Sergio Busquets", "busquets", "Spain"),
]

# Every StatsBomb position name, mapped to a broad role. A player's *main position* is the role
# in which he played the most minutes, so a midfielder who rotates between "Left Center
# Midfield" and "Center Defensive Midfield" is still a central midfielder.
# Central midfield = defensive, centre and attacking midfield (left/right/centre variants).
# Wingers and wide midfielders ("Left/Right Wing", "Left/Right Midfield") are NOT included.
CENTRAL_MIDFIELD = "Central midfield"
POSITION_GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Right Back": "Defender", "Right Center Back": "Defender", "Center Back": "Defender",
    "Left Center Back": "Defender", "Left Back": "Defender",
    "Right Wing Back": "Defender", "Left Wing Back": "Defender",
    "Right Defensive Midfield": CENTRAL_MIDFIELD, "Center Defensive Midfield": CENTRAL_MIDFIELD,
    "Left Defensive Midfield": CENTRAL_MIDFIELD,
    "Right Center Midfield": CENTRAL_MIDFIELD, "Center Midfield": CENTRAL_MIDFIELD,
    "Left Center Midfield": CENTRAL_MIDFIELD,
    "Right Attacking Midfield": CENTRAL_MIDFIELD, "Center Attacking Midfield": CENTRAL_MIDFIELD,
    "Left Attacking Midfield": CENTRAL_MIDFIELD,
    "Right Midfield": "Wide", "Left Midfield": "Wide", "Right Wing": "Wide", "Left Wing": "Wide",
    "Right Center Forward": "Forward", "Left Center Forward": "Forward", "Center Forward": "Forward",
    "Striker": "Forward", "Secondary Striker": "Forward",
}

# (key, label on the chart, group). The order is the order of the slices, clockwise from the top.
METRICS = [
    ("passes_completed", "Passes\ncompleted", "Passing"),
    ("pass_accuracy", "Pass\naccuracy %", "Passing"),
    ("progressive_passes", "Progressive\npasses", "Passing"),
    ("long_passes_completed", "Long passes\ncompleted", "Passing"),
    ("key_passes", "Key\npasses", "Creating"),
    ("xa", "Expected\nassists (xA)", "Creating"),
    ("passes_final_third", "Passes into\nfinal third", "Creating"),
    ("dribbles_completed", "Successful\ndribbles", "Creating"),
    ("pressures", "Pressures", "Defending"),
    ("tackles_interceptions", "Tackles +\ninterceptions", "Defending"),
    ("ball_recoveries", "Ball\nrecoveries", "Defending"),
    ("aerial_duels_won", "Aerial\nduels won", "Defending"),
]
METRIC_KEYS = [key for key, _, _ in METRICS]

# Chart colours. The three group colours (blue, orange, aqua) were checked to stay distinguishable
# from one another for colour-blind readers, for every pair of the three. Aqua is a little pale
# against the background (2.7:1), so the numbers sit in white pills with dark text and every
# slice carries its value instead of relying on colour alone.
GROUP_COLORS = {"Passing": "#2a78d6", "Creating": "#eb6834", "Defending": "#1baf7a"}
SURFACE, INK, INK_2, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
NOTE = "Compared with 2022 World Cup central midfielders, 270+ minutes, per 90"

# Only these event types / columns are kept after downloading (the full feed is ~100 columns wide).
EVENT_TYPES = {
    "Pass", "Dribble", "Pressure", "Duel", "Interception", "Ball Recovery", "Clearance",
    "Miscontrol", "Shot",                                                    # metrics
    "Starting XI", "Substitution", "Player Off", "Player On", "Foul Committed",
    "Bad Behaviour", "Half End",                                             # minutes played
}
EVENT_COLUMNS = [
    "id", "index", "period", "timestamp", "type", "team", "player_id", "location",
    "pass_end_location", "pass_length", "pass_outcome", "pass_type", "pass_assisted_shot_id",
    "pass_aerial_won", "dribble_outcome", "duel_type", "duel_outcome", "interception_outcome",
    "ball_recovery_recovery_failure", "clearance_aerial_won", "miscontrol_aerial_won",
    "shot_aerial_won", "shot_statsbomb_xg", "tactics", "substitution_replacement_id",
    "foul_committed_card", "bad_behaviour_card",
]

# Match clock (seconds) at which each period starts on the lineup "from"/"to" times. The clock
# restarts at 45:00 in the second half, so first-half stoppage time is not on it (see compute_minutes).
CLOCK_START = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
DISMISSALS = {"Red Card", "Second Yellow"}


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def _fetch_match(match_id: int, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return match_id, sb.events(match_id=match_id), sb.lineups(match_id=match_id)
        except Exception:  # transient network problem: wait and retry
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def load_tournament(refresh: bool = False):
    """Return (matches, events, lineups) for all 2022 World Cup matches.

    events:  one row per event (periods 1-5), only EVENT_TYPES / EVENT_COLUMNS, plus match_id.
    lineups: one row per squad player per match; `positions` holds that player's position stints.
    """
    if CACHE_FILE.exists() and not refresh:
        cached = pickle.loads(CACHE_FILE.read_bytes())
        if cached.get("version") == CACHE_VERSION:
            return cached["matches"], cached["events"], cached["lineups"]

    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    print(f"Downloading events and lineups for {len(matches)} matches (first run only) ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(_fetch_match, matches["match_id"]))

    events, lineups = [], []
    for match_id, match_events, match_lineups in fetched:
        kept = match_events[match_events["type"].isin(EVENT_TYPES)]
        events.append(kept.reindex(columns=EVENT_COLUMNS).assign(match_id=match_id))
        for team, squad in match_lineups.items():
            lineups.append(squad.assign(match_id=match_id, team=team))
    events = pd.concat(events, ignore_index=True)
    lineups = pd.concat(lineups, ignore_index=True)[
        ["match_id", "team", "player_id", "player_name", "player_nickname", "positions"]
    ]

    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_bytes(
        pickle.dumps({"version": CACHE_VERSION, "matches": matches, "events": events, "lineups": lineups})
    )
    return matches, events, lineups


# ---------------------------------------------------------------------------
# 2. Minutes played
# ---------------------------------------------------------------------------
def _ts_seconds(timestamp: str) -> float:
    """StatsBomb event timestamp "HH:MM:SS.mmm" (time since the period started) -> seconds."""
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def build_clock(events: pd.DataFrame):
    """Real length of every period (1-4; the penalty shoot-out is period 5 and is ignored).

    Returns (period_start, match_end): seconds of play before each (match_id, period) starts, and
    seconds of play in the whole match. Extra time is included whenever the match had it.
    """
    ends = events[(events["type"] == "Half End") & (events["period"] <= 4)]
    length = ends["timestamp"].map(_ts_seconds).groupby([ends["match_id"], ends["period"]]).max()
    period_start, match_end = {}, {}
    for match_id, periods in length.groupby(level="match_id"):
        elapsed = 0.0
        for (_, period), seconds in periods.items():
            period_start[(match_id, period)] = elapsed
            elapsed += seconds
        match_end[match_id] = elapsed
    return period_start, match_end


def on_pitch_spans(events: pd.DataFrame, period_start: dict, match_end: dict) -> dict:
    """Spans of play (seconds) during which each player was on the pitch, rebuilt from the events.

    Returns {(match_id, player_id): [(start, end), ...]}. A player is on from the Starting XI or
    from the Substitution that brings him on, until he is substituted off or sent off, the match
    ends, or he leaves briefly for treatment ("Player Off" ... "Player On").
    """
    kinds = ["Starting XI", "Substitution", "Player Off", "Player On", "Foul Committed", "Bad Behaviour"]
    ev = events[events["type"].isin(kinds) & (events["period"] <= 4)]
    spans: dict = {}
    for (match_id, _team), team_events in ev.groupby(["match_id", "team"], sort=False):
        opened: dict = {}  # player_id -> second at which his current span started
        gone: set = set()  # substituted off or sent off: cannot come back

        def enter(player_id, t):
            if player_id not in opened and player_id not in gone:
                opened[player_id] = t

        def leave(player_id, t, for_good):
            if player_id in opened:
                spans.setdefault((match_id, player_id), []).append((opened.pop(player_id), t))
            if for_good:
                gone.add(player_id)

        for e in team_events.sort_values(["period", "index"]).to_dict("records"):
            t = period_start[(match_id, e["period"])] + _ts_seconds(e["timestamp"])
            if e["type"] == "Starting XI":
                for slot in e["tactics"]["lineup"]:
                    enter(slot["player"]["id"], 0.0)
            elif e["type"] == "Substitution":
                leave(int(e["player_id"]), t, for_good=True)
                enter(int(e["substitution_replacement_id"]), t)
            elif e["type"] == "Player Off":
                leave(int(e["player_id"]), t, for_good=False)
            elif e["type"] == "Player On":
                enter(int(e["player_id"]), t)
            elif e["foul_committed_card"] in DISMISSALS or e["bad_behaviour_card"] in DISMISSALS:
                leave(int(e["player_id"]), t, for_good=True)
        for player_id, start in opened.items():
            spans.setdefault((match_id, player_id), []).append((start, match_end[match_id]))
    return spans


def compute_minutes(lineups: pd.DataFrame, events: pd.DataFrame, clock):
    """Minutes played by every player in every position, from the lineup position stints.

    Minutes are real time on the pitch: stoppage time and extra time count, the penalty
    shoot-out does not. (Stoppage was long at this World Cup: a regulation match averaged about
    101 minutes of play, so a full 90-minute game is worth ~101 minutes here, and per-90 rates
    are per 90 minutes of that real time.) `clock` is the pair returned by build_clock.

    How a stint's length is found. The lineup "from"/"to" fields are not reliable on their own:
    in some matches (mostly the extra-time games) "to" values and end reasons are attached to
    the wrong stint, red-carded players are credited until the final whistle, and Lucas
    Hernandez gets ~91 phantom minutes after being substituted off in the 12th minute. The
    stint *start* times and position names are sound, so each position is held from its own start
    until the player's next stint begins, and only the time the player was really on the pitch
    (on_pitch_spans, rebuilt from starting XIs, substitutions, treatment exits and red cards)
    is counted. Check: the players' minutes then add up to 11 x the real match length for every
    team (less red cards and brief treatment absences).

    Returns (position_minutes, n_checked, n_disagree). position_minutes has one row per
    match/player/position (columns: match_id, team, player_id, position, minutes). The two counts
    say how many player-matches the plain lineup arithmetic (to - from) gets wrong by over 30 s.
    """
    period_start, match_end = clock
    spans = on_pitch_spans(events, period_start, match_end)

    def elapsed(match_id, period, lineup_clock):  # lineup clock "MM:SS" + period -> seconds of play
        minutes, seconds = lineup_clock.split(":")
        into_period = max(0, int(minutes) * 60 + int(seconds) - CLOCK_START[int(period)])
        return period_start[(match_id, int(period))] + into_period

    rows, plain = [], {}
    for squad in lineups.itertuples(index=False):
        stints = []
        for stint in squad.positions:
            if int(stint["from_period"]) > 4:  # nothing happens in the shoot-out
                continue
            start = elapsed(squad.match_id, stint["from_period"], stint["from"])
            if stint["to"] is None or int(stint["to_period"]) > 4:
                end = match_end[squad.match_id]
            else:
                end = elapsed(squad.match_id, stint["to_period"], stint["to"])
            stints.append((start, stint["position"]))
            key = (squad.match_id, squad.player_id)
            plain[key] = plain.get(key, 0.0) + (end - start)
        if not stints:
            continue
        stints.sort(key=lambda s: s[0])  # chronological; ties keep the lineup order
        player_spans = spans.get((squad.match_id, squad.player_id), [])
        for i, (start, position) in enumerate(stints):
            lo = -np.inf if i == 0 else start  # first stint also covers any time before it
            hi = np.inf if i == len(stints) - 1 else stints[i + 1][0]
            seconds = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in player_spans)
            rows.append((squad.match_id, squad.team, squad.player_id, position, seconds / 60))
    position_minutes = pd.DataFrame(rows, columns=["match_id", "team", "player_id", "position", "minutes"])

    guarded = position_minutes.groupby(["match_id", "player_id"])["minutes"].sum() * 60
    n_disagree = sum(abs(guarded.get(key, 0.0) - secs) > 30 for key, secs in plain.items())
    return position_minutes, len(plain), n_disagree


def build_player_table(lineups: pd.DataFrame, position_minutes: pd.DataFrame) -> pd.DataFrame:
    """One row per player: team, name, minutes, matches, main position and positions played."""
    unknown = set(position_minutes["position"]) - set(POSITION_GROUPS)
    if unknown:
        raise ValueError(f"Add these StatsBomb positions to POSITION_GROUPS: {sorted(unknown)}")
    pm = position_minutes.assign(role=position_minutes["position"].map(POSITION_GROUPS))

    players = (lineups.drop_duplicates("player_id")
               .set_index("player_id")[["team", "player_name", "player_nickname"]])
    players["minutes"] = pm.groupby("player_id")["minutes"].sum()
    players["matches"] = pm[pm["minutes"] > 0].groupby("player_id")["match_id"].nunique()
    role_minutes = pm.pivot_table(index="player_id", columns="role", values="minutes", aggfunc="sum", fill_value=0)
    players["main_position"] = role_minutes.idxmax(axis=1)
    by_position = (pm.groupby(["player_id", "position"])["minutes"].sum().reset_index()
                   .query("minutes >= 0.5").sort_values(["player_id", "minutes"], ascending=[True, False]))
    players["positions"] = pd.Series({
        player_id: list(zip(g["position"], g["minutes"])) for player_id, g in by_position.groupby("player_id")
    })
    players = players.dropna(subset=["minutes"])  # squad members who never got on the pitch
    players["matches"] = players["matches"].fillna(0).astype(int)
    players["display_name"] = players["player_nickname"].fillna(players["player_name"])
    return players


# ---------------------------------------------------------------------------
# 3. Metrics (per player, totals over the tournament; periods 1-4 only, no shoot-out)
# ---------------------------------------------------------------------------
# StatsBomb pitch: 120 x 80 yards. x = 0 is the team's own goal line and x = 120 the opposition's
# (every team is shown attacking left to right), y runs across the pitch, the opposition goal
# is centred at (120, 40). Penalty area: x >= 102 and 18 <= y <= 62. Final third: x >= 80.
#
# Set pieces: corners, free kicks, throw-ins, goal kicks and kick-offs.
SET_PIECES = ["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"]
SUCCESSFUL = ["Won", "Success In Play", "Success Out"]  # StatsBomb outcomes of a won tackle / interception


def _xy(locations: pd.Series) -> np.ndarray:
    """StatsBomb [x, y] lists -> (n, 2) float array (NaN where there is no location)."""
    return np.array([loc[:2] if isinstance(loc, list) else [np.nan, np.nan] for loc in locations], dtype=float)


def _in_box(x, y):
    return (x >= 102) & (y >= 18) & (y <= 62)


def compute_event_totals(events: pd.DataFrame) -> pd.DataFrame:
    """Tournament totals for every player, one column per ingredient of the 12 metrics."""
    ev = events[(events["period"] <= 4) & events["player_id"].notna()].copy()
    ev["player_id"] = ev["player_id"].astype(int)

    # --- Passing -----------------------------------------------------------------------------
    p = ev[ev["type"] == "Pass"]
    x0, y0 = _xy(p["location"]).T
    x1, y1 = _xy(p["pass_end_location"]).T
    outcome, pass_type = p["pass_outcome"], p["pass_type"]

    # Passes completed: a Pass event with no outcome (StatsBomb only records an outcome when the
    # pass failed: Incomplete, Out, Pass Offside, Unknown or Injury Clearance). Set pieces count.
    completed = outcome.isna().to_numpy()

    # Pass accuracy %: completed / attempted x 100. Attempted = every Pass event except
    # "Injury Clearance" (the ball is kicked out so a player can be treated, not a pass to a
    # teammate). Incomplete, Out, Offside and Unknown passes all count as failed.
    attempted = (outcome != "Injury Clearance").to_numpy()

    # Progressive passes (FBref/Opta-style, simplified). A completed open-play pass that
    #   - starts outside the passer's own defensive 40% of the pitch (x >= 48), and
    #   - either moves the ball at least 10 yards towards the opposition goal line (end x minus
    #     start x >= 10), or ends inside the opposition penalty area after starting outside it.
    # Differences from FBref: Opta measures the 10 yards from the furthest point the ball reached
    # in the previous six passes, so a ball recycled back and forth is not counted twice. Here
    # each pass is judged against its own start point, which gives somewhat more progressive
    # passes (about 61 per team per match at this World Cup). Set pieces (corners, free kicks,
    # throw-ins, goal kicks, kick-offs) are excluded so a corner into the box is not "progress".
    open_play = ~pass_type.isin(SET_PIECES).to_numpy()
    progressive = (completed & open_play & (x0 >= 48)
                   & ((x1 - x0 >= 10) | (_in_box(x1, y1) & ~_in_box(x0, y0))))

    # Long passes completed: completed passes longer than 30 yards (FBref's "long" band; about
    # 27 m). StatsBomb's pass_length is in yards. Set pieces included.
    long_completed = completed & (p["pass_length"].to_numpy() > 30)

    # --- Creating ----------------------------------------------------------------------------
    # Key passes: passes that directly led to a shot, whether or not it scored (StatsBomb marks
    # them pass_shot_assist, or pass_goal_assist when the shot was a goal; in this data both
    # always coincide with pass_assisted_shot_id being filled). Set pieces included.
    key_pass = p["pass_assisted_shot_id"].notna().to_numpy()

    # Expected assists (xA): StatsBomb has no xA column, so it is built the usual way: each key
    # pass is credited with the xG (shot_statsbomb_xg) of the shot it created, summed per player.
    shot_xg = ev.loc[ev["type"] == "Shot"].set_index("id")["shot_statsbomb_xg"]
    xa = p["pass_assisted_shot_id"].map(shot_xg).fillna(0.0).to_numpy()

    # Passes into the final third: completed open-play passes that start outside the final third
    # (x < 80) and end inside it (x >= 80). Passes already inside the third do not count.
    into_final_third = completed & open_play & (x0 < 80) & (x1 >= 80)

    passes = pd.DataFrame({
        "passes_attempted": attempted, "passes_completed": completed,
        "progressive_passes": progressive, "long_passes_completed": long_completed,
        "key_passes": key_pass, "xa": xa, "passes_final_third": into_final_third,
    }, index=p.index).groupby(p["player_id"]).sum()

    # --- Other events --------------------------------------------------------------------------
    # Successful dribbles: Dribble events (the player took on an opponent) with outcome "Complete".
    dribbles = (ev["type"] == "Dribble") & (ev["dribble_outcome"] == "Complete")

    # Pressures: Pressure events, i.e. each time the player closed down an opponent who was
    # receiving, carrying or releasing the ball.
    pressures = ev["type"] == "Pressure"

    # Tackles + interceptions: *successful* ones only. StatsBomb logs failed attempts as well
    # (about 40% of its tackle and interception events), outcomes "Lost In Play" / "Lost Out", and
    # they are not counted here. Tackles are Duel events of type "Tackle"; interceptions are
    # Interception events. "Won", "Success In Play" and "Success Out" count as successful.
    tackles = (ev["type"] == "Duel") & (ev["duel_type"] == "Tackle") & ev["duel_outcome"].isin(SUCCESSFUL)
    interceptions = (ev["type"] == "Interception") & ev["interception_outcome"].isin(SUCCESSFUL)

    # Ball recoveries: Ball Recovery events (the player picked up a loose ball), leaving out
    # attempts StatsBomb flags as a recovery failure.
    recoveries = (ev["type"] == "Ball Recovery") & ~ev["ball_recovery_recovery_failure"].eq(True)

    # Aerial duels won: StatsBomb logs the loser as a Duel ("Aerial Lost"); the winner is whatever
    # event followed the header and carries an aerial_won flag (a pass, clearance, shot or
    # miscontrol). Counting those flags gives exactly one win per Aerial Lost in this data.
    aerial_flags = ["pass_aerial_won", "clearance_aerial_won", "shot_aerial_won", "miscontrol_aerial_won"]
    aerial_won = ev[aerial_flags].eq(True).any(axis=1)

    def count(mask):
        return ev[mask].groupby("player_id").size()

    others = pd.DataFrame({
        "dribbles_completed": count(dribbles), "pressures": count(pressures),
        "tackles_interceptions": count(tackles | interceptions),
        "ball_recoveries": count(recoveries), "aerial_duels_won": count(aerial_won),
    })
    return passes.join(others, how="outer").fillna(0)


def metrics_per90(totals: pd.DataFrame, minutes: pd.Series) -> pd.DataFrame:
    """The 12 metrics per 90 minutes (pass accuracy stays a percentage)."""
    totals = totals.reindex(minutes.index).fillna(0)
    per90 = totals.div(minutes, axis=0) * 90
    per90["pass_accuracy"] = totals["passes_completed"] / totals["passes_attempted"] * 100
    return per90[METRIC_KEYS]


def percentile_ranks(values: pd.DataFrame, group_values: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0-100) of every row of `values` within `group_values`, per metric.

    percentileofscore(kind="mean") = (players below + half the players level) / group size, so
    a lone best value scores just under 100, the lowest just over 0, and ties share the middle.
    """
    return pd.DataFrame({
        key: [percentileofscore(group_values[key], v, kind="mean") for v in values[key]]
        for key in METRIC_KEYS
    }, index=values.index)


# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]  # first one installed

# Layout in inches. Every card is PANEL_W wide, so the side-by-side figure is just three cards.
PANEL_W, FIG_H, PIZZA_SIZE = 7.8, 9.8, 5.3


def draw_pizza(ax, percentiles):
    """One pizza on a polar axes: slices coloured by group, the percentile written on each slice."""
    slice_colors = [GROUP_COLORS[group] for _, _, group in METRICS]
    baker = PyPizza(
        params=[label for _, label, _ in METRICS],
        background_color=SURFACE,
        straight_line_color=SURFACE, straight_line_lw=2,   # the gap between slices
        last_circle_color=BASELINE, last_circle_lw=1, last_circle_ls="-",
        other_circle_color=GRID, other_circle_lw=1, other_circle_ls="-",
        inner_circle_size=16,
    )
    baker.make_pizza(
        [int(round(v)) for v in percentiles], ax=ax, param_location=118,
        slice_colors=slice_colors, color_blank_space="same", blank_alpha=0.13,
        value_colors=[INK] * len(METRICS), value_bck_colors=[SURFACE] * len(METRICS),
        kwargs_slices=dict(edgecolor=SURFACE, linewidth=2, zorder=2),
        kwargs_params=dict(color=INK, fontsize=10, va="center", linespacing=1.2),
        kwargs_values=dict(fontsize=10.5, fontweight="bold", va="center", zorder=5,
                           bbox=dict(boxstyle="round,pad=0.25", linewidth=1.5, facecolor=SURFACE)),
    )
    for text, color in zip(baker.get_value_texts(), slice_colors):  # outline each value like its slice
        text.get_bbox_patch().set_edgecolor(color)
    ax.set_facecolor(SURFACE)
    # PyPizza hides the outer spine but not the ring around the centre hole. matplotlib switches that
    # spine back on every time it draws, so make it invisible through its colour instead.
    ax.spines["inner"].set_edgecolor("none")


def add_footer(fig, n_group):
    """Group legend, what the slices mean, the comparison note and the data credit."""
    handles = [Patch(facecolor=color, label=group) for group, color in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=11.5, labelcolor=INK,
               handlelength=1.1, handleheight=1.1, columnspacing=2.2, bbox_to_anchor=(0.5, 1.5 / FIG_H))
    fig.text(0.5, 1.05 / FIG_H, f"Slice length and number: percentile rank (0–100) among {n_group} players",
             ha="center", va="center", fontsize=10, color=INK_2)
    fig.text(0.5, 0.72 / FIG_H, NOTE, ha="center", va="center", fontsize=10, color=INK_2)
    fig.text(0.5, 0.40 / FIG_H, "Data: StatsBomb", ha="center", va="center", fontsize=10, color=INK_2)


def pizza_figure(cards, n_group):
    """A figure with one card (name, team, minutes, pizza) per entry of `cards`, plus the footer."""
    fig_w = PANEL_W * len(cards)
    fig = plt.figure(figsize=(fig_w, FIG_H), facecolor=SURFACE)
    for i, card in enumerate(cards):
        centre = (i + 0.5) / len(cards)  # horizontal centre of the card, as a fraction of the figure
        ax = fig.add_axes([centre - PIZZA_SIZE / fig_w / 2, 2.55 / FIG_H, PIZZA_SIZE / fig_w, PIZZA_SIZE / FIG_H],
                          projection="polar")
        draw_pizza(ax, card["percentiles"])
        fig.text(centre, 9.28 / FIG_H, card["name"], ha="center", va="center",
                 fontsize=22, fontweight="bold", color=INK)
        fig.text(centre, 8.84 / FIG_H,
                 f"{card['team']}  ·  {card['minutes']:.0f} minutes played (incl. stoppage time)",
                 ha="center", va="center", fontsize=12.5, color=INK_2)
    add_footer(fig, n_group)
    return fig


def slug(name: str) -> str:
    return "_".join(_plain(name).split())


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
def _plain(text) -> str:
    """Lower-case text without accents, for matching names."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def find_player(players: pd.DataFrame, key: str, team: str) -> int:
    searchable = (players["player_name"].map(_plain) + " " + players["player_nickname"].map(_plain))
    hits = players[(players["team"] == team) & searchable.str.contains(key, regex=False)]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one {team} player matching '{key}', found {len(hits)}")
    return hits.index[0]


def _fmt(key: str, value: float) -> str:
    if key == "pass_accuracy":
        return f"{value:.1f}%"
    return f"{value:.3f}" if key == "xa" else f"{value:.2f}"


def minutes_by_match(position_minutes, matches, match_end, player_ids) -> pd.DataFrame:
    """Minutes each of `player_ids` played in every match, next to the real length of that match."""
    info = matches.set_index("match_id")
    rows = (position_minutes[position_minutes["player_id"].isin(player_ids)]
            .groupby(["player_id", "match_id", "team"], as_index=False)["minutes"].sum())
    rows = rows[rows["minutes"] > 0].copy()
    rows["opponent"] = [info.at[m, "away_team"] if info.at[m, "home_team"] == t else info.at[m, "home_team"]
                        for m, t in zip(rows["match_id"], rows["team"])]
    rows["stage"] = rows["match_id"].map(info["competition_stage"])
    rows["date"] = rows["match_id"].map(info["match_date"])
    rows["match_length"] = rows["match_id"].map(lambda m: match_end[m] / 60)
    rows["order"] = rows["player_id"].map({player_id: i for i, player_id in enumerate(player_ids)})
    return rows.sort_values(["order", "date"])


def print_report(cards, players, raw, pct, group_ids, match_minutes, n_central, n_played):
    names = [c["name"] for c in cards]
    print("\n" + "=" * 100)
    print("COMPARISON GROUP")
    print(f"  Players who played at all:                        {n_played}")
    print(f"  ... with central midfield as main position:        {n_central}")
    print(f"  ... and {MIN_MINUTES}+ minutes played (the group):            {len(group_ids)}")

    print("\nPLAYERS (minutes = real time on the pitch, incl. stoppage time and extra time)")
    print(f"  {'Player':<17}{'Team':<13}{'Minutes':>8}{'Matches':>9}   Main position")
    for c in cards:
        row = players.loc[c["player_id"]]
        print(f"  {c['name']:<17}{c['team']:<13}{row['minutes']:>8.1f}{row['matches']:>9}   {row['main_position']}")
    print("\nPOSITIONS PLAYED (minutes)")
    for c in cards:
        played = ", ".join(f"{pos} {mins:.1f}" for pos, mins in players.loc[c["player_id"], "positions"])
        print(f"  {c['name']:<17}{played}")

    print("\nMINUTES BY MATCH (minutes played / real length of the match)")
    names_by_id = {c["player_id"]: c["name"] for c in cards}
    for row in match_minutes.itertuples():
        print(f"  {names_by_id[row.player_id]:<17}{row.opponent:<15}{row.stage:<15}"
              f"{row.minutes:>8.1f} / {row.match_length:.1f}")

    print("\nMETRICS PER 90 (pass accuracy in %) AND PERCENTILE RANK WITHIN THE GROUP")
    w_val, w_pct, gap = 8, 5, "   "
    print(f"  {'':<10} {'':<24}" + "".join(f"{gap}{n:^{w_val + 1 + w_pct}}" for n in names) + f"{gap}{'Group':>{w_val}}")
    header = f"  {'Group':<10} {'Metric':<24}" + "".join(f"{gap}{'per 90':>{w_val}} {'pctl':>{w_pct}}" for _ in names)
    print(header + f"{gap}{'median':>{w_val}}")
    print("  " + "-" * (len(header) + len(gap) + w_val - 2))
    medians = raw.loc[group_ids].median()
    for key, label, group in METRICS:
        line = f"  {group:<10} {label.replace(chr(10), ' '):<24}"
        for c in cards:
            line += f"{gap}{_fmt(key, raw.loc[c['player_id'], key]):>{w_val}} {int(round(pct.loc[c['player_id'], key])):>{w_pct}}"
        print(line + f"{gap}{_fmt(key, medians[key]):>{w_val}}")


def save_group_csv(path, players, raw, pct, group_ids):
    out = players.loc[group_ids, ["display_name", "player_name", "team", "minutes", "matches", "main_position"]].copy()
    out["positions"] = players.loc[group_ids, "positions"].map(lambda ps: "; ".join(f"{p} {m:.0f}" for p, m in ps))
    out = out.join(raw.loc[group_ids].add_suffix("_per90")).join(pct.loc[group_ids].round(1).add_suffix("_pctl"))
    out.sort_values("minutes", ascending=False).round(3).to_csv(path, index_label="player_id", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--refresh", action="store_true", help="download the StatsBomb data again")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where to write PNGs and CSV")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    matches, events, lineups = load_tournament(refresh=args.refresh)
    print(f"Loaded {len(matches)} matches, {len(events):,} relevant events, "
          f"{lineups['player_id'].nunique()} squad players.")

    # 2. minutes
    clock = build_clock(events)
    position_minutes, n_checked, n_disagree = compute_minutes(lineups, events, clock)
    players = build_player_table(lineups, position_minutes)
    print(f"Minutes: the plain lineup from/to arithmetic is off by more than 30 s for {n_disagree} of "
          f"{n_checked} player-matches (extra time, red cards, one phantom stint); those use the event-checked values.")

    # 3. comparison group
    played = players[players["minutes"] > 0]
    central = played[played["main_position"] == CENTRAL_MIDFIELD]
    group_ids = central.index[central["minutes"] >= MIN_MINUTES]

    # 4. metrics and percentiles
    target_ids = [find_player(players, key, team) for _, key, team in TARGET_PLAYERS]
    totals = compute_event_totals(events)
    raw = metrics_per90(totals, played["minutes"])
    pct = percentile_ranks(raw.loc[group_ids.union(target_ids)], raw.loc[group_ids])

    cards = []
    for (label, _key, team), player_id in zip(TARGET_PLAYERS, target_ids):
        if player_id not in group_ids:
            print(f"WARNING: {label} is not in the comparison group "
                  f"(main position {players.loc[player_id, 'main_position']}, "
                  f"{players.loc[player_id, 'minutes']:.0f} minutes)")
        cards.append({"player_id": player_id, "name": label, "team": team,
                      "minutes": players.loc[player_id, "minutes"], "percentiles": pct.loc[player_id].tolist()})

    # 5. charts and files
    saved = []
    for card in cards:
        fig = pizza_figure([card], len(group_ids))
        saved.append(args.output_dir / f"pizza_{slug(card['name'])}.png")
        fig.savefig(saved[-1], dpi=200, facecolor=SURFACE)
        plt.close(fig)
    fig = pizza_figure(cards, len(group_ids))
    saved.append(args.output_dir / "pizza_comparison.png")
    fig.savefig(saved[-1], dpi=200, facecolor=SURFACE)
    plt.close(fig)
    saved.append(args.output_dir / "comparison_group.csv")
    save_group_csv(saved[-1], players, raw, pct, group_ids)

    match_minutes = minutes_by_match(position_minutes, matches, clock[1], target_ids)
    print_report(cards, players, raw, pct, group_ids, match_minutes, n_central=len(central), n_played=len(played))
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
