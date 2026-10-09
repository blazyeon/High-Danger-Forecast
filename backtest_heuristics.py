#!/usr/bin/env python3
"""Backtest the props heuristics against last season's PBP shot store.

Validates two heuristics used by the player-props model:

  1. Matchup funnel -- a shooter's goal/SOG rate is multiplied by
     (opponent's allowed-by-position) / (league average for that position),
     clamped to [0.75, 1.35].
  2. PP scoring lift -- goals/assists/points are boosted x1.08 on PP1 and
     x1.03 on PP2.

Data: ``pbp_cache/shots/shots_2025_2.parquet`` (2025-26) +
``static/data/pbp_skater_stats_20252026.json``.

This is a signal test, not an out-of-sample edge estimate: the "allowed by
position" rates are ground-truth PBP (the same source shotpropz repackages),
and each shooter's baseline is their own full-season per-game rate. The
question answered is "does the multiplier point the right way, and is the
relationship monotonic?" -- i.e. do shooters actually score / shoot more
against opponents that give up more to their position?
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

SEASON = 2025
SHOTS_PATH = f"pbp_cache/shots/shots_{SEASON}_2.parquet"
SKATERS_PATH = f"static/data/pbp_skater_stats_{SEASON}{SEASON + 1}.json"

# NHL team_id -> abbreviation (Utah relocates ARI 53 -> 68).
TEAM_ID = {
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT", 6: "BOS", 7: "BUF",
    8: "MTL", 9: "OTT", 10: "TOR", 12: "CAR", 13: "FLA", 14: "TBL", 15: "WSH",
    16: "CHI", 17: "DET", 18: "NSH", 19: "STL", 20: "CGY", 21: "COL", 22: "EDM",
    23: "VAN", 24: "ANA", 25: "DAL", 26: "LAK", 28: "SJS", 29: "CBJ", 30: "MIN",
    52: "WPG", 54: "VGK", 55: "SEA", 68: "UTA",
}

POS_BUCKET = {"C": "C", "L": "LW", "R": "RW", "D": "D"}
MATCHUP_MIN, MATCHUP_MAX = 0.75, 1.35
MIN_GP = 20  # only trust baselines from players who played a real season


def load_shots() -> pd.DataFrame:
    df = pd.read_parquet(SHOTS_PATH)
    # A goal is a shot; drop the non-shot play rows (the store also carries
    # some non-shot events we don't need here).
    df = df[df["is_shot"] == 1].copy()
    df["team_abbr"] = df["team_id"].map(TEAM_ID)
    # Sanity: every shooter's mapped team must be one of the two teams in-game.
    assert (((df["team_abbr"] == df["homeTeam"]) | (df["team_abbr"] == df["awayTeam"])).all()), \
        "team_id -> abbr mapping inconsistent with home/away"
    # Opponent + the location the opponent is defending (shotpropz's split).
    df["is_home_shooter"] = df["team_abbr"] == df["homeTeam"]
    df["opponent"] = df["awayTeam"].where(df["is_home_shooter"], df["homeTeam"])
    df["defending_location"] = df["is_home_shooter"].map({True: "Away", False: "Home"})
    # Power-play flag: shooter's side has more skaters than the opponent.
    shooter_sk = df["home_skaters"].where(df["is_home_shooter"], df["away_skaters"])
    opp_sk = df["away_skaters"].where(df["is_home_shooter"], df["home_skaters"])
    df["on_pp"] = shooter_sk > opp_sk
    df["five_on_five"] = (shooter_sk == 5) & (opp_sk == 5)
    return df


def load_skaters() -> dict:
    data = json.load(open(SKATERS_PATH, encoding="utf-8"))["data"]
    out = {}
    for p in data:
        out[p["name"]] = {
            "position": p["position"],
            "team": p["team"],
            "gp": p["gp"],
            "gpg": p["goals"] / p["gp"] if p["gp"] else 0.0,
            "sogpg": p["shots"] / p["gp"] if p["gp"] else 0.0,
        }
    return out


def build_allowed(df: pd.DataFrame) -> dict:
    """{metric: {(team, location, pos): per_game_rate}} plus games-per-location.

    `df` is the (already position-tagged) shot frame restricted to the train
    window. Per-game rate = total allowed / number of games the team played at
    that location.
    """
    team_locations = {}
    for _, row in df[["game_id", "homeTeam", "awayTeam"]].drop_duplicates().iterrows():
        team_locations.setdefault((row["homeTeam"], "Home"), set()).add(row["game_id"])
        team_locations.setdefault((row["awayTeam"], "Away"), set()).add(row["game_id"])
    games = {k: len(v) for k, v in team_locations.items()}

    allowed = {"goals": defaultdict(float), "sog": defaultdict(float)}
    for metric, col in (("goals", "is_goal"), ("sog", "is_shot")):
        agg = df.groupby(["opponent", "defending_location", "pos_bucket"])[col].sum()
        for (team, loc, pos), total in agg.items():
            n = games.get((team, loc), 0)
            allowed[metric][(team, loc, pos)] = total / n if n else 0.0
        # "All" fallback (home + away combined), matching shotpropz's third split.
        for (team, loc, pos), total in agg.items():
            n_home = games.get((team, "Home"), 0)
            n_away = games.get((team, "Away"), 0)
            all_key = (team, "All", pos)
            allowed[metric][all_key] = allowed[metric].get(all_key, 0.0) + total / (n_home + n_away)
    return allowed


def multiplier(allowed: dict, metric: str, opponent: str, loc: str, pos: str) -> float | None:
    key = (opponent, loc, pos)
    bucket = allowed[metric].get(key)
    if bucket is None:
        key = (opponent, "All", pos)
        bucket = allowed[metric].get(key)
    if bucket is None:
        return None
    # League average over every team's same-position bucket (same location split).
    league = [v for (t, l, p), v in allowed[metric].items() if l == loc and p == pos]
    if not league:
        return None
    avg = sum(league) / len(league)
    if avg <= 0:
        return None
    return max(MATCHUP_MIN, min(MATCHUP_MAX, bucket / avg))


def tag_positions(df: pd.DataFrame, skaters: dict) -> pd.DataFrame:
    """Attach position bucket and drop shots we can't position (name mismatch)."""
    df = df[df["shooter_name"].isin(skaters)].copy()
    df["position"] = df["shooter_name"].map(lambda n: skaters[n]["position"])
    df["pos_bucket"] = df["position"].map(POS_BUCKET)
    return df[df["pos_bucket"].notna()].copy()


def build_player_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, game): goals, shots, opponent, position."""
    return df.groupby(["shooter_name", "game_id"]).agg(
        goals=("is_goal", "sum"),
        shots=("is_shot", "sum"),
        opponent=("opponent", "first"),
        loc=("defending_location", "first"),
        pos=("pos_bucket", "first"),
    ).reset_index()


def build_baselines(train: pd.DataFrame) -> pd.DataFrame:
    """Per-player baseline rate from the train window, conditioned on shot-games.

    Both baseline and evaluation count only games where the player took a shot,
    so "actual/baseline" sits near 1.0 when the multiplier is uninformative
    (avoids the level shift a goals/gp baseline introduces).
    """
    b = train.groupby("shooter_name").agg(
        games=("game_id", "nunique"),
        goals=("is_goal", "sum"),
        shots=("is_shot", "sum"),
    )
    b["gpg"] = b["goals"] / b["games"]
    b["sogpg"] = b["shots"] / b["games"]
    return b[b["games"] >= MIN_GP].reset_index()


def binned_ratio(pg: pd.DataFrame, metric: str, allowed: dict) -> None:
    y_col = "goals" if metric == "goals" else "shots"
    base_col = "gpg" if metric == "goals" else "sogpg"
    m = pg.apply(lambda r: multiplier(allowed, metric, r["opponent"], r["loc"], r["pos"]), axis=1)
    pg = pg.assign(m=m).dropna(subset=["m"])
    pg = pg.assign(base=pg[base_col], y=pg[y_col])

    print(f"\n=== Matchup funnel -- {metric.upper()} (n={len(pg)} player-games) ===")
    pg["bin"] = pd.qcut(pg["m"], 5, duplicates="drop")
    rows = []
    for b, g in pg.groupby("bin", observed=True):
        rows.append({
            "multiplier range": f"{g.m.min():.2f}-{g.m.max():.2f}",
            "n": len(g),
            "mean baseline": round(g["base"].mean(), 4),
            "mean actual": round(g["y"].mean(), 4),
            "actual/baseline": round(g["y"].mean() / g["base"].mean(), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Spearman between multiplier and (actual - baseline): does the funnel move
    # outcomes the right way relative to the player's own rate?
    resid = pg["y"] - pg["base"]
    print(f"Spearman(multiplier, actual-baseline) = {pg['m'].corr(resid, method='spearman'):+.3f}")
    print(f"Pearson(multiplier, actual/baseline)   = "
          f"{pg['m'].corr(pg['y'] / pg['base'], method='pearson'):+.3f}")


def pp_lift(df: pd.DataFrame) -> None:
    print("\n=== PP scoring lift (league-wide, per-shot conversion) ===")
    skater = df[df["shooter_name"].notna()]
    for label, mask in (("5v5", skater["five_on_five"]), ("PP", skater["on_pp"])):
        sub = skater[mask]
        shots = sub["is_shot"].sum()
        goals = sub["is_goal"].sum()
        print(f"  {label}: {goals} goals on {shots} shots = {goals/shots:.4f} shots->goal")
    es = skater[skater["five_on_five"]]
    pp = skater[skater["on_pp"]]
    es_conv = es["is_goal"].sum() / es["is_shot"].sum()
    pp_conv = pp["is_goal"].sum() / pp["is_shot"].sum()
    print(f"  PP conversion lift over 5v5: {pp_conv/es_conv:.2f}x "
          f"(heuristic boosts per-game rate x1.08 PP1 / x1.03 PP2 -- different unit)")


def main() -> None:
    skaters = load_skaters()
    df = load_shots()

    # Out-of-sample: train the funnel on the first 60% of games (chronological),
    # evaluate on the last 40% so the multiplier is not self-fulfilling.
    ordered = sorted(df["game_id"].unique())
    split = ordered[int(len(ordered) * 0.6)]
    train = tag_positions(df[df["game_id"] <= split], skaters)
    test = tag_positions(df[df["game_id"] > split], skaters)
    print(f"games: train {train.game_id.nunique()} / test {test.game_id.nunique()}")

    baseline = build_baselines(train)
    pg_test = build_player_games(test).merge(baseline[["shooter_name", "gpg", "sogpg"]],
                                             on="shooter_name", how="inner")

    for metric in ("goals", "sog"):
        allowed = build_allowed(train)
        binned_ratio(pg_test, metric, allowed)

    pp_lift(df)


if __name__ == "__main__":
    main()
