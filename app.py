"""
NHL Game Predictor — Flask backend.
Serves the dark-themed SPA frontend and provides REST API endpoints
for team data, predictions, lookups, stats, and player props.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory

from NHL.AppState import (
    get_app_state, get_state_info, check_elo_data_availability,
    get_team_elo, get_team_recent_form_rating,
)
from NHL.Config import (
    DIVISIONS, TEAM_ABBR_MAPPING, NST_ABBR_TO_FULL, NHL_SEASON_START_MONTH,
    DEFAULT_SIMULATIONS, DEFAULT_TREND_GAMES, _season_options,
)
from NHL.MatchupUtils import (
    build_team_options, build_teams_api_data,
    score_combo_distribution, choose_non_tie_split,
    build_skater_stats_df, merge_goalie_options,
    detect_game_type,
)
from NHL.Simulation import simulate_matchup
from NHL.StatsFromPBP import load_skater_rates_from_json
from NHL.ApiScrape import (
    get_confirmed_or_predicted_lineup,
    get_roster_goalies_for_override,
    get_games_on_date, get_boxscore,
)
from NHL.GoaliePrediction import predict_starting_goalie
from NHL.Errors import safe_api_call
from NHL.Utils import (
    season_from_date, get_data_season_for_game,
    sanitize_text, format_initial_last,
)
from NHL.OddsAPI import fetch_nhl_player_props_by_date, OddsAPIError
from NHL.StatsFromPBP import load_skater_rates_from_json, load_cached_stats
from NHL.PlayByPlay import count_pp_opportunities, count_faceoffs

logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)

_state_initialised = False


@app.before_request
def _ensure_state():
    global _state_initialised
    if not _state_initialised:
        try:
            get_app_state()
        except Exception as e:
            logger.error(f"State init error: {e}")
        _state_initialised = True


# ── Global error logging ────────────────────────────────────────────────

@app.errorhandler(Exception)
def _handle_exception(e):
    logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    return jsonify({"error": "Internal server error"}), 500


# ── JSON helper ─────────────────────────────────────────────────────────

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, float) and (pd.isna(obj) if hasattr(pd, 'isna') else False):
        return None
    if isinstance(obj, bool):
        return obj
    return obj


# ── Pages ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Logo serving ───────────────────────────────────────────────────────

IMAGES_DIR = Path(__file__).parent / "Images"


@app.route("/api/logos/<team>.png")
def team_logo(team: str):
    """Serve team logo PNG from the Images directory."""
    abbr = TEAM_ABBR_MAPPING.get(team.upper(), team.upper())
    filename = f"{abbr}.png"
    logo_path = IMAGES_DIR / filename
    if logo_path.exists():
        return send_from_directory(str(IMAGES_DIR), filename)
    fallback = IMAGES_DIR / f"{team.upper()}.png"
    if fallback.exists():
        return send_from_directory(str(IMAGES_DIR), f"{team.upper()}.png")
    return "", 404


# ── API: Teams ─────────────────────────────────────────────────────────

@app.route("/api/teams")
def api_teams():
    """Return team list with divisions, abbreviations, and full names."""
    data = build_teams_api_data()
    return jsonify({"divisions": data})


# ── API: Goalies ────────────────────────────────────────────────────────

@app.route("/api/goalies/<team>/<date_str>")
def api_goalies(team: str, date_str: str):
    """Return predicted starter + backup goalies for a team on a given date."""
    try:
        abbr = TEAM_ABBR_MAPPING.get(team.upper(), team.upper())
        opponent = request.args.get("opponent") or None
        is_b2b = request.args.get("b2b", "false").lower() in ("1", "true", "yes")

        all_goalies = get_roster_goalies_for_override(abbr, date_str)
        if all_goalies is None:
            all_goalies = []

        predicted = predict_starting_goalie(abbr, date_str, opponent_abbr=opponent, is_b2b=is_b2b)

        ordered = []
        if predicted and predicted in all_goalies:
            ordered.append(predicted)
        for g in all_goalies:
            if g not in ordered:
                ordered.append(g)

        return jsonify({"goalies": ordered})
    except Exception as e:
        logger.error(f"Error fetching goalies for {team}: {e}")
        return jsonify({"error": str(e)}), 500


# ── API: Predict ───────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Run a full simulation for a matchup.

    JSON body:
    {
        "home_team": "TOR",
        "away_team": "MTL",
        "date": "2025-01-15",
        "simulations": 10000,
        "trend_games": 25,
        "nst_window": 14,
        "season_type": 2,
        "home_goalie": null,
        "away_goalie": null,
        "home_b2b": false,
        "away_b2b": false
    }
    """
    try:
        data = request.get_json(force=True)
        home_abbr = data.get("home_team", "").upper()
        away_abbr = data.get("away_team", "").upper()

        if not home_abbr or not away_abbr:
            return jsonify({"error": "home_team and away_team are required"}), 400

        home_raw = TEAM_ABBR_MAPPING.get(home_abbr, home_abbr)
        away_raw = TEAM_ABBR_MAPPING.get(away_abbr, away_abbr)

        date_str = data.get("date")
        game_date = _date.fromisoformat(date_str) if date_str else _date.today()

        stype = data.get("season_type", 2)
        sims = data.get("simulations", DEFAULT_SIMULATIONS)
        trend_games = data.get("trend_games", DEFAULT_TREND_GAMES)
        nst_days_window = data.get("nst_window")
        home_goalie = data.get("home_goalie") or None
        away_goalie = data.get("away_goalie") or None
        home_b2b = bool(data.get("home_b2b", False))
        away_b2b = bool(data.get("away_b2b", False))

        game_season, data_season, use_previous = get_data_season_for_game(
            game_date, NHL_SEASON_START_MONTH
        )

        fd_str = td_str = ""
        if nst_days_window:
            data_season_year = int(data_season[:4])
            season_end_date = _date(data_season_year + 1, 4, 30)
            td_day = min(game_date - timedelta(days=1), _date.today(), season_end_date)
            fd_day = td_day - timedelta(days=nst_days_window - 1)
            fd_str = fd_day.isoformat()
            td_str = td_day.isoformat()

        # Get Elo ratings
        try:
            home_elo_base = get_team_elo(home_raw)
            away_elo_base = get_team_elo(away_raw)
            if nst_days_window:
                home_elo = get_team_recent_form_rating(home_raw, days=nst_days_window)
                away_elo = get_team_recent_form_rating(away_raw, days=nst_days_window)
            else:
                home_elo = home_elo_base
                away_elo = away_elo_base
        except Exception as e:
            logger.warning(f"Failed to get Elo ratings: {e}")
            home_elo = home_elo_base = 1500.0
            away_elo = away_elo_base = 1500.0

        # Get lineups
        away_lineup_full = safe_api_call(
            get_confirmed_or_predicted_lineup,
            away_raw, game_date.isoformat(), None,
            service_name="NHL Lineup API (Away)",
            fallback={"forwards": [], "defense": [], "goalies": []},
        )
        home_lineup_full = safe_api_call(
            get_confirmed_or_predicted_lineup,
            home_raw, game_date.isoformat(), None,
            service_name="NHL Lineup API (Home)",
            fallback={"forwards": [], "defense": [], "goalies": []},
        )

        away_skaters = away_lineup_full.get("forwards", []) + away_lineup_full.get("defense", [])
        home_skaters = home_lineup_full.get("forwards", []) + home_lineup_full.get("defense", [])

        away_sk = [
            {"Name": format_initial_last(sanitize_text(p.get("name", ""))),
             "Position": sanitize_text(p.get("position", "")),
             "Confirmed": p.get("confirmed", False)}
            for p in away_skaters
        ]
        home_sk = [
            {"Name": format_initial_last(sanitize_text(p.get("name", ""))),
             "Position": sanitize_text(p.get("position", "")),
             "Confirmed": p.get("confirmed", False)}
            for p in home_skaters
        ]
        df_away_sk = pd.DataFrame(away_sk, columns=["Name", "Position", "Confirmed"])
        df_home_sk = pd.DataFrame(home_sk, columns=["Name", "Position", "Confirmed"])

        # Skater rates: use the lightweight exported JSON instead of loading
        # the full PBP parquet and running xG inference per request. This is the
        # main memory/CPU win for Render.
        try:
            data_season_year = int(data_season[:4])
            season_skill_cur = load_skater_rates_from_json(data_season_year, stype)
            if not season_skill_cur:
                logger.warning(f"JSON skater rates empty for {data_season}")
        except Exception as e:
            logger.warning(f"JSON skater rates failed: {e}")
            season_skill_cur = {}

        # Run simulation
        sim = simulate_matchup(
            home_abbr=home_raw,
            away_abbr=away_raw,
            game_date=game_date,
            stype=stype,
            sims=sims,
            trend_games=trend_games,
            use_recent_window_days=nst_days_window,
            season_skill_for_lineups=season_skill_cur if season_skill_cur else None,
            home_lineup_df=df_home_sk if not df_home_sk.empty else None,
            away_lineup_df=df_away_sk if not df_away_sk.empty else None,
            selected_home_goalie=home_goalie,
            selected_away_goalie=away_goalie,
            home_elo_override=home_elo,
            away_elo_override=away_elo,
            home_b2b=home_b2b,
            away_b2b=away_b2b,
        )

        result = _make_json_safe(sim)
        result["home_lineup"] = home_lineup_full
        result["away_lineup"] = away_lineup_full
        result["home_elo_base"] = float(home_elo_base)
        result["away_elo_base"] = float(away_elo_base)
        result["home_elo_adj"] = float(home_elo)
        result["away_elo_adj"] = float(away_elo)
        result["data_season"] = data_season

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── API: State ─────────────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    """Return application state info (Elo status, model status, etc.)."""
    try:
        state_info = get_state_info()
        elo_check = check_elo_data_availability()
        return jsonify({"state": state_info, "elo": elo_check})
    except Exception as e:
        logger.error(f"State API error: {e}")
        return jsonify({"error": str(e)}), 500


# ── API: Elo Leaderboard ───────────────────────────────────────────────

@app.route("/api/elo-leaderboard")
def api_elo_leaderboard():
    """Return the latest Elo ratings for all teams, highest first."""
    try:
        state = get_app_state()
        current_season = state.get("current_season")
        db = state.get("db")
        if db is None or not hasattr(db, "conn"):
            return jsonify({"error": "Database unavailable"}), 500

        cursor = db.conn.cursor()
        # Latest row per team for the current season, then sorted by rating.
        cursor.execute(
            """
            SELECT team_abbr, rating, games_played, date
            FROM (
                SELECT team_abbr, rating, games_played, date,
                       ROW_NUMBER() OVER (PARTITION BY team_abbr ORDER BY date DESC, rowid DESC) as rn
                FROM team_elo
                WHERE season = ?
            )
            WHERE rn = 1
            ORDER BY rating DESC
            """,
            (current_season,),
        )
        rows = [dict(r) for r in cursor.fetchall()]

        # If the current season has no ratings yet, fall back to the latest rows overall.
        if not rows:
            cursor.execute(
                """
                SELECT team_abbr, rating, games_played, date
                FROM (
                    SELECT team_abbr, rating, games_played, date,
                           ROW_NUMBER() OVER (PARTITION BY team_abbr ORDER BY date DESC, rowid DESC) as rn
                    FROM team_elo
                )
                WHERE rn = 1
                ORDER BY rating DESC
                """
            )
            rows = [dict(r) for r in cursor.fetchall()]

        return jsonify({
            "season": current_season,
            "teams": _make_json_safe(rows),
        })
    except Exception as e:
        logger.error(f"Elo leaderboard error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── API: Lookup ────────────────────────────────────────────────────────

@app.route("/api/lookup")
def api_lookup():
    """
    Search for games on a date.

    Query params:
        date (required): YYYY-MM-DD
    """
    from NHL.Lookup import (
        get_team_full_name, display_abbr_for_game,
    )

    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "date parameter is required (YYYY-MM-DD)"}), 400

    try:
        games = safe_api_call(
            get_games_on_date, date_str,
            service_name="NHL Schedule API", fallback=[],
        )

        result_games = []
        if games:
            for game in games:
                home_team = game.get("homeTeam") or {}
                away_team = game.get("awayTeam") or {}
                home_abbr = display_abbr_for_game(
                    home_team.get("abbrev", home_team.get("name", ""))
                )
                away_abbr = display_abbr_for_game(
                    away_team.get("abbrev", away_team.get("name", ""))
                )
                result_games.append({
                    "id": game.get("id"),
                    "home": home_abbr,
                    "away": away_abbr,
                    "home_name": get_team_full_name(home_team),
                    "away_name": get_team_full_name(away_team),
                    "home_score": home_team.get("score"),
                    "away_score": away_team.get("score"),
                    "startTime": game.get("startTimeUTC", game.get("gameDate", "")),
                    "state": game.get("gameState",
                              game.get("status", {}).get("abstractGameState", "")),
                })

        return jsonify({"date": date_str, "games": result_games})
    except Exception as e:
        logger.error(f"Lookup error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── API: Stats ─────────────────────────────────────────────────────────

def _current_season_start_year() -> int:
    """Return the start year of the current NHL season."""
    today = _date.today()
    return today.year if today.month >= NHL_SEASON_START_MONTH else today.year - 1


@app.route("/api/stats/teams")
def api_stats_teams():
    """Return PBP-derived team statistics."""
    season = request.args.get("season", "")
    stype = int(request.args.get("stype", "2"))
    return _fetch_pbp_stats("teams", season, stype)


@app.route("/api/stats/skaters")
def api_stats_skaters():
    """Return PBP-derived skater statistics."""
    season = request.args.get("season", "")
    stype = int(request.args.get("stype", "2"))
    return _fetch_pbp_stats("skaters", season, stype)


@app.route("/api/stats/goalies")
def api_stats_goalies():
    """Return PBP-derived goalie statistics."""
    season = request.args.get("season", "")
    stype = int(request.args.get("stype", "2"))
    return _fetch_pbp_stats("goalies", season, stype)


def _fetch_pbp_stats(table_type: str, season: str, stype: int):
    """
    Fetch PBP-derived stats and return as JSON.

    `season` is the YYYYYYYY form ("20242025"); if omitted, the current
    NHL season is used. We translate to the start year (2024) for the
    PBP shot-store lookup.

    Stats are served from the daily JSON cache first; only a cold cache
    falls back to on-the-fly computation from the PBP shot store.
    """
    try:
        try:
            start_year = int(str(season)[:4])
        except (ValueError, TypeError):
            start_year = _current_season_start_year()

        payload = load_cached_stats(table_type, start_year, stype)
        data = payload.get("data", [])
        if not data:
            return jsonify({"error": f"No {table_type} data available"}), 404

        return jsonify({
            "type": table_type,
            "data": _make_json_safe(data),
            "meta": {
                "season": payload.get("season"),
                "stype": payload.get("stype"),
                "source": payload.get("source"),
                "updated_at": payload.get("updated_at"),
            },
        })
    except Exception as e:
        logger.error(f"Stats API error ({table_type}): {e}")
        return jsonify({"error": str(e)}), 500


# ── API: Boxscore ──────────────────────────────────────────────────────

@app.route("/api/boxscore/<game_id>")
def api_boxscore(game_id: str):
    """Return a normalized boxscore for a given NHL game ID."""
    try:
        from NHL.ApiScrape import get_boxscore

        box = safe_api_call(
            get_boxscore, game_id,
            service_name="NHL Boxscore API", fallback={},
        )
        if not box:
            return jsonify({"error": "Boxscore unavailable"}), 404

        home = box.get("homeTeam", {})
        away = box.get("awayTeam", {})

        # Modern boxscore endpoint no longer exposes summary.teamGameStats,
        # so derive real special-teams and faceoff numbers from PBP.
        try:
            home_pp_opps, away_pp_opps = count_pp_opportunities(game_id)
        except Exception as e:
            logger.warning(f"Failed to count PP opportunities for {game_id}: {e}")
            home_pp_opps, away_pp_opps = 0, 0
        try:
            home_fo_wins, away_fo_wins = count_faceoffs(
                game_id,
                home_team_id=home.get("id"),
                away_team_id=away.get("id"),
            )
        except Exception as e:
            logger.warning(f"Failed to count faceoffs for {game_id}: {e}")
            home_fo_wins, away_fo_wins = 0, 0

        def _norm_team_info(t):
            if not isinstance(t, dict):
                return {}
            name_field = t.get("name") or t.get("commonName") or {}
            return {
                "id": t.get("id"),
                "abbrev": t.get("abbrev"),
                "name": name_field.get("default") if isinstance(name_field, dict) else str(name_field),
                "score": t.get("score"),
            }

        def _extract_summary_team_stats(side: str) -> Dict[str, Any]:
            """Pull clean team-level stats from the boxscore summary when available."""
            summary = box.get("summary", {}) if isinstance(box, dict) else {}
            team_game_stats = summary.get("teamGameStats", [])
            prefix = "home" if side == "homeTeam" else "away"
            stats: Dict[str, Any] = {}
            key_map = {
                "sog": "sog",
                "hits": "hits",
                "blockedShots": "blocked_shots",
                "giveaways": "giveaways",
                "takeaways": "takeaways",
                "penaltyMinutes": "pim",
                "powerPlay": "power_play",
                "faceoffWinningPctg": "faceoff_pct_raw",
            }
            for item in team_game_stats:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "")
                key = key_map.get(cat)
                if not key:
                    continue
                val = item.get(f"{prefix}Value")
                if val is None:
                    continue
                stats[key] = val
            return stats

        def _aggregate_team_stats(side: str) -> Dict[str, Any]:
            """Sum player boxscore stats into team totals, preferring summary data."""
            raw_groups = box.get("playerByGameStats", {}).get(side, {}) if isinstance(box, dict) else {}
            if isinstance(raw_groups, dict):
                skaters = raw_groups.get("forwards", []) + raw_groups.get("defense", [])
                goalies = raw_groups.get("goalies", [])
                raw = skaters + goalies
            elif isinstance(raw_groups, list):
                skaters = [p for p in raw_groups if str(p.get("positionCode", p.get("position", ""))).upper() != "G"]
                goalies = [p for p in raw_groups if str(p.get("positionCode", p.get("position", ""))).upper() == "G"]
                raw = raw_groups
            else:
                skaters, goalies, raw = [], [], []

            summary_stats = _extract_summary_team_stats(side)

            totals: Dict[str, Any] = {
                "sog": 0, "hits": 0, "blocked_shots": 0, "giveaways": 0,
                "takeaways": 0, "pim": 0, "goals": 0, "assists": 0,
                "power_play_goals": 0, "power_play_opps": 0,
                "faceoff_wins": 0, "faceoff_total": 0,
                "_faceoff_players": 0, "_faceoff_sum": 0.0,
            }
            for p in raw:
                if not isinstance(p, dict):
                    continue
                totals["sog"] += int(p.get("sog", 0) or 0)
                totals["hits"] += int(p.get("hits", 0) or 0)
                totals["blocked_shots"] += int(p.get("blockedShots", 0) or 0)
                totals["giveaways"] += int(p.get("giveaways", 0) or 0)
                totals["takeaways"] += int(p.get("takeaways", 0) or 0)
                totals["pim"] += int(p.get("pim", 0) or 0)
                totals["goals"] += int(p.get("goals", 0) or 0)
                totals["assists"] += int(p.get("assists", 0) or 0)
                totals["power_play_goals"] += int(p.get("powerPlayGoals", 0) or 0)
                fop = p.get("faceoffWinningPctg")
                fot = p.get("faceoffTaken")
                # The modern boxscore exposes per-player faceoffWinningPctg but
                # not the raw number of draws. Each player who took a draw has a
                # non-None percentage; we cannot derive an exact total without
                # raw counts. We used to sum 1 per player (giving 18 total for
                # a full roster), which produced wildly wrong records. Instead,
                # keep the percentage as an average and skip the fabricated
                # wins/total. When a real summary stat is available, it wins.
                if fop is not None and fot is not None:
                    try:
                        totals["faceoff_total"] += int(fot)
                        totals["faceoff_wins"] += round(float(fop) * int(fot))
                    except (TypeError, ValueError):
                        pass
                elif fop is not None:
                    # Tally how many players had a percentage so we can at least
                    # produce a team average when no summary stat exists.
                    totals["_faceoff_players"] = totals.get("_faceoff_players", 0) + 1
                    totals["_faceoff_sum"] = totals.get("_faceoff_sum", 0.0) + float(fop)

            # Prefer summary stats for display categories.
            for key in ("sog", "hits", "blocked_shots", "giveaways", "takeaways", "pim"):
                if key in summary_stats:
                    totals[key] = _to_int_safe(summary_stats[key])

            # Power-play display string (e.g. "1/4") from summary if present,
            # otherwise use the play-by-play-derived opportunities.
            pp_summary = summary_stats.get("power_play")
            if isinstance(pp_summary, str) and "/" in pp_summary:
                totals["power_play"] = pp_summary
                try:
                    made, attempted = pp_summary.split("/", 1)
                    totals["power_play_goals"] = int(made)
                    totals["power_play_opps"] = int(attempted)
                except Exception:
                    pass
            else:
                side_pp_opps = home_pp_opps if side == "homeTeam" else away_pp_opps
                totals["power_play_opps"] = side_pp_opps
                totals["power_play"] = f"{totals['power_play_goals']}/{totals['power_play_opps']}"

            # Faceoff data from PBP is authoritative. Player boxscore percentages
            # no longer include raw totals, so we count wins from PBP and the
            # total is home_wins + away_wins. The per-side wins are passed in via
            # the closure from the caller.
            is_home = side == "homeTeam"
            fo_wins = home_fo_wins if is_home else away_fo_wins
            fo_losses = away_fo_wins if is_home else home_fo_wins
            fo_total = fo_wins + fo_losses

            if fo_total > 0:
                totals["faceoff_wins"] = fo_wins
                totals["faceoff_total"] = fo_total
                totals["faceoff_pct"] = round(100 * fo_wins / fo_total, 1)
                totals["faceoff_record"] = f"{fo_wins}/{fo_total}"
            elif totals["faceoff_total"] > 0:
                totals["faceoff_pct"] = round(100 * totals["faceoff_wins"] / totals["faceoff_total"], 1)
                totals["faceoff_record"] = f"{int(totals['faceoff_wins'])}/{int(totals['faceoff_total'])}"
            else:
                totals["faceoff_pct"] = None
                totals["faceoff_record"] = "-"

            return totals

        def _to_int_safe(v: Any) -> int:
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return 0

        def _get_raw_players(side):
            raw_groups = box.get("playerByGameStats", {}).get(side, {}) if isinstance(box, dict) else {}
            if isinstance(raw_groups, dict):
                return (
                    raw_groups.get("forwards", []) + raw_groups.get("defense", []),
                    raw_groups.get("goalies", []),
                )
            elif isinstance(raw_groups, list):
                skaters = [p for p in raw_groups if str(p.get("positionCode", p.get("position", ""))).upper() != "G"]
                goalies = [p for p in raw_groups if str(p.get("positionCode", p.get("position", ""))).upper() == "G"]
                return skaters, goalies
            return [], []

        def _norm_skaters(side):
            skaters, _ = _get_raw_players(side)
            out = []
            for p in skaters:
                if not isinstance(p, dict):
                    continue
                nm = p.get("name") or {}
                name = nm.get("default") if isinstance(nm, dict) else str(nm)
                out.append({
                    "name": name,
                    "position": p.get("position", p.get("positionCode", "?")),
                    "goals": p.get("goals"),
                    "assists": p.get("assists"),
                    "sog": p.get("sog"),
                    "hits": p.get("hits"),
                    "blocked_shots": p.get("blockedShots"),
                    "takeaways": p.get("takeaways"),
                    "giveaways": p.get("giveaways"),
                    "pim": p.get("pim"),
                    "toi": p.get("toi"),
                })
            return out

        def _norm_goalies(side):
            _, goalies = _get_raw_players(side)
            out = []
            for p in goalies:
                if not isinstance(p, dict):
                    continue
                nm = p.get("name") or {}
                name = nm.get("default") if isinstance(nm, dict) else str(nm)
                saves = p.get("saves")
                shots_against = p.get("shotsAgainst") or p.get("shotsAgainstByStrength")
                goals_against = p.get("goalsAgainst")
                save_pct = p.get("savePctg")
                if save_pct is not None:
                    try:
                        save_pct = float(save_pct)
                        if save_pct > 1:
                            save_pct = save_pct / 100.0
                    except Exception:
                        save_pct = None
                out.append({
                    "name": name,
                    "position": "G",
                    "starter": False,
                    "saves": saves,
                    "shots_against": shots_against,
                    "goals_against": goals_against,
                    "save_pct": round(save_pct, 3) if save_pct is not None else None,
                    "toi": p.get("toi"),
                    "goals": p.get("goals"),
                    "assists": p.get("assists"),
                    "pim": p.get("pim"),
                })
            # Sort by time on ice descending so the starter is on top.
            out.sort(key=lambda g: _toi_seconds(g.get("toi") or "0:00"), reverse=True)
            if out:
                out[0]["starter"] = True
            return out

        def _toi_seconds(toi: str) -> int:
            if not toi or not isinstance(toi, str):
                return 0
            parts = toi.split(":")
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0

        period = box.get("periodDescriptor", {}) if isinstance(box, dict) else {}
        clock = box.get("clock", {}) if isinstance(box, dict) else {}
        outcome = box.get("gameOutcome", {}) if isinstance(box, dict) else {}
        home_stats = _aggregate_team_stats("homeTeam")
        away_stats = _aggregate_team_stats("awayTeam")
        return jsonify({
            "game_id": game_id,
            "state": box.get("gameState", box.get("status", {}).get("abstractGameState", "")),
            "period": period.get("number"),
            "period_type": period.get("periodType"),
            "clock": clock.get("timeRemaining"),
            "last_period_type": outcome.get("lastPeriodType"),
            "home_team": {**_norm_team_info(home), **home_stats},
            "away_team": {**_norm_team_info(away), **away_stats},
            "home_roster": _norm_skaters("homeTeam"),
            "away_roster": _norm_skaters("awayTeam"),
            "home_goalies": _norm_goalies("homeTeam"),
            "away_goalies": _norm_goalies("awayTeam"),
        })
    except Exception as e:
        logger.error(f"Boxscore API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── API: Player Props ─────────────────────────────────────────────────

@app.route("/api/player-props/<date_str>")
def api_player_props(date_str: str):
    """Return player prop odds for a given date."""
    try:
        from NHL.PlayerLinePredictor import (
            load_player_props_multi_day, DEFAULT_PLAYER_MARKETS,
        )

        game_date = _date.fromisoformat(date_str)
        markets = request.args.getlist("markets")
        if not markets:
            markets = list(DEFAULT_PLAYER_MARKETS)

        props = load_player_props_multi_day(tuple(markets), game_date)
        if not props:
            return jsonify({"date": date_str, "props": []})

        result = _make_json_safe(list(props))
        return jsonify({"date": date_str, "props": result})

    except Exception as e:
        logger.error(f"Player props error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── API: Season Options ────────────────────────────────────────────────

@app.route("/api/seasons")
def api_seasons():
    """Return available season options for stats."""
    options = _season_options()
    return jsonify({"seasons": [{"label": label, "key": key} for label, key in options]})


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NHL Game Predictor")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)