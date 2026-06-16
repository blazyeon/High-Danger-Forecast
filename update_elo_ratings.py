"""
Update Elo ratings and game_results using NHL API PBP-derived stats.

Run:
  python update_elo_ratings.py --current-season --reset --initial-only
  python update_elo_ratings.py --season <SEASON> --training --reset
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from NHL.Utils import season_from_date
from NHL.StatsFromPBP import compute_skater_rates, TEAM_ID_TO_ABBR
from NHL.PlayByPlay import load_shot_store
from NHL.xGModel import load_xg_model, predict_xg
from EloMl.Database import EloDatabase
from EloMl.Ratings import EloConfig, PlayerEloSystem, TeamEloSystem
# NST.Cache no longer imported — we use PBP-derived stats instead.
# The function is still called load_nst_player_stats for compatibility
# with the rest of the file but is now backed by the NHL API PBP.

NHL_API_BASE = "https://api-web.nhle.com/v1"

VALID_NHL_TEAMS = {
    "ANA","ARI","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA",
    "LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS",
    "STL","TBL","TOR","VAN","VGK","WPG","WSH","UTA"
}
# NOTE: ARI (Arizona Coyotes) moved to Utah (UTA) for 2024-25+.
# Both abbreviations are included so historical data is not dropped.
# ARI should be mapped to UTA via Config.TEAM_ABBR_MAPPING when processing current-season games.

def get_current_season() -> Tuple[str, date, date]:
    """Get current NHL season."""
    today = date.today()
    year = today.year
    month = today.month

    if month >= 10:
        season_start_year = year
        season_str = f"{year}{year+1}"
    else:
        season_start_year = year - 1
        season_str = f"{year-1}{year}"

    start_date = date(season_start_year, 10, 7)
    end_date = today

    return season_str, start_date, end_date

def normalize_position(pos: str) -> str:
    """Normalize position codes."""
    pos = str(pos).upper().strip()
    if pos in ("C", "LW", "RW", "W", "L", "R"):
        return "F"
    elif pos in ("D", "LD", "RD"):
        return "D"
    elif pos in ("G",):
        return "G"
    else:
        if "D" in pos:
            return "D"
        elif "G" in pos:
            return "G"
        else:
            return "F"

def safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        if pd.isna(val):
            return default
        return float(str(val).replace('%', '').replace(',', ''))
    except:
        return default

def safe_int(val, default=0):
    """Safely convert to int."""
    try:
        if pd.isna(val):
            return default
        return int(float(str(val).replace(',', '')))
    except:
        return default

def load_nst_player_stats(season: str) -> Dict[str, Dict]:
    """
    Load comprehensive player stats.

    Now backed by NHL API PBP (was NST HTML scrape). The function name
    is kept for compatibility with the rest of this file; new code
    should call `compute_skater_rates` directly from `NHL.StatsFromPBP`.

    `season` is the YYYYYYYY form ("20242025"). We call
    `compute_skater_rates(start_year, stype=2)` which returns
    {name_key: {name, gpg, apg, sogpg, xgf_pg, gp, goals, shots, assists}}
    and reshape into the format the Elo initial-rating code expects.

    Fields we can't compute from PBP (toi, blocks, takeaways, giveaways,
    shots-blocked) are set to 0 — the consumer treats 0 as "no data" and
    applies a neutral adjustment, which is the right default.
    """
    logger.info(f"📊 Loading PBP player stats for {season}...")
    try:
        start_year = int(str(season)[:4])
    except (ValueError, TypeError):
        logger.error(f"❌ Invalid season format {season!r}, expected YYYYYYYY")
        return {}

    try:
        rates = compute_skater_rates(start_year, 2)
    except Exception as e:
        logger.error(f"❌ Failed to compute skater rates for {season}: {e}")
        return {}

    if not rates:
        logger.warning(f"⚠️ No PBP stats for {season}, returning empty dict")
        return {}

    # Per-player ixG: the xG model gives us per-shot xg summed to a
    # player. We need a separate groupby because compute_skater_rates
    # returns aggregate xgf per player but no per-player ixg. ixg is
    # the model's per-shot xG average — we approximate as xgf / shots
    # (which is essentially "expected goals per shot attempt", i.e. ixG).
    player_stats: Dict[str, Dict] = {}
    for name_key, d in rates.items():
        try:
            gp = int(d.get("gp", 0))
            if gp == 0:
                continue
            goals = int(d.get("goals", 0))
            assists = int(d.get("assists", 0))
            shots = int(d.get("shots", 0))
            # xgf_pg is xG per game. We want total xG to compute ixG
            # (which the Elo formula treats as expected goals per shot).
            xgf_total = float(d.get("xgf_pg", 0.0)) * gp
            ixg = (xgf_total / shots) if shots > 0 else 0.0
            points = goals + assists
            # Position default: 'F' (forward). The Elo formula treats
            # 'D' (defense) differently, so the misclassification is
            # ~uniform noise. We do not have roster-position data here
            # without an extra API call; PBP doesn't carry position
            # per event. NHL roster lookup could be added as a
            # follow-up via the api-web.nhle.com/v1/roster endpoint.
            player_stats[name_key] = {
                "name": d.get("name", ""),
                "position": "F",
                "team": "",  # would need a roster lookup to fill
                "gp": gp,
                "toi": 0.0,
                "goals": goals,
                "assists": assists,
                "points": points,
                "isf": shots,
                "ixg": ixg,
                "ish_pct": (goals / shots) if shots > 0 else 0.0,
                "cf": 0,
                "ca": 0,
                "cf_pct": 50.0,
                "xgf": xgf_total,
                "xga": 0.0,
                "xg_pct": 50.0,
                "hdcf": 0,
                "hdca": 0,
                "blocks": 0,
                "takeaways": 0,
                "giveaways": 0,
                "plus_minus": 0,
            }
        except Exception as e:
            logger.debug(f"Error parsing player row: {e}")

    logger.info(f"✓ Loaded {len(player_stats)} players from PBP")
    return player_stats

def calculate_player_initial_rating(stats: Dict, config: EloConfig) -> float:
    """Calculate initial Elo rating based on performance."""
    try:
        gp = stats.get('gp', 0)
        if gp == 0:
            return config.initial_player_rating
        
        position = stats.get('position', 'F')
        points_pg = stats.get('points', 0) / gp
        goals_pg = stats.get('goals', 0) / gp
        ixg_pg = stats.get('ixg', 0) / gp
        toi_total = stats.get('toi', 0)
        toi_pg = toi_total / gp if gp > 0 else 0
        blocks_pg = stats.get('blocks', 0) / gp
        
        if position == "F":
            performance = 0.0
            
            if points_pg >= 1.75:
                performance = 3.0 + (points_pg - 1.75) * 5.0
            elif points_pg >= 1.4:
                performance = 2.5 + (points_pg - 1.4) * 1.43
            elif points_pg >= 1.2:
                performance = 2.0 + (points_pg - 1.2) * 2.5
            elif points_pg >= 1.0:
                performance = 1.5 + (points_pg - 1.0) * 2.5
            elif points_pg >= 0.7:
                performance = 0.8 + (points_pg - 0.7) * 2.33
            else:
                performance = (points_pg - 0.5) * 1.6
            
            if goals_pg >= 0.7:
                performance += 0.5
            elif goals_pg >= 0.5:
                performance += 0.3
            
            if ixg_pg >= 0.35:
                performance += 0.3
            elif ixg_pg >= 0.25:
                performance += 0.15
            
            if toi_pg >= 20:
                performance += 0.3
            elif toi_pg >= 18:
                performance += 0.2
            
        elif position == "D":
            performance = 0.0
            
            if points_pg >= 1.0:
                performance = 2.0 + (points_pg - 1.0) * 2.5
            elif points_pg >= 0.7:
                performance = 1.5 + (points_pg - 0.7) * 1.67
            elif points_pg >= 0.5:
                performance = 1.0 + (points_pg - 0.5) * 2.5
            elif points_pg >= 0.3:
                performance = 0.5 + (points_pg - 0.3) * 2.5
            else:
                performance = (points_pg - 0.25) * 2.0
            
            if blocks_pg >= 2.0:
                performance += 0.5
            elif blocks_pg >= 1.5:
                performance += 0.3
            
            if toi_pg >= 25:
                performance += 0.5
            elif toi_pg >= 23:
                performance += 0.3
        else:
            return config.initial_player_rating
        
        elo_adjustment = performance * 100
        elo_adjustment = max(-400, min(450, elo_adjustment))
        
        initial_rating = config.initial_player_rating + elo_adjustment
        initial_rating = max(1100, min(1900, initial_rating))
        
        return float(initial_rating)
        
    except Exception as e:
        logger.warning(f"Error calculating initial rating: {e}")
        return config.initial_player_rating

def populate_initial_player_elo(
    season: str,
    db: EloDatabase,
    player_stats: Dict[str, Dict],
    min_games: int = 1
) -> int:
    """Populate initial player Elo ratings from PBP-derived stats."""
    logger.info(f"\n⚙️  Setting initial player Elo ratings for {season}...")
    
    config = EloConfig()
    processed = 0
    skipped_low_gp = 0
    
    for name_key, stats in player_stats.items():
        try:
            gp = stats.get('gp', 0)
            if gp < min_games:
                skipped_low_gp += 1
                continue
            
            name = stats['name']
            position = stats['position']
            team = stats.get('team', 'UNK')
            
            rating = calculate_player_initial_rating(stats, config)
            
            db.save_player_elo(
                player_name=name,
                position=position,
                team_abbr=team,
                rating=rating,
                games_played=0,
                game_date=date.today(),
                season=season,
                rating_change=0.0,
                recent_form=[]
            )
            
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"   Set initial ratings for {processed} players...")
            
        except Exception as e:
            logger.warning(f"Error setting initial rating: {e}")
    
    logger.info(f"✓ Set initial ratings for {processed} players")
    if skipped_low_gp > 0:
        logger.info(f"  (Skipped {skipped_low_gp} players with <{min_games} GP)")
    
    return processed

def get_games_on_date(date_str: str) -> list:
    """Get games from NHL API for a specific date."""
    try:
        url = f"{NHL_API_BASE}/score/{date_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get('games', [])
    except Exception as e:
        logger.debug(f"Error fetching games for {date_str}: {e}")
        return []

def populate_team_elo_from_games(season: str, db: EloDatabase) -> int:
    """Calculate team Elo from actual game results."""
    logger.info(f"\n⚙️  Calculating team Elo from game results for {season}...")
    
    config = EloConfig()
    season_start = date(int(season[:4]), 10, 7)
    today = date.today()
    
    team_records = {}
    current_date = season_start
    games_found = 0
    
    logger.info(f"Scanning games from {season_start} to {today}...")
    
    while current_date <= today:
        date_str = current_date.isoformat()
        
        try:
            games = get_games_on_date(date_str)
            
            for game in games:
                game_state = str(game.get('gameState', '')).upper()
                if game_state not in ('OFF', 'FINAL', 'OVER'):
                    continue
                
                home_team = game.get('homeTeam', {}) or {}
                away_team = game.get('awayTeam', {}) or {}
                
                home_abbr = (home_team.get('abbrev') or '').upper()
                away_abbr = (away_team.get('abbrev') or '').upper()
                
                if not home_abbr or not away_abbr:
                    continue
                
                if home_abbr not in VALID_NHL_TEAMS or away_abbr not in VALID_NHL_TEAMS:
                    continue
                
                home_score = home_team.get('score')
                away_score = away_team.get('score')
                
                if home_score is None or away_score is None:
                    continue
                
                if home_abbr not in team_records:
                    team_records[home_abbr] = {'wins': 0, 'losses': 0, 'otl': 0, 'points': 0}
                if away_abbr not in team_records:
                    team_records[away_abbr] = {'wins': 0, 'losses': 0, 'otl': 0, 'points': 0}
                
                was_overtime = False
                period_descriptor = game.get('periodDescriptor', {}) or {}
                period_type = period_descriptor.get('periodType', '').upper()
                
                if period_type in ('OT', 'SO'):
                    was_overtime = True
                
                home_won = home_score > away_score
                
                if home_won:
                    team_records[home_abbr]['wins'] += 1
                    team_records[home_abbr]['points'] += 2
                    
                    if was_overtime:
                        team_records[away_abbr]['otl'] += 1
                        team_records[away_abbr]['points'] += 1
                    else:
                        team_records[away_abbr]['losses'] += 1
                else:
                    team_records[away_abbr]['wins'] += 1
                    team_records[away_abbr]['points'] += 2
                    
                    if was_overtime:
                        team_records[home_abbr]['otl'] += 1
                        team_records[home_abbr]['points'] += 1
                    else:
                        team_records[home_abbr]['losses'] += 1
                
                games_found += 1
        
        except Exception as e:
            logger.debug(f"Error processing games for {date_str}: {e}")
        
        current_date += timedelta(days=1)
    
    logger.info(f"\n✓ Found {games_found} completed games")
    
    if not team_records:
        for team in VALID_NHL_TEAMS:
            db.save_team_elo(
                team_abbr=team,
                rating=1500.0,
                games_played=0,
                game_date=date.today(),
                season=season,
                rating_change=0.0,
                recent_form=[]
            )
        return len(VALID_NHL_TEAMS)
    
    processed = 0
    team_list = sorted(team_records.items(), key=lambda x: x[1]['points'], reverse=True)
    
    logger.info("\n🏆 Team Standings → Elo Ratings:")
    
    for team_abbr, record in team_list:
        gp = record['wins'] + record['losses'] + record['otl']
        if gp == 0:
            initial_rating = 1500.0
        else:
            win_pct = (record['wins'] + 0.5 * record['otl']) / gp
            elo_adjustment = (win_pct - 0.50) * 1000
            elo_adjustment = max(-300, min(300, elo_adjustment))
            initial_rating = config.initial_team_rating + elo_adjustment
        
        db.save_team_elo(
            team_abbr=team_abbr,
            rating=initial_rating,
            games_played=gp,
            game_date=date.today(),
            season=season,
            rating_change=0.0,
            recent_form=[]
        )
        
        processed += 1
        record_str = f"{record['wins']}-{record['losses']}-{record['otl']}"
        logger.info(f"  {processed:2d}. {team_abbr}: {record_str} ({record['points']} pts) → {initial_rating:.0f} Elo")
    
    logger.info(f"\n✓ Set initial ratings for {processed} teams")
    return processed

def _build_abbr_to_team_id() -> Dict[str, int]:
    """Build a best-effort abbreviation -> team_id lookup from the PBP table."""
    out: Dict[str, int] = {}
    for tid, abbr in TEAM_ID_TO_ABBR.items():
        if abbr not in out:
            out[abbr] = tid
    out["UTA"] = 59
    out["ARI"] = 30
    return out


def _team_ids_for_abbr(abbr: str, mapping: Dict[str, int]) -> List[int]:
    """Return candidate team_ids for an abbreviation (handles franchise moves)."""
    abbr = abbr.upper()
    if abbr == "ARI":
        return [30, 53]
    tid = mapping.get(abbr)
    return [tid] if tid else []


def _build_game_xg_lookup(
    season_start: int,
    stype: int = 2
) -> Tuple[Optional[pd.DataFrame], Dict[int, Dict[int, Dict[str, float]]]]:
    """
    Build a lookup of per-game xG and shot-on-goal counts by team_id.

    Returns:
        (shots_df_or_none, {game_id: {team_id: {'xgf': float, 'sf': int}}})
    """
    try:
        shots = load_shot_store(season_start, stype)
    except Exception as e:
        logger.warning(f"Could not load shot store for {season_start}: {e}")
        return None, {}

    if shots.empty or "game_id" not in shots.columns or "team_id" not in shots.columns:
        logger.warning(f"No PBP shots available for {season_start}")
        return shots, {}

    try:
        xg_model = load_xg_model()
        shots = shots.copy()
        shots["xg"] = predict_xg(shots, xg_model)
    except Exception as e:
        logger.warning(f"xG model unavailable for {season_start}: {e}")
        shots = shots.copy()
        shots["xg"] = 0.092

    shots["is_shot_on_goal"] = (
        shots.get("is_shot", pd.Series([0] * len(shots))).fillna(0).astype(int)
    )

    grouped = shots.groupby(["game_id", "team_id"], as_index=False).agg(
        xgf=("xg", "sum"),
        sf=("is_shot_on_goal", "sum"),
    )

    lookup: Dict[int, Dict[int, Dict[str, float]]] = {}
    for _, row in grouped.iterrows():
        gid = int(row["game_id"])
        tid = int(row["team_id"])
        lookup.setdefault(gid, {})[tid] = {
            "xgf": float(row["xgf"]),
            "sf": int(row["sf"]),
        }

    logger.info(f"✓ Built PBP xG lookup for {season_start}: {len(lookup)} games")
    return shots, lookup


def fetch_and_process_games(
    season_str: str,
    start_date: date,
    end_date: date,
    db: EloDatabase,
    player_elo: PlayerEloSystem,
    team_elo: TeamEloSystem,
    config: EloConfig
) -> int:
    """Fetch games and process them to update Elo and save to database."""

    logger.info("\n" + "=" * 60)
    logger.info("📊 PROCESSING GAMES")
    logger.info("=" * 60)
    logger.info(f"\nSeason: {season_str}")
    logger.info(f"Date range: {start_date} to {end_date}")

    logger.info(f"\nFetching games from NHL API...")

    all_games = []
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.isoformat()
        games = get_games_on_date(date_str)

        for game in games:
            game_state = str(game.get('gameState', '')).upper()
            if game_state in ('OFF', 'FINAL', 'OVER'):
                all_games.append(game)

        current_date += timedelta(days=1)

    logger.info(f"✓ Found {len(all_games)} completed games")

    if len(all_games) == 0:
        logger.warning("⚠️  No games found in date range")
        return 0

    # Load PBP-derived xG/shot counts for the season
    season_start = int(season_str[:4])
    _, game_xg_lookup = _build_game_xg_lookup(season_start, stype=2)
    abbr_to_tid = _build_abbr_to_team_id()

    logger.info(f"\n🔄 Processing {len(all_games)} games...")
    processed = 0
    errors = 0

    # Track games played manually
    team_games_count = {}

    for i, game in enumerate(all_games, 1):
        if i % 100 == 0:
            logger.info(f"   Progress: {i}/{len(all_games)} games...")
        
        try:
            game_id = game.get('id')
            game_date_str = game.get('gameDate')
            
            if not game_id or not game_date_str:
                errors += 1
                continue
            
            # Parse game date
            try:
                game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00')).date()
            except:
                try:
                    game_date = datetime.strptime(game_date_str[:10], '%Y-%m-%d').date()
                except:
                    errors += 1
                    continue
            
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            if home_team is None:
                home_team = {}
            if away_team is None:
                away_team = {}
            
            home_abbr = (home_team.get('abbrev') or '').upper()
            away_abbr = (away_team.get('abbrev') or '').upper()
            home_score = home_team.get('score')
            away_score = away_team.get('score')
            
            if not home_abbr or not away_abbr:
                errors += 1
                continue
            
            if home_score is None or away_score is None:
                errors += 1
                continue

            if home_abbr not in VALID_NHL_TEAMS or away_abbr not in VALID_NHL_TEAMS:
                errors += 1
                continue

            # Detect OT/SO from game data (must compute BEFORE the INSERT)
            period_descriptor = game.get('periodDescriptor', {}) or {}
            period_type = str(period_descriptor.get('periodType', '')).upper()
            is_ot_so = period_type in ('OT', 'SO')

            # Pull real xG and shot counts from PBP when available
            home_tids = _team_ids_for_abbr(home_abbr, abbr_to_tid)
            away_tids = _team_ids_for_abbr(away_abbr, abbr_to_tid)
            game_lookup = game_xg_lookup.get(int(game_id), {})

            home_xgf = 0.0
            away_xgf = 0.0
            home_sf_est = 0
            away_sf_est = 0

            for tid in home_tids:
                if tid in game_lookup:
                    home_xgf = game_lookup[tid]["xgf"]
                    home_sf_est = game_lookup[tid]["sf"]
                    break
            for tid in away_tids:
                if tid in game_lookup:
                    away_xgf = game_lookup[tid]["xgf"]
                    away_sf_est = game_lookup[tid]["sf"]
                    break

            # Fallback to score-based estimates when PBP data is missing
            if home_sf_est == 0:
                home_sf_est = max(25, min(42, int(30 + (home_score - away_score) * 1.5)))
            if away_sf_est == 0:
                away_sf_est = max(25, min(42, int(30 + (away_score - home_score) * 1.5)))
            if home_xgf == 0.0:
                home_xgf = float(home_score)
            if away_xgf == 0.0:
                away_xgf = float(away_score)

            # Save to game_results table
            cursor = db.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO game_results (
                    game_id, game_date, season,
                    home_team, away_team,
                    home_score, away_score,
                    home_xgf, away_xgf,
                    home_xga, away_xga,
                    is_ot_so, home_sf, away_sf
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id,
                game_date.isoformat(),
                season_str,
                home_abbr,
                away_abbr,
                home_score,
                away_score,
                home_xgf,
                away_xgf,
                away_xgf,
                home_xgf,
                1 if is_ot_so else 0,
                home_sf_est,
                away_sf_est,
            ))
            db.conn.commit()

            # Update team Elo
            home_team_obj = team_elo.get_or_create_team(home_abbr)
            away_team_obj = team_elo.get_or_create_team(away_abbr)

            if home_score > away_score:
                home_result = 1.0
                away_result = 0.25 if is_ot_so else 0.0
            elif away_score > home_score:
                home_result = 0.25 if is_ot_so else 0.0
                away_result = 1.0
            else:
                home_result = 0.5
                away_result = 0.5

            home_team_obj.update(
                opponent_rating=away_team_obj.rating,
                team_gf=home_score,
                team_ga=away_score,
                team_xgf=home_xgf,
                team_xga=away_xgf,
                team_sf=home_sf_est,
                team_sa=away_sf_est,
                result=home_result,
                config=config
            )

            away_team_obj.update(
                opponent_rating=home_team_obj.rating,
                team_gf=away_score,
                team_ga=home_score,
                team_xgf=away_xgf,
                team_xga=home_xgf,
                team_sf=away_sf_est,
                team_sa=home_sf_est,
                result=away_result,
                config=config
            )
            
            # Track games played manually
            team_games_count[home_abbr] = team_games_count.get(home_abbr, 0) + 1
            team_games_count[away_abbr] = team_games_count.get(away_abbr, 0) + 1
            
            # Save updated team Elo to database
            db.save_team_elo(
                team_abbr=home_abbr,
                rating=home_team_obj.rating,
                games_played=team_games_count[home_abbr],  # ✅ Use manual counter
                game_date=game_date,
                season=season_str,
                rating_change=0.0,
                recent_form=[]
            )
            
            db.save_team_elo(
                team_abbr=away_abbr,
                rating=away_team_obj.rating,
                games_played=team_games_count[away_abbr],  # ✅ Use manual counter
                game_date=game_date,
                season=season_str,
                rating_change=0.0,
                recent_form=[]
            )
            
            processed += 1
            
        except Exception as e:
            if errors < 5:
                logger.error(f"   Game {i}: Exception: {e}")
            errors += 1
            continue
    
    logger.info(f"\n" + "=" * 60)
    logger.info("✅ PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Games processed: {processed}")
    logger.info(f"   Errors: {errors}")
    
    return processed

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update NHL Elo ratings with comprehensive NST data"
    )
    parser.add_argument("--current-season", action="store_true",
                       help="Update current season (for predictions)")
    parser.add_argument("--season", type=str,
                       help="Specific season (e.g., 20242025)")
    parser.add_argument("--training", action="store_true",
                       help="Training data (separate database)")
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of parallel workers")
    parser.add_argument("--reset", action="store_true",
                       help="Reset Elo ratings before updating")
    parser.add_argument("--initial-only", action="store_true",
                       help="Set initial ratings only, don't process games")
    parser.add_argument("--min-games", type=int, default=1,
                       help="Minimum games played to include player")
    args = parser.parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("🏒 NHL Elo Rating Updater (NST Integration)")
    logger.info("=" * 60 + "\n")

    if args.current_season or (not args.season and not args.training):
        season_str, start_date, end_date = get_current_season()
        db_path = "elo_ratings.db"
    elif args.training or args.season:
        season_str = args.season
        start_year = int(args.season[:4])
        start_date = date(start_year, 10, 1)
        end_date = date(start_year + 1, 6, 30)
        db_path = f"training_data_{season_str}.db"
    else:
        season_str, start_date, end_date = get_current_season()
        db_path = "elo_ratings.db"

    logger.info(f"📅 Season: {season_str}")
    logger.info(f"   {start_date} to {end_date}")
    logger.info(f"   💾 Database: {db_path}")

    db = EloDatabase(db_path)
    config = EloConfig()

    if args.reset:
        logger.info("\n🔄 Resetting Elo ratings...")
        cursor = db.conn.cursor()
        cursor.execute("DELETE FROM team_elo WHERE season = ?", (season_str,))
        cursor.execute("DELETE FROM player_elo WHERE season = ?", (season_str,))
        db.conn.commit()
        logger.info(f"✅ Database cleared for {season_str}")

    player_stats = load_nst_player_stats(season_str)
    if player_stats:
        populate_initial_player_elo(season_str, db, player_stats, min_games=args.min_games)
    else:
        logger.warning("⚠️  PBP player data unavailable. Skipping player Elo init.")

    populate_team_elo_from_games(season_str, db)

    if args.initial_only:
        logger.info("\n" + "=" * 60)
        logger.info("✅ Initial Ratings Set Successfully!")
        logger.info("=" * 60)
        logger.info(f"\nMode: Initial ratings only (--initial-only)")
        logger.info(f"Database: {db_path}")
        logger.info(f"\nNext: python run_app.py\n")
        db.close()
        return 0

    player_elo = PlayerEloSystem(config)
    team_elo = TeamEloSystem(config)
    
    processed = fetch_and_process_games(
        season_str,
        start_date,
        end_date,
        db,
        player_elo,
        team_elo,
        config
    )
    
    logger.info(f"\n🎯 Next step:")
    if args.training:
        logger.info(f"   python train_model.py --training-db {db_path}")
    else:
        logger.info(f"   python run_app.py")
    logger.info("")
    
    db.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())