"""
Persistent application state for Elo ratings and ML model.
Uses module-level singleton instead of Streamlit session state.

MODIFIED: Only loads CURRENT SEASON Elo ratings (2025-2026).
Includes early-season fallback to previous season for stability.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import logging
from pathlib import Path
from datetime import date

from EloMl.Database import EloDatabase
from EloMl.Ratings import PlayerEloSystem, TeamEloSystem, EloConfig
from EloMl.MLModel import EloMLPredictor, ModelConfig
from EloMl.Features import EloFeatureEngine
from Calibration import Calibrator
from NHL.Utils import season_from_date

logger = logging.getLogger(__name__)

# Module-level singleton
_app_state_instance: Optional[Dict[str, Any]] = None


def get_app_state() -> Dict[str, Any]:
    """
    Get or initialize persistent application state.
    Uses module-level singleton instead of Streamlit session state.
    """
    global _app_state_instance

    if _app_state_instance is not None:
        return _app_state_instance

    logger.info("🔧 Initializing NHL prediction application state...")

    try:
        # Get current season
        today = date.today()
        current_season = season_from_date(today.isoformat())
        logger.info(f"📅 Current Season: {current_season}")

        # Initialize database
        db_path = Path("elo_ratings.db")
        db_missing = not db_path.exists() or db_path.stat().st_size == 0
        db = EloDatabase(str(db_path))
        logger.info(f"✓ Database connected: {db_path}")

        # Initialize Elo systems
        config = EloConfig()
        player_elo = PlayerEloSystem(config=config)
        team_elo = TeamEloSystem(config=config)

        # Check season bleed / stale data
        season_bleed_warning = None
        try:
            cur = db.conn.cursor()
            cur.execute("SELECT MAX(season) FROM team_elo")
            max_season_row = cur.fetchone()
            max_season = max_season_row[0] if max_season_row else None
            if max_season and max_season != current_season:
                season_bleed_warning = (
                    f"Found Elo data for season {max_season} but current season is {current_season}. "
                    "Run: python update_elo_ratings.py --current-season --reset"
                )
        except Exception:
            pass

        # Load CURRENT SEASON ratings from database (with early-season fallback)
        teams_loaded = _load_team_ratings_from_db(team_elo, db, current_season)
        players_loaded = _load_player_ratings_from_db(player_elo, db, current_season)

        if db_missing or (teams_loaded == 0 and players_loaded == 0):
            msg = (
                "No Elo data found for the current season. "
                "Run: python update_elo_ratings.py --current-season --reset --initial-only"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"✓ Loaded Elo ratings from {current_season}: {teams_loaded} teams, {players_loaded} players")

        if teams_loaded == 0:
            logger.warning(
                f"⚠️  No team Elo data found for {current_season}. "
                f"Run: python update_elo_ratings.py --current-season"
            )

        if players_loaded == 0:
            logger.warning(
                f"⚠️  No player Elo data found for {current_season}. "
                f"Run: python update_elo_ratings.py --current-season"
            )

        # Initialize feature engine
        feature_engine = EloFeatureEngine(player_elo, team_elo, config)
        logger.info("✓ Feature engine initialized")

        # Initialize ML model
        ml_model = EloMLPredictor(model_id="main", config=ModelConfig())

        # Try to load trained model
        model_path = Path("models/main_model.pkl")
        if model_path.exists():
            try:
                ml_model.load(str(model_path))
                logger.info(f"✓ ML model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load ML model: {e}")
                logger.info("   Model will use baseline predictions until trained")
        else:
            logger.info(f"ℹ️  No trained model found at {model_path}")
            logger.info(f"   Run: python train_model.py --season {current_season}")

        # Try to load fitted calibrator
        calibrator = None
        calib_path = Path("models/calibrator.pkl")
        if calib_path.exists():
            try:
                calibrator = Calibrator.load(str(calib_path))
                logger.info(f"✓ Calibrator loaded from {calib_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load calibrator: {e}")

        from datetime import datetime
        state = {
            'db': db,
            'player_elo': player_elo,
            'team_elo': team_elo,
            'feature_engine': feature_engine,
            'ml_model': ml_model,
            'calibrator': calibrator,
            'config': config,
            'current_season': current_season,
            'initialized_at': datetime.now().isoformat(),
            'season_bleed_warning': season_bleed_warning,
        }

        _app_state_instance = state
        logger.info("✅ Application state initialized successfully")

        return state

    except Exception as e:
        logger.error(f"❌ Failed to initialize application state: {e}", exc_info=True)
        logger.warning("⚠️  Using fallback state with default values")
        fallback_state = _create_fallback_state()
        fallback_state['init_error'] = str(e)
        _app_state_instance = fallback_state
        return fallback_state


def _load_team_ratings_from_db(
    team_elo: TeamEloSystem,
    db: EloDatabase,
    current_season: str
) -> int:
    """
    Load most recent team ratings from database.
    Loads from CURRENT SEASON with early-season fallback to previous season.
    """
    try:
        cursor = db.conn.cursor()

        cursor.execute("""
            SELECT team_abbr, rating, games_played, recent_form
            FROM (
                SELECT team_abbr, rating, games_played, recent_form,
                       ROW_NUMBER() OVER (PARTITION BY team_abbr ORDER BY date DESC) as rn
                FROM team_elo
                WHERE season = ?
            ) t
            WHERE rn = 1
        """, (current_season,))

        count = 0
        for row in cursor.fetchall():
            team_abbr = row['team_abbr']
            rating = row['rating']
            games_played = row['games_played']
            recent_form_json = row['recent_form']

            team = team_elo.get_or_create_team(team_abbr)
            team.rating = rating
            team.games_played = games_played

            if recent_form_json:
                try:
                    import json
                    team.recent_form = json.loads(recent_form_json)
                except Exception:
                    pass

            count += 1

        # Early season fallback: load previous season if insufficient data
        if count > 0:
            avg_games = sum(t.games_played for t in team_elo.teams.values()) / count

            if avg_games < 5:
                logger.warning(
                    f"🔄 Early season detected (avg {avg_games:.1f} games played). "
                    f"Loading previous season as baseline for missing teams..."
                )
                prev_season = _get_previous_season(current_season)
                fallback_count = _load_previous_season_fallback(
                    team_elo, cursor, prev_season
                )

                if fallback_count > 0:
                    logger.info(f"✓ Added {fallback_count} teams from previous season ({prev_season})")
                    count += fallback_count

        elif count == 0:
            logger.warning(
                f"⚠️  No team data found for {current_season}. "
                f"Attempting to load previous season..."
            )
            prev_season = _get_previous_season(current_season)
            count = _load_previous_season_fallback(
                team_elo, cursor, prev_season, full_fallback=True
            )

            if count > 0:
                logger.info(
                    f"✓ Loaded {count} teams from previous season ({prev_season}) "
                    f"with 10% regression to mean"
                )

        return count

    except Exception as e:
        logger.error(f"Error loading team ratings from {current_season}: {e}")
        return 0


def _load_previous_season_fallback(
    team_elo: TeamEloSystem,
    cursor,
    prev_season: str,
    full_fallback: bool = False
) -> int:
    """Load previous season ratings as fallback for missing teams."""
    try:
        import json

        cursor.execute("""
            SELECT team_abbr, rating, games_played, recent_form
            FROM (
                SELECT team_abbr, rating, games_played, recent_form,
                       ROW_NUMBER() OVER (PARTITION BY team_abbr ORDER BY date DESC) as rn
                FROM team_elo
                WHERE season = ?
            ) t
            WHERE rn = 1
        """, (prev_season,))

        fallback_count = 0
        for row in cursor.fetchall():
            team_abbr = row['team_abbr']
            prev_rating = row['rating']
            prev_games_played = row['games_played']
            recent_form_json = row['recent_form']

            if full_fallback or team_abbr not in team_elo.teams:
                team = team_elo.get_or_create_team(team_abbr)
                team.rating = prev_rating * 0.9 + 1500 * 0.1
                team.games_played = 0
                team.recent_form = []
                fallback_count += 1

                logger.debug(
                    f"  {team_abbr}: {prev_rating:.0f} → {team.rating:.0f} "
                    f"(regressed from {prev_season})"
                )

        return fallback_count

    except Exception as e:
        logger.error(f"Error loading previous season fallback from {prev_season}: {e}")
        return 0


def _load_player_ratings_from_db(
    player_elo: PlayerEloSystem,
    db: EloDatabase,
    current_season: str,
    limit: int = 5000
) -> int:
    """Load most recent player ratings from database."""
    try:
        cursor = db.conn.cursor()

        cursor.execute("""
            SELECT player_name, position, rating, games_played, recent_form
            FROM (
                SELECT player_name, position, rating, games_played, recent_form,
                       ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC) as rn
                FROM player_elo
                WHERE season = ?
            ) p
            WHERE rn = 1
            ORDER BY games_played DESC
            LIMIT ?
        """, (current_season, limit))

        count = 0
        for row in cursor.fetchall():
            player_name = row['player_name']
            position = row['position']
            rating = row['rating']
            games_played = row['games_played']
            recent_form_json = row['recent_form']

            player = player_elo.get_or_create_player(player_name, position)
            player.rating = rating
            player.games_played = games_played

            if recent_form_json:
                try:
                    import json
                    player.recent_form = json.loads(recent_form_json)
                except Exception:
                    pass

            count += 1

        # Early season fallback for players
        if count < 500:
            logger.warning(
                f"🔄 Few players found for {current_season} ({count}). "
                f"Loading top players from previous season..."
            )
            prev_season = _get_previous_season(current_season)
            fallback_count = _load_previous_season_players_fallback(
                player_elo, cursor, prev_season, limit=2000
            )

            if fallback_count > 0:
                logger.info(f"✓ Added {fallback_count} players from previous season ({prev_season})")
                count += fallback_count

        return count

    except Exception as e:
        logger.error(f"Error loading player ratings from {current_season}: {e}")
        return 0


def _load_previous_season_players_fallback(
    player_elo: PlayerEloSystem,
    cursor,
    prev_season: str,
    limit: int = 2000
) -> int:
    """Load previous season player ratings as fallback for missing players."""
    try:
        import json

        cursor.execute("""
            SELECT player_name, position, rating, games_played, recent_form
            FROM (
                SELECT player_name, position, rating, games_played, recent_form,
                       ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC) as rn
                FROM player_elo
                WHERE season = ?
            ) p
            WHERE rn = 1
            ORDER BY games_played DESC
            LIMIT ?
        """, (prev_season, limit))

        fallback_count = 0
        for row in cursor.fetchall():
            player_name = row['player_name']
            position = row['position']
            prev_rating = row['rating']

            if player_name not in player_elo.players:
                player = player_elo.get_or_create_player(player_name, position)
                player.rating = prev_rating * 0.85 + 1500 * 0.15
                player.games_played = 0
                player.recent_form = []
                fallback_count += 1

        return fallback_count

    except Exception as e:
        logger.error(f"Error loading previous season players from {prev_season}: {e}")
        return 0


def _get_previous_season(current_season: str) -> str:
    """Get previous season string from current season (e.g., '20252026' → '20242025')."""
    try:
        start_year = int(current_season[:4])
        prev_start = start_year - 1
        prev_end = start_year
        return f"{prev_start}{prev_end}"
    except (ValueError, IndexError):
        logger.error(f"Invalid season format: {current_season}")
        today = date.today()
        year = today.year if today.month >= 10 else today.year - 1
        return f"{year - 1}{year}"


def _create_fallback_state() -> Dict[str, Any]:
    """Create minimal fallback state if initialization fails."""
    logger.warning("Creating fallback state with default values")

    config = EloConfig()
    player_elo = PlayerEloSystem(config=config)
    team_elo = TeamEloSystem(config=config)
    feature_engine = EloFeatureEngine(player_elo, team_elo, config)
    ml_model = EloMLPredictor(model_id="fallback", config=ModelConfig())

    # Create in-memory database
    db = EloDatabase(":memory:")

    # Get current season
    today = date.today()
    current_season = season_from_date(today.isoformat())

    return {
        'db': db,
        'player_elo': player_elo,
        'team_elo': team_elo,
        'feature_engine': feature_engine,
        'ml_model': ml_model,
        'calibrator': None,
        'config': config,
        'current_season': current_season,
        'is_fallback': True,
    }


def reset_app_state():
    """Reset application state (useful for testing or forcing reload)."""
    global _app_state_instance
    if _app_state_instance is not None:
        try:
            _app_state_instance['db'].close()
        except Exception:
            pass
    _app_state_instance = None
    logger.info("Application state reset")


def get_state_info() -> Dict[str, Any]:
    """Get information about current application state."""
    global _app_state_instance

    if _app_state_instance is None:
        return {
            'initialized': False,
            'status': 'not_initialized',
        }

    state = _app_state_instance

    try:
        team_elo = state['team_elo']
        player_elo = state['player_elo']
        ml_model = state['ml_model']
        current_season = state.get('current_season', 'unknown')

        return {
            'initialized': True,
            'status': 'ready',
            'is_fallback': state.get('is_fallback', False),
            'current_season': current_season,
            'n_teams': len(team_elo.teams),
            'n_players': len(player_elo.players),
            'ml_model_trained': ml_model.is_trained,
            'ml_model_id': ml_model.model_id,
            'db_path': str(state['db'].db_path) if hasattr(state['db'], 'db_path') else 'in-memory',
        }
    except Exception as e:
        return {
            'initialized': True,
            'status': 'error',
            'error': str(e),
        }


def get_team_elo(team_abbr: str) -> float:
    """Get base Elo rating for a team (current season only)."""
    state = get_app_state()

    if state.get('is_fallback'):
        return 1500.0

    team_elo_system = state['team_elo']
    return team_elo_system.get_team_rating(team_abbr.upper())


def get_team_recent_form_rating(team_abbr: str, days: int = 14) -> float:
    """Get team's Elo rating adjusted for recent form."""
    state = get_app_state()

    if state.get('is_fallback'):
        return 1500.0

    team_elo_system = state['team_elo']
    team = team_elo_system.get_or_create_team(team_abbr.upper())

    return team.get_recent_form_rating(days=days)


def get_player_elo(player_name: str) -> float:
    """Get Elo rating for a player (current season only)."""
    state = get_app_state()

    if state.get('is_fallback'):
        return 1500.0

    player_elo_system = state['player_elo']
    return player_elo_system.get_player_rating(player_name)


def check_elo_data_availability() -> Dict[str, Any]:
    """Check if Elo data is available for current season."""
    state = get_app_state()
    current_season = state.get('current_season', 'unknown')

    team_elo = state['team_elo']
    player_elo = state['player_elo']

    n_teams = len(team_elo.teams)
    n_players = len(player_elo.players)

    avg_team_games = 0
    if n_teams > 0:
        total_games = sum(t.games_played for t in team_elo.teams.values())
        avg_team_games = total_games / n_teams

    warnings = []

    if n_teams == 0:
        warnings.append(
            f"⚠️  No team Elo data for {current_season}. "
            f"Run: python update_elo_ratings.py --current-season --reset"
        )
    elif n_teams < 32:
        warnings.append(
            f"⚠️  Only {n_teams}/32 teams have Elo data. "
            f"Some teams might be missing."
        )

    if avg_team_games < 5 and n_teams > 0:
        warnings.append(
            f"ℹ️  Early season: Teams have only played ~{avg_team_games:.0f} games. "
            f"Elo ratings may include previous season data with regression to mean."
        )

    if n_players == 0:
        warnings.append(
            f"⚠️  No player Elo data for {current_season}. "
            f"Player-level features will use defaults."
        )
    elif n_players < 500:
        warnings.append(
            f"ℹ️  Limited player data ({n_players} players). "
            f"May include previous season data for missing players."
        )

    if state.get("season_bleed_warning"):
        warnings.append(state["season_bleed_warning"])

    return {
        'season': current_season,
        'n_teams': n_teams,
        'n_players': n_players,
        'avg_team_games': avg_team_games,
        'warnings': warnings,
        'data_available': n_teams > 0,
    }


__all__ = [
    'get_app_state',
    'reset_app_state',
    'get_state_info',
    'get_team_elo',
    'get_team_recent_form_rating',
    'get_player_elo',
    'check_elo_data_availability',
]