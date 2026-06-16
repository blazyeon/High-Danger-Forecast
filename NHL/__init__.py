"""
NHL Prediction System - Core modules
"""

from .AppState import get_app_state, get_state_info
from .Config import *
from .Errors import safe_api_call, retry_on_failure, safe_division, DataValidationError, SimulationError
from .Utils import (
    season_from_date,
    prev_season_key,
    normalize_name_key,
    sanitize_text,
    format_initial_last,
    get_column_safe,
)
from .ApiScrape import get_games_on_date, get_confirmed_or_predicted_lineup

__version__ = "2.0.0"

__all__ = [
    # Config
    'NHL_API_BASE',
    'NHL_STATS_API_BASE',
    'SIMULATION_PARAMS',
    'LEAGUE_AVERAGES',
    'MODEL_WEIGHTS',
    'PLAYER_PARAMS',
    'GOALIE_PARAMS',

    # Errors
    'DataValidationError',
    'SimulationError',
    'safe_api_call',
    'retry_on_failure',
    'safe_division',

    # Utils
    'season_from_date',
    'prev_season_key',
    'normalize_name_key',
    'sanitize_text',
    'format_initial_last',
    'get_column_safe',

    # ApiScrape
    'get_games_on_date',
    'get_confirmed_or_predicted_lineup',

    # AppState
    'get_app_state',
    'get_state_info',
]