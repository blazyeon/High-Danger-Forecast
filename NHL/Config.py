"""
Configuration constants for NHL Game Predictor.
Consolidated, deduplicated, and tuned for 2024-25 season baselines.
"""

from typing import Dict, Any, List, Tuple
import os

# ── APIs ────────────────────────────────────────────────────────────
NHL_API_BASE = "https://api-web.nhle.com/v1"
NHL_STATS_API_BASE = "https://statsapi.web.nhl.com/api/v1"
NST_BASE = "https://www.naturalstattrick.com"

# ── Network ─────────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 25
RATE_LIMIT_SLEEP_SECONDS = float(os.environ.get("RATE_LIMIT_SLEEP", "0.35"))
RATE_LIMIT_JITTER_SECONDS = float(os.environ.get("RATE_LIMIT_JITTER", "0.12"))
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Cache ───────────────────────────────────────────────────────────
CACHE_DIR = os.environ.get("NST_CACHE_DIR", "nst_cache")
CACHE_TTL_SECONDS = 6 * 60 * 60
CACHE_MAX_AGE_DAYS = 7

# ── Season ────────────────────────────────────────────────────────────
CURRENT_SEASON_YEAR = 2025
NHL_SEASON_START_MONTH = 10

# ── Team Mapping ────────────────────────────────────────────────────
TEAM_ABBR_MAPPING = {"ARI": "UTA"}

DIVISIONS: Dict[str, List[str]] = {
    "Atlantic":     ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    "Central":      ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"],
    "Pacific":      ["ANA", "CGY", "EDM", "LAK", "SJS", "SEA", "VAN", "VGK"],
}

NST_ABBR_TO_FULL: Dict[str, str] = {
    "BOS": "Boston Bruins",      "BUF": "Buffalo Sabres",
    "DET": "Detroit Red Wings",  "FLA": "Florida Panthers",
    "MTL": "Montreal Canadiens","OTT": "Ottawa Senators",
    "TBL": "Tampa Bay Lightning","TOR": "Toronto Maple Leafs",
    "CAR": "Carolina Hurricanes","CBJ": "Columbus Blue Jackets",
    "NJD": "New Jersey Devils",  "NYI": "New York Islanders",
    "NYR": "New York Rangers",   "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins","WSH": "Washington Capitals",
    "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",       "MIN": "Minnesota Wild",
    "NSH": "Nashville Predators","STL": "St. Louis Blues",
    "UTA": "Utah Mammoth",       "WPG": "Winnipeg Jets",
    "ANA": "Anaheim Ducks",      "CGY": "Calgary Flames",
    "EDM": "Edmonton Oilers",    "LAK": "Los Angeles Kings",
    "SJS": "San Jose Sharks",    "SEA": "Seattle Kraken",
    "VAN": "Vancouver Canucks",  "VGK": "Vegas Golden Knights",
    "T.B": "Tampa Bay Lightning","ARI": "Arizona Coyotes",
    "UTAH": "Utah Mammoth",
}

# ── Simulation ──────────────────────────────────────────────────────
DEFAULT_SIMULATIONS = 10000
DEFAULT_TREND_GAMES = 6

SIMULATION_PARAMS = {
    "min_goals": 0.05,
    "max_goals": 10.0,
    "shock_sigma": 0.22,
    "empty_net_probability": 0.28,     # dynamic in code; this is the ceiling
    "empty_net_2goal_probability": 0.15,
    "blowout_probability": 0.06,
    "blowout_boost": 1.35,
    "correlation_rho": 0.38,
}

# ── League Baseline (2024-25) ─────────────────────────────────────
LEAGUE_AVERAGES = {
    "sv_pct": 0.905,
    "goals_per_game": 3.0,
    "shots_per_game": 30.0,
    "shooting_pct": 0.092,
    "powerplay_minutes": 6.0,
    "min_shots": 22.0,
    "max_shots": 40.0,
}

# ── Model Weights ─────────────────────────────────────────────────
MODEL_WEIGHTS = {
    "xg_for_weight": 0.55,
    "gf_weight": 0.20,
    "xga_weight": 0.25,
    "recent_form_weight": 0.12,
    "opponent_defense_weight": 0.06,
    "lineup_impact_weight": 0.10,
    "pp_pk_weight": 0.70,
    "goalie_impact_weight": 0.75,
    "season_blend_current": 0.70,
    "season_blend_previous": 0.30,
    "elo_winprob_weight": 0.34,      # weight for Elo win probability
    "simulation_winprob_weight": 0.51, # weight for simulation win probability
    "ml_winprob_weight": 0.15,       # weight for ML win probability
}

# ── Player / Goalie / Special Teams ───────────────────────────────
PLAYER_PARAMS = {
    "line_weights": {
        "forward": {1: 1.00, 2: 0.82, 3: 0.66, 4: 0.52},
        "defense": {1: 0.55, 2: 0.50, 3: 0.45},
    },
    "toi_estimates": {
        "forward": {1: 19.5, 2: 17.0, 3: 13.5, 4: 10.0},
        "defense": {1: 24.0, 2: 20.0, 3: 14.0},
    },
    "hot_streak_boost": 0.20,
    "cold_streak_penalty": -0.06,
    "pp_boost": 1.6,
    "top6_boost": 1.4,
    "min_shooting_pct": 0.02,
    "max_shooting_pct": 0.35,
}

GOALIE_PARAMS = {
    "default_sv_pct": 0.905,
    "default_dsv_pct": 0.0,
    "sv_pct_noise_std": 0.02,
    "min_sv_pct": 0.01,
    "max_sv_pct": 0.999,
    "max_dsv_pct_impact": 0.5,
}

SPECIAL_TEAMS_PARAMS = {
    "max_pp_goal_impact": 0.30,
}

VENUE_ADV_PARAMS = {
    "league_home_win_pct": 0.545,
    "league_away_win_pct": 0.455,
    "baseline_goals": 0.30,
    "overperformance_scale": 1.2,
    "max_adjustment": 0.70,
}

REST_TRAVEL_PARAMS = {
    "back_to_back_penalty": -0.08,      # ~8% reduction in expected goals
    "rest_diff_penalty": -0.02,          # per day of rest disadvantage
    "travel_penalty_per_km": -0.0002,    # fatigue per km traveled
    "cross_country_penalty": -0.04,     # extra penalty for 2500+ km / 2.5+ tz
    "min_multiplier": 0.85,
    "max_multiplier": 1.08,
}

# ── Connection / Retry ──────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 0.75
CONNECTION_POOL_SIZE = 10
CONNECTION_POOL_MAXSIZE = 20

# ── Position Lists ──────────────────────────────────────────────────
FORWARD_POSITIONS = ["C", "L", "R", "LW", "RW", "W"]
DEFENSE_POSITIONS = ["D", "LD", "RD"]
GOALIE_POSITIONS = ["G", "Goalie"]

# ── Season Range ────────────────────────────────────────────────────
EARLIEST_SEASON_YEAR = 2010


def _season_options(
    start_year: int = EARLIEST_SEASON_YEAR,
    end_year: int = CURRENT_SEASON_YEAR
) -> List[Tuple[str, str]]:
    """Generate season options for dropdown."""
    opts: List[Tuple[str, str]] = []
    for y in range(start_year, end_year + 1):
        key = f"{y}{y+1}"
        label = f"{y}-{str(y+1)[-2:]}"
        opts.append((label, key))
    return opts


# ── UI ──────────────────────────────────────────────────────────────
UI_PARAMS = {
    "logo_width": 110,
    "logo_height": 74,
}

# ── Column Name Variations (for NST scraping) ───────────────────────
COLUMN_VARIATIONS = {
    "player": ["Player", "Name", "Skater", "Goalie", "player name", "playername"],
    "team": ["Team", "Tm", "team"],
    "games_played": ["GP", "Games Played", "gp"],
    "games_started": ["GS", "Games Started", "gs"],
    "goals": ["G", "Goals", "goals"],
    "assists": ["A", "Assists", "Total Assists"],
    "points": ["P", "Points", "PTS", "Total Points"],
    "shots": ["S", "Shots", "SOG", "Shots On Goal", "SF", "ISF"],
    "save_pct": [
        "Sv%", "SV%", "Save%", "Save Pct", "SavePct", "Save Percentage",
        "SavePercentage", "savePercentage", "SV PCT", "Sv Pct", "SVP", "SVPCT",
    ],
    "goals_for": ["GF", "Goals For"],
    "goals_against": ["GA", "Goals Against", "goalsAgainst"],
    "expected_goals_for": ["xGF", "Expected Goals For"],
    "expected_goals_against": ["xGA", "Expected Goals Against"],
    "shots_for": ["SF", "Shots For"],
    "shots_against": ["SA", "Shots Against", "shotsAgainst"],
    "toi": ["TOI", "Time On Ice", "ATOI", "Average TOI"],
}
