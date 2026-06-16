"""
Predict the likely starting goalie for an upcoming game.

Rules of thumb in the NHL:
- Back-to-back → backup almost always starts.
- Heavy workload / 3 games in 4 nights → backup likely.
- Strong opponent + normal rest → starter.
- Weak opponent + comfortable rest → teams often rest the starter.

This module uses each team's most recent game TOI to identify the starter and
backup, then applies the heuristics above.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from NHL.ApiScrape import _get_last_game_roster, _try_get_json
from NHL.Config import NHL_API_BASE
from NHL.Utils import format_initial_last, sanitize_text, season_from_date

logger = logging.getLogger(__name__)


def _parse_game_date(g: Dict[str, Any]) -> Optional[date]:
    gd_raw = g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime")
    if not isinstance(gd_raw, str):
        return None
    try:
        return datetime.fromisoformat(gd_raw.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.fromisoformat(gd_raw).date()
        except Exception:
            return None


def _team_schedule(team_abbr: str, season: str) -> List[Dict[str, Any]]:
    url = f"{NHL_API_BASE}/club-schedule-season/{team_abbr}/{season}"
    data = _try_get_json(url)
    if not isinstance(data, dict):
        return []
    games = data.get("games")
    return games if isinstance(games, list) else []


def _rest_and_next_game(
    team_abbr: str, game_date: date, season: str
) -> Tuple[Optional[int], Optional[int]]:
    """Return (days since previous game, days until next game)."""
    games = _team_schedule(team_abbr, season)
    if not games:
        return None, None

    prev_date: Optional[date] = None
    next_date: Optional[date] = None
    for g in games:
        gd = _parse_game_date(g)
        if gd is None:
            continue
        if gd < game_date and (prev_date is None or gd > prev_date):
            prev_date = gd
        if gd > game_date and (next_date is None or gd < next_date):
            next_date = gd

    days_rest = (game_date - prev_date).days if prev_date else None
    days_next = (next_date - game_date).days if next_date else None
    return days_rest, days_next


def _last_game_goalies(team_abbr: str, game_date: str) -> List[str]:
    """Return goalies from the most recent completed game, starter first (by TOI)."""
    _, goalies = _get_last_game_roster(team_abbr, game_date)
    return [
        format_initial_last(sanitize_text(g["name"]))
        for g in goalies
        if g.get("name")
    ]


def predict_starting_goalie(
    team_abbr: str,
    game_date: Any,
    opponent_abbr: Optional[str] = None,
    is_b2b: bool = False,
) -> Optional[str]:
    """
    Predict the likely starting goalie for `team_abbr` on `game_date`.

    Returns the formatted name (e.g. "A. Vasilevskiy") or None if unavailable.
    """
    try:
        if isinstance(game_date, str):
            target_date = date.fromisoformat(game_date)
        elif isinstance(game_date, date):
            target_date = game_date
        else:
            target_date = date.today()
    except Exception:
        target_date = date.today()

    season = season_from_date(target_date.isoformat())

    # Most recent game goalies, ordered by TOI (starter first).
    names = _last_game_goalies(team_abbr, target_date.isoformat())

    # Fallback to roster if we have no recent game data.
    if not names:
        from NHL.ApiScrape import get_roster_goalies_for_override
        names = get_roster_goalies_for_override(team_abbr, target_date.isoformat())

    if not names:
        return None
    if len(names) == 1:
        return names[0]

    starter = names[0]
    backup = names[1]

    days_rest, days_next = _rest_and_next_game(team_abbr, target_date, season)

    # Opponent strength (Elo) when known.
    opponent_elo: Optional[float] = None
    if opponent_abbr:
        try:
            from NHL.Prediction import get_team_elo
            opponent_elo = get_team_elo(opponent_abbr, season)
        except Exception as e:
            logger.debug(f"Could not fetch opponent Elo for {opponent_abbr}: {e}")

    # Back-to-back or no rest → backup.
    if is_b2b or (days_rest is not None and days_rest < 2):
        logger.info(
            f"Predicting backup {backup} for {team_abbr}: b2b={is_b2b}, rest={days_rest}d"
        )
        return backup

    # Strong opponent + normal rest → starter.
    if opponent_elo is not None and opponent_elo >= 1525:
        if days_rest is not None and days_rest >= 2:
            logger.info(
                f"Predicting starter {starter} for {team_abbr}: strong opponent (Elo {opponent_elo:.0f})"
            )
            return starter

    # Weak opponent + comfortable rest → often rest the starter.
    if opponent_elo is not None and opponent_elo <= 1475:
        if days_rest is not None and days_rest >= 3:
            logger.info(
                f"Predicting backup {backup} for {team_abbr}: weak opponent (Elo {opponent_elo:.0f}), rest={days_rest}d"
            )
            return backup

    # Busy upcoming schedule with enough rest → save starter for next game.
    if (
        days_next is not None
        and days_next <= 1
        and days_rest is not None
        and days_rest >= 3
    ):
        logger.info(
            f"Predicting backup {backup} for {team_abbr}: back-to-back tomorrow, rest={days_rest}d"
        )
        return backup

    logger.info(f"Predicting starter {starter} for {team_abbr}: default")
    return starter
