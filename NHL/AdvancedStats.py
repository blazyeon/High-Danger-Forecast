"""
NHL Advanced Stats utilities — URL builders and column configuration.

The actual data fetching has moved to ``NHL.PlayByPlay`` and
``NHL.StatsFromPBP`` (NHL API PBP + computed stats). This module
retains the URL builders for backwards compatibility with code that
still constructs NST URLs; new code should not import them.
"""
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

# NST.Cache is no longer imported here; URL builders below are kept
# as a back-compat shim but the data path goes through StatsFromPBP.
from NHL.Config import (
    EARLIEST_SEASON_YEAR,
    CURRENT_SEASON_YEAR,
)
from NHL.Errors import (
    safe_api_call,
    validate_dataframe,
    DataValidationError,
    safe_division
)

logger = logging.getLogger(__name__)

# ---------------------- Configuration ----------------------


def _season_options(
    start_year: int = EARLIEST_SEASON_YEAR,
    end_year: int = CURRENT_SEASON_YEAR
) -> List[Tuple[str, str]]:
    """
    Generate season options for dropdown.

    Args:
        start_year: Earliest season year
        end_year: Latest season year

    Returns:
        List of (label, key) tuples
    """
    opts: List[Tuple[str, str]] = []
    for y in range(start_year, end_year + 1):
        key = f"{y}{y+1}"
        label = f"{y}-{str(y+1)[-2:]}"
        opts.append((label, key))
    return opts


# ---------------------- URL builders ----------------------

def build_team_url(
    season_key: str,
    stype: int,
    sit: str,
    fd: str = "",
    td: str = ""
) -> str:
    """Build NST team stats URL"""
    return (
        f"https://www.naturalstattrick.com/teamtable.php?"
        f"fromseason={season_key}&thruseason={season_key}"
        f"&stype={stype}&sit={sit}&score=all&rate=n&team=all&loc=B&fd={fd}&td={td}"
    )

def build_player_url(
    season_key: str,
    stype: int,
    sit: str,
    pos: str,
    fd: str = "",
    td: str = "",
    gpfilt: str = "none",
) -> str:
    """
    Build NST player stats URL.

    Args:
        season_key: Season key (e.g., "20242025")
        stype: Season type (1=pre, 2=reg, 3=playoffs)
        sit: Situation (all, ev, pp, pk)
        pos: Position (S=skaters, G=goalies)
        fd: From date (optional)
        td: To date (optional)
        gpfilt: Game filter (gpdate when using fd/td)

    Returns:
        NST URL string
    """
    stdoi = "g" if str(pos).upper() == "G" else "std"
    return (
        f"https://www.naturalstattrick.com/playerteams.php?"
        f"fromseason={season_key}&thruseason={season_key}"
        f"&stype={stype}&sit={sit}&score=all&stdoi={stdoi}&rate=n"
        f"&team=ALL&pos={pos}&loc=B&toi=0&gpfilt={gpfilt}&fd={fd}&td={td}"
        f"&tgp=410&lines=single&draftteam=ALL"
    )