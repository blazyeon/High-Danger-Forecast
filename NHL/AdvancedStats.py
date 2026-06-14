"""
NHL Advanced Stats utilities for building URLs, processing data,
and configuring column displays from Natural Stat Trick.
"""
from __future__ import annotations
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

from NST.Cache import get_nst_table_from_url
from NHL.Config import (
    EARLIEST_SEASON_YEAR,
    CURRENT_SEASON_YEAR,
    UI_PARAMS,
    COLUMN_VARIATIONS
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

def _table_height_for_rows(visible_rows: int) -> int:
    """Calculate table height based on number of rows"""
    row_px = UI_PARAMS["table_row_height"]
    header_px = UI_PARAMS["table_header_height"]
    padding_px = UI_PARAMS["table_padding"]
    return header_px + padding_px + max(0, visible_rows) * row_px

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Safely coerce series to numeric"""
    try:
        return pd.to_numeric(series, errors="coerce").fillna(0)
    except Exception:
        return series

# ---------------------- Column label maps ----------------------

TEAM_LABELS = {
    "Team": "Team",
    "GP": "Games Played (GP)",
    "TOI": "Time on Ice (TOI)",
    "CF": "Shot Attempts For (CF)",
    "CA": "Shot Attempts Against (CA)",
    "CF%": "Shot Attempts Percentage (CF%)",
    "FF": "Unblocked Shot Attempts For (FF)",
    "FA": "Unblocked Shot Attempts Against (FA)",
    "FF%": "Unblocked Shot Attempts Percentage (FF%)",
    "SF": "Shots For (SF)",
    "SA": "Shots Against (SA)",
    "SF%": "Shots For Percentage (SF%)",
    "GF": "Goals For (GF)",
    "GA": "Goals Against (GA)",
    "GF%": "Goals For Percentage (GF%)",
    "xGF": "Expected Goals For (xGF)",
    "xGA": "Expected Goals Against (xGA)",
    "xGF%": "Expected Goals Percentage (xGF%)",
    "SCF": "Scoring Chances For (SCF)",
    "SCA": "Scoring Chances Against (SCA)",
    "SCF%": "Scoring Chances Percentage (SCF%)",
    "HDCF": "High-Danger Chances For (HDCF)",
    "HDCA": "High-Danger Chances Against (HDCA)",
    "HDCF%": "High-Danger Chances Percentage (HDCF%)",
    "HDGF": "High-Danger Goals For (HDGF)",
    "HDGA": "High-Danger Goals Against (HDGA)",
    "HDGF%": "High-Danger Goals Percentage (HDGF%)",
    "Sh%": "Shooting Percentage (Sh%)",
    "Sv%": "Save Percentage (Sv%)",
    "SV%": "Save Percentage (SV%)",
    "PDO": "PDO",
    "PTS": "Points (PTS)",
    "W": "Wins (W)",
    "OT": "Overtime or Shootout Losses (OT)",
}

SKATER_LABELS = {
    "Player": "Player",
    "Team": "Team",
    "GP": "Games Played (GP)",
    "TOI": "Time on Ice (TOI)",
    "G": "Goals (G)",
    "A": "Assists (A)",
    "P": "Points (P)",
    "P/GP": "Points per Game (P/GP)",
    "S": "Shots on Goal (SOG)",
    "Sh%": "Shooting Percentage (Sh%)",
    "iCF": "Individual Shot Attempts (iCF)",
    "iFF": "Individual Unblocked Shot Attempts (iFF)",
    "ixG": "Individual Expected Goals (ixG)",
    "IPP": "Individual Points Percentage (IPP)",
    "CF%": "On-Ice Shot Attempts Percentage (CF%)",
    "FF%": "On-Ice Unblocked Shot Attempts Percentage (FF%)",
    "xGF%": "On-Ice Expected Goals Percentage (xGF%)",
    "PTS": "Points (PTS)",
}

GOALIE_LABELS = {
    "Player": "Player",
    "Team": "Team",
    "GP": "Games Played (GP)",
    "GS": "Games Started (GS)",
    "W": "Wins (W)",
    "L": "Losses (L)",
    "OT": "Overtime or Shootout Losses (OT)",
    "SA": "Shots Against (SA)",
    "SV": "Saves (SV)",
    "Sv%": "Save Percentage (Sv%)",
    "SV%": "Save Percentage (SV%)",
    "GAA": "Goals Against Average (GAA)",
    "GA": "Goals Against (GA)",
    "SO": "Shutouts (SO)",
    "QS": "Quality Starts (QS)",
    "RBS": "Really Bad Starts (RBS)",
    "xSV%": "Expected Save Percentage (xSV%)",
    "dSV%": "Save Percentage Above Expected (dSV%)",
    "PTS": "Points (PTS)",
}

def _build_column_config(
    df: pd.DataFrame,
    labels: Dict[str, str],
    pinned_cols: Optional[Union[List[str], Tuple[str, ...]]] = (),
    numeric_formats: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build column configuration metadata for a DataFrame.

    Args:
        df: DataFrame to configure
        labels: Column label mapping
        pinned_cols: Columns to pin
        numeric_formats: Format strings for numeric columns

    Returns:
        Dictionary mapping column names to config dicts with keys:
        'label' (str), 'pinned' (bool), 'format' (str|None), 'type' (str: 'number' or 'text')
    """
    config: Dict[str, Dict[str, Any]] = {}
    labels_ci = {str(k).strip().lower(): v for k, v in labels.items()}
    pinned_set = {str(c).strip().lower() for c in (pinned_cols or ())}
    numeric_formats = numeric_formats or {}
    fmt_ci = {k.lower(): v for k, v in numeric_formats.items()}

    for col in df.columns:
        col_str = str(col).strip()
        label = labels.get(col_str) or labels_ci.get(col_str.lower(), col_str)
        fmt = fmt_ci.get(col_str.lower())
        is_pinned = col_str.lower() in pinned_set

        if fmt:
            config[col] = {
                "label": label,
                "pinned": is_pinned,
                "format": fmt,
                "type": "number",
            }
        else:
            config[col] = {
                "label": label,
                "pinned": is_pinned,
                "format": None,
                "type": "text",
            }

    return config

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