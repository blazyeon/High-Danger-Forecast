"""
NHL Lookup helpers with enhanced error handling, validation, and robust parsing.
Pure utility functions for game information, lineups, and statistics.
Streamlit UI rendering has been removed — use these helpers from your Flask backend.
"""
import os
import re
import html
import base64
from datetime import datetime, timezone, date as _date
from typing import Any, Dict, List, Optional

import pandas as pd
import logging

from NHL.ApiScrape import get_games_on_date, get_boxscore
from NHL.Config import (
    TEAM_ABBR_MAPPING,
    UI_PARAMS,
    FORWARD_POSITIONS,
    DEFENSE_POSITIONS,
    GOALIE_POSITIONS,
    COLUMN_VARIATIONS
)
from NHL.Errors import (
    safe_api_call,
    validate_dataframe,
    DataValidationError,
    safe_division,
    validate_date_range
)
from NHL.Utils import (
    sanitize_text as _sanitize_text,
    format_initial_last as _format_initial_last,
    parse_minutes as _parse_minutes,
    extract_name_from_dict as _extract_name_util,
)

logger = logging.getLogger(__name__)

# --------------------------- Styles ---------------------------
CSS = """
<style>
.block-container { max-width: 100% !important; padding-left: 0.75rem; padding-right: 0.75rem; }
div[data-testid="stExpander"] { width: 100% !important; }
div[data-testid="stExpander"] .stExpanderContent { width: 100% !important; padding-left: 0 !important; padding-right: 0 !important; }
div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }

/* Header row for each game card */
.team-row {
  display:flex;
  justify-content:center;
  align-items:center;
  gap:16px;
  margin-bottom:0.6em;
  flex-wrap:wrap;
  position:relative;
}
.team-card { text-align:center; width:300px; color: #fff; }
.team-logo {
  width:""" + str(UI_PARAMS["logo_width"]) + """px;
  height:auto;
  display:inline-block;
  object-fit:contain;
  background:transparent;
}
.team-name { display:block; font-size:1.8em; font-weight:900; margin-top:0.15em; word-break:break-word; color:#fff; }
.team-score { font-size:3.2em; font-weight:900; margin:0.15em 0 0.45em 0; line-height:1; color:#fff; }

/* Bigger VS and centered between team names */
.vs {
  display:flex;
  align-items:center;
  justify-content:center;
  font-size:4.0em;
  font-weight:900;
  padding:0 10px;
  color:#fff;
  line-height:1;
  margin:0 8px;
}

.game-time { position:absolute; left:8px; top:8px; font-size:0.95em; font-weight:700; color:#fff; }

.element-container .stDataFrame div { font-size:14px !important; white-space:normal !important; color: #fff !important; }
.stDataFrame > div { overflow: hidden !important; }
.stDataFrame table { border-collapse: collapse !important; }
.stDataFrame tbody { display: table-row-group !important; }

/* Fallback small logo block */
.logo-fallback {
  width:""" + str(UI_PARAMS["logo_width"]) + """px;
  height:""" + str(UI_PARAMS["logo_height"]) + """px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:transparent;
  color:#fff;
  font-weight:800;
  font-size:1.05em;
}
</style>
"""

# --------------------------- Helpers (safe HTML/text) ---------------------------

def sanitize_text(text: Any) -> str:
    """Sanitize text for safe display — delegates to NHL.Utils."""
    return _sanitize_text(text)

def escape_for_html(text: Any) -> str:
    """HTML-escape text for safe rendering"""
    return html.escape(sanitize_text(text))

def extract_name(name_field: Any) -> str:
    """Extract name from various API formats — delegates to NHL.Utils."""
    return _extract_name_util(name_field)

def get_team_logo_path(abbrev: str) -> Optional[str]:
    """
    Get path to team logo image.

    Args:
        abbrev: Team abbreviation

    Returns:
        Path to logo or None
    """
    if not abbrev:
        return None

    img_path = os.path.join("Images", f"{abbrev.upper()}.png")

    if os.path.exists(img_path):
        return img_path

    logger.debug(f"Logo not found: {img_path}")
    return None

def img_to_base64(img_path: Optional[str]) -> Optional[str]:
    """Convert image to base64 for embedding"""
    if not img_path or not os.path.exists(img_path):
        return None

    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {img_path} - {e}")
        return None

def parse_game_datetime(g: Dict[str, Any]) -> Optional[datetime]:
    """Parse game datetime from API response"""
    raw = g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime")
    if not raw:
        return None

    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        logger.debug(f"Failed to parse game datetime: {raw}")
        return None

def display_abbr_for_game(raw_abbr: str) -> str:
    """
    Get display abbreviation with team mapping (e.g., ARI -> UTA).

    Args:
        raw_abbr: Raw team abbreviation

    Returns:
        Display abbreviation
    """
    abbr = (raw_abbr or "").upper()
    return TEAM_ABBR_MAPPING.get(abbr, abbr)

def get_logo_b64_for_display_abbr(display_abbr: str) -> Optional[str]:
    """
    Get base64 logo for display abbreviation with fallback.

    Args:
        display_abbr: Display abbreviation

    Returns:
        Base64 encoded logo or None
    """
    # Try primary logo
    primary = get_team_logo_path(display_abbr)
    b64 = img_to_base64(primary)

    if b64:
        return b64

    # Fallback for mapped teams (e.g., UTA -> ARI)
    for original, mapped in TEAM_ABBR_MAPPING.items():
        if display_abbr == mapped:
            fallback = get_team_logo_path(original)
            b64 = img_to_base64(fallback)
            if b64:
                logger.debug(f"Using fallback logo {original} for {display_abbr}")
                return b64

    return None

def get_team_full_name(team_info: Any) -> str:
    """Extract full team name from API response"""
    if isinstance(team_info, dict):
        raw = team_info.get('teamName') or team_info.get('name') or team_info.get('abbrev')
        resolved = extract_name(raw)
    else:
        resolved = str(team_info) if team_info is not None else "Unknown"

    return sanitize_text(resolved or "Unknown")

def is_game_started(game: Optional[Dict[str, Any]], box: Optional[Dict[str, Any]]) -> bool:
    """
    Check if game has started.

    Args:
        game: Game info dict
        box: Boxscore dict

    Returns:
        True if game started
    """
    # If we have boxscore, game has started
    if box:
        return True

    if not game:
        return False

    # Check status
    try:
        status = game.get("status", {}) or {}
        state = ""

        if isinstance(status, dict):
            t = status.get("type") or {}
            state = (t.get("state") or t.get("name") or "") if isinstance(t, dict) else ""

            if not state:
                state = status.get("detailedState") or status.get("shortState") or ""

        if isinstance(state, str) and state.strip():
            state_upper = state.strip().upper()
            if state_upper in ("FINAL", "IN_PROGRESS", "LIVE", "COMPLETED", "PLAYED"):
                return True
    except Exception as e:
        logger.debug(f"Error checking game status: {e}")

    # Check start time
    start_raw = game.get("startTimeUTC") or game.get("gameDate") or game.get("startTime")
    if start_raw:
        try:
            start_dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
            return start_dt <= datetime.now(timezone.utc)
        except Exception:
            pass

    return False

def format_start_time(game: Optional[Dict[str, Any]]) -> str:
    """Format game start time for display"""
    if not game:
        return ""

    start_raw = game.get("startTimeUTC") or game.get("gameDate") or game.get("startTime")
    if not start_raw:
        return ""

    try:
        start_dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
        local_dt = start_dt.astimezone()
        return local_dt.strftime("%I:%M %p").lstrip("0").lower()
    except Exception:
        return sanitize_text(start_raw)

# --------------------------- Live status helpers ---------------------------

def _ordinal(n: int) -> str:
    """Convert number to ordinal string"""
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    if n == 3:
        return "3rd"
    return f"{n}th"

def _period_label(period_num: Optional[int], period_type: str) -> str:
    """Get period label (1st, 2nd, 3rd, OT, SO)"""
    ptype = (period_type or "").upper()

    if ptype == "SO":
        return "SO"
    if ptype == "OT":
        return "OT"

    try:
        n = int(period_num) if period_num is not None else None
    except Exception:
        n = None

    if n is None:
        return "Period"
    if n <= 3:
        return _ordinal(n)
    if n == 4:
        return "OT"
    return f"OT{n-3}"

def _extract_clock_info(game: Optional[Dict[str, Any]], box: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract clock and period information"""
    obj = None
    for cand in (game, box):
        if isinstance(cand, dict) and cand:
            obj = cand
            break

    if not obj:
        return {
            "time": "",
            "intermission": False,
            "period_num": None,
            "period_type": ""
        }

    clk = obj.get("clock")
    time_str = ""
    intermission = False

    if isinstance(clk, dict):
        time_str = clk.get("timeRemaining") or clk.get("timeRemainingInPeriod") or clk.get("time") or ""
        intermission = bool(clk.get("inIntermission") or False)
    elif isinstance(clk, str):
        time_str = clk

    pd_desc = obj.get("periodDescriptor")
    period_num = None
    period_type = ""

    if isinstance(pd_desc, dict):
        period_num = pd_desc.get("number") or pd_desc.get("period")
        period_type = pd_desc.get("periodType") or pd_desc.get("type") or ""

    if period_num is None:
        period_num = obj.get("currentPeriod") or obj.get("period")

    if not period_type:
        period_type = obj.get("periodType") or ""

    return {
        "time": str(time_str) if time_str else "",
        "intermission": intermission,
        "period_num": period_num,
        "period_type": period_type,
    }

def _game_state(game: Optional[Dict[str, Any]], box: Optional[Dict[str, Any]]) -> str:
    """Determine game state (FINAL, LIVE, SCHEDULED)"""
    state = ""

    try:
        if game and isinstance(game, dict):
            status = game.get("status", {}) or {}

            if isinstance(status, dict):
                t = status.get("type") or {}
                state = (t.get("state") or t.get("name") or "") if isinstance(t, dict) else ""

                if not state:
                    state = status.get("detailedState") or status.get("shortState") or ""

            if not state:
                state = game.get("gameState") or ""
    except Exception as e:
        logger.debug(f"Error determining game state: {e}")
        state = ""

    s = (state or "").strip().upper()

    if s in ("FINAL", "COMPLETED", "PLAYED"):
        return "FINAL"
    if s in ("LIVE", "IN_PROGRESS"):
        return "LIVE"

    return "SCHEDULED"

def live_or_start_text(game: Optional[Dict[str, Any]], box: Optional[Dict[str, Any]]) -> str:
    """Get live status or start time text"""
    state = _game_state(game, box)

    if state == "LIVE":
        info = _extract_clock_info(game, box)
        label = _period_label(info.get("period_num"), info.get("period_type"))

        if info.get("intermission"):
            return f"Active — Intermission (End {label})"

        t = info.get("time")
        if t:
            return f"Active — {label} {t}"

        return f"Active — {label}"

    if state == "FINAL":
        return "Final"

    return format_start_time(game)

# --------------------------- Player table helpers ---------------------------

SKATER_COLUMNS = [
    "Name", "Position", "Goals", "Assists", "Points", "+/-", "PIM", "TOI",
    "Shots", "Blocks", "Hits", "FO%", "GV", "TK", "PP Goals"
]

def parse_minutes(toi: Any) -> float:
    """Parse time on ice string to minutes — delegates to NHL.Utils."""
    return _parse_minutes(toi)

def _name_from_frag(val: Any) -> str:
    """Extract name from fragment"""
    if isinstance(val, dict):
        return val.get("default") or next(iter(val.values()), "")
    return val or ""

def get_player_display_name(rec: Dict[str, Any]) -> str:
    """Get player display name with fallbacks"""
    nm = rec.get("name")
    if nm:
        disp = _name_from_frag(nm)
        if disp:
            return disp

    first = _name_from_frag(rec.get("firstName")) or _name_from_frag(rec.get("firstInitial"))
    last = _name_from_frag(rec.get("lastName"))
    full = " ".join([first, last]).strip()

    if full:
        return full

    bio = rec.get("playerBio") or rec.get("player") or {}
    nm = bio.get("name")

    if nm:
        disp = _name_from_frag(nm)
        if disp:
            return disp

    first = _name_from_frag(bio.get("firstName")) or _name_from_frag(bio.get("firstInitial"))
    last = _name_from_frag(bio.get("lastName"))
    full = " ".join([first, last]).strip()

    return full if full else "Unknown"

def map_position(rec: Dict[str, Any], stats: Optional[Dict[str, Any]] = None) -> str:
    """
    Map player position with normalization.

    Args:
        rec: Player record
        stats: Optional stats dict

    Returns:
        Normalized position string
    """
    pos = (rec.get("position")
           or (stats or {}).get("position")
           or rec.get("positionCode")
           or "")
    pos = str(pos).upper().strip()

    # Direct mappings
    if pos in ("LW", "RW", "C", "LD", "RD"):
        return pos

    if pos in ("L", "R"):
        return "LW" if pos == "L" else "RW"

    if pos == "C":
        return "C"

    if pos in DEFENSE_POSITIONS:
        # Try to determine side
        side = (rec.get("positionSide")
                or rec.get("side")
                or rec.get("leftRight")
                or (rec.get("playerBio") or {}).get("positionSide")
                or "")
        side = str(side).upper()[:1]

        if side in ("L", "R"):
            return "LD" if side == "L" else "RD"

        # Fallback to shoots/catches
        shoots = (rec.get("shootsCatches")
                  or (rec.get("playerBio") or {}).get("shootsCatches")
                  or (rec.get("player") or {}).get("shootsCatches")
                  or "")
        shoots = str(shoots).upper()[:1]

        if shoots in ("L", "R"):
            return "LD" if shoots == "L" else "RD"

        return "D"

    return pos or "NA"

def normalize_skater_from_stats(name: str, pos_label: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize skater stats from API response.

    Args:
        name: Player name
        pos_label: Position label
        stats: Stats dictionary

    Returns:
        Normalized stats dictionary
    """
    goals = stats.get("goals", stats.get("g", 0)) or 0
    assists = stats.get("assists", stats.get("a", 0)) or 0
    points = stats.get("points")

    if points is None:
        try:
            points = int(goals) + int(assists)
        except Exception:
            points = 0

    plusminus = stats.get("plusMinus", stats.get("plusMinusRating", 0)) or 0
    pim = stats.get("pim", stats.get("penaltyMinutes", 0)) or 0
    toi = stats.get("toi", stats.get("timeOnIce", "0:00")) or "0:00"
    shots = stats.get("sog", stats.get("shotsOnGoal", stats.get("shots", 0))) or 0
    blocks = stats.get("blockedShots", stats.get("blocks", 0)) or 0
    hits = stats.get("hits", 0) or 0

    # Faceoff percentage
    faceoff_raw = (
        stats.get("faceoffWinningPctg") or
        stats.get("faceoffWinPct") or
        stats.get("foPct") or
        None
    )

    if faceoff_raw is None:
        fo_pct = "N/A"
    else:
        try:
            val = float(faceoff_raw)
            # Convert to percentage if needed
            fo_pct = f"{(val*100.0 if val <= 1.0 else val):.1f}"
        except Exception:
            fo_pct = "N/A"

    gv = stats.get("giveaways", stats.get("gv", 0)) or 0
    tk = stats.get("takeaways", stats.get("tk", 0)) or 0
    pp_goals = stats.get("powerPlayGoals", stats.get("ppGoals", 0)) or 0

    return {
        "Name": name,
        "Position": pos_label,
        "Goals": goals,
        "Assists": assists,
        "Points": points,
        "+/-": plusminus,
        "PIM": pim,
        "TOI": toi,
        "Shots": shots,
        "Blocks": blocks,
        "Hits": hits,
        "FO%": fo_pct,
        "GV": gv,
        "TK": tk,
        "PP Goals": pp_goals,
    }

def _collect_records_from_players_map(players_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect player records from players map (excludes goalies)"""
    recs: List[Dict[str, Any]] = []

    for rec in (players_map or {}).values():
        if not isinstance(rec, dict):
            continue

        # Skip goalies
        if (rec.get("goalieStats") or
            rec.get("position") in GOALIE_POSITIONS or
            rec.get("positionCode") == "G"):
            continue

        r = dict(rec)

        # Normalize stats location
        if "stats" not in r:
            if "skaterStats" in r:
                r["stats"] = r["skaterStats"]
            elif "gameStats" in r and isinstance(r["gameStats"], dict):
                if "skaterStats" in r["gameStats"]:
                    r["stats"] = r["gameStats"]["skaterStats"]

        recs.append(r)

    return recs

def extract_team_skaters(
    team_stats: Dict[str, Any],
    include_zero_toi: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract skater stats from team stats dictionary.

    Args:
        team_stats: Team stats dictionary from boxscore
        include_zero_toi: Whether to include players with 0 TOI

    Returns:
        List of normalized skater dictionaries
    """
    results: List[Dict[str, Any]] = []

    def played_enough(toi_str: str) -> bool:
        """Check if player has sufficient TOI"""
        return parse_minutes(toi_str) > 0 if not include_zero_toi else True

    def add_from_record_if_valid(rec: Dict[str, Any]):
        """Add player record if valid"""
        stats = rec.get("stats") or rec.get("skaterStats") or rec
        game_stats = rec.get("gameStats")

        if isinstance(game_stats, dict):
            stats = game_stats.get("skaterStats") or game_stats.get("stats") or stats

        toi_str = (
            (stats or {}).get("toi") or
            (stats or {}).get("timeOnIce") or
            rec.get("toi") or
            "0:00"
        )

        if not played_enough(str(toi_str)):
            return

        name = get_player_display_name(rec)
        pos_label = map_position(rec, stats or {})

        results.append(normalize_skater_from_stats(name, pos_label, stats or {}))

    # Extract from forwards and defense arrays
    for key in ("forwards", "defense"):
        arr = team_stats.get(key)
        if isinstance(arr, list) and arr:
            for rec in arr:
                if isinstance(rec, dict):
                    add_from_record_if_valid(rec)

    # Extract from skaters array
    skaters = team_stats.get("skaters", [])
    players_map = team_stats.get("players", {}) or {}

    if isinstance(skaters, list) and skaters:
        if isinstance(skaters[0], dict):
            # Array of player objects
            for rec in skaters:
                add_from_record_if_valid(rec)
        else:
            # Array of player IDs - look up in players map
            for pid in skaters:
                key = str(pid)
                rec = players_map.get(key) or players_map.get(pid)
                if isinstance(rec, dict):
                    add_from_record_if_valid(rec)

    # Fallback to players map if no results yet
    if not results and isinstance(players_map, dict) and players_map:
        for rec in _collect_records_from_players_map(players_map):
            add_from_record_if_valid(rec)

    logger.info(f"Extracted {len(results)} skaters")
    return results

def get_goalie_stats(goalies: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract and format goalie stats.

    Args:
        goalies: List of goalie dictionaries

    Returns:
        DataFrame with goalie stats
    """
    rows = []
    toi_list = []

    # Calculate TOI for starter determination
    for g in goalies:
        toi = g.get('toi', g.get('timeOnIce', '0:00')) or "0:00"
        toi_list.append(parse_minutes(toi))

    # Find starter(s) - highest TOI
    starter_indices = []
    if toi_list:
        max_toi = max(toi_list)
        starter_indices = [i for i, t in enumerate(toi_list) if t == max_toi and max_toi > 0]

    for i, g in enumerate(goalies):
        # Extract name
        name_field = g.get('name') or (g.get('playerBio') or {}).get('name')
        if name_field:
            raw_name = extract_name(name_field) if isinstance(name_field, dict) else str(name_field)
        else:
            raw_name = get_player_display_name(g)

        name = sanitize_text(raw_name)

        # Extract stats
        saves = g.get('saves', g.get('sv', 0)) or 0
        shots = g.get('shotsAgainst', g.get('sa', 0)) or 0
        goals_against = g.get('goalsAgainst', g.get('ga', 0)) or 0
        toi = g.get('toi', g.get('timeOnIce', '0:00')) or "0:00"

        # Calculate save percentage
        if shots > 0:
            sv_pct = f"{safe_division(float(saves), float(shots), 0.0):.3f}"
        else:
            sv_pct = "N/A"

        # Calculate GAA
        total_minutes = parse_minutes(toi)
        if total_minutes > 0:
            gaa = safe_division(float(goals_against) * 60.0, total_minutes, 0.0)
            gaa_str = f"{gaa:.2f}"
        else:
            gaa_str = "N/A"

        # Starter indicator
        is_starter = "Y" if i in starter_indices else ""

        rows.append({
            "Name": name,
            "Save %": sv_pct,
            "Shots": shots,
            "Saves": saves,
            "GA Average": gaa_str,
            "TOI": toi,
            "Starter": is_starter
        })

    df = pd.DataFrame(rows, columns=["Name", "Save %", "Shots", "Saves", "GA Average", "TOI", "Starter"])
    return df.reset_index(drop=True)

def format_initial_last(name_str: str) -> str:
    """Format name as 'F. Last' — delegates to NHL.Utils."""
    return _format_initial_last(name_str)

def style_name_column(df_or_styler):
    """Apply styling to name column for better display"""
    styler = getattr(df_or_styler, "style", df_or_styler)

    styler = styler.set_properties(
        subset=["Name"],
        **{
            "white-space": "normal",
            "overflow-wrap": "anywhere",
            "word-break": "break-word",
            "text-overflow": "initial",
            "overflow": "visible",
            "line-height": "1.2",
        },
    )

    styler = styler.set_table_styles(
        [
            {"selector": "th.col_heading.level0.col0",
             "props": [("min-width", "220px"), ("width", "280px")]},
            {"selector": "td.col0",
             "props": [("min-width", "220px"), ("width", "280px"),
                       ("white-space", "normal"), ("overflow-wrap", "anywhere"),
                       ("word-break", "break-word")]},
        ],
        overwrite=False,
    )

    return styler

def df_height_exact_fit(df: pd.DataFrame, row_height: int = None) -> int:
    """
    Calculate exact DataFrame height for display.

    Args:
        df: DataFrame
        row_height: Height per row in pixels

    Returns:
        Total height in pixels
    """
    if row_height is None:
        row_height = UI_PARAMS["table_row_height"]

    if df is None or df.empty:
        return 120

    header_height = UI_PARAMS["table_header_height"]
    n_data_rows = len(df)
    total_height = header_height + (n_data_rows * row_height)

    return max(UI_PARAMS["min_table_height"], total_height)

# --------------------------- Rendering helpers ---------------------------

def build_logo_html(b64: Optional[str], abbr: str) -> str:
    """Build HTML for team logo with fallback"""
    safe_abbr = escape_for_html(abbr or "")

    if b64:
        alt = safe_abbr or "team logo"
        return f'<img class="team-logo" src="data:image/png;base64,{b64}" alt="{alt}">'

    # Fallback text
    return f'<div class="logo-fallback">{safe_abbr}</div>'