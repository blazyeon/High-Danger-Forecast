"""
Centralized utility functions to eliminate code duplication across the NHL application.
"""
import re
import html
from typing import Any, Optional, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

# ===================== NAME NORMALIZATION =====================

def normalize_name_key(name: str) -> str:
    """
    Normalize name to alphanumeric lowercase for matching.
    Centralized implementation used across all modules.
    
    Args:
        name: Name to normalize
    
    Returns:
        Normalized name string (alphanumeric lowercase only)
    
    Example:
        >>> normalize_name_key("Connor McDavid")
        'connormcdavid'
    """
    if not name:
        return ""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def last_token_norm(name: str) -> str:
    """
    Get normalized last name token for fuzzy matching.
    
    Args:
        name: Full name string
    
    Returns:
        Normalized last name token
    
    Example:
        >>> last_token_norm("Connor McDavid")
        'mcdavid'
    """
    if not name:
        return ""
    
    s = "".join(ch.lower() for ch in str(name) if ch.isalpha() or ch.isspace()).strip()
    if not s:
        return ""
    
    parts = [p for p in s.split() if p]
    return parts[-1] if parts else ""


# ===================== TEXT SANITIZATION =====================

def sanitize_text(text: Any) -> str:
    """
    Sanitize text for safe display (remove HTML, normalize whitespace).
    
    Args:
        text: Text to sanitize
    
    Returns:
        Cleaned text string
    """
    try:
        s = "" if text is None else str(text)
        unescaped = html.unescape(s)
        cleaned = re.sub(r'<[^>]*>', '', unescaped)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception as e:
        logger.debug(f"Error sanitizing text: {e}")
        return ""


def escape_for_html(text: Any) -> str:
    """HTML-escape text for safe rendering"""
    return html.escape(sanitize_text(text))


def format_initial_last(name_str: str) -> str:
    """
    Format name as 'F. Last'.
    
    Args:
        name_str: Full name string
    
    Returns:
        Formatted name
    
    Example:
        >>> format_initial_last("Connor McDavid")
        'C. McDavid'
    """
    try:
        parts = str(name_str).split()
        if not parts:
            return name_str or "Unknown"
        
        first = parts[0]
        last = " ".join(parts[1:]).strip()
        
        if not last:
            return f"{first[:1].upper()}."
        
        initial = first[:1].upper()
        return f"{initial}. {last}"
    except Exception:
        return name_str or "Unknown"


# ===================== SEASON UTILITIES =====================

def season_from_date(date_str: str, season_start_month: int = 10) -> str:
    """
    Get NHL season key from date string with proper handling.
    
    Args:
        date_str: ISO format date (YYYY-MM-DD)
        season_start_month: Month when NHL season starts (default 10 = October)
    
    Returns:
        Season key like '20252026'
    
    Example:
        >>> season_from_date("2025-11-15")
        '20252026'
        >>> season_from_date("2026-03-15")
        '20252026'
    """
    try:
        dt = datetime.fromisoformat(date_str).date()
        
        # October through December: use current year as season start
        if dt.month >= season_start_month:
            start_year = dt.year
            end_year = dt.year + 1
        # January through June: use previous year as season start
        elif dt.month <= 6:
            start_year = dt.year - 1
            end_year = dt.year
        # July-September (offseason): use next season
        else:
            start_year = dt.year
            end_year = dt.year + 1
        
        return f"{start_year}{end_year}"
    except Exception as e:
        logger.error(f"Error parsing season from date {date_str}: {e}")
        today = date.today()
        if today.month >= season_start_month:
            start_year = today.year
        elif today.month <= 6:
            start_year = today.year - 1
        else:
            start_year = today.year
        return f"{start_year}{start_year + 1}"


def prev_season_key(season_key: str) -> str:
    """
    Get previous season key.
    
    Args:
        season_key: Current season (e.g., "20252026")
    
    Returns:
        Previous season key (e.g., "20242025")
    """
    try:
        a = int(season_key[:4])
        b = int(season_key[4:])
        return f"{a-1}{b-1}"
    except Exception as e:
        logger.error(f"Error calculating previous season from {season_key}: {e}")
        return season_key


def get_data_season_for_game(
    game_date: date, 
    season_start_month: int = 10,
    current_date: Optional[date] = None
) -> Tuple[str, str, bool]:
    """
    Determine which season's data to use for a game prediction.
    
    This is the FIXED version that properly handles future season games.
    
    Args:
        game_date: Date of the game
        season_start_month: Month when NHL season starts
        current_date: Current date (defaults to today)
    
    Returns:
        Tuple of (game_season, data_season, use_previous_season_flag)
    
    Logic:
        - If game is in future season -> use most recent completed season
        - If game is in current season but season hasn't started -> use previous season
        - If game is in current season and started -> use current season
        - If game is in past season -> use that season's data
    """
    if current_date is None:
        current_date = date.today()
    
    # Determine game's season
    game_season = season_from_date(game_date.isoformat(), season_start_month)
    game_season_start = int(game_season[:4])
    
    # Determine current season
    if current_date.month >= season_start_month:
        current_season_start = current_date.year
    elif current_date.month <= 6:
        current_season_start = current_date.year - 1
    else:
        current_season_start = current_date.year
    
    current_season = f"{current_season_start}{current_season_start + 1}"
    
    # DECISION LOGIC
    if game_season_start > current_season_start:
        # Game is in FUTURE season - use most recent completed season
        data_season = f"{current_season_start - 1}{current_season_start}"
        use_previous = True
        logger.info(
            f"Game in future season {game_season}, using completed season {data_season}"
        )
    elif game_season_start == current_season_start:
        # Game is in CURRENT season - check if season has started
        season_start_date = date(current_season_start, season_start_month, 1)
        
        if current_date < season_start_date:
            # Current season hasn't started yet
            data_season = f"{current_season_start - 1}{current_season_start}"
            use_previous = True
            logger.info(
                f"Current season {game_season} hasn't started, using {data_season}"
            )
        else:
            # Current season has started
            data_season = game_season
            use_previous = False
            logger.info(f"Using current season {data_season} data")
    else:
        # Game is in PAST season
        data_season = game_season
        use_previous = False
        logger.info(f"Using historical season {data_season} data")
    
    return game_season, data_season, use_previous


# ===================== COLUMN UTILITIES =====================

def get_column_safe(df, col_variations: dict, col_key: str) -> Optional[str]:
    """
    Get column name from DataFrame using variations (case-insensitive).
    
    Args:
        df: DataFrame to search
        col_variations: Dictionary mapping keys to list of possible column names
        col_key: Key to look up in col_variations
    
    Returns:
        Actual column name from DataFrame, or None if not found
    """
    if df is None or df.empty:
        return None
    
    if col_key in col_variations:
        candidates = col_variations[col_key]
    else:
        candidates = [col_key]
    
    # Try exact match first
    for c in candidates:
        if c in df.columns:
            return c
    
    # Case-insensitive fallback
    cols_lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    
    return None


def normalize_sv_column(df):
    """
    Normalize save percentage column name to 'Sv%' across DataFrame.
    Handles both 'SV%' and 'Sv%' variations.
    
    Args:
        df: DataFrame with goalie stats
    
    Returns:
        DataFrame with normalized column name
    """
    if df is None or df.empty:
        return df
    
    # Check if we need to rename
    if "SV%" in df.columns and "Sv%" not in df.columns:
        return df.rename(columns={"SV%": "Sv%"})
    
    return df


# ===================== PARSING UTILITIES =====================

def parse_minutes(toi: Any) -> float:
    """
    Parse time on ice string to minutes.
    
    Args:
        toi: Time on ice string (MM:SS or HH:MM:SS)
    
    Returns:
        Minutes as float
    
    Example:
        >>> parse_minutes("18:45")
        18.75
        >>> parse_minutes("1:02:30")
        62.5
    """
    try:
        parts = str(toi).split(":")
        
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return m + s / 60.0
        
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 60.0 + m + s / 60.0
        
        return 0.0
    except Exception:
        return 0.0


def extract_name_from_dict(name_field: Any) -> str:
    """
    Extract name from various API response formats.
    
    Args:
        name_field: Name field from API (could be dict or string)
    
    Returns:
        Name as string
    """
    if isinstance(name_field, dict):
        return name_field.get('default') or next(iter(name_field.values()), "Unknown")
    return str(name_field) if name_field is not None else "Unknown"


# ===================== VALIDATION UTILITIES =====================

def safe_numeric(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to numeric with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Int value or default
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default