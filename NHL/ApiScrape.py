"""
NHL API scraping module with improved error handling, connection pooling, validation, and injury tracking.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, date
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
import json

from NHL.Errors import (
    safe_api_call, retry_on_failure, APIError, DataValidationError,
    validate_date_range
)
from NHL.Config import (
    NHL_API_BASE, REQUEST_HEADERS, DEFAULT_TIMEOUT, MAX_RETRIES,
    RETRY_BACKOFF_BASE, FORWARD_POSITIONS, DEFENSE_POSITIONS, GOALIE_POSITIONS,
    CONNECTION_POOL_SIZE, CONNECTION_POOL_MAXSIZE
)
from NHL.Utils import (
    normalize_name_key, format_initial_last, sanitize_text,
    season_from_date, parse_minutes
)

logger = logging.getLogger(__name__)

# ===================== CONNECTION POOLING =====================

def create_session() -> requests.Session:
    """Create a requests session with connection pooling and retry logic"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_BASE,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=CONNECTION_POOL_SIZE,
        pool_maxsize=CONNECTION_POOL_MAXSIZE
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(REQUEST_HEADERS)
    
    return session

_session = create_session()

# ===================== API FUNCTIONS =====================

@retry_on_failure(max_attempts=MAX_RETRIES, backoff_base=RETRY_BACKOFF_BASE)
def get_games_on_date(date_str: str) -> List[Dict[str, Any]]:
    """Get all games on a specific date with validation and error handling"""
    try:
        validate_date_range(date_str)
    except Exception as e:
        logger.error(f"Invalid date: {date_str} - {e}")
        raise
    
    url = f"{NHL_API_BASE}/score/{date_str}"
    
    try:
        resp = _session.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        games = data.get("games", [])
        if not isinstance(games, list):
            logger.warning(f"Unexpected games format from API: {type(games)}")
            return []
        
        logger.info(f"Retrieved {len(games)} games for {date_str}")
        return games
        
    except requests.exceptions.RequestException as e:
        raise APIError("NHL API", f"Failed to fetch games for {date_str}: {str(e)}", e)
    except Exception as e:
        raise APIError("NHL API", f"Unexpected error fetching games: {str(e)}", e)


@retry_on_failure(max_attempts=MAX_RETRIES, backoff_base=RETRY_BACKOFF_BASE)
def get_boxscore(game_id: Any) -> Dict[str, Any]:
    """Get boxscore for a specific game"""
    if not game_id:
        raise DataValidationError("game_id cannot be empty")
    
    url = f"{NHL_API_BASE}/gamecenter/{game_id}/boxscore"
    
    try:
        resp = _session.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        if not isinstance(data, dict):
            raise DataValidationError(f"Invalid boxscore format: {type(data)}")
        
        logger.info(f"Retrieved boxscore for game {game_id}")
        return data
        
    except requests.exceptions.RequestException as e:
        raise APIError("NHL API", f"Failed to fetch boxscore for {game_id}: {str(e)}", e)
    except Exception as e:
        raise APIError("NHL API", f"Unexpected error fetching boxscore: {str(e)}", e)


# ===================== HELPER FUNCTIONS =====================

def _try_get_json(url: str) -> Optional[Dict[str, Any]]:
    """Safely attempt to get JSON from URL using pooled session"""
    try:
        resp = _session.get(url, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.debug(f"Failed to get JSON from {url[:80]}: {e}")
        return None


def _name_from_frag(val: Any) -> str:
    """Extract name from various formats"""
    if isinstance(val, dict):
        return val.get("default") or next(iter(val.values()), "") if val else ""
    return str(val) if val else ""


def _display_name(rec: Dict[str, Any]) -> str:
    """Get display name from player record with fallbacks"""
    for key in ("name",):
        nm = rec.get(key)
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


def _position_code(rec: Dict[str, Any]) -> str:
    """Determine position code with better normalization"""
    pos = (
        rec.get("positionCode")
        or rec.get("position")
        or (rec.get("playerBio") or {}).get("positionCode")
        or ""
    )
    pos = str(pos).upper().strip()
    
    if pos in ("LW", "L"):
        return "LW"
    if pos in ("RW", "R"):
        return "RW"
    if pos in ("C",):
        return "C"
    if pos in ("LD", "RD"):
        return pos
    if pos in DEFENSE_POSITIONS:
        return "D"
    if pos in GOALIE_POSITIONS:
        return "G"
    
    return pos or "NA"


# ===================== INJURY TRACKING =====================

def get_team_injuries(team_abbr: str) -> Dict[str, str]:
    """
    Get injured players from local JSON override or live scrape.
    Uses multiple name formats for better matching against NST data.
    """
    injuries = {}
    
    # 1. Try local JSON override first
    if os.path.exists("injuries.json"):
        try:
            with open("injuries.json", "r") as f:
                data = json.load(f)
                # Handle both list format and dict format
                items = data.get('injuries', []) if isinstance(data, dict) else data
                
                for item in items:
                    # Filter by team and ensure 'injured' is true
                    if item.get("team") == team_abbr.upper() and item.get("injured", True):
                        name = item.get("player")
                        status = item.get("status", "Manual Override")
                        if name:
                            # Store injury under multiple normalized formats for better matching
                            sanitized = sanitize_text(name)
                            
                            # 1. Standard normalized key (e.g., "laurentbrossoit")
                            injuries[normalize_name_key(name)] = status
                            
                            # 2. Initial + Last format (e.g., "L. Brossoit" -> "lbrossoit")
                            formatted = format_initial_last(sanitized)
                            injuries[normalize_name_key(formatted)] = status
                            
                            # 3. Also store the formatted version without punctuation
                            # for display comparison (e.g., "l.brossoit" -> "lbrossoit")
                            injuries[normalize_name_key(formatted.replace(". ", "").replace(".", ""))] = status
                            
                            logger.debug(f"Added injury mappings for: {name} -> {status}")
            
            if injuries:
                logger.info(f"Loaded {len(injuries)} injuries from injuries.json for {team_abbr}")
                return injuries  # Return immediately if we found overrides
        except Exception as e:
            logger.warning(f"Failed to read injuries.json: {e}")
    
    # 2. Fallback to API scraping if no local override
    try:
        # Try current roster endpoint
        roster_url = f"{NHL_API_BASE}/roster/{team_abbr}/current"
        roster_data = _try_get_json(roster_url)
        
        if not roster_data:
            # Fallback to club-stats endpoint
            from datetime import date
            from NHL.Utils import season_from_date
            season = season_from_date(date.today().isoformat())
            roster_url = f"{NHL_API_BASE}/club-stats/{team_abbr}/{season}/2"
            roster_data = _try_get_json(roster_url)
        
        if roster_data:
            # Check all position groups
            for position_group in ["forwards", "defensemen", "goalies"]:
                players = roster_data.get(position_group, [])
                if isinstance(players, list):
                    for player in players:
                        # Check multiple status fields
                        status = (
                            player.get("status") or 
                            player.get("healthStatus") or 
                            player.get("injuryStatus")
                        )
                        
                        # Also check if player is on IR
                        is_injured = False
                        if status:
                            status_upper = str(status).upper()
                            if status_upper not in ("ACTIVE", "HEALTHY", ""):
                                is_injured = True
                        
                        # Check roster status
                        roster_status = player.get("rosterStatus", "").upper()
                        if "IR" in roster_status or "INJURED" in roster_status:
                            is_injured = True
                            if not status:
                                status = roster_status
                        
                        if is_injured:
                            name = _display_name(player)
                            injuries[normalize_name_key(name)] = str(status)
                            logger.info(f"Found injury: {name} - {status}")
        
        logger.info(f"Found {len(injuries)} injured players for {team_abbr}")
        
    except Exception as e:
        logger.warning(f"Could not fetch injuries for {team_abbr}: {e}")
    
    return injuries

# ===================== ROSTER FUNCTIONS =====================

def _split_roster_from_json(roster_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split roster JSON into skaters and goalies with deduplication"""
    skaters: List[Dict[str, Any]] = []
    goalies: List[Dict[str, Any]] = []

    if not isinstance(roster_json, dict):
        return skaters, goalies

    candidates: List[List[Dict[str, Any]]] = []
    for k in ("forwards", "defensemen", "defense", "skaters", "players", "roster"):
        v = roster_json.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            candidates.append(v)

    forwards = roster_json.get("forwards", [])
    defense = roster_json.get("defensemen", roster_json.get("defense", []))
    gls = roster_json.get("goalies", [])
    
    if isinstance(forwards, list) and forwards:
        candidates.append(forwards)
    if isinstance(defense, list) and defense:
        candidates.append(defense)

    seen_ids = set()
    flat: List[Dict[str, Any]] = []
    for arr in candidates:
        for rec in arr:
            pid = rec.get("id") or rec.get("playerId") or rec.get("personId") or _display_name(rec)
            key = f"{pid}|{_position_code(rec)}"
            if key in seen_ids:
                continue
            seen_ids.add(key)
            flat.append(rec)

    if isinstance(gls, list) and gls and isinstance(gls[0], dict):
        for g in gls:
            name = _display_name(g)
            pos = _position_code(g) or "G"
            goalies.append({"name": name, "position": pos})

    for rec in flat:
        pos = _position_code(rec)
        name = _display_name(rec)
        if pos == "G":
            goalies.append({"name": name, "position": pos})
        else:
            skaters.append({"name": name, "position": pos})

    def _dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        s = set()
        out = []
        for r in rows:
            k = (r.get("name") or "", r.get("position") or "")
            if k in s:
                continue
            s.add(k)
            out.append(r)
        return out

    return _dedup(skaters), _dedup(goalies)


def _get_team_roster_by_season_or_current(abbrev: str, season: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Get team roster with multiple endpoint fallbacks"""
    urls = [
        f"{NHL_API_BASE}/roster/{abbrev}/{season}",
        f"{NHL_API_BASE}/roster/{abbrev}/current",
        f"{NHL_API_BASE}/club-roster/{abbrev}/{season}",
        f"{NHL_API_BASE}/club-roster/{abbrev}/current",
    ]
    
    for url in urls:
        js = _try_get_json(url)
        if js:
            sk, gl = _split_roster_from_json(js)
            if sk or gl:
                logger.info(f"Retrieved roster for {abbrev} from {url}")
                return sk, gl
    
    logger.warning(f"No roster found for {abbrev} season {season}")
    return [], []


def _get_last_game_roster(abbrev: str, date_str: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Get roster from most recent game before date, including TOI for skaters"""
    season = season_from_date(date_str)
    sched_url = f"{NHL_API_BASE}/club-schedule-season/{abbrev}/{season}"
    sched = _try_get_json(sched_url)
    
    if not sched:
        return [], []

    games = sched.get("games", [])
    if not isinstance(games, list) or not games:
        return [], []

    try:
        target_date = datetime.fromisoformat(date_str).date()
    except Exception:
        target_date = date.today()

    last_game = None
    for g in games:
        gd_raw = g.get("gameDate") or g.get("startTimeUTC") or g.get("startTime")
        gd = None
        try:
            gd = datetime.fromisoformat(gd_raw.replace("Z", "+00:00")).date() if isinstance(gd_raw, str) else None
        except Exception:
            try:
                gd = datetime.fromisoformat(gd_raw).date() if isinstance(gd_raw, str) else None
            except Exception:
                gd = None
        
        if gd and gd < target_date:
            if (last_game is None) or (gd > datetime.fromisoformat(last_game.get("_gd")).date()):
                g["_gd"] = gd.isoformat()
                last_game = g

    if not last_game:
        return [], []

    gid = last_game.get("id") or last_game.get("gameId") or last_game.get("gamePk")
    if not gid:
        return [], []

    box = _try_get_json(f"{NHL_API_BASE}/gamecenter/{gid}/boxscore")
    if not box:
        return [], []

    home = box.get("homeTeam", {}) or {}
    away = box.get("awayTeam", {}) or {}
    pstats = box.get("playerByGameStats", {}) or {}
    home_stats = pstats.get("homeTeam", {}) or {}
    away_stats = pstats.get("awayTeam", {}) or {}
    home_abv = (home.get("abbrev") or "").upper()
    away_abv = (away.get("abbrev") or "").upper()

    team_block = home_stats if home_abv == abbrev.upper() else away_stats if away_abv == abbrev.upper() else {}

    skaters: List[Dict[str, Any]] = []
    goalies: List[Dict[str, Any]] = []

    gls = team_block.get("goalies") or []
    if isinstance(gls, list):
        for g in gls:
            name = _display_name(g)
            goalies.append({"name": name, "position": "G"})

    added = set()
    for key in ("forwards", "defense", "skaters"):
        arr = team_block.get(key) or []
        if isinstance(arr, list):
            for rec in arr:
                nm = _display_name(rec)
                pos = _position_code(rec)
                stats = rec.get("stats") or rec.get("skaterStats") or rec
                toi_str = stats.get("toi") or rec.get("toi") or "0:00"
                toi_min = parse_minutes(toi_str)
                
                if pos != "G" and toi_min > 0:
                    k = (nm, pos)
                    if k not in added:
                        added.add(k)
                        skaters.append({"name": nm, "position": pos, "toi": str(toi_str), "toi_min": float(toi_min)})

    if not skaters:
        players_map = team_block.get("players", {}) or {}
        for rec in players_map.values():
            nm = _display_name(rec)
            pos = _position_code(rec)
            if pos != "G":
                k = (nm, pos)
                if k not in added:
                    added.add(k)
                    skaters.append({"name": nm, "position": pos, "toi": "0:00", "toi_min": 0.0})

    return skaters, goalies


def get_predicted_roster(abbrev: str, date_str: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get predicted roster for a team on a specific date"""
    season = season_from_date(date_str)
    sk, gl = _get_team_roster_by_season_or_current(abbrev, season)
    
    if not sk and not gl:
        sk, gl = _get_last_game_roster(abbrev, date_str)

    def _norm(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            out.append({
                "name": r.get("name") if isinstance(r.get("name"), str) else _display_name(r),
                "position": _position_code(r)
            })
        return out

    return {
        "skaters": _norm(sk),
        "goalies": _norm(gl),
    }


# ===================== LINEUP SELECTION =====================

def _is_forward(pos: str) -> bool:
    """Check if position is forward"""
    return (pos or "").upper() in FORWARD_POSITIONS


def _is_defense(pos: str) -> bool:
    """Check if position is defense"""
    return (pos or "").upper() in DEFENSE_POSITIONS


def _is_goalie(pos: str) -> bool:
    """Check if position is goalie"""
    return (pos or "").upper() in GOALIE_POSITIONS


def get_predicted_lineup(abbrev: str, date_str: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get predicted 18-skater + 2-goalie lineup for a team"""
    season = season_from_date(date_str)
    roster = get_predicted_roster(abbrev, date_str)
    roster_skaters = roster.get("skaters", [])
    roster_goalies = roster.get("goalies", [])

    last_skaters, last_goalies = _get_last_game_roster(abbrev, date_str)

    def uniq(seq):
        seen = set()
        out = []
        for r in seq:
            key = (r.get("name"), r.get("position"))
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": r.get("name"), "position": _position_code(r)})
        return out

    last_skaters = uniq(last_skaters)
    last_goalies = uniq(last_goalies)
    roster_skaters = uniq(roster_skaters)
    roster_goalies = uniq(roster_goalies)

    last_forwards = [r for r in last_skaters if _is_forward(r["position"])]
    last_defense = [r for r in last_skaters if _is_defense(r["position"])]
    bench_forwards = [r for r in roster_skaters if _is_forward(r["position"]) and r not in last_forwards]
    bench_defense = [r for r in roster_skaters if _is_defense(r["position"]) and r not in last_defense]
    bench_other = [r for r in roster_skaters if not _is_forward(r["position"]) and not _is_defense(r["position"])]

    chosen_forwards = (last_forwards + bench_forwards)[:12]
    chosen_defense = (last_defense + bench_defense)[:6]
    skaters = chosen_forwards + chosen_defense

    if len(skaters) < 18:
        remaining = []
        for r in last_skaters:
            if r not in skaters and not _is_goalie(r["position"]):
                remaining.append(r)
        for r in roster_skaters:
            if r not in skaters and not _is_goalie(r["position"]):
                remaining.append(r)
        remaining += [r for r in bench_other if r not in remaining]
        needed = 18 - len(skaters)
        skaters += remaining[:needed]

    skaters = skaters[:18]

    goalies = (last_goalies + [g for g in roster_goalies if g not in last_goalies])
    goalies = [g for g in goalies if _is_goalie(g["position"])]
    
    if len(goalies) < 2 and roster_goalies:
        for g in roster_goalies:
            if _is_goalie(g["position"]) and g not in goalies:
                goalies.append(g)
            if len(goalies) >= 2:
                break
    
    goalies = goalies[:2]

    logger.info(f"Generated lineup for {abbrev}: {len(skaters)} skaters, {len(goalies)} goalies")
    
    return {"skaters": skaters, "goalies": goalies}


# ===================== ENHANCED LINEUP PREDICTION WITH INJURIES =====================

def _build_lineup_from_last_game_records(
    skaters: List[Dict[str, Any]],
    goalies: List[Dict[str, Any]],
    injuries: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a lineup dict (forwards/defense/goalies) from last game's roster,
    using Time On Ice (TOI) to select lines: top 12 F and top 6 D by TOI.
    """
    forwards_raw = [r for r in skaters if _is_forward(_position_code(r))]
    defense_raw = [r for r in skaters if _is_defense(_position_code(r))]
    forwards_sorted = sorted(
        forwards_raw, key=lambda r: float(r.get("toi_min", 0.0)), reverse=True
    )
    defense_sorted = sorted(
        defense_raw, key=lambda r: float(r.get("toi_min", 0.0)), reverse=True
    )

    forwards: List[Dict[str, Any]] = []
    defense: List[Dict[str, Any]] = []
    gl_out: List[Dict[str, Any]] = []

    for r in forwards_sorted[:12]:
        name = str(r.get("name") or "").strip()
        pos = _position_code(r)
        nk = normalize_name_key(name)
        forwards.append({
            "name": name,
            "position": pos,
            "confirmed": False,
            "injured": nk in injuries,
            "injury_status": injuries.get(nk, "")
        })

    for r in defense_sorted[:6]:
        name = str(r.get("name") or "").strip()
        pos = _position_code(r)
        nk = normalize_name_key(name)
        defense.append({
            "name": name,
            "position": pos,
            "confirmed": False,
            "injured": nk in injuries,
            "injury_status": injuries.get(nk, "")
        })

    for g in goalies[:2]:
        name = str(g.get("name") or "").strip()
        nk = normalize_name_key(name)
        gl_out.append({
            "name": name,
            "position": "G",
            "confirmed": False,
            "injured": nk in injuries,
            "injury_status": injuries.get(nk, "")
        })

    return {"forwards": forwards, "defense": defense, "goalies": gl_out}


def get_confirmed_or_predicted_lineup(
    abbrev: str,
    game_date: str,
    game_id: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get confirmed starting lineup if available; otherwise:
    - Prefer last game's lineup based on TOI (top 12 F, top 6 D)
    - Fall back to model-predicted lineup based on roster and recent usage
    """
    lineup = {
        "forwards": [],
        "defense": [],
        "goalies": []
    }
    
    injuries = get_team_injuries(abbrev)
    logger.info(f"Found {len(injuries)} injured players for {abbrev}")
    
    if game_id:
        try:
            preview_url = f"{NHL_API_BASE}/gamecenter/{game_id}/landing"
            preview = _try_get_json(preview_url)
            
            if preview:
                confirmed = _extract_confirmed_lineup(preview, abbrev, injuries)
                if confirmed:
                    logger.info(f"Found confirmed lineup for {abbrev}")
                    return confirmed
        except Exception as e:
            logger.debug(f"Could not get confirmed lineup: {e}")
    
    try:
        last_skaters, last_goalies = _get_last_game_roster(abbrev, game_date)
        if last_skaters or last_goalies:
            lg_lineup = _build_lineup_from_last_game_records(last_skaters, last_goalies, injuries)
            if lg_lineup["forwards"] or lg_lineup["defense"] or lg_lineup["goalies"]:
                logger.info(f"Using last game's TOI-based lineup for {abbrev}")
                return lg_lineup
    except Exception as e:
        logger.debug(f"Failed to use last game's lineup for {abbrev}: {e}")
    
    roster = get_predicted_roster(abbrev, game_date)
    recent_usage = _get_recent_lineup_usage(abbrev, game_date, n=5)
    
    return _build_predicted_lineup(roster, recent_usage, injuries)


def _extract_confirmed_lineup(
    preview_data: Dict[str, Any],
    team_abbr: str,
    injuries: Dict[str, str]
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Extract confirmed lineup from game preview data"""
    try:
        home_team = (preview_data.get("homeTeam") or {})
        away_team = (preview_data.get("awayTeam") or {})
        
        team_data = None
        if (home_team.get("abbrev") or "").upper() == team_abbr.upper():
            team_data = home_team
        elif (away_team.get("abbrev") or "").upper() == team_abbr.upper():
            team_data = away_team
        
        if not team_data:
            return None
        
        lineup_data = team_data.get("lineup") or team_data.get("roster")
        if not lineup_data:
            return None
        
        forwards = []
        defense = []
        goalies = []
        
        for player in lineup_data:
            pos = _position_code(player)
            name = _display_name(player)
            name_key = normalize_name_key(name)
            
            is_injured = name_key in injuries
            injury_status = injuries.get(name_key, "")
            
            player_info = {
                "name": name,
                "position": pos,
                "confirmed": True,
                "injured": is_injured,
                "injury_status": injury_status
            }
            
            if pos in ("C", "LW", "RW"):
                forwards.append(player_info)
            elif pos in ("D", "LD", "RD"):
                defense.append(player_info)
            elif pos == "G":
                goalies.append(player_info)
        
        if forwards or defense or goalies:
            return {
                "forwards": forwards[:12],
                "defense": defense[:6],
                "goalies": goalies[:2]
            }
        
        return None
    
    except Exception as e:
        logger.debug(f"Error extracting confirmed lineup: {e}")
        return None


def _get_recent_lineup_usage(
    team_abbr: str,
    game_date: str,
    n: int = 5
) -> Dict[str, Dict[str, Any]]:
    """Get player usage stats from recent games"""
    usage = {}
    season = season_from_date(game_date)
    
    try:
        sched_url = f"{NHL_API_BASE}/club-schedule-season/{team_abbr}/{season}"
        sched = _try_get_json(sched_url)
        
        if not sched:
            return usage
        
        games = sched.get("games", [])
        target_date = datetime.fromisoformat(game_date).date()
        
        recent_games = []
        for g in games:
            gd = None
            try:
                gd_raw = g.get("gameDate") or g.get("startTimeUTC")
                gd = datetime.fromisoformat(gd_raw.replace("Z", "+00:00")).date()
            except:
                continue
            
            if gd and gd < target_date:
                recent_games.append((gd, g))
        
        recent_games.sort(key=lambda x: x[0], reverse=True)
        recent_games = recent_games[:n]
        
        for _, game in recent_games:
            gid = game.get("id") or game.get("gameId")
            if not gid:
                continue
            
            box = _try_get_json(f"{NHL_API_BASE}/gamecenter/{gid}/boxscore")
            if not box:
                continue
            
            home_abbr = (box.get("homeTeam") or {}).get("abbrev", "").upper()
            is_home = (home_abbr == team_abbr.upper())
            
            team_key = "homeTeam" if is_home else "awayTeam"
            team_stats = (box.get("playerByGameStats") or {}).get(team_key) or {}
            
            for fwd in (team_stats.get("forwards") or []):
                name = _display_name(fwd)
                pos = _position_code(fwd)
                stats = fwd.get("stats") or {}
                toi = stats.get("toi") or "0:00"
                
                try:
                    parts = toi.split(":")
                    minutes = int(parts[0]) + int(parts[1]) / 60.0
                except:
                    minutes = 0.0
                
                if name not in usage:
                    usage[name] = {
                        "games": 0,
                        "total_toi": 0.0,
                        "position": pos,
                        "is_forward": True
                    }
                
                usage[name]["games"] += 1
                usage[name]["total_toi"] += minutes
            
            for dman in (team_stats.get("defense") or []):
                name = _display_name(dman)
                pos = _position_code(dman)
                stats = dman.get("stats") or {}
                toi = stats.get("toi") or "0:00"
                
                try:
                    parts = toi.split(":")
                    minutes = int(parts[0]) + int(parts[1]) / 60.0
                except:
                    minutes = 0.0
                
                if name not in usage:
                    usage[name] = {
                        "games": 0,
                        "total_toi": 0.0,
                        "position": pos,
                        "is_forward": False
                    }
                
                usage[name]["games"] += 1
                usage[name]["total_toi"] += minutes
            
            for goalie in (team_stats.get("goalies") or []):
                name = _display_name(goalie)
                
                if name not in usage:
                    usage[name] = {
                        "games": 0,
                        "total_toi": 0.0,
                        "position": "G",
                        "is_goalie": True
                    }
                
                usage[name]["games"] += 1
        
        for player in usage.values():
            if player["games"] > 0:
                player["avg_toi"] = player["total_toi"] / player["games"]
    
    except Exception as e:
        logger.error(f"Error getting recent usage for {team_abbr}: {e}")
    
    return usage


def _build_predicted_lineup(
    roster: Dict[str, List[Dict[str, Any]]],
    recent_usage: Dict[str, Dict[str, Any]],
    injuries: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Build predicted lineup by combining roster, recent usage, and filtering out injuries"""
    predicted = {
        "forwards": [],
        "defense": [],
        "goalies": []
    }
    
    forward_candidates = []
    for player in roster.get("skaters", []):
        if player.get("position") in ("C", "LW", "RW"):
            name = player.get("name")
            name_key = normalize_name_key(name)
            usage = recent_usage.get(name, {})
            
            if name_key in injuries:
                logger.info(f"Excluding injured player: {name} ({injuries[name_key]})")
                continue
            
            forward_candidates.append({
                "name": name,
                "position": player.get("position"),
                "games": usage.get("games", 0),
                "avg_toi": usage.get("avg_toi", 0.0),
                "confirmed": False,
                "injured": False,
                "injury_status": ""
            })
    
    forward_candidates.sort(
        key=lambda x: (x["games"], x["avg_toi"]),
        reverse=True
    )
    predicted["forwards"] = forward_candidates[:12]
    
    defense_candidates = []
    for player in roster.get("skaters", []):
        if player.get("position") in ("D", "LD", "RD"):
            name = player.get("name")
            name_key = normalize_name_key(name)
            usage = recent_usage.get(name, {})
            
            if name_key in injuries:
                logger.info(f"Excluding injured player: {name} ({injuries[name_key]})")
                continue
            
            defense_candidates.append({
                "name": name,
                "position": player.get("position"),
                "games": usage.get("games", 0),
                "avg_toi": usage.get("avg_toi", 0.0),
                "confirmed": False,
                "injured": False,
                "injury_status": ""
            })
    
    defense_candidates.sort(
        key=lambda x: (x["games"], x["avg_toi"]),
        reverse=True
    )
    predicted["defense"] = defense_candidates[:6]
    
    goalie_candidates = []
    for player in roster.get("goalies", []):
        name = player.get("name")
        name_key = normalize_name_key(name)
        usage = recent_usage.get(name, {})
        
        if name_key in injuries:
            logger.info(f"Excluding injured goalie: {name} ({injuries[name_key]})")
            continue
        
        goalie_candidates.append({
            "name": name,
            "position": "G",
            "games": usage.get("games", 0),
            "confirmed": False,
            "injured": False,
            "injury_status": ""
        })
    
    goalie_candidates.sort(key=lambda x: x["games"], reverse=True)
    predicted["goalies"] = goalie_candidates[:2]
    
    logger.info(f"Built predicted lineup: {len(predicted['forwards'])}F, {len(predicted['defense'])}D, {len(predicted['goalies'])}G")
    
    return predicted


# ===================== GOALIE OVERRIDE (ROSTER-BASED) =====================

def get_roster_goalies_for_override(team_abbr: str, game_date: str) -> List[str]:
    """
    Return up to two goaltenders from the team's roster for goalie override selection.
    Prefers the current/season roster; falls back to last game's goalies.
    """
    season = season_from_date(game_date)
    names: List[str] = []
    try:
        _sk, gl = _get_team_roster_by_season_or_current(team_abbr, season)
        if gl:
            for g in gl:
                nm = g.get("name") if isinstance(g.get("name"), str) else _display_name(g)
                if nm:
                    nm_fmt = format_initial_last(sanitize_text(nm))
                    if nm_fmt not in names:
                        names.append(nm_fmt)
                if len(names) >= 2:
                    break
    except Exception as e:
        logger.debug(f"Roster goalies not found for {team_abbr}: {e}")
    
    if len(names) < 2:
        try:
            _sk_last, gl_last = _get_last_game_roster(team_abbr, game_date)
            for g in gl_last:
                nm = g.get("name")
                if nm:
                    nm_fmt = format_initial_last(sanitize_text(nm))
                    if nm_fmt not in names:
                        names.append(nm_fmt)
                if len(names) >= 2:
                    break
        except Exception as e:
            logger.debug(f"Fallback last-game goalies failed for {team_abbr}: {e}")
    
    return names[:2]


# ===================== SESSION MANAGEMENT =====================

def close_session():
    """Close the global session (call on application shutdown)"""
    global _session
    if _session:
        _session.close()
        logger.info("Closed NHL API session")


def reset_session():
    """Reset the global session (useful for testing or error recovery)"""
    global _session
    close_session()
    _session = create_session()
    logger.info("Reset NHL API session")