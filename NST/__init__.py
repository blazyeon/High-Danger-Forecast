"""
NST — Natural Stat Trick data fetching and scraping.

Primary interface:
    get_nst_table_from_url  — Fetch and parse any NST table (cached, robust)
    build_nst_team_url      — Build team stats URL
    build_nst_player_url    — Build player stats URL
    test_nst_connection      — Health check
    clear_cache             — Clear the in-memory cache

Backward-compatible aliases:
    fetch_nst_table         — Alias for get_nst_table_from_url
    get_table               — Alias for get_nst_table_from_url
"""
from NST.Cache import (
    get_nst_table_from_url,
    scrape_nst_player_table,
    get_nst_player_stats,
    build_nst_team_url,
    build_nst_player_url,
    validate_nst_url,
    test_nst_connection,
    clear_cache,
)

# Backward-compatible aliases (previously from TeamStats)
fetch_nst_table = get_nst_table_from_url
get_table = get_nst_table_from_url

__all__ = [
    "get_nst_table_from_url",
    "scrape_nst_player_table",
    "get_nst_player_stats",
    "build_nst_team_url",
    "build_nst_player_url",
    "validate_nst_url",
    "test_nst_connection",
    "clear_cache",
    "fetch_nst_table",
    "get_table",
]