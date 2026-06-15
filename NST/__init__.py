"""
NST — Natural Stat Trick data fetching and scraping.

.. deprecated::
    NST HTML scraping has been replaced by the NHL API play-by-play
    pipeline (see ``NHL.PlayByPlay`` + ``NHL.StatsFromPBP``). This
    package is kept as a thin re-export layer for backwards
    compatibility — any code that still imports from ``NST`` will
    continue to work but should migrate to the new pipeline.
"""
import warnings
warnings.warn(
    "The NST package is deprecated; use NHL.PlayByPlay + NHL.StatsFromPBP instead.",
    DeprecationWarning,
    stacklevel=2,
)

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