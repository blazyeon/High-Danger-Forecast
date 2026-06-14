# NHL Predictor — Data Sources Guide

## Natural Stat Trick (NST) — CURRENT STATUS

**How it works:** `NST.Cache.py` scrapes HTML tables from `naturalstattrick.com` using `pandas.read_html` with a BeautifulSoup fallback. Results are cached in-memory for 6 hours and written to `nst_cache/*.parquet`.

**Is it working?** Yes, for now. The scraper is functional and returns team/skater/goalie stats. However, it is **fragile** — if NST changes their table structure or adds anti-bot measures, the scraper will break.

**Risk level:** Medium-High. NST is a free community site with no official API. Scraping is against their ToS in spirit, though they tolerate light traffic.

## Alternatives for Live Data / APIs

| Source | Type | API? | Notes |
|--------|------|------|-------|
| **NHL API** (`api-web.nhle.com/v1`) | Official | ✅ Yes, undocumented but stable | Schedule, boxscores, rosters, play-by-play. **Use this.** |
| **MoneyPuck** (`moneypuck.com`) | Advanced stats | ❌ No official API | Also scraped. More advanced models (xG, GAR, WAR). |
| **Hockey-Reference** | Historical | ❌ No API | Scraped. Great for historical Elo/backtesting. |
| **The Odds API** (`the-odds-api.com`) | Betting odds | ✅ Yes, free tier (500 calls/mo) | For props and live odds. Much better than scraping sportsbooks. |
| **RapidAPI / API-NHL** | Aggregated | ✅ Paid tier | Paid but reliable. Covers scores, stats, odds. |
| **Sportradar / Stats Perform** | Enterprise | ✅ Paid only | Official data providers. $$$ but bulletproof. |
| **PuckPedia** | Rosters/cap | ❌ No API | Scraped. Good for confirmed lineups and injuries. |

## Recommended Action Items

1. **Keep NST** as a secondary source for xGF/xGA/PDO but add robust error handling and fallback to league averages when it fails.
2. **Primary data:** Use the NHL API (`api-web.nhle.com/v1`) for schedules, rosters, boxscores, and live scores. It is fast, stable, and free.
3. **Odds/Props:** Sign up for The Odds API free tier and replace `NHL.OddsAPI.py` with their official endpoints.
4. **Backtesting:** Use MoneyPuck CSV exports (they publish season CSVs) to calibrate your model weights instead of hand-tuning.

## How to check if NST is alive

```bash
python -c "from NST.Cache import test_nst_connection; print(test_nst_connection())"
```

If this returns `False`, your NST scraper is broken. Check:
- Are you being rate-limited? (Wait 30s between requests)
- Did the site HTML structure change?
- Is `beautifulsoup4` / `lxml` / `html5lib` installed?
