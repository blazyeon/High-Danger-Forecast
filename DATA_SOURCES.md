# NHL Predictor — Data Sources Guide

## Primary Pipeline (v2+): NHL API + MoneyPuck CSVs

The Natural Stat Trick HTML scraper has been replaced. The new pipeline
has three legs:

1. **NHL API play-by-play** (`api-web.nhle.com/v1/gamecenter/{id}/play-by-play`)
   — the same raw data MoneyPuck and NST are built on top of. We hit it
   directly, no scraping. Used to **compute** all advanced stats (xG,
   Corsi, Fenwick, HDCF, GSAx). No auth, no rate limit issues, JSON,
   server-side only (no CORS).

2. **xG model** (`NHL/xGModel.py`) — logistic regression on shot features
   (distance, angle, shot type, situation, rebound/rush flags). Trained
   quarterly on the latest shot parquet. Validated against MoneyPuck's
   `xGoal` column.

3. **MoneyPuck CSVs** (`NHL/MoneyPuck.py`) — `peter-tanner.com/moneypuck/downloads/shots_{year}.zip`
   downloaded as a ZIP + extracted to CSV. Used as a **validation source**
   for our computed xG (NHL/Validation.py) and as the source for season-level
   aggregates (GAR, WAR).

**Why this is the best free option:** xG is just a logistic regression on
shot coordinates + shot type + situation. The NHL API gives us all those
features. MoneyPuck's published xG is computed from the same data. We can
match their model within ~2-3% AUC and **improve it** because we control
the features and model. NST is dropped because it adds no value over
NHL API + MP, and its scraper is the actual fragility point.

**Pipeline modules:**

- `NHL/PlayByPlay.py` — fetcher + disk cache (`pbp_cache/raw/{id}.json`),
  schedule walker, shot store builder (`pbp_cache/shots/shots_{year}_{stype}.parquet`).
- `NHL/StatsFromPBP.py` — `compute_team_rates`, `compute_skater_rates`,
  `compute_goalie_rates` from the shot store. Drop-in compatible with the
  old NST-driven aggregator (same return shape: team DataFrame, skater
  dict by name_key, goalie DataFrame).
- `NHL/xGModel.py` — `build_features`, `train_xg_model`, `predict_xg`,
  `load_xg_model`. Trained model saved to `models/xg_model.pkl`,
  training metrics to `models/xg_model_report.json`.
- `NHL/MoneyPuck.py` — `download_shots_zip(years)`, `parse_mp_shots(csv)`.
- `NHL/Validation.py` — `validate_xg_against_money_puck(season_year)`,
  cross-checks our xG vs MP's, writes `models/xg_validation.json`.
- `update_pbp_stats.py` — full refresh script. Idempotent, resumable from
  on-disk cache.

**Refresh cadence:**
- Daily: `update_pbp_stats.py --season <current>` — appends new games to
  the shot store, re-writes frontend JSON.
- Quarterly: `update_pbp_stats.py --train-xg` — retrain xG on latest data.
- On-demand: validation cross-check (runs by default as part of refresh).

## Natural Stat Trick (NST) — DEPRECATED

NST HTML scraping was the original data source. The `NST/` package is
still importable for backwards compatibility but logs a deprecation
warning and the rest of the codebase no longer calls into it. Do not
add new NST dependencies. The two main things NST gave us that we
needed:
- **xG / Corsi / Fenwick / HDCF** — replaced by our own xG model + the
  derived stats in `NHL.StatsFromPBP.compute_team_rates`.
- **Team / skater / goalie rates** — replaced by `compute_team_rates`,
  `compute_skater_rates`, `compute_goalie_rates` (PBP-derived).

## Other Sources

| Source | Used for | Status |
|--------|----------|--------|
| **NHL API** (`api-web.nhle.com/v1`) | Play-by-play, schedule, rosters, boxscores | ✅ Primary, `NHL/PlayByPlay.py`, `NHL/ApiScrape.py` |
| **MoneyPuck** (`peter-tanner.com/moneypuck`) | xG validation, season aggregates | ✅ Validation only, `NHL/MoneyPuck.py` |
| **NST** (`naturalstattrick.com`) | Was: advanced stats tables | ❌ Deprecated, `NST/Cache.py` shim only |
| **The Odds API** (`the-odds-api.com`) | Player prop odds, moneyline | ✅ In use, `NHL/OddsAPI.py` |
| **Hockey-Reference** | Historical Elo backfill | Not in use; CSV exports work fine for one-off imports |
| **RapidAPI / API-NHL** | Aggregated | Not used; NHL API covers the need |
| **Sportradar / Stats Perform** | Enterprise | Not used; $$$ |
| **PuckPedia** | Rosters/cap | Not used |
