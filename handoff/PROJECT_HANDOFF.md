# NHL Game Predictor — Full Project Handoff

**Project:** High Danger Forecast / NHL Game Predictor  
**Repository:** `C:\Users\Nicholas Siapas\OneDrive\Documents\NHLPredictionV1.0.0\NHLPrediction-main`  
**Handoff date:** 2026-06-17  
**Status:** Active; Betting Edge and Player Props table refactor deployed.

---

## 1. What This Project Is

A Flask-based single-page application that predicts NHL game outcomes using a hybrid ensemble of:

- **Elo ratings** (`EloMl/`)
- **Monte-Carlo simulation** (`NHL/Simulation.py`)
- **Machine-learning win-probability model** (`EloMl/MLModel.py`)

The site serves a dark-themed hockey UI with tabs for Matchup Predictor, Game Lookup, Advanced Stats, Player Props (new table layout), and (new) Betting Edge.

---

## 2. Quick Start

```bash
pip install -r requirements.txt
python run_app.py
```

Open http://localhost:8501

Run tests:

```bash
python test_xg_pipeline.py
python test_betting_edge.py
```

---

## 3. Repository Layout

```
NHLPrediction-main/
├── app.py                 # Flask server + REST API endpoints
├── run_app.py             # Launcher (port 8501 by default)
├── requirements.txt
├── README.md
├── daily_update.bat       # Windows Task Scheduler daily update script
│
├── NHL/                   # Core hockey logic
│   ├── Simulation.py          # Monte-Carlo game simulator
│   ├── Prediction.py          # Matchup prediction orchestration
│   ├── MatchupUtils.py        # Matchup helpers
│   ├── Lookup.py              # NHL API schedule/team lookups
│   ├── ApiScrape.py           # NHL API scraping
│   ├── PlayByPlay.py          # PBP ingestion
│   ├── StatsFromPBP.py        # PBP-derived team/skater/goalie stats
│   ├── xGModel.py             # Expected-goals model
│   ├── Features.py            # Rest/travel/fatigue features
│   ├── GoaliePrediction.py    # Goalie impact modeling
│   ├── PlayerLinePredictor.py # Projected lineups
│   ├── InjuryScraper.py       # Injury data
│   ├── OddsAPI.py             # Player props odds
│   ├── BettingEdge.py         # NEW: value-bet engine
│   ├── TeamsMeta.py, Utils.py, Config.py, Errors.py
│
├── EloMl/                 # Elo + ML subsystems
│   ├── Ratings.py
│   ├── Features.py
│   ├── MLModel.py
│   ├── Database.py
│   └── __init__.py
│
├── templates/
│   └── index.html         # SPA shell
├── static/
│   ├── app.js             # Frontend logic
│   ├── style.css          # Dark hockey theme
│   └── data/              # Cached stats + demo data
│       ├── demo_odds.json        # NEW
│       ├── demo_props.json
│       └── pbp_*_stats*.json
│
├── update_pbp_stats.py
├── update_elo_ratings.py
├── update_injuries.py
├── update_odds.py            # NEW
├── train_model.py
├── backtest.py
├── check_model.py
├── check_elo_data.py
├── view_elo_ratings.py
├── cleanup_database.py
├── tune_elo.py
├── test_xg_pipeline.py
├── test_betting_edge.py      # NEW
└── handoff/                  # THIS FOLDER
```

---

## 4. Architecture Overview

### 4.1 Data flow

1. **Daily update** (`daily_update.bat`, ~5:03 AM)
   - `update_pbp_stats.py` → refreshes play-by-play derived stats
   - `update_elo_ratings.py --current-season` → refreshes Elo ratings
   - `update_odds.py` → NEW: fetches and caches sportsbook odds

2. **Web request**
   - Frontend hits Flask REST endpoints
   - Endpoints call `NHL.Simulation.simulate_matchup()` or cached stats
   - Results returned as JSON and rendered client-side

3. **Model prediction** (`simulate_matchup`)
   - Pulls team xG/goal stats from PBP data
   - Applies rest/travel, injuries, goalie impact, special teams, venue advantage
   - Runs correlated Poisson/normal Monte Carlo simulations
   - Blends Elo, simulation, and ML win probabilities using calibrated ensemble weights

### 4.2 Ensemble weights

Current calibrated weights (from 2024-25 walk-forward backtest):

| Component | Weight |
|-----------|--------|
| Elo win probability | 34% |
| Simulation win probability | 51% |
| ML win probability | 15% |

These are encoded in `NHL/Config.py` (`MODEL_WEIGHTS`) and applied inside the simulation/calibration layer.

---

## 5. Key Configuration

Central config: **`NHL/Config.py`**

Important knobs:

| Setting | Location | Notes |
|---------|----------|-------|
| `CURRENT_SEASON_YEAR` | `Config.py` | 2025 for 2025-26 season |
| `DEFAULT_SIMULATIONS` | `Config.py` | 10,000 default sims |
| `SIMULATION_PARAMS` | `Config.py` | Shock sigma, empty-net, blowout, correlation rho |
| `MODEL_WEIGHTS` | `Config.py` | xG/gf/pp/goalie weights and ensemble weights |
| `VENUE_ADV_PARAMS` | `Config.py` | Home-ice baseline calibrated to 2025-26 observed rates |
| `REST_TRAVEL_PARAMS` | `Config.py` | B2B, rest diff, travel fatigue |
| `DIVISIONS` / `NST_ABBR_TO_FULL` | `Config.py` | Team mappings (ARI → UTA handled) |

Environment variables:

- `ODDS_API_KEY` — required for live odds fetch in `update_odds.py` / Betting Edge
- `RATE_LIMIT_SLEEP`, `RATE_LIMIT_JITTER` — optional throttling overrides
- `NST_CACHE_DIR` — optional Natural Stat Trick cache path

---

## 6. Backend Modules

### 6.1 `app.py` endpoints

| Endpoint | Purpose |
|----------|---------|
| `/` | Serves `index.html` |
| `/api/teams` | Team list |
| `/api/schedule?date=YYYY-MM-DD` | Games on a date |
| `/api/matchup?home=...&away=...` | Full matchup prediction |
| `/api/simulate` | Simulation-only call |
| `/api/stats?type=team/skater/goalie` | PBP advanced stats |
| `/api/player-props?date=...` | Player prop lines |
| `/api/betting-edge?date=...&edge_threshold=0.03&demo=0/1` | **NEW** value bets |
| `/api/elo` | Elo ratings |
| `/api/logos/<team>.png` | Team logos |

### 6.2 `NHL/Simulation.py`

The main simulator. Key output fields used by UI and Betting Edge:

- `home_win_pct`, `away_win_pct`
- `home_win_2plus_pct`, `away_win_2plus_pct` (puck-line proxy)
- `totals_distribution` (over/under probabilities)
- `expected_home_goals`, `expected_away_goals`
- `regulation_prob`, `ot_prob`, `so_prob`
- `confidence`, `component_breakdown`

### 6.3 `EloMl/`

- `Ratings.py` — Elo rating engine
- `MLModel.py` — trained win-probability model
- `Database.py` — SQLite Elo storage
- `Features.py` — feature engineering for ML

### 6.4 `NHL/BettingEdge.py` (NEW)

- Fetches/caches odds from The Odds API v4
- Removes vig via two-way normalization
- Matches schedule games to odds events
- Computes edges for moneyline, puck line, totals
- Provides demo odds + demo schedule fallback for offseason

---

## 7. Frontend

- **Shell:** `templates/index.html`
- **Logic:** `static/app.js`
- **Styles:** `static/style.css`

SPA tabs: Matchup, Lookup, Stats, Props, Elo, **Betting Edge**.

Shared helpers in `static/app.js`:
- `safeFetchJson(url)`
- `formatAmerican(num)`
- `escapeHtml(str)`
- `showLoading(container)`

---

## 8. Data Pipeline & Scheduling

### 8.1 Daily update (`daily_update.bat`)

Run at ~5:03 AM by Windows Task Scheduler.

Order of operations:
1. `update_pbp_stats.py`
2. `update_elo_ratings.py --current-season`
3. `update_odds.py`

Log file: `daily_update.log`

### 8.2 Manual refresh commands

```bash
# PBP stats
python update_pbp_stats.py --season 2024 --stype 2 --out static/data

# Elo ratings (current season)
python update_elo_ratings.py --current-season

# Odds
python update_odds.py

# Injuries
python update_injuries.py
```

### 8.3 Model retraining

```bash
python train_model.py
python backtest.py --season 2024202025
```

---

## 9. Testing

| Test file | Coverage |
|-----------|----------|
| `test_xg_pipeline.py` | Full xG/PBP/Stats/Elo/Simulation pipeline |
| `test_betting_edge.py` | Odds matching, vig removal, edge computation, endpoint |

Run both before any deploy:

```bash
python test_xg_pipeline.py
python test_betting_edge.py
```

---

## 10. Deployment Notes

- App runs on Flask directly via `run_app.py`.
- Currently targeted for Render deployment (per recent commits).
- Environment variable `ODDS_API_KEY` must be set in production for live odds.
- Ensure Windows Task Scheduler job points to the correct `daily_update.bat` path after repo moves.

---

## 11. Recent Changes (last few commits)

1. **Betting Edge tab** — full implementation across backend/frontend/scheduler/tests.
2. **Model logic fixes** — removed pre-sim score effects, symmetric streaks, season-blend decay, ML total-inflation fix.
3. **Ensemble recalibration** — 34/51/15 Elo/Sim/ML weights.
4. **Venue advantage recalibration** — based on 2025-26 observed home/away win rates.
5. **B2B/rest/travel recalibration**.
6. **KeyError fix** in injury stats aggregation (`assists`).

---

## 12. Known Issues & Watch Items

1. **Offseason demo mode** — Betting Edge currently shows synthetic games because the NHL season is over. When the season restarts, live schedule + `ODDS_API_KEY` will provide real data.
2. **Puck-line proxy** — uses `win by 2+` probability as puck-line cover proxy; accuracy should be monitored.
3. **Odds API rate limits / availability** — cache is designed to absorb this.
4. **NHL API changes** — the project depends on `api-web.nhle.com` and `statsapi.web.nhl.com`. Monitor for breaking changes.
5. **Task Scheduler path** — if the repo is moved, the scheduled task must be updated.

---

## 13. Next Steps / Recommended Work

- [ ] Deploy current branch to Render and smoke-test all tabs.
- [ ] Verify `ODDS_API_KEY` in production.
- [ ] When season starts, run `update_odds.py` manually once and confirm live events match schedule.
- [ ] Consider backtesting Betting Edge threshold (currently 3%) once a month of live odds is available.
- [ ] Evaluate replacing puck-line proxy with explicit -1.5/+1.5 cover modeling.
- [ ] Add automated alerting if `daily_update.bat` fails or odds cache goes stale > 12 hours.

---

## 14. Contact / Context

- Author: `blazyeon`
- Branch: `main`
- Recent commits visible in git log; full history retained.
- Claude memory folder (hidden): `C:\Users\Nicholas Siapas\.claude\projects\...\memory\`

---

*End of handoff.*
