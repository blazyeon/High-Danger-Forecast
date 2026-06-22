# CLAUDE.md — NHL Game Predictor

Instructions for Claude Code and future maintainers working on the `High-Danger-Forecast` / NHL Game Predictor repository.

## Project overview

Flask SPA that predicts NHL games using an ensemble of Elo ratings, Monte-Carlo simulation, and a trained ML model. Serves a dark-themed hockey UI with tabs for Matchup Predictor, Game Lookup, Advanced Stats, Player Props, Betting Edge, and Elo Leaderboard.

- Entry point: `python run_app.py` → http://localhost:8501
- Flask app: `app.py`
- Frontend: `templates/index.html`, `static/app.js`, `static/style.css`
- Core logic: `NHL/` and `EloMl/`

## Common commands

```bash
# Run locally
python run_app.py

# Run tests (must pass before deploy)
python test_xg_pipeline.py
python test_betting_edge.py
python test_api_endpoints.py

# Refresh data caches
python update_pbp_stats.py --season 2025 --stype 2 --out static/data
python update_elo_ratings.py --current-season
python update_odds.py
python update_injuries.py

# Retrain / tune
python train_model.py
python backtest.py --season 20242025
python tune_elo.py
```

## Architecture highlights

- Ensemble weights (34% Elo / 51% Sim / 15% ML) live in `NHL/Config.py`.
- PBP-derived advanced stats are cached in `static/data/pbp_*_stats_YYYYYYYY.json`.
- Shot stores are tracked in Git at `pbp_cache/shots/*.parquet` (`.gitignore` keeps raw/schedule local-only).
- Daily scheduler: `daily_update.bat` run by Windows Task Scheduler at ~5:03 AM.

## Key conventions

- American odds integers (e.g. `-135`, `+220`) are converted to decimal internally.
- Vig is removed via two-way normalization: `true_prob = implied_prob / (p1 + p2)`.
- Player Props use `model probability − book-implied probability` for edges.
- Goals props are intentionally Over-only.
- Historical team mapping: `ARI` is normalized to `UTA` where applicable.

## Deployment

- Target: Render (auto-deploys from `main` pushes).
- Required production env var: `ODDS_API_KEY` for live odds / Betting Edge.
- Verify Windows Task Scheduler path if the repo is moved.

## Things to watch

- NHL API endpoints can change without notice (`api-web.nhle.com`, `statsapi.web.nhl.com`).
- `update_pbp_stats.py` for older seasons can take 10–15 min per full regular season.
- If `elo_ratings.db` is locked by a running `run_app.py`, merges/pushes may fail until the process is stopped.

## Memory

Project context is stored in `C:\Users\Nicholas Siapas\.claude\projects\...NHLPrediction-main\memory\`. Consult `MEMORY.md` for persistent facts, recent fixes, and deployment status.

---

# Behavioral Guidelines

Rules to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
