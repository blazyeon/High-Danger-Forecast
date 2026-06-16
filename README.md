# NHL Game Predictor

Elo & ML-powered NHL game prediction system with a professional dark-themed web interface.

## Quick Start

```bash
pip install -r requirements.txt
python run_app.py
```

Then open **http://localhost:8501** in your browser.

## Features

- **Matchup Predictor** — Select two teams, optionally choose goalies and simulation parameters, and get win probabilities, expected goals, score distributions, and projected lineups
- **Game Lookup** — Search NHL games by date with scores and status
- **Advanced Stats** — Browse NHL API PBP-derived team, skater, and goalie statistics
- **Player Props** — View player prop lines and odds from sportsbooks

## Configuration

Key settings are in `NHL/Config.py`:
- Simulation parameters (sim count, shock sigma, home ice advantage)
- Elo rating weights and decay
- Model weights for xG, goals, shot quality
- Injury impact thresholds

## CLI Tools

```bash
# Update Elo ratings from NHL data
python update_elo_ratings.py --current-season --reset

# Refresh PBP-derived stats
python update_pbp_stats.py --season 2024 --stype 2 --out static/data

# Train the ML model (default: all available seasons)
python train_model.py

# Backtest and tune weights
python backtest.py --season 20242025

# Check Elo database status
python check_elo_data.py

# View Elo ratings
python view_elo_ratings.py

# Check ML model status
python check_model.py

# Run the test suite
python test_xg_pipeline.py
```

## Architecture

| Component | Description |
|-----------|-------------|
| `app.py` | Flask web server with REST API |
| `NHL/` | Core prediction logic |
| `EloMl/` | Elo rating system & ML model |
| `templates/index.html` | SPA frontend |
| `static/style.css` | Dark hockey theme |
| `static/app.js` | Frontend application logic |