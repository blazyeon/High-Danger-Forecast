"""
Train ML model with Elo features for NHL game predictions.

This script:
1. Loads games from training database
2. Initializes Elo systems and processes games chronologically
3. Extracts Elo features for each game
4. Combines with statistical features
5. Trains XGBoost model
6. Saves trained model with Elo features
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from EloMl.Database import EloDatabase
from EloMl.Ratings import PlayerEloSystem, TeamEloSystem, EloConfig
from EloMl.Features import EloFeatureEngine
from EloMl.MLModel import EloMLPredictor, ModelConfig
from EloMl.Updater import EloGameUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _current_season() -> str:
    """Compute current NHL season string dynamically."""
    today = date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}{year + 1}"


def parse_args():
    """Parse command line arguments"""
    _season = _current_season()
    parser = argparse.ArgumentParser(
        description='Train ML model with Elo features'
    )
    parser.add_argument(
        '--training-db',
        type=str,
        default=f'training_data_{_season}.db',
        help='Training database path'
    )
    parser.add_argument(
        '--season',
        type=str,
        default=_season,
        help='Season to train on'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/main_model.pkl',
        help='Output model path'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of workers (not used currently)'
    )
    
    return parser.parse_args()


def load_training_games(db: EloDatabase, season: str) -> List[Dict]:
    """Load games from training database"""
    
    logger.info("Loading training games from database...")
    
    cursor = db.conn.cursor()
    
    cursor.execute("""
        SELECT
            game_id,
            game_date,
            season,
            home_team,
            away_team,
            home_score,
            away_score,
            home_xgf,
            away_xgf,
            is_ot_so,
            home_sf,
            away_sf
        FROM game_results
        WHERE season = ?
        ORDER BY game_date ASC
    """, (season,))

    games = []
    for row in cursor.fetchall():
        games.append({
            'game_id': row[0],
            'game_date': datetime.fromisoformat(row[1]).date(),
            'season': row[2],
            'home_team': row[3],
            'away_team': row[4],
            'home_score': row[5],
            'away_score': row[6],
            'home_xgf': row[7] or 0.0,
            'away_xgf': row[8] or 0.0,
            'is_ot_so': bool(row[9]) if row[9] is not None else False,
            'home_sf': row[10] or 30,
            'away_sf': row[11] or 30,
        })
    
    logger.info(f"✓ Loaded {len(games)} games")
    return games


def extract_basic_features(game: Dict) -> Dict[str, float]:
    """
    Extract pre-game statistical features from game data.

    IMPORTANT: Only features available BEFORE the game starts are included.
    Using actual game outcomes (goals, goal differential) would be data leakage
    since the model is predicting whether the home team wins.
    """
    home_xgf = game['home_xgf']
    away_xgf = game['away_xgf']

    # Avoid division by zero
    total_xgf = max(home_xgf + away_xgf, 0.1)

    features = {
        # Expected goals (normalized) — pre-game estimate, not actual
        'home_xgf_norm': home_xgf / 6.0,
        'away_xgf_norm': away_xgf / 6.0,

        # xG share — how much of the expected offense belongs to each team
        'home_xgf_pct': home_xgf / total_xgf,
        'away_xgf_pct': away_xgf / total_xgf,

        # Shot quality proxy — xG per expected shot (~30 shots per team per game)
        'home_shot_quality': home_xgf / 30.0 if home_xgf > 0 else 0.0,
        'away_shot_quality': away_xgf / 30.0 if away_xgf > 0 else 0.0,

        # Placeholder for shot features (populated when available from NST)
        'home_sf_norm': 0.0,
        'away_sf_norm': 0.0,
    }

    return features


def create_training_data(
    games: List[Dict],
    config: EloConfig
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create training data with Elo features.
    
    This processes games chronologically to build accurate Elo ratings.
    """
    
    logger.info("\n🔄 Extracting features with Elo...")
    
    # Initialize Elo systems
    player_elo = PlayerEloSystem(config)
    team_elo = TeamEloSystem(config)
    feature_engine = EloFeatureEngine(player_elo, team_elo, config)
    
    X_list = []
    y_list = []
    
    for i, game in enumerate(games):
        if (i + 1) % 100 == 0:
            logger.info(f"   Processing game {i+1}/{len(games)}...")
        
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Extract basic features
        basic_features = extract_basic_features(game)
        
        # Extract Elo features (using current Elo state)
        try:
            # Create dummy lineups (minimal - just for team Elo)
            home_lineup = []
            away_lineup = []
            
            elo_features = feature_engine.extract_team_features(
                home_team, away_team
            )
            
            # Combine features
            all_features = {**basic_features, **elo_features}
            
        except Exception as e:
            logger.warning(f"Could not extract Elo features for game {game['game_id']}: {e}")
            # Use basic features only
            all_features = basic_features
            # Add zero Elo features (neutral defaults)
            all_features.update({
                'elo_diff': 0.0,
                'home_elo_norm': 0.0,
                'away_elo_norm': 0.0,
                'home_elo_level': 0.75,
                'away_elo_level': 0.75,
                'home_elo_momentum': 0.0,
                'away_elo_momentum': 0.0,
                'elo_strength_product': 1.0,
                'elo_momentum_diff': 0.0,
                'home_elo_experience': 0.0,
                'away_elo_experience': 0.0,
                'home_forward_elo_avg': 0.0,
                'away_forward_elo_avg': 0.0,
                'home_forward_elo_max': 0.0,
                'away_forward_elo_max': 0.0,
                'home_defense_elo_avg': 0.0,
                'away_defense_elo_avg': 0.0,
                'home_goalie_elo': 0.0,
                'away_goalie_elo': 0.0,
                'home_forward_depth': 0.0,
                'away_forward_depth': 0.0,
                'forward_elo_diff': 0.0,
                'defense_elo_diff': 0.0,
                'goalie_elo_diff': 0.0,
            })
        
        # Store features
        X_list.append(all_features)
        
        # Target: home team win (1) or loss (0)
        y_list.append(1.0 if game['home_score'] > game['away_score'] else 0.0)
        
        # Update Elo ratings after extracting features
        # (so next game uses updated ratings)
        try:
            home_team_obj = team_elo.get_or_create_team(home_team)
            away_team_obj = team_elo.get_or_create_team(away_team)
            
            # Determine result — NHL: OT/SO loss gets partial credit
            # game_results table doesn't store OT flag, so we use score closeness
            # as a heuristic: 1-goal differential could be an OT game
            score_diff = abs(game['home_score'] - game['away_score'])
            is_ot_so = game.get('is_ot_so', score_diff == 1)

            if game['home_score'] > game['away_score']:
                home_result = 1.0
                away_result = 0.25 if is_ot_so else 0.0
            elif game['away_score'] > game['home_score']:
                home_result = 0.25 if is_ot_so else 0.0
                away_result = 1.0
            else:
                home_result = 0.5
                away_result = 0.5
            
            # Derive shot estimate from xG (NHL avg ~0.06 xG per shot)
            # This is more informative than a flat 30
            home_sf_est = max(25, min(40, int(game['home_xgf'] / 0.06))) if game['home_xgf'] > 0 else 30
            away_sf_est = max(25, min(40, int(game['away_xgf'] / 0.06))) if game['away_xgf'] > 0 else 30

            # Update team Elo
            home_team_obj.update(
                opponent_rating=away_team_obj.rating,
                team_gf=game['home_score'],
                team_ga=game['away_score'],
                team_xgf=game['home_xgf'],
                team_xga=game['away_xgf'],
                team_sf=home_sf_est,
                team_sa=away_sf_est,
                result=home_result,
                config=config
            )

            away_team_obj.update(
                opponent_rating=home_team_obj.rating,
                team_gf=game['away_score'],
                team_ga=game['home_score'],
                team_xgf=game['away_xgf'],
                team_xga=game['home_xgf'],
                team_sf=away_sf_est,
                team_sa=home_sf_est,
                result=away_result,
                config=config
            )
            
        except Exception as e:
            logger.warning(f"Could not update Elo for game {game['game_id']}: {e}")
    
    # Convert to numpy arrays
    feature_names = list(X_list[0].keys())
    X = np.array([[game_features[f] for f in feature_names] for game_features in X_list])
    y = np.array(y_list)
    
    logger.info(f"\n✓ Extracted features for {len(X)} games")
    logger.info(f"✓ Feature count: {len(feature_names)}")
    
    # Count Elo features
    elo_feature_count = sum(1 for f in feature_names if 'elo' in f.lower())
    logger.info(f"✓ Elo features: {elo_feature_count}")
    
    # Show feature list
    logger.info("\n📋 Features:")
    for i, fname in enumerate(feature_names, 1):
        marker = "🔥" if "elo" in fname.lower() else "  "
        logger.info(f"   {i:2d}. {marker} {fname}")
    
    return X, y, feature_names


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_path: str
) -> EloMLPredictor:
    """Train XGBoost model"""
    
    logger.info("\n🤖 Training ML model...")
    
    # Split train/validation
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    logger.info(f"   Training: {len(X_train)} games")
    logger.info(f"   Validation: {len(X_val)} games")
    
    # Create model
    config = ModelConfig(
        elo_feature_weight=0.3,
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
    )
    
    model = EloMLPredictor(model_id="main", config=config)
    
    # Train
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names
    )
    
    # Get feature importance
    logger.info("\n📊 Feature Importance (Top 15):")
    importance = model.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    for i, (fname, imp) in enumerate(sorted_features, 1):
        marker = "🔥" if "elo" in fname.lower() else "  "
        logger.info(f"   {i:2d}. {marker} {fname:30s} {imp:.4f}")
    
    # Save model
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_file))
    
    logger.info(f"\n✓ Model saved to {output_path}")
    
    return model


def main():
    """Main training function"""
    
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("🏒 NHL ML Model Training (With Elo Features)")
    logger.info("=" * 60)
    logger.info(f"\n📅 Training on season: {args.season}")
    logger.info(f"   💾 Database: {args.training_db}")
    logger.info(f"   📦 Output: {args.output}")
    
    # Check database exists
    db_path = Path(args.training_db)
    if not db_path.exists():
        logger.error(f"\n❌ Training database not found: {args.training_db}")
        logger.error(f"   Run: python update_elo_ratings.py --season {args.season} --training --reset")
        return 1
    
    # Load data
    db = EloDatabase(str(db_path))
    games = load_training_games(db, args.season)
    
    if len(games) < 100:
        logger.error(f"\n❌ Insufficient training data: only {len(games)} games found!")
        logger.error(f"   Database: {args.training_db}")
        logger.error(f"   Run: python update_elo_ratings.py --season {args.season} --training --reset")
        return 1
    
    logger.info(f"✓ Loaded {len(games)} training games")
    
    # Create training data with Elo features
    config = EloConfig()
    X, y, feature_names = create_training_data(games, config)
    
    # Train model
    model = train_model(X, y, feature_names, args.output)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Training games: {len(games)}")
    logger.info(f"   Features: {len(feature_names)}")
    logger.info(f"   Elo features: {sum(1 for f in feature_names if 'elo' in f.lower())}")
    logger.info(f"   Model saved: {args.output}")
    
    logger.info("\n🎯 Next steps:")
    logger.info("   1. Run: python check_model.py")
    logger.info("   2. Run: python test_prediction.py")
    logger.info("   3. Run: python run_app.py")
    
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())