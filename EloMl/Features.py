"""
Feature engineering: Convert Elo ratings into ML-ready features.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import date
import logging

from .Ratings import PlayerEloSystem, TeamEloSystem, EloConfig

logger = logging.getLogger(__name__)

class EloFeatureEngine:
    """Converts Elo ratings into normalized features for ML"""
    
    def __init__(
        self,
        player_elo_system: PlayerEloSystem,
        team_elo_system: TeamEloSystem,
        config: Optional[EloConfig] = None
    ):
        self.player_elo = player_elo_system
        self.team_elo = team_elo_system
        self.config = config or EloConfig()
        
    def extract_team_features(
        self,
        home_abbr: str,
        away_abbr: str
    ) -> Dict[str, float]:
        """Extract Elo-based features for teams"""
        
        home_rating = self.team_elo.get_team_rating(home_abbr)
        away_rating = self.team_elo.get_team_rating(away_abbr)
        
        home_team = self.team_elo.get_or_create_team(home_abbr)
        away_team = self.team_elo.get_or_create_team(away_abbr)
        
        # Normalized ratings (mean=0, std=1 approximately)
        mean_rating = self.config.regression_mean
        std_rating = 200.0  # Approximate std dev of Elo ratings
        
        features = {
            # Raw Elo ratings (normalized)
            'home_elo_norm': (home_rating - mean_rating) / std_rating,
            'away_elo_norm': (away_rating - mean_rating) / std_rating,
            
            # Elo difference (primary signal)
            'elo_diff': (home_rating - away_rating) / std_rating,
            
            # Absolute Elo levels
            'home_elo_level': home_rating / 2000.0,  # Scale to ~0.75-1.2
            'away_elo_level': away_rating / 2000.0,
            
            # Recent form (momentum)
            'home_elo_momentum': home_team.get_recent_form_avg() / 20.0,
            'away_elo_momentum': away_team.get_recent_form_avg() / 20.0,
            
            # Experience (games played) - log scale
            'home_elo_experience': np.log1p(home_team.games_played) / 6.0,
            'away_elo_experience': np.log1p(away_team.games_played) / 6.0,
            
            # Combined signals
            'elo_strength_product': (home_rating * away_rating) / (mean_rating ** 2),
            'elo_momentum_diff': (home_team.get_recent_form_avg() - away_team.get_recent_form_avg()) / 20.0,
        }
        
        return features
    
    def extract_player_features(
        self,
        home_lineup: List[Dict],
        away_lineup: List[Dict]
    ) -> Dict[str, float]:
        """Extract Elo-based features from lineups"""
        
        # Aggregate player Elo for each team
        home_forward_elo = []
        home_defense_elo = []
        home_goalie_elo = []
        
        away_forward_elo = []
        away_defense_elo = []
        away_goalie_elo = []
        
        for player in home_lineup:
            name = player.get('name', '')
            pos = player.get('position', '').upper()
            rating = self.player_elo.get_player_rating(name)
            
            if pos in ('C', 'LW', 'RW'):
                home_forward_elo.append(rating)
            elif pos in ('D', 'LD', 'RD'):
                home_defense_elo.append(rating)
            elif pos == 'G':
                home_goalie_elo.append(rating)
        
        for player in away_lineup:
            name = player.get('name', '')
            pos = player.get('position', '').upper()
            rating = self.player_elo.get_player_rating(name)
            
            if pos in ('C', 'LW', 'RW'):
                away_forward_elo.append(rating)
            elif pos in ('D', 'LD', 'RD'):
                away_defense_elo.append(rating)
            elif pos == 'G':
                away_goalie_elo.append(rating)
        
        # Calculate aggregated features
        mean_rating = self.config.regression_mean
        std_rating = 200.0
        
        def safe_mean(arr):
            return np.mean(arr) if arr else mean_rating
        
        def safe_max(arr):
            return np.max(arr) if arr else mean_rating
        
        features = {
            # Forward strength
            'home_forward_elo_avg': (safe_mean(home_forward_elo) - mean_rating) / std_rating,
            'away_forward_elo_avg': (safe_mean(away_forward_elo) - mean_rating) / std_rating,
            'home_forward_elo_max': (safe_max(home_forward_elo) - mean_rating) / std_rating,
            'away_forward_elo_max': (safe_max(away_forward_elo) - mean_rating) / std_rating,
            
            # Defense strength
            'home_defense_elo_avg': (safe_mean(home_defense_elo) - mean_rating) / std_rating,
            'away_defense_elo_avg': (safe_mean(away_defense_elo) - mean_rating) / std_rating,
            
            # Goalie strength
            'home_goalie_elo': (safe_max(home_goalie_elo) - mean_rating) / std_rating,
            'away_goalie_elo': (safe_max(away_goalie_elo) - mean_rating) / std_rating,
            
            # Lineup depth (std dev indicates depth)
            'home_forward_depth': np.std(home_forward_elo) / std_rating if len(home_forward_elo) > 1 else 0.0,
            'away_forward_depth': np.std(away_forward_elo) / std_rating if len(away_forward_elo) > 1 else 0.0,
            
            # Matchup advantages
            'forward_elo_diff': (safe_mean(home_forward_elo) - safe_mean(away_forward_elo)) / std_rating,
            'defense_elo_diff': (safe_mean(home_defense_elo) - safe_mean(away_defense_elo)) / std_rating,
            'goalie_elo_diff': (safe_max(home_goalie_elo) - safe_max(away_goalie_elo)) / std_rating,
        }
        
        return features
    
    def combine_all_features(
        self,
        home_abbr: str,
        away_abbr: str,
        home_lineup: List[Dict],
        away_lineup: List[Dict],
        existing_features: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Combine all Elo features with existing features"""
        
        team_features = self.extract_team_features(home_abbr, away_abbr)
        player_features = self.extract_player_features(home_lineup, away_lineup)
        
        # Combine
        all_features = {**team_features, **player_features}
        
        # Add existing features if provided
        if existing_features:
            all_features.update(existing_features)
        
        return all_features
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for importance analysis"""
        return {
            'team_elo': [
                'home_elo_norm', 'away_elo_norm', 'elo_diff',
                'home_elo_level', 'away_elo_level'
            ],
            'momentum': [
                'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff'
            ],
            'player_strength': [
                'home_forward_elo_avg', 'away_forward_elo_avg',
                'home_defense_elo_avg', 'away_defense_elo_avg',
                'home_goalie_elo', 'away_goalie_elo'
            ],
            'matchup_advantages': [
                'forward_elo_diff', 'defense_elo_diff', 'goalie_elo_diff'
            ],
            'depth': [
                'home_forward_depth', 'away_forward_depth'
            ]
        }