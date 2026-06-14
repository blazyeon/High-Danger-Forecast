"""
Core Elo rating system for players and teams.
Tracks performance over time without situational factors.
Includes recent form tracking, adjustment capabilities, and season reset methods.

MODIFIED: Added reset methods to support fresh Elo each season.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class EloConfig:
    """Configuration for Elo rating system"""
    # Base K-factors (learning rates)
    k_factor_player: float = 32.0
    k_factor_team: float = 20.0
    k_factor_goalie: float = 28.0
    
    # Initial ratings
    initial_player_rating: float = 1500.0
    initial_team_rating: float = 1500.0
    
    # Regression to mean
    regression_factor: float = 0.02  # Pull toward mean each game
    regression_mean: float = 1500.0
    
    # Performance weights for players
    goal_weight: float = 1.0
    assist_weight: float = 0.6
    shot_weight: float = 0.15
    xg_weight: float = 0.8
    toi_weight: float = 0.3
    
    # Position multipliers
    forward_multiplier: float = 1.0
    defense_multiplier: float = 0.7
    goalie_multiplier: float = 1.2
    
    # Team performance weights
    xgf_pct_weight: float = 1.0
    gf_weight: float = 0.8
    shot_attempt_weight: float = 0.4
    pp_pk_weight: float = 0.6
    
    # Bounds
    min_rating: float = 800.0
    max_rating: float = 2400.0
    
    # Recent form tracking
    recent_form_window: int = 10  # Track last 10 games


@dataclass
class PlayerElo:
    """Individual player Elo rating tracker"""
    name: str
    position: str  # F, D, G
    rating: float = 1500.0
    games_played: int = 0
    rating_history: List[float] = field(default_factory=list)
    recent_form: List[float] = field(default_factory=list)  # Last N game deltas
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if not self.rating_history:
            self.rating_history = [self.rating]
    
    def add_result(self, delta: float):
        """Add a game result delta to recent form"""
        self.recent_form.append(delta)
        if len(self.recent_form) > 10:
            self.recent_form.pop(0)
    
    def update(
        self,
        goals: int,
        assists: int,
        shots: int,
        xg: float,
        toi_minutes: float,
        team_result: float,  # 1.0 = win, 0.5 = OT/SO loss, 0.0 = loss
        config: EloConfig
    ) -> float:
        """
        Update player Elo based on performance.
        Returns the rating change.
        """
        # Calculate performance score (normalized 0-1)
        perf_score = self._calculate_performance_score(
            goals, assists, shots, xg, toi_minutes, config
        )
        
        # Combine individual performance with team result
        # 70% individual performance, 30% team result
        combined_score = 0.7 * perf_score + 0.3 * team_result
        
        # Position-based K-factor adjustment
        k = self._get_k_factor(config)
        
        # Expected score (0.5 = neutral expectation)
        expected = 0.5
        
        # Rating change
        delta = k * (combined_score - expected)
        
        # Apply regression to mean
        regression = config.regression_factor * (config.regression_mean - self.rating)
        
        # Update rating with bounds
        old_rating = self.rating
        self.rating = np.clip(
            self.rating + delta + regression,
            config.min_rating,
            config.max_rating
        )
        
        # Track history and recent form
        self.games_played += 1
        self.rating_history.append(self.rating)
        self.add_result(delta)
        
        return delta
    
    def _calculate_performance_score(
        self,
        goals: int,
        assists: int,
        shots: int,
        xg: float,
        toi_minutes: float,
        config: EloConfig
    ) -> float:
        """Calculate normalized performance score (0-1)"""
        
        # Normalize each metric
        goal_score = min(1.0, goals / 3.0)  # 3 goals = perfect
        assist_score = min(1.0, assists / 3.0)
        shot_score = min(1.0, shots / 8.0)  # 8 shots = perfect
        xg_score = min(1.0, xg / 2.0)  # 2.0 xG = perfect
        toi_score = min(1.0, toi_minutes / 22.0)  # 22 min = perfect
        
        # Weighted combination
        score = (
            config.goal_weight * goal_score +
            config.assist_weight * assist_score +
            config.shot_weight * shot_score +
            config.xg_weight * xg_score +
            config.toi_weight * toi_score
        )
        
        # Normalize to 0-1
        total_weight = (
            config.goal_weight + config.assist_weight + 
            config.shot_weight + config.xg_weight + config.toi_weight
        )
        
        return score / total_weight
    
    def _get_k_factor(self, config: EloConfig) -> float:
        """Get position-adjusted K-factor"""
        base_k = config.k_factor_player
        
        if self.position in ('C', 'LW', 'RW', 'F'):
            return base_k * config.forward_multiplier
        elif self.position in ('D', 'LD', 'RD'):
            return base_k * config.defense_multiplier
        elif self.position == 'G':
            return config.k_factor_goalie * config.goalie_multiplier
        else:
            return base_k
    
    def get_recent_form_avg(self) -> float:
        """Get average rating change over recent games"""
        if not self.recent_form:
            return 0.0
        return float(np.mean(self.recent_form))
    
    def get_volatility(self) -> float:
        """Get rating volatility (std dev of recent changes)"""
        if len(self.recent_form) < 3:
            return 0.0
        return float(np.std(self.recent_form))
    
    def reset(self, initial_rating: float = 1500.0):
        """Reset player to initial rating (new season)"""
        self.rating = initial_rating
        self.games_played = 0
        self.rating_history = [self.rating]
        self.recent_form = []


@dataclass
class TeamElo:
    """Team Elo rating tracker"""
    team: str
    rating: float = 1500.0
    games_played: int = 0
    rating_history: List[float] = field(default_factory=list)
    recent_form: List[float] = field(default_factory=list)  # Last N game deltas
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if not self.rating_history:
            self.rating_history = [self.rating]
    
    def add_result(self, delta: float):
        """Add a game result delta to recent form"""
        self.recent_form.append(delta)
        if len(self.recent_form) > 10:
            self.recent_form.pop(0)
    
    def update(
        self,
        opponent_rating: float,
        team_gf: int,
        team_ga: int,
        team_xgf: float,
        team_xga: float,
        team_sf: int,
        team_sa: int,
        result: float,  # 1.0 = win, 0.5 = OT/SO, 0.0 = loss
        config: EloConfig
    ) -> float:
        """
        Update team Elo based on game result and performance.
        Returns rating change.
        """
        # Expected score based on rating difference
        rating_diff = self.rating - opponent_rating
        expected = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
        
        # Performance adjustment based on underlying stats
        perf_factor = self._calculate_performance_factor(
            team_gf, team_ga, team_xgf, team_xga, team_sf, team_sa, config
        )
        
        # Adjusted result (blend actual result with performance)
        adjusted_result = 0.6 * result + 0.4 * perf_factor
        
        # Rating change
        delta = config.k_factor_team * (adjusted_result - expected)
        
        # Regression to mean
        regression = config.regression_factor * (config.regression_mean - self.rating)
        
        # Update with bounds
        old_rating = self.rating
        self.rating = np.clip(
            self.rating + delta + regression,
            config.min_rating,
            config.max_rating
        )
        
        # Track history and recent form
        self.games_played += 1
        self.rating_history.append(self.rating)
        self.add_result(delta)
        
        return delta
    
    def _calculate_performance_factor(
        self,
        gf: int, ga: int,
        xgf: float, xga: float,
        sf: int, sa: int,
        config: EloConfig
    ) -> float:
        """Calculate performance factor (0-1) from underlying stats"""
        
        # xGF% component
        xgf_pct = xgf / (xgf + xga) if (xgf + xga) > 0 else 0.5
        
        # GF% component
        gf_pct = gf / (gf + ga) if (gf + ga) > 0 else 0.5
        
        # SF% component
        sf_pct = sf / (sf + sa) if (sf + sa) > 0 else 0.5
        
        # Weighted combination
        perf = (
            config.xgf_pct_weight * xgf_pct +
            config.gf_weight * gf_pct +
            config.shot_attempt_weight * sf_pct
        )
        
        total_weight = config.xgf_pct_weight + config.gf_weight + config.shot_attempt_weight
        
        return perf / total_weight
    
    def get_recent_form_avg(self) -> float:
        """Get average rating change over recent games"""
        if not self.recent_form:
            return 0.0
        return float(np.mean(self.recent_form))
    
    def get_recent_form_rating(self, days: int = 14) -> float:
        """
        Get rating adjusted for recent form.
        
        Args:
            days: Number of days to consider (14 days ≈ 5 games)
        
        Returns:
            Adjusted rating based on recent performance
        """
        if not self.recent_form:
            return self.rating
        
        # Determine how many games to look at based on days
        # Average team plays ~3.5 games per week (0.5 games per day)
        games_to_consider = max(3, min(len(self.recent_form), int(days * 0.5)))
        
        recent_deltas = self.recent_form[-games_to_consider:]
        
        # Calculate cumulative effect of recent form
        cumulative_adjustment = sum(recent_deltas)
        
        # Cap adjustment at ±100 Elo to prevent extreme swings
        cumulative_adjustment = max(-100, min(100, cumulative_adjustment))
        
        # Apply 50% weight to recent form (don't fully override base rating)
        adjusted_rating = self.rating + cumulative_adjustment * 0.5
        
        logger.debug(
            f"{self.team} recent form: {cumulative_adjustment:+.1f} "
            f"(last {games_to_consider} games) → "
            f"Rating: {self.rating:.0f} → {adjusted_rating:.0f}"
        )
        
        return adjusted_rating
    
    def get_momentum(self) -> float:
        """
        Get momentum indicator (-1.0 to 1.0).
        Positive = hot streak, Negative = cold streak
        """
        if len(self.recent_form) < 3:
            return 0.0
        
        recent = self.recent_form[-3:]
        avg_delta = float(np.mean(recent))
        
        # Map average delta to -1 to +1 scale
        # ±20 Elo per game = maximum momentum
        return np.clip(avg_delta / 20.0, -1.0, 1.0)
    
    def reset(self, initial_rating: float = 1500.0):
        """Reset team to initial rating (new season)"""
        self.rating = initial_rating
        self.games_played = 0
        self.rating_history = [self.rating]
        self.recent_form = []


class PlayerEloSystem:
    """Manages Elo ratings for all players"""
    
    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        self.players: Dict[str, PlayerElo] = {}
        logger.info("Initialized Player Elo System")
    
    def get_or_create_player(
        self, 
        player_name: str, 
        position: str
    ) -> PlayerElo:
        """Get existing player or create new one"""
        key = player_name.lower().strip()
        
        if key not in self.players:
            self.players[key] = PlayerElo(
                name=player_name, 
                position=position, 
                rating=self.config.initial_player_rating
            )
            logger.debug(f"Created new player: {player_name} ({position})")
        
        return self.players[key]
    
    def get_player_rating(self, player_name: str) -> float:
        """Get current rating for a player"""
        key = player_name.lower().strip()
        
        if key in self.players:
            return self.players[key].rating
        
        return self.config.initial_player_rating
    
    def get_top_players(self, n: int = 50) -> List[Tuple[str, float]]:
        """Get top N players by rating"""
        sorted_players = sorted(
            [(p.name, p.rating) for p in self.players.values()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_players[:n]
    
    def get_player_by_name(self, player_name: str) -> Optional[PlayerElo]:
        """Get player object by name"""
        key = player_name.lower().strip()
        return self.players.get(key)
    
    def reset_all_ratings(self):
        """Reset all players to initial rating (new season)"""
        initial = self.config.initial_player_rating
        for player in self.players.values():
            player.reset(initial)
        
        logger.info(f"🔄 Reset {len(self.players)} players to {initial}")
    
    def reset_to_season(self, season: str):
        """Clear all players (will rebuild from season data)"""
        count = len(self.players)
        self.players.clear()
        logger.info(f"🔄 Cleared {count} players - will rebuild from {season} data")


class TeamEloSystem:
    """Manages Elo ratings for all teams"""
    
    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        self.teams: Dict[str, TeamElo] = {}
        logger.info("Initialized Team Elo System")
    
    def get_or_create_team(self, team_abbr: str) -> TeamElo:
        """Get existing team or create new one"""
        key = team_abbr.upper().strip()
        
        if key not in self.teams:
            self.teams[key] = TeamElo(
                team=team_abbr.upper(),
                rating=self.config.initial_team_rating
            )
            logger.debug(f"Created new team: {key}")
        
        return self.teams[key]
    
    def get_team_rating(self, team_abbr: str) -> float:
        """Get current rating for a team"""
        key = team_abbr.upper().strip()
        
        if key in self.teams:
            return self.teams[key].rating
        
        return self.config.initial_team_rating
    
    def get_team_rankings(self) -> List[Tuple[str, float]]:
        """Get all teams ranked by rating"""
        sorted_teams = sorted(
            [(t.team, t.rating) for t in self.teams.values()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_teams
    
    def get_team_by_abbr(self, team_abbr: str) -> Optional[TeamElo]:
        """Get team object by abbreviation"""
        key = team_abbr.upper().strip()
        return self.teams.get(key)
    
    def reset_all_ratings(self):
        """Reset all teams to initial rating (new season)"""
        initial = self.config.initial_team_rating
        for team in self.teams.values():
            team.reset(initial)
        
        logger.info(f"🔄 Reset {len(self.teams)} teams to {initial}")
    
    def reset_to_season(self, season: str):
        """Clear all teams (will rebuild from season data)"""
        count = len(self.teams)
        self.teams.clear()
        logger.info(f"🔄 Cleared {count} teams - will rebuild from {season} data")