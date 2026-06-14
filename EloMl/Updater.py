"""
Elo Game Updater: Updates player and team Elo ratings from game results.
Also saves game results to database for ML training.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from datetime import date
import logging

from .Ratings import PlayerEloSystem, TeamEloSystem, EloConfig
from .Database import EloDatabase

logger = logging.getLogger(__name__)


class EloGameUpdater:
    """
    Updates Elo ratings for players and teams based on game results.
    Also records game data to database for ML training.
    """
    
    def __init__(
        self,
        player_elo: PlayerEloSystem,
        team_elo: TeamEloSystem,
        database: EloDatabase,
        config: Optional[EloConfig] = None
    ):
        self.player_elo = player_elo
        self.team_elo = team_elo
        self.db = database
        self.config = config or EloConfig()
        
        logger.info("Initialized EloGameUpdater")
    
    def update_from_game(
        self,
        game_date: date,
        season: str,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        home_stats: Dict,
        away_stats: Dict,
        home_players: List[Dict],
        away_players: List[Dict],
        is_ot_so: bool = False
    ):
        """
        Update Elo ratings from a completed game.

        Args:
            game_date: Date of the game
            season: Season string (e.g., "20252026")
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_score: Home team final score
            away_score: Away team final score
            home_stats: Home team stats dict (xGF, SF, etc.)
            away_stats: Away team stats dict
            home_players: List of home player dicts
            away_players: List of away player dicts
            is_ot_so: Whether the game went to overtime or shootout
        """

        # Determine game result
        # NHL rules: regulation win = full points, OT/SO loss = partial credit
        if home_score > away_score:
            home_result = 1.0
            away_result = 0.25 if is_ot_so else 0.0  # OT/SO loser gets partial credit
        elif away_score > home_score:
            home_result = 0.25 if is_ot_so else 0.0  # OT/SO loser gets partial credit
            away_result = 1.0
        else:
            # True tie (only possible in older seasons; treated as draw)
            home_result = 0.5
            away_result = 0.5
        
        # Get teams
        home_team_obj = self.team_elo.get_or_create_team(home_team)
        away_team_obj = self.team_elo.get_or_create_team(away_team)
        
        # Update team Elo
        home_rating_before = home_team_obj.rating
        away_rating_before = away_team_obj.rating
        
        home_delta = home_team_obj.update(
            opponent_rating=away_team_obj.rating,
            team_gf=home_score,
            team_ga=away_score,
            team_xgf=float(home_stats.get('xGF', 0.0)),
            team_xga=float(away_stats.get('xGF', 0.0)),
            team_sf=int(home_stats.get('SF', 0)),
            team_sa=int(away_stats.get('SF', 0)),
            result=home_result,
            config=self.config
        )
        
        away_delta = away_team_obj.update(
            opponent_rating=home_team_obj.rating,
            team_gf=away_score,
            team_ga=home_score,
            team_xgf=float(away_stats.get('xGF', 0.0)),
            team_xga=float(home_stats.get('xGF', 0.0)),
            team_sf=int(away_stats.get('SF', 0)),
            team_sa=int(home_stats.get('SF', 0)),
            result=away_result,
            config=self.config
        )
        
        # Save team Elo to database
        self.db.save_team_elo(
            team_abbr=home_team,
            rating=home_team_obj.rating,
            games_played=home_team_obj.games_played,
            game_date=game_date,
            season=season,
            rating_change=home_delta,
            recent_form=home_team_obj.recent_form
        )
        
        self.db.save_team_elo(
            team_abbr=away_team,
            rating=away_team_obj.rating,
            games_played=away_team_obj.games_played,
            game_date=game_date,
            season=season,
            rating_change=away_delta,
            recent_form=away_team_obj.recent_form
        )
        
        logger.debug(
            f"{home_team} vs {away_team} ({home_score}-{away_score}): "
            f"{home_team} {home_rating_before:.0f} -> {home_team_obj.rating:.0f} ({home_delta:+.1f}), "
            f"{away_team} {away_rating_before:.0f} -> {away_team_obj.rating:.0f} ({away_delta:+.1f})"
        )
        
        # Update player Elo
        for player_data in home_players:
            self._update_player(
                player_data=player_data,
                team_abbr=home_team,
                team_result=home_result,
                game_date=game_date,
                season=season
            )
        
        for player_data in away_players:
            self._update_player(
                player_data=player_data,
                team_abbr=away_team,
                team_result=away_result,
                game_date=game_date,
                season=season
            )
        
        # Save game result to database for ML training
        self._save_game_result(
            game_date=game_date,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            home_stats=home_stats,
            away_stats=away_stats
        )
    
    def _update_player(
        self,
        player_data: Dict,
        team_abbr: str,
        team_result: float,
        game_date: date,
        season: str
    ):
        """Update individual player Elo rating"""
        
        try:
            name = player_data.get('name', '')
            position = player_data.get('position', 'F')
            
            if not name:
                return
            
            # Get player stats
            goals = int(player_data.get('goals', 0))
            assists = int(player_data.get('assists', 0))
            shots = int(player_data.get('shots', 0))
            xg = float(player_data.get('xG', 0.0))
            toi_minutes = float(player_data.get('toi_minutes', 0.0))
            
            # Skip if player didn't play enough
            if toi_minutes < 0.5:
                return
            
            # Get or create player
            player = self.player_elo.get_or_create_player(name, position)
            
            # Update rating
            rating_before = player.rating
            
            delta = player.update(
                goals=goals,
                assists=assists,
                shots=shots,
                xg=xg,
                toi_minutes=toi_minutes,
                team_result=team_result,
                config=self.config
            )
            
            # Save to database
            self.db.save_player_elo(
                player_name=name,
                position=position,
                team_abbr=team_abbr,
                rating=player.rating,
                games_played=player.games_played,
                game_date=game_date,
                season=season,
                rating_change=delta,
                recent_form=player.recent_form
            )
            
            logger.debug(
                f"  {name} ({position}): {rating_before:.0f} -> {player.rating:.0f} ({delta:+.1f}) "
                f"[{goals}G {assists}A {shots}S {toi_minutes:.1f}TOI]"
            )
            
        except Exception as e:
            logger.error(f"Error updating player {player_data.get('name', 'unknown')}: {e}")
    
    def _save_game_result(
        self,
        game_date: date,
        season: str,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        home_stats: Dict,
        away_stats: Dict,
        is_ot_so: bool = False
    ):
        """
        Save game result to database for ML training.

        This creates a record that train_model.py can use to train on historical games
        without using historical Elo ratings.
        """

        try:
            # Create unique game ID
            game_id = f"{season}_{game_date.isoformat()}_{away_team}@{home_team}"

            cursor = self.db.conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO game_results
                (game_id, game_date, season, home_team, away_team,
                 home_score, away_score, home_xgf, away_xgf, is_ot_so, home_sf, away_sf, elo_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                game_id,
                game_date.isoformat(),
                season,
                home_team,
                away_team,
                home_score,
                away_score,
                float(home_stats.get('xGF', 0.0)),
                float(away_stats.get('xGF', 0.0)),
                1 if is_ot_so else 0,
                int(home_stats.get('SF', 30)),
                int(away_stats.get('SF', 30)),
            ))

            self.db.conn.commit()

            logger.debug(f"Saved game result: {game_id}")
            
        except Exception as e:
            logger.error(f"Error saving game result: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics of current Elo state"""
        
        team_ratings = self.team_elo.get_team_rankings()
        player_ratings = self.player_elo.get_top_players(10)
        
        return {
            'n_teams': len(self.team_elo.teams),
            'n_players': len(self.player_elo.players),
            'top_team': team_ratings[0] if team_ratings else None,
            'top_player': player_ratings[0] if player_ratings else None,
            'avg_team_rating': sum(r for _, r in team_ratings) / len(team_ratings) if team_ratings else 1500.0
        }


__all__ = ['EloGameUpdater']