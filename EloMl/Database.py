"""
Database storage for Elo ratings with SQLite backend.
Supports context manager protocol for safe connection handling.
"""
from __future__ import annotations
import sqlite3
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
import logging
import atexit

logger = logging.getLogger(__name__)


class EloDatabase:
    """SQLite database for Elo ratings"""

    def __init__(self, db_path: str = "elo_ratings.db"):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._closed = False
        self._initialize_db()
        # Register cleanup on process exit as a safety net
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Ensure connection is closed on garbage collection"""
        self.close()

    def _ensure_connection(self):
        """Reconnect if the connection was closed or is unusable"""
        if self._closed or self.conn is None:
            self._closed = False
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        try:
            # Quick liveness check
            self.conn.execute("SELECT 1")
        except sqlite3.Error:
            logger.warning("Reconnecting to Elo database (stale connection)")
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

    def _initialize_db(self):
        """Create database tables if they don't exist"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        # Player Elo history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_elo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL,
                team_abbr TEXT,
                rating REAL NOT NULL,
                games_played INTEGER DEFAULT 0,
                date TEXT NOT NULL,
                season TEXT NOT NULL,
                rating_change REAL DEFAULT 0.0,
                recent_form TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Team Elo history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_elo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_abbr TEXT NOT NULL,
                rating REAL NOT NULL,
                games_played INTEGER DEFAULT 0,
                date TEXT NOT NULL,
                season TEXT NOT NULL,
                rating_change REAL DEFAULT 0.0,
                recent_form TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Game results table for updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT UNIQUE NOT NULL,
                game_date TEXT NOT NULL,
                season TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                home_xgf REAL,
                away_xgf REAL,
                is_ot_so INTEGER DEFAULT 0,
                home_sf INTEGER,
                away_sf INTEGER,
                elo_updated INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ML model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                game_date TEXT NOT NULL,
                prediction_home_win REAL,
                actual_home_win INTEGER,
                brier_score REAL,
                log_loss REAL,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_name_date
            ON player_elo(player_name, date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_team_date
            ON team_elo(team_abbr, date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_game_date
            ON game_results(game_date)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_game_season
            ON game_results(season)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_perf_date
            ON model_performance(model_id, game_date)
        """)

        self.conn.commit()
        logger.info(f"Initialized Elo database at {self.db_path}")

    def save_player_elo(
        self,
        player_name: str,
        position: str,
        team_abbr: str,
        rating: float,
        games_played: int,
        game_date: date,
        season: str,
        rating_change: float = 0.0,
        recent_form: Optional[List[float]] = None
    ):
        """Save player Elo rating to database"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO player_elo
            (player_name, position, team_abbr, rating, games_played, date, season, rating_change, recent_form)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            player_name,
            position,
            team_abbr,
            rating,
            games_played,
            game_date.isoformat(),
            season,
            rating_change,
            json.dumps(recent_form) if recent_form else None
        ))

        self.conn.commit()

    def save_team_elo(
        self,
        team_abbr: str,
        rating: float,
        games_played: int,
        game_date: date,
        season: str,
        rating_change: float = 0.0,
        recent_form: Optional[List[float]] = None
    ):
        """Save team Elo rating to database"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO team_elo
            (team_abbr, rating, games_played, date, season, rating_change, recent_form)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            team_abbr,
            rating,
            games_played,
            game_date.isoformat(),
            season,
            rating_change,
            json.dumps(recent_form) if recent_form else None
        ))

        self.conn.commit()

    def get_latest_player_elo(
        self,
        player_name: str,
        before_date: Optional[date] = None
    ) -> Optional[float]:
        """Get most recent Elo rating for a player"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        if before_date:
            cursor.execute("""
                SELECT rating FROM player_elo
                WHERE player_name = ? AND date < ?
                ORDER BY date DESC LIMIT 1
            """, (player_name, before_date.isoformat()))
        else:
            cursor.execute("""
                SELECT rating FROM player_elo
                WHERE player_name = ?
                ORDER BY date DESC LIMIT 1
            """, (player_name,))

        row = cursor.fetchone()
        return row['rating'] if row else None

    def get_latest_team_elo(
        self,
        team_abbr: str,
        before_date: Optional[date] = None
    ) -> Optional[float]:
        """Get most recent Elo rating for a team"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        if before_date:
            cursor.execute("""
                SELECT rating FROM team_elo
                WHERE team_abbr = ? AND date < ?
                ORDER BY date DESC LIMIT 1
            """, (team_abbr, before_date.isoformat()))
        else:
            cursor.execute("""
                SELECT rating FROM team_elo
                WHERE team_abbr = ?
                ORDER BY date DESC LIMIT 1
            """, (team_abbr,))

        row = cursor.fetchone()
        return row['rating'] if row else None

    def get_player_rating_history(
        self,
        player_name: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get rating history for a player"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT date, rating, rating_change, games_played
            FROM player_elo
            WHERE player_name = ?
            ORDER BY date DESC LIMIT ?
        """, (player_name, limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_team_rating_history(
        self,
        team_abbr: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get rating history for a team"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT date, rating, rating_change, games_played
            FROM team_elo
            WHERE team_abbr = ?
            ORDER BY date DESC LIMIT ?
        """, (team_abbr, limit))

        return [dict(row) for row in cursor.fetchall()]

    def save_model_performance(
        self,
        model_id: str,
        model_version: str,
        game_date: date,
        prediction: float,
        actual: int,
        brier: float,
        logloss: float,
        features: Dict
    ):
        """Save ML model performance metrics"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO model_performance
            (model_id, model_version, game_date, prediction_home_win,
             actual_home_win, brier_score, log_loss, features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            model_version,
            game_date.isoformat(),
            prediction,
            actual,
            brier,
            logloss,
            json.dumps(features)
        ))

        self.conn.commit()

    def get_model_performance_summary(
        self,
        model_id: str,
        days: int = 30
    ) -> Dict:
        """Get performance summary for a model"""
        self._ensure_connection()
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as n_predictions,
                AVG(brier_score) as avg_brier,
                AVG(log_loss) as avg_logloss,
                AVG(CASE WHEN
                    (prediction_home_win >= 0.5 AND actual_home_win = 1) OR
                    (prediction_home_win < 0.5 AND actual_home_win = 0)
                    THEN 1 ELSE 0 END) as accuracy
            FROM model_performance
            WHERE model_id = ? AND date(game_date) >= date('now', '-' || ? || ' days')
        """, (model_id, days))

        row = cursor.fetchone()
        return dict(row) if row else {}

    def close(self):
        """Close database connection safely"""
        if self.conn and not self._closed:
            try:
                self.conn.close()
            except Exception:
                pass
            self._closed = True
            self.conn = None