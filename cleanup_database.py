"""
Clean up Elo database - remove duplicates and invalid entries.
"""
import logging
from EloMl.Database import EloDatabase
from EloMl.Ratings import PlayerEloSystem, TeamEloSystem, EloConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_database():
    """Clean up database and show statistics"""
    
    logger.info("Opening database...")
    db = EloDatabase("elo_ratings.db")
    cursor = db.conn.cursor()
    
    # Valid NHL teams (32 teams for 2024-25 season)
    VALID_TEAMS = {
        'ANA', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL', 
        'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NJD',
        'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SEA', 'SJS',
        'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WPG', 'WSH', 'UTA'
    }
    
    logger.info("\n📊 Current Database State:")
    
    # Check teams
    cursor.execute("SELECT COUNT(DISTINCT team_abbr) FROM team_elo")
    n_teams = cursor.fetchone()[0]
    logger.info(f"  Unique teams: {n_teams}")
    
    cursor.execute("SELECT DISTINCT team_abbr FROM team_elo ORDER BY team_abbr")
    all_teams = [row[0] for row in cursor.fetchall()]
    logger.info(f"  Teams: {', '.join(all_teams)}")
    
    # Check for invalid teams
    invalid_teams = [t for t in all_teams if t not in VALID_TEAMS]
    if invalid_teams:
        logger.warning(f"  ⚠️  Invalid teams found: {', '.join(invalid_teams)}")
    
    # Check players
    cursor.execute("SELECT COUNT(DISTINCT player_name) FROM player_elo")
    n_players = cursor.fetchone()[0]
    logger.info(f"  Unique players: {n_players}")
    
    cursor.execute("""
        SELECT COUNT(DISTINCT player_name) 
        FROM player_elo 
        WHERE games_played >= 5
    """)
    active_players = cursor.fetchone()[0]
    logger.info(f"  Active players (5+ games): {active_players}")
    
    # Check for players with < 2 games
    cursor.execute("""
        SELECT COUNT(DISTINCT player_name)
        FROM (
            SELECT player_name, MAX(games_played) as max_gp
            FROM player_elo
            GROUP BY player_name
            HAVING max_gp < 2
        )
    """)
    inactive_players = cursor.fetchone()[0]
    logger.info(f"  Inactive players (<2 games): {inactive_players}")
    
    # Show top teams
    logger.info("\n🏆 Top 10 Teams:")
    cursor.execute("""
        SELECT team_abbr, rating, games_played
        FROM team_elo
        WHERE id IN (
            SELECT MAX(id) FROM team_elo GROUP BY team_abbr
        )
        ORDER BY rating DESC
        LIMIT 10
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        logger.info(f"  {i}. {row[0]}: {row[1]:.0f} ({row[2]} games)")
    
    # Show top players
    logger.info("\n⭐ Top 10 Players:")
    cursor.execute("""
        SELECT player_name, rating, games_played
        FROM player_elo
        WHERE id IN (
            SELECT MAX(id) FROM player_elo GROUP BY player_name
        )
        ORDER BY rating DESC
        LIMIT 10
    """)
    for i, row in enumerate(cursor.fetchall(), 1):
        logger.info(f"  {i}. {row[0]}: {row[1]:.0f} ({row[2]} games)")
    
    # Cleanup options
    logger.info("\n🧹 Cleanup Options:")
    logger.info("  1. Remove players with < 2 games")
    logger.info("  2. Remove invalid team entries")
    logger.info("  3. Keep all data (no cleanup)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        logger.info("\nRemoving players with < 2 games...")
        cursor.execute("""
            DELETE FROM player_elo
            WHERE player_name IN (
                SELECT player_name FROM (
                    SELECT player_name, MAX(games_played) as max_gp
                    FROM player_elo
                    GROUP BY player_name
                    HAVING max_gp < 2
                )
            )
        """)
        deleted = cursor.rowcount
        db.conn.commit()
        logger.info(f"✓ Deleted {deleted} entries for {inactive_players} inactive players")
        
    elif choice == "2":
        if invalid_teams:
            logger.info(f"\nRemoving invalid teams: {', '.join(invalid_teams)}")
            placeholders = ','.join('?' * len(invalid_teams))
            cursor.execute(
                f"DELETE FROM team_elo WHERE team_abbr IN ({placeholders})",
                invalid_teams
            )
            deleted = cursor.rowcount
            db.conn.commit()
            logger.info(f"✓ Deleted {deleted} entries for invalid teams")
        else:
            logger.info("No invalid teams to remove")
    
    elif choice == "3":
        logger.info("No cleanup performed")
    else:
        logger.error("Invalid choice")
        return
    
    # Show final stats
    logger.info("\n📊 Final Database State:")
    cursor.execute("SELECT COUNT(DISTINCT team_abbr) FROM team_elo")
    n_teams = cursor.fetchone()[0]
    logger.info(f"  Teams: {n_teams}")
    
    cursor.execute("SELECT COUNT(DISTINCT player_name) FROM player_elo")
    n_players = cursor.fetchone()[0]
    logger.info(f"  Players: {n_players}")
    
    logger.info("\n✅ Cleanup complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Start the app: python run_app.py")
    logger.info("  2. Check sidebar - should show correct counts")
    
    db.close()


if __name__ == "__main__":
    cleanup_database()