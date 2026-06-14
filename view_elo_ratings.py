"""
Quick viewer for Elo ratings in the database.
Updated to work with new database structure.
"""
from EloMl.Database import EloDatabase
import pandas as pd
from datetime import datetime, timedelta

def view_elo_ratings():
    print("\n" + "="*60)
    print("🏒 NHL ELO RATINGS VIEWER")
    print("="*60 + "\n")
    
    db = EloDatabase("elo_ratings.db")
    cursor = db.conn.cursor()
    
    # Get current season
    today = datetime.now().date()
    year = today.year
    month = today.month
    if month >= 10:
        current_season = f"{year}{year+1}"
    else:
        current_season = f"{year-1}{year}"
    
    print(f"📅 Current Season: {current_season}\n")
    
    # ============================================
    # TEAM RATINGS
    # ============================================
    print("🏆 TEAM RATINGS (Current Season)\n")

    cursor.execute("""
        SELECT team_abbr, rating, games_played, date
        FROM (
            SELECT team_abbr, rating, games_played, date,
                   ROW_NUMBER() OVER (PARTITION BY team_abbr ORDER BY date DESC, id DESC) as rn
            FROM team_elo
            WHERE season = ?
        )
        WHERE rn = 1
        ORDER BY rating DESC
    """, (current_season,))
    
    teams_data = []
    for row in cursor.fetchall():
        teams_data.append({
            'Rank': len(teams_data) + 1,
            'Team': row[0],
            'Elo': f"{row[1]:.0f}",
            'Games': row[2],
            'Last Updated': row[3]
        })
    
    if teams_data:
        df_teams = pd.DataFrame(teams_data)
        print(df_teams.to_string(index=False))
        print(f"\nTotal Teams: {len(teams_data)}")
    else:
        print("❌ No team data found for current season")
        print(f"   Run: python update_elo_ratings.py --current-season --reset --initial-only\n")
    
    # ============================================
    # TOP FORWARDS
    # ============================================
    print("\n" + "="*60)
    print("⭐ TOP 15 FORWARDS BY ELO\n")
    
    cursor.execute("""
        SELECT player_name, team_abbr, rating, games_played, date
        FROM (
            SELECT player_name, team_abbr, rating, games_played, date,
                   ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC, id DESC) as rn
            FROM player_elo
            WHERE season = ? AND position = 'F'
        )
        WHERE rn = 1
        ORDER BY rating DESC
        LIMIT 15
    """, (current_season,))
    
    forwards_data = []
    for row in cursor.fetchall():
        forwards_data.append({
            'Rank': len(forwards_data) + 1,
            'Player': row[0][:23],  # Truncate long names
            'Team': row[1],
            'Elo': f"{row[2]:.0f}",
            'Games': row[3],
            'Last Updated': row[4]
        })
    
    if forwards_data:
        df_forwards = pd.DataFrame(forwards_data)
        print(df_forwards.to_string(index=False))
    else:
        print("❌ No forward data found")
    
    # ============================================
    # TOP DEFENSEMEN
    # ============================================
    print("\n" + "="*60)
    print("🛡️  TOP 15 DEFENSEMEN BY ELO\n")
    
    cursor.execute("""
        SELECT player_name, team_abbr, rating, games_played, date
        FROM (
            SELECT player_name, team_abbr, rating, games_played, date,
                   ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC, id DESC) as rn
            FROM player_elo
            WHERE season = ? AND position = 'D'
        )
        WHERE rn = 1
        ORDER BY rating DESC
        LIMIT 15
    """, (current_season,))
    
    defense_data = []
    for row in cursor.fetchall():
        defense_data.append({
            'Rank': len(defense_data) + 1,
            'Player': row[0][:23],
            'Team': row[1],
            'Elo': f"{row[2]:.0f}",
            'Games': row[3],
            'Last Updated': row[4]
        })
    
    if defense_data:
        df_defense = pd.DataFrame(defense_data)
        print(df_defense.to_string(index=False))
    else:
        print("❌ No defenseman data found")
    
    # ============================================
    # TOP 25 OVERALL
    # ============================================
    print("\n" + "="*60)
    print("🏆 TOP 25 PLAYERS OVERALL\n")
    
    cursor.execute("""
        SELECT player_name, position, team_abbr, rating, games_played, date
        FROM (
            SELECT player_name, position, team_abbr, rating, games_played, date,
                   ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC, id DESC) as rn
            FROM player_elo
            WHERE season = ?
        )
        WHERE rn = 1
        ORDER BY rating DESC
        LIMIT 25
    """, (current_season,))
    
    overall_data = []
    for row in cursor.fetchall():
        overall_data.append({
            'Rank': len(overall_data) + 1,
            'Player': row[0][:23],
            'Pos': row[1],
            'Team': row[2],
            'Elo': f"{row[3]:.0f}",
            'GP': row[4],
        })
    
    if overall_data:
        df_overall = pd.DataFrame(overall_data)
        print(df_overall.to_string(index=False))
    else:
        print("❌ No player data found")
    
    # ============================================
    # ELO DISTRIBUTION
    # ============================================
    print("\n" + "="*60)
    print("📊 PLAYER ELO DISTRIBUTION\n")
    
    cursor.execute("""
        SELECT rating
        FROM (
            SELECT rating,
                   ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC, id DESC) as rn
            FROM player_elo
            WHERE season = ?
        )
        WHERE rn = 1
    """, (current_season,))
    
    ratings = [row[0] for row in cursor.fetchall()]
    
    if ratings:
        import numpy as np
        
        print(f"Total Players: {len(ratings)}")
        print(f"Average Elo:   {np.mean(ratings):.0f}")
        print(f"Median Elo:    {np.median(ratings):.0f}")
        print(f"Std Dev:       {np.std(ratings):.0f}")
        print(f"Min Elo:       {min(ratings):.0f}")
        print(f"Max Elo:       {max(ratings):.0f}")
        
        # Distribution breakdown
        elite = sum(1 for r in ratings if r >= 1650)
        very_good = sum(1 for r in ratings if 1600 <= r < 1650)
        good = sum(1 for r in ratings if 1550 <= r < 1600)
        average = sum(1 for r in ratings if 1450 <= r < 1550)
        below_avg = sum(1 for r in ratings if 1400 <= r < 1450)
        poor = sum(1 for r in ratings if r < 1400)
        
        print(f"\nDistribution:")
        print(f"  Elite (1650+):        {elite:3d} ({elite/len(ratings)*100:.1f}%)")
        print(f"  Very Good (1600-1649): {very_good:3d} ({very_good/len(ratings)*100:.1f}%)")
        print(f"  Good (1550-1599):      {good:3d} ({good/len(ratings)*100:.1f}%)")
        print(f"  Average (1450-1549):   {average:3d} ({average/len(ratings)*100:.1f}%)")
        print(f"  Below Avg (1400-1449): {below_avg:3d} ({below_avg/len(ratings)*100:.1f}%)")
        print(f"  Poor (<1400):          {poor:3d} ({poor/len(ratings)*100:.1f}%)")
    
    # ============================================
    # BIGGEST MOVERS (if game updates exist)
    # ============================================
    print("\n" + "="*60)
    print("📈 BIGGEST MOVERS (Last 7 Days)\n")
    
    seven_days_ago = (datetime.now() - timedelta(days=7)).date().isoformat()
    
    cursor.execute("""
        WITH recent_changes AS (
            SELECT 
                player_name,
                position,
                team_abbr,
                rating,
                rating_change,
                date
            FROM player_elo
            WHERE season = ?
            AND date >= ?
            AND rating_change IS NOT NULL
            AND ABS(rating_change) > 0
        )
        SELECT 
            player_name,
            position,
            team_abbr,
            SUM(rating_change) as total_change,
            MAX(rating) as current_rating
        FROM recent_changes
        GROUP BY player_name
        ORDER BY ABS(SUM(rating_change)) DESC
        LIMIT 10
    """, (current_season, seven_days_ago))
    
    movers = []
    for row in cursor.fetchall():
        movers.append({
            'Player': row[0][:23],
            'Pos': row[1],
            'Team': row[2],
            'Change': f"{row[3]:+.0f}",
            'Current Elo': f"{row[4]:.0f}"
        })
    
    if movers:
        df_movers = pd.DataFrame(movers)
        print(df_movers.to_string(index=False))
    else:
        print("No rating changes in last 7 days (initial ratings only)")
    
    # ============================================
    # DATABASE STATS
    # ============================================
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS\n")
    
    cursor.execute("SELECT COUNT(DISTINCT team_abbr) FROM team_elo WHERE season = ?", (current_season,))
    total_teams = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT player_name) FROM player_elo WHERE season = ?", (current_season,))
    total_players = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM team_elo WHERE season = ?", (current_season,))
    total_team_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM player_elo WHERE season = ?", (current_season,))
    total_player_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(date), MAX(date) FROM team_elo WHERE season = ?", (current_season,))
    date_range = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) FROM game_results WHERE season = ?", (current_season,))
    total_games = cursor.fetchone()[0]
    
    print(f"Season: {current_season}")
    print(f"Teams tracked: {total_teams}")
    print(f"Players tracked: {total_players}")
    print(f"Total team records: {total_team_records}")
    print(f"Total player records: {total_player_records}")
    print(f"Games recorded: {total_games}")
    if date_range[0] and date_range[1]:
        print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    # Check for other seasons
    cursor.execute("SELECT DISTINCT season FROM player_elo ORDER BY season DESC")
    all_seasons = [row[0] for row in cursor.fetchall()]
    
    if len(all_seasons) > 1:
        print(f"\nOther seasons in database:")
        for season in all_seasons:
            if season != current_season:
                cursor.execute("SELECT COUNT(DISTINCT player_name) FROM player_elo WHERE season = ?", (season,))
                player_count = cursor.fetchone()[0]
                print(f"  {season}: {player_count} players")
    
    db.close()
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        view_elo_ratings()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run:")
        print("  python update_elo_ratings.py --current-season --reset --initial-only\n")