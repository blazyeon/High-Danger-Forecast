"""
Check Elo database status
"""
import sys
from pathlib import Path
import sqlite3
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("ELO DATABASE DIAGNOSTIC")
    print("=" * 60)

    # Compute current season dynamically
    today = datetime.now().date()
    year = today.year
    month = today.month
    if month >= 10:
        current_season = f"{year}{year+1}"
    else:
        current_season = f"{year-1}{year}"
    print(f"Current season: {current_season}\n")

    db_path = Path("elo_ratings.db")

    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        print("   Run: python update_elo_ratings.py --current-season --reset --initial-only")
        return 1

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check team Elo
        cursor.execute("""
            SELECT COUNT(DISTINCT team_abbr)
            FROM team_elo
            WHERE season = ?
        """, (current_season,))
        n_teams = cursor.fetchone()[0]

        cursor.execute("""
            SELECT MAX(date)
            FROM team_elo
            WHERE season = ?
        """, (current_season,))
        last_team_update = cursor.fetchone()[0]

        # Check player Elo
        cursor.execute("""
            SELECT COUNT(DISTINCT player_name)
            FROM player_elo
            WHERE season = ?
        """, (current_season,))
        n_players = cursor.fetchone()[0]

        cursor.execute("""
            SELECT MAX(date)
            FROM player_elo
            WHERE season = ?
        """, (current_season,))
        last_player_update = cursor.fetchone()[0]

        # Get top teams (latest rating per team)
        cursor.execute("""
            SELECT team_abbr, rating, games_played
            FROM (
                SELECT team_abbr, rating, games_played,
                       ROW_NUMBER() OVER (PARTITION BY team_abbr ORDER BY date DESC) as rn
                FROM team_elo
                WHERE season = ?
            ) t
            WHERE rn = 1
            ORDER BY rating DESC
            LIMIT 5
        """, (current_season,))
        top_teams = cursor.fetchall()

        # Get top players (latest rating per player)
        cursor.execute("""
            SELECT player_name, position, rating, games_played
            FROM (
                SELECT player_name, position, rating, games_played,
                       ROW_NUMBER() OVER (PARTITION BY player_name ORDER BY date DESC) as rn
                FROM player_elo
                WHERE season = ?
            ) p
            WHERE rn = 1
            ORDER BY rating DESC
            LIMIT 5
        """, (current_season,))
        top_players = cursor.fetchall()

        conn.close()

        # Display results
        season_display = f"{current_season[:4]}-{current_season[4:]}"
        print(f"\n📊 Season {season_display} Data:")
        print(f"   Teams: {n_teams}/32")
        print(f"   Players: {n_players}")
        print(f"   Last team update: {last_team_update}")
        print(f"   Last player update: {last_player_update}")

        if n_teams < 32:
            print(f"\n⚠️  Warning: Only {n_teams} teams have Elo data")
            print("   Run: python update_elo_ratings.py --current-season --continue")
        else:
            print("\n✅ All 32 teams have Elo data")

        if last_team_update:
            try:
                last_date = datetime.fromisoformat(last_team_update).date()
                days_old = (datetime.now().date() - last_date).days
                if days_old > 7:
                    print(f"\n⚠️  Elo data is {days_old} days old")
                    print("   Run: python update_elo_ratings.py --current-season --continue")
                else:
                    print(f"\n✅ Elo data is up to date ({days_old} days old)")
            except (ValueError, TypeError):
                pass

        print(f"\n🏆 Top 5 Teams by Elo:")
        for i, (team, rating, gp) in enumerate(top_teams, 1):
            print(f"   {i}. {team:4s} - {rating:4.0f} Elo ({gp} GP)")

        print(f"\n⭐ Top 5 Players by Elo:")
        for i, (name, pos, rating, gp) in enumerate(top_players, 1):
            print(f"   {i}. {name:20s} ({pos}) - {rating:4.0f} Elo ({gp} GP)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())