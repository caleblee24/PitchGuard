#!/usr/bin/env python3
"""
Load Real MLB Pitchers into Database
Creates pitcher records for all real MLB pitchers found in the pitch data.
"""

import sys
import os
import sqlite3
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import get_db_session_context
from database.models import Pitcher

def load_real_mlb_pitchers():
    """Load real MLB pitchers from pitch data into pitchers table."""
    print("🔍 Loading real MLB pitchers from pitch data...")
    
    with get_db_session_context() as session:
        
        # Get unique pitcher IDs from pitches that aren't in pitchers table
        conn = sqlite3.connect('pitchguard_dev.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT p.pitcher_id, 
                   COUNT(*) as total_pitches,
                   MIN(p.game_date) as first_game,
                   MAX(p.game_date) as last_game,
                   AVG(p.release_speed) as avg_velocity
            FROM pitches p
            LEFT JOIN pitchers pr ON p.pitcher_id = pr.pitcher_id
            WHERE pr.pitcher_id IS NULL 
              AND p.pitcher_id > 500000  -- Real MLB IDs
            GROUP BY p.pitcher_id
            ORDER BY COUNT(*) DESC
        """)
        
        pitcher_stats = cursor.fetchall()
        conn.close()
        
        print(f"Found {len(pitcher_stats)} real MLB pitchers to add")
        
        for pitcher_id, total_pitches, first_game, last_game, avg_velocity in pitcher_stats:
            try:
                # Determine role based on pitch patterns (simplified heuristic)
                role = "starter" if total_pitches > 50 else "reliever"
                
                # Create pitcher name (in production, you'd get this from MLB API)
                pitcher_name = f"MLB Pitcher {pitcher_id}"
                
                # Create pitcher record
                pitcher = Pitcher(
                    pitcher_id=int(pitcher_id),
                    name=pitcher_name,
                    team="MLB",  # Would get real team from roster API
                    role=role,
                    baseline_velocity=float(avg_velocity) if avg_velocity else None,
                    baseline_spin_rate=2200  # Default estimate
                )
                
                session.add(pitcher)
                
                print(f"   ✅ Added {pitcher_name} (ID: {pitcher_id}) - {role}, {total_pitches} pitches")
                
            except Exception as e:
                print(f"   ⚠️ Error adding pitcher {pitcher_id}: {e}")
                continue
        
        session.commit()
        
        # Print summary
        total_pitchers = session.query(Pitcher).count()
        print(f"\n📊 Database now has {total_pitchers} total pitchers")
        
        # Show some real MLB pitchers for testing
        real_pitchers = session.query(Pitcher).filter(Pitcher.pitcher_id > 500000).limit(5).all()
        print("\n🎯 Real MLB pitchers ready for API testing:")
        for pitcher in real_pitchers:
            print(f"   - {pitcher.name} (ID: {pitcher.pitcher_id}) - {pitcher.role}")

if __name__ == "__main__":
    load_real_mlb_pitchers()

