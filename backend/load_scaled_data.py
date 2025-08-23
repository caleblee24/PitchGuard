#!/usr/bin/env python3
"""
Load Scaled Historical Data into Database
Loads the generated scaled data into the SQLite database for API testing.
"""

import sys
import os
import pandas as pd
from datetime import datetime
import logging

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import get_db_session_context
from database.models import Pitch, Appearance, Injury, Pitcher
from scaled_historical_integration import ScaledHistoricalDataIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_scaled_data_to_db():
    """Load scaled historical data into the database."""
    print("🚀 Loading scaled historical data into database...")
    
    # Generate scaled data
    integrator = ScaledHistoricalDataIntegrator()
    
    print("📊 Generating scaled Statcast data...")
    pitches_df = integrator.generate_scaled_statcast_data()
    
    print("🏥 Generating scaled injury data...")
    injuries_df = integrator.generate_scaled_injury_data()
    
    print("📈 Aggregating to appearances...")
    appearances_df = integrator.aggregate_to_appearances(pitches_df)
    
    # Load into database
    with get_db_session_context() as session:
        print("🗄️ Loading pitchers...")
        
        # Create pitcher records
        unique_pitchers = pitches_df['pitcher_id'].unique()
        for pitcher_id in unique_pitchers[:50]:  # Load first 50 for testing
            # Determine if starter or reliever based on appearances
            pitcher_appearances = appearances_df[appearances_df['pitcher_id'] == pitcher_id]
            avg_pitches = pitcher_appearances['pitches_thrown'].mean()
            is_starter = avg_pitches > 50  # Starters throw more pitches per appearance
            
            pitcher = Pitcher(
                pitcher_id=pitcher_id,
                name=f"Pitcher {pitcher_id}",
                team="MLB",
                role="starter" if is_starter else "reliever"
            )
            session.add(pitcher)
        
        print("📊 Loading pitches...")
        # Load pitches in batches
        batch_size = 1000
        for i in range(0, len(pitches_df), batch_size):
            batch = pitches_df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                pitch = Pitch(
                    pitcher_id=int(row['pitcher_id']),
                    game_date=row['game_date'],
                    pitch_number=int(row['pitch_number']),
                    at_bat_id=int(row['at_bat_id']),
                    pitch_type=str(row['pitch_type']),
                    release_speed=float(row['release_speed']),
                    release_spin_rate=float(row['release_spin_rate'])
                )
                session.add(pitch)
            
            session.commit()
            print(f"   ✅ Loaded batch {i//batch_size + 1}/{(len(pitches_df) + batch_size - 1)//batch_size}")
        
        print("📈 Loading appearances...")
        # Load appearances
        for _, row in appearances_df.iterrows():
            appearance = Appearance(
                pitcher_id=int(row['pitcher_id']),
                game_date=row['game_date'],
                pitches_thrown=int(row['pitches_thrown']),
                avg_vel=float(row['avg_vel']),
                avg_spin=float(row['avg_spin']),
                vel_std=float(row['vel_std']),
                spin_std=float(row['spin_std']),
                outs_recorded=int(row['outs_recorded']),
                innings_pitched=float(row['innings_pitched']),
                created_at=datetime.now()
            )
            session.add(appearance)
        
        print("🏥 Loading injuries...")
        # Load injuries
        for _, row in injuries_df.iterrows():
            injury = Injury(
                pitcher_id=int(row['pitcher_id']),
                il_start=row['il_start'],
                il_end=row['il_end'],
                stint_type=str(row['stint_type'])
            )
            session.add(injury)
        
        session.commit()
        print("✅ All data loaded successfully!")
        
        # Print summary
        pitcher_count = session.query(Pitcher).count()
        pitch_count = session.query(Pitch).count()
        appearance_count = session.query(Appearance).count()
        injury_count = session.query(Injury).count()
        
        print(f"\n📊 Database Summary:")
        print(f"   - Pitchers: {pitcher_count}")
        print(f"   - Pitches: {pitch_count}")
        print(f"   - Appearances: {appearance_count}")
        print(f"   - Injuries: {injury_count}")

if __name__ == "__main__":
    load_scaled_data_to_db()
