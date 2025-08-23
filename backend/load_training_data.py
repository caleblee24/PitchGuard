"""
Load training data into the database for enhanced model integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.mock_data_generator import MockDataGenerator
from database.connection import get_db_session_context, create_tables
from database.models import Pitcher, Pitch, Appearance, Injury

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_and_load_training_data():
    """Generate and load comprehensive training data."""
    
    logger.info("Starting training data generation and loading...")
    
    # Generate comprehensive data for enhanced model
    config = {
        'random_seed': 42,
        'output_dir': 'data/training',
        'num_pitchers': 10,  # Start with fewer pitchers for testing
        'start_date': '2024-01-01',
        'end_date': '2024-08-20'
    }
    
    generator = MockDataGenerator(config)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(
        num_pitchers=10,
        start_date='2024-01-01',
        end_date='2024-08-20',
        injury_scenario=True
    )
    
    # Convert to DataFrames
    pitchers_data = dataset['pitchers']
    pitches_data = dataset['pitches']
    appearances_data = dataset['appearances']
    injuries_data = dataset['injuries']
    
    logger.info(f"Generated training data:")
    logger.info(f"  - {len(pitchers_data)} pitchers")
    logger.info(f"  - {len(pitches_data)} pitches")
    logger.info(f"  - {len(appearances_data)} appearances")
    logger.info(f"  - {len(injuries_data)} injuries")
    
    # Load into database
    with get_db_session_context() as db:
        # Create tables if they don't exist
        create_tables()
        
        # Load pitchers
        logger.info("Loading pitchers...")
        for pitcher_data in pitchers_data:
            # Check if pitcher already exists
            existing = db.query(Pitcher).filter(
                Pitcher.pitcher_id == pitcher_data.pitcher_id
            ).first()
            
            if not existing:
                pitcher = Pitcher(
                    pitcher_id=pitcher_data.pitcher_id,
                    name=pitcher_data.name,
                    team=pitcher_data.team,
                    role=pitcher_data.role,
                    baseline_velocity=pitcher_data.baseline_velocity,
                    baseline_spin_rate=pitcher_data.baseline_spin_rate,
                    injury_risk_profile=pitcher_data.injury_risk_profile
                )
                db.add(pitcher)
        
        db.commit()
        logger.info(f"Loaded {len(pitchers_data)} pitchers")
        
        # Load pitches (data comes as DataFrame)
        logger.info("Loading pitches...")
        pitch_count = 0
        pitches_df = pitches_data
        
        for _, pitch_row in pitches_df.iterrows():
            pitch = Pitch(
                game_date=datetime.strptime(str(pitch_row['game_date']), '%Y-%m-%d').date(),
                pitcher_id=int(pitch_row['pitcher_id']),
                pitch_type=str(pitch_row['pitch_type']),
                release_speed=float(pitch_row['release_speed']),
                release_spin_rate=int(pitch_row['release_spin_rate']),
                pitch_number=int(pitch_row['pitch_number']),
                at_bat_id=int(pitch_row['at_bat_id'])
            )
            db.add(pitch)
            pitch_count += 1
            
            # Commit in batches to avoid memory issues
            if pitch_count % 1000 == 0:
                db.commit()
                logger.info(f"Loaded {pitch_count} pitches...")
        
        db.commit()
        logger.info(f"Loaded {pitch_count} total pitches")
        
        # Load appearances (data comes as DataFrame)
        logger.info("Loading appearances...")
        appearances_df = appearances_data
        
        for _, appearance_row in appearances_df.iterrows():
            appearance = Appearance(
                game_date=datetime.strptime(str(appearance_row['game_date']), '%Y-%m-%d').date(),
                pitcher_id=int(appearance_row['pitcher_id']),
                pitches_thrown=int(appearance_row['pitches_thrown']),
                avg_vel=float(appearance_row['avg_vel']),
                avg_spin=int(appearance_row['avg_spin']),
                vel_std=float(appearance_row['vel_std']),
                spin_std=int(appearance_row['spin_std']),
                outs_recorded=int(appearance_row['outs_recorded']),
                innings_pitched=float(appearance_row['innings_pitched'])
            )
            db.add(appearance)
        
        db.commit()
        logger.info(f"Loaded {len(appearances_df)} appearances")
        
        # Load injuries (data comes as list of dictionaries)
        logger.info("Loading injuries...")
        
        for injury_data in injuries_data:
            injury = Injury(
                pitcher_id=int(injury_data['pitcher_id']),
                il_start=datetime.strptime(str(injury_data['il_start']), '%Y-%m-%d').date(),
                il_end=datetime.strptime(str(injury_data['il_end']), '%Y-%m-%d').date() if injury_data['il_end'] else None,
                stint_type=str(injury_data['stint_type'])
            )
            db.add(injury)
        
        db.commit()
        logger.info(f"Loaded {len(injuries_data)} injuries")
    
    logger.info("Training data loading completed successfully!")
    
    return {
        'pitchers': len(pitchers_data),
        'pitches': pitch_count,
        'appearances': len(appearances_data),
        'injuries': len(injuries_data)
    }

def verify_data_loading():
    """Verify that data was loaded correctly."""
    
    logger.info("Verifying data loading...")
    
    with get_db_session_context() as db:
        # Check counts
        pitcher_count = db.query(Pitcher).count()
        pitch_count = db.query(Pitch).count()
        appearance_count = db.query(Appearance).count()
        injury_count = db.query(Injury).count()
        
        logger.info(f"Database contains:")
        logger.info(f"  - {pitcher_count} pitchers")
        logger.info(f"  - {pitch_count} pitches")
        logger.info(f"  - {appearance_count} appearances")
        logger.info(f"  - {injury_count} injuries")
        
        # Check sample data
        sample_pitcher = db.query(Pitcher).first()
        if sample_pitcher:
            logger.info(f"Sample pitcher: {sample_pitcher.name} ({sample_pitcher.team})")
            
            # Check related data
            pitcher_appearances = db.query(Appearance).filter(
                Appearance.pitcher_id == sample_pitcher.pitcher_id
            ).count()
            
            pitcher_pitches = db.query(Pitch).filter(
                Pitch.pitcher_id == sample_pitcher.pitcher_id
            ).count()
            
            logger.info(f"  - {pitcher_appearances} appearances")
            logger.info(f"  - {pitcher_pitches} pitches")
        
        return {
            'pitchers': pitcher_count,
            'pitches': pitch_count,
            'appearances': appearance_count,
            'injuries': injury_count
        }

if __name__ == "__main__":
    try:
        # Load training data
        load_stats = generate_and_load_training_data()
        
        # Verify loading
        db_stats = verify_data_loading()
        
        logger.info("=== Training Data Loading Summary ===")
        logger.info(f"Generated and loaded:")
        logger.info(f"  - {load_stats['pitchers']} pitchers")
        logger.info(f"  - {load_stats['pitches']} pitches")
        logger.info(f"  - {load_stats['appearances']} appearances")
        logger.info(f"  - {load_stats['injuries']} injuries")
        
        logger.info("Enhanced model integration ready!")
        
    except Exception as e:
        logger.error(f"Error during training data loading: {e}")
        raise
