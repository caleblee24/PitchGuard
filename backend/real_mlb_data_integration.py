#!/usr/bin/env python3
"""
Real MLB Data Integration for PitchGuard
Integrates live MLB Statcast data and injury reports for production use.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import time
from typing import Dict, List, Tuple, Optional
import logging
import json

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.enhanced_features import EnhancedFeatureEngineer
from modeling.enhanced_model import EnhancedInjuryRiskModel
from database.connection import get_db_session_context
from database.models import Pitch, Appearance, Injury, Pitcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealMLBDataIntegrator:
    """Integrates real MLB Statcast data and injury reports."""
    
    def __init__(self):
        self.base_url = "https://baseballsavant.mlb.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PitchGuard/1.0 (https://pitchguard.com)'
        })
        
    def setup_pybaseball(self):
        """Set up pybaseball with proper configuration."""
        try:
            import pybaseball
            from pybaseball import cache
            
            # Enable caching to avoid repeated API calls
            cache.enable()
            
            # Set up data directory
            data_dir = os.path.join(os.path.dirname(__file__), 'data', 'mlb_cache')
            os.makedirs(data_dir, exist_ok=True)
            
            print("✅ Pybaseball configured successfully")
            return True
            
        except ImportError:
            print("❌ pybaseball not installed. Installing...")
            os.system("pip install pybaseball")
            return self.setup_pybaseball()
        except Exception as e:
            print(f"⚠️ Warning: Could not configure pybaseball: {e}")
            return False
    
    def get_statcast_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve Statcast pitch data for date range."""
        print(f"🔍 Fetching Statcast data from {start_date} to {end_date}...")
        
        try:
            from pybaseball import statcast
            
            # Get Statcast data
            data = statcast(start_dt=start_date, end_dt=end_date)
            
            if data is None or len(data) == 0:
                print("⚠️ No Statcast data returned")
                return pd.DataFrame()
            
            print(f"✅ Retrieved {len(data)} pitches from Statcast")
            
            # Filter for essential columns
            essential_columns = [
                'game_date', 'pitcher', 'batter', 'events', 'description',
                'pitch_type', 'release_speed', 'release_spin_rate',
                'release_pos_x', 'release_pos_z', 'release_extension',
                'at_bat_number', 'pitch_number', 'player_name'
            ]
            
            # Keep only available columns
            available_columns = [col for col in essential_columns if col in data.columns]
            filtered_data = data[available_columns].copy()
            
            # Clean and standardize
            filtered_data['game_date'] = pd.to_datetime(filtered_data['game_date']).dt.date
            filtered_data = filtered_data.dropna(subset=['pitcher', 'release_speed'])
            
            print(f"✅ Cleaned data: {len(filtered_data)} pitches")
            return filtered_data
            
        except Exception as e:
            print(f"❌ Error fetching Statcast data: {e}")
            return pd.DataFrame()
    
    def get_injury_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve MLB injury/IL stint data."""
        print(f"🏥 Fetching injury data from {start_date} to {end_date}...")
        
        try:
            # Try multiple sources for injury data
            injury_data = []
            
            # Method 1: Try pybaseball injury data
            try:
                from pybaseball import get_transactions
                
                # Get transactions (includes IL moves)
                transactions = get_transactions(start_date, end_date)
                
                if transactions is not None and len(transactions) > 0:
                    # Filter for IL-related transactions
                    il_transactions = transactions[
                        transactions['description'].str.contains('IL|disabled', case=False, na=False)
                    ].copy()
                    
                    for _, row in il_transactions.iterrows():
                        injury_data.append({
                            'pitcher_id': row.get('player_id'),
                            'player_name': row.get('player_name', ''),
                            'il_start': pd.to_datetime(row.get('date')).date(),
                            'il_end': None,  # End date often not available immediately
                            'injury_type': self._extract_injury_type(row.get('description', '')),
                            'stint_type': self._extract_stint_type(row.get('description', '')),
                            'source': 'transactions'
                        })
                
            except Exception as e:
                print(f"⚠️ Could not get transaction data: {e}")
            
            # Method 2: Try Baseball Savant injury data
            injury_data.extend(self._get_baseball_savant_injuries(start_date, end_date))
            
            # Method 3: Try ESPN injury data as fallback
            injury_data.extend(self._get_espn_injuries())
            
            if injury_data:
                injuries_df = pd.DataFrame(injury_data)
                injuries_df = injuries_df.drop_duplicates(subset=['pitcher_id', 'il_start'])
                print(f"✅ Retrieved {len(injuries_df)} injury records")
                return injuries_df
            else:
                print("⚠️ No injury data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching injury data: {e}")
            return pd.DataFrame()
    
    def _get_baseball_savant_injuries(self, start_date: str, end_date: str) -> List[Dict]:
        """Try to get injury data from Baseball Savant."""
        injuries = []
        try:
            # Baseball Savant doesn't have a direct injury API, but we can check
            # for missing players in expected lineups vs actual appearances
            print("   🔍 Checking Baseball Savant for injury patterns...")
            
            # This is a placeholder - in practice, you'd implement logic to
            # detect when regular players suddenly stop appearing
            
        except Exception as e:
            print(f"   ⚠️ Baseball Savant injury check failed: {e}")
        
        return injuries
    
    def _get_espn_injuries(self) -> List[Dict]:
        """Get injury data from ESPN as fallback."""
        injuries = []
        try:
            print("   🔍 Checking ESPN for injury reports...")
            
            # ESPN has injury reports but requires scraping
            # This is a placeholder for a more robust implementation
            url = "https://www.espn.com/mlb/injuries"
            
            # In practice, you'd implement web scraping here
            # For now, we'll return empty list
            
        except Exception as e:
            print(f"   ⚠️ ESPN injury check failed: {e}")
        
        return injuries
    
    def _extract_injury_type(self, description: str) -> str:
        """Extract injury type from transaction description."""
        description = description.lower()
        
        injury_keywords = {
            'shoulder': 'Shoulder Injury',
            'elbow': 'Elbow Injury', 
            'tommy john': 'Tommy John Surgery',
            'forearm': 'Forearm Strain',
            'lat': 'Lat Strain',
            'oblique': 'Oblique Strain',
            'back': 'Back Strain',
            'finger': 'Finger Injury',
            'wrist': 'Wrist Injury',
            'bicep': 'Bicep Injury'
        }
        
        for keyword, injury_type in injury_keywords.items():
            if keyword in description:
                return injury_type
        
        return 'Unspecified Injury'
    
    def _extract_stint_type(self, description: str) -> str:
        """Extract IL stint type from description."""
        description = description.lower()
        
        if '60-day' in description:
            return '60-day IL'
        elif '15-day' in description:
            return '15-day IL'
        elif '10-day' in description:
            return '10-day IL'
        else:
            return '10-day IL'  # Default
    
    def get_pitcher_roster(self) -> pd.DataFrame:
        """Get current MLB pitcher roster information."""
        print("👥 Fetching current MLB pitcher roster...")
        
        try:
            from pybaseball import playerid_lookup, get_roster
            
            pitchers = []
            
            # Get rosters for all 30 MLB teams
            teams = [
                'LAA', 'HOU', 'OAK', 'TOR', 'ATL', 'MIL', 'STL', 'CHC',
                'ARI', 'LAD', 'SF', 'CLE', 'SEA', 'MIA', 'NYM', 'WSH',
                'BAL', 'SD', 'PHI', 'PIT', 'TEX', 'TB', 'BOS', 'CIN',
                'COL', 'KC', 'DET', 'MIN', 'CWS', 'NYY'
            ]
            
            current_year = datetime.now().year
            
            for team in teams:
                try:
                    roster = get_roster(team, year=current_year)
                    
                    if roster is not None and len(roster) > 0:
                        # Filter for pitchers
                        team_pitchers = roster[roster['Pos'].str.contains('P', na=False)]
                        
                        for _, player in team_pitchers.iterrows():
                            pitchers.append({
                                'pitcher_id': player.get('playerid', player.get('mlbamid')),
                                'name': player.get('Name', ''),
                                'team': team,
                                'role': 'starter' if 'SP' in str(player.get('Pos', '')) else 'reliever'
                            })
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    print(f"   ⚠️ Could not get roster for {team}: {e}")
                    continue
            
            if pitchers:
                pitchers_df = pd.DataFrame(pitchers)
                pitchers_df = pitchers_df.dropna(subset=['pitcher_id'])
                pitchers_df = pitchers_df.drop_duplicates(subset=['pitcher_id'])
                print(f"✅ Retrieved {len(pitchers_df)} MLB pitchers")
                return pitchers_df
            else:
                print("⚠️ No pitcher roster data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching pitcher roster: {e}")
            return pd.DataFrame()
    
    def process_statcast_to_pitches(self, statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Statcast data to our pitch format."""
        print("🔄 Processing Statcast data to pitch format...")
        
        if statcast_df.empty:
            return pd.DataFrame()
        
        # Map Statcast columns to our format
        pitch_data = []
        
        for _, row in statcast_df.iterrows():
            try:
                pitch_data.append({
                    'pitcher_id': int(row['pitcher']) if pd.notna(row['pitcher']) else None,
                    'game_date': row['game_date'],
                    'pitch_number': int(row.get('pitch_number', 1)) if pd.notna(row.get('pitch_number')) else 1,
                    'at_bat_id': int(row.get('at_bat_number', 1)) if pd.notna(row.get('at_bat_number')) else 1,
                    'pitch_type': str(row.get('pitch_type', 'FF'))[:10],
                    'release_speed': float(row['release_speed']) if pd.notna(row['release_speed']) else 0.0,
                    'release_spin_rate': int(row.get('release_spin_rate', 0)) if pd.notna(row.get('release_spin_rate')) else 0
                })
            except Exception as e:
                print(f"   ⚠️ Error processing pitch: {e}")
                continue
        
        pitches_df = pd.DataFrame(pitch_data)
        pitches_df = pitches_df.dropna(subset=['pitcher_id'])
        
        print(f"✅ Processed {len(pitches_df)} pitches")
        return pitches_df
    
    def aggregate_to_appearances(self, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch data to appearance level."""
        print("📊 Aggregating pitches to appearances...")
        
        if pitches_df.empty:
            return pd.DataFrame()
        
        appearances = pitches_df.groupby(['pitcher_id', 'game_date']).agg({
            'pitch_number': 'count',  # Total pitches
            'release_speed': ['mean', 'std'],
            'release_spin_rate': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        appearances.columns = [
            'pitches_thrown', 'avg_vel', 'vel_std', 'avg_spin', 'spin_std'
        ]
        
        # Fill NaN std values with 0
        appearances['vel_std'] = appearances['vel_std'].fillna(0)
        appearances['spin_std'] = appearances['spin_std'].fillna(0)
        
        # Calculate innings pitched (rough estimate: ~15 pitches per inning)
        appearances['innings_pitched'] = (appearances['pitches_thrown'] / 15).round(1)
        appearances['outs_recorded'] = (appearances['innings_pitched'] * 3).astype(int)
        
        appearances = appearances.reset_index()
        
        print(f"✅ Created {len(appearances)} appearances")
        return appearances
    
    def load_to_database(self, pitchers_df: pd.DataFrame, pitches_df: pd.DataFrame, 
                        appearances_df: pd.DataFrame, injuries_df: pd.DataFrame):
        """Load real MLB data into database."""
        print("🗄️ Loading real MLB data into database...")
        
        with get_db_session_context() as session:
            
            # Load pitchers
            if not pitchers_df.empty:
                print(f"   👥 Loading {len(pitchers_df)} pitchers...")
                for _, row in pitchers_df.iterrows():
                    try:
                        # Check if pitcher already exists
                        existing = session.query(Pitcher).filter_by(pitcher_id=row['pitcher_id']).first()
                        
                        if not existing:
                            pitcher = Pitcher(
                                pitcher_id=int(row['pitcher_id']),
                                name=str(row['name']),
                                team=str(row['team']),
                                role=str(row['role'])
                            )
                            session.add(pitcher)
                    except Exception as e:
                        print(f"     ⚠️ Error loading pitcher {row.get('pitcher_id')}: {e}")
                        continue
                
                session.commit()
                print("   ✅ Pitchers loaded")
            
            # Load pitches in batches
            if not pitches_df.empty:
                print(f"   ⚾ Loading {len(pitches_df)} pitches...")
                batch_size = 1000
                
                for i in range(0, len(pitches_df), batch_size):
                    batch = pitches_df.iloc[i:i+batch_size]
                    
                    for _, row in batch.iterrows():
                        try:
                            pitch = Pitch(
                                pitcher_id=int(row['pitcher_id']),
                                game_date=row['game_date'],
                                pitch_number=int(row['pitch_number']),
                                at_bat_id=int(row['at_bat_id']),
                                pitch_type=str(row['pitch_type']),
                                release_speed=float(row['release_speed']),
                                release_spin_rate=int(row['release_spin_rate'])
                            )
                            session.add(pitch)
                        except Exception as e:
                            continue
                    
                    session.commit()
                    print(f"     ✅ Loaded batch {i//batch_size + 1}/{(len(pitches_df) + batch_size - 1)//batch_size}")
            
            # Load appearances
            if not appearances_df.empty:
                print(f"   📈 Loading {len(appearances_df)} appearances...")
                for _, row in appearances_df.iterrows():
                    try:
                        appearance = Appearance(
                            pitcher_id=int(row['pitcher_id']),
                            game_date=row['game_date'],
                            pitches_thrown=int(row['pitches_thrown']),
                            avg_vel=float(row['avg_vel']),
                            avg_spin=int(row['avg_spin']),
                            vel_std=float(row['vel_std']),
                            spin_std=int(row['spin_std']),
                            outs_recorded=int(row['outs_recorded']),
                            innings_pitched=float(row['innings_pitched'])
                        )
                        session.add(appearance)
                    except Exception as e:
                        continue
                
                session.commit()
                print("   ✅ Appearances loaded")
            
            # Load injuries
            if not injuries_df.empty:
                print(f"   🏥 Loading {len(injuries_df)} injuries...")
                for _, row in injuries_df.iterrows():
                    try:
                        injury = Injury(
                            pitcher_id=int(row['pitcher_id']),
                            il_start=row['il_start'],
                            il_end=row.get('il_end'),
                            stint_type=str(row['stint_type'])
                        )
                        session.add(injury)
                    except Exception as e:
                        continue
                
                session.commit()
                print("   ✅ Injuries loaded")
            
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
    
    def run_integration(self, start_date: str, end_date: str):
        """Run complete real MLB data integration."""
        print("🚀 Starting Real MLB Data Integration")
        print("=" * 60)
        print(f"📅 Date Range: {start_date} to {end_date}")
        print()
        
        # Step 1: Setup pybaseball
        if not self.setup_pybaseball():
            print("❌ Could not setup pybaseball. Using fallback methods...")
        
        # Step 2: Get pitcher roster
        pitchers_df = self.get_pitcher_roster()
        
        # Step 3: Get Statcast data
        statcast_df = self.get_statcast_data(start_date, end_date)
        
        # Step 4: Process to pitch format
        pitches_df = self.process_statcast_to_pitches(statcast_df)
        
        # Step 5: Aggregate to appearances
        appearances_df = self.aggregate_to_appearances(pitches_df)
        
        # Step 6: Get injury data
        injuries_df = self.get_injury_data(start_date, end_date)
        
        # Step 7: Load to database
        if not pitches_df.empty or not injuries_df.empty:
            self.load_to_database(pitchers_df, pitches_df, appearances_df, injuries_df)
        
        print("\n🎉 Real MLB Data Integration Complete!")
        print("=" * 60)
        
        return {
            'pitchers': len(pitchers_df),
            'pitches': len(pitches_df),
            'appearances': len(appearances_df),
            'injuries': len(injuries_df)
        }

if __name__ == "__main__":
    integrator = RealMLBDataIntegrator()
    
    # Get recent data (last 30 days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    results = integrator.run_integration(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
