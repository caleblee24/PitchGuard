#!/usr/bin/env python3
"""
Multi-Season MLB Data Loader for PitchGuard
Loads 2022-2024 regular seasons with idempotent ETL and data quality checks.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import sqlite3
import logging
from typing import Dict, List, Tuple, Optional
import time

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import get_db_session_context
from database.models import Pitch, Appearance, Pitcher
from real_mlb_data_integration import RealMLBDataIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSeasonDataLoader:
    """Loads multi-season MLB data with idempotent ETL and quality checks."""
    
    def __init__(self):
        self.integrator = RealMLBDataIntegrator()
        self.db_path = 'pitchguard_dev.db'
        
        # Season configurations
        self.seasons = {
            2022: {'start': '2022-03-31', 'end': '2022-10-05'},
            2023: {'start': '2023-03-30', 'end': '2023-10-01'},
            2024: {'start': '2024-03-28', 'end': '2024-09-29'}
        }
        
        # Monthly batches for each season
        self.monthly_batches = self._generate_monthly_batches()
        
    def _generate_monthly_batches(self) -> List[Dict]:
        """Generate monthly batches for all seasons."""
        batches = []
        
        for year, season_config in self.seasons.items():
            start_date = datetime.strptime(season_config['start'], '%Y-%m-%d')
            end_date = datetime.strptime(season_config['end'], '%Y-%m-%d')
            
            current_date = start_date
            while current_date <= end_date:
                # Create monthly batch
                batch_start = current_date
                
                # Calculate next month's start date safely
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    next_month = current_date.replace(month=current_date.month + 1, day=1)
                
                batch_end = min(next_month - timedelta(days=1), end_date)
                
                batches.append({
                    'year': year,
                    'start_date': batch_start.strftime('%Y-%m-%d'),
                    'end_date': batch_end.strftime('%Y-%m-%d'),
                    'description': f"{year} {batch_start.strftime('%B')}"
                })
                
                # Move to next month
                current_date = next_month
        
        return batches
    
    def setup_database_indices(self):
        """Create database indices for performance."""
        print("🔧 Setting up database indices...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create indices for performance
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_pitches_pid_date ON pitches(pitcher_id, game_date)",
                "CREATE INDEX IF NOT EXISTS idx_pitches_date ON pitches(game_date)",
                "CREATE INDEX IF NOT EXISTS idx_appearances_pid_date ON appearances(pitcher_id, game_date)",
                "CREATE INDEX IF NOT EXISTS idx_appearances_date ON appearances(game_date)",
                "CREATE INDEX IF NOT EXISTS idx_pitches_unique ON pitches(pitcher_id, game_date, at_bat_id, pitch_number)"
            ]
            
            for index_sql in indices:
                cursor.execute(index_sql)
                print(f"   ✅ Created index: {index_sql.split('ON')[1].strip()}")
            
            conn.commit()
            print("   ✅ Database indices created")
    
    def check_existing_data(self) -> Dict:
        """Check what data already exists in the database."""
        print("🔍 Checking existing data...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check pitches by year
            cursor.execute("""
                SELECT 
                    strftime('%Y', game_date) as year,
                    COUNT(*) as pitch_count,
                    COUNT(DISTINCT pitcher_id) as unique_pitchers
                FROM pitches 
                WHERE game_date IS NOT NULL
                GROUP BY strftime('%Y', game_date)
                ORDER BY year
            """)
            
            existing_pitches = cursor.fetchall()
            
            # Check appearances by year
            cursor.execute("""
                SELECT 
                    strftime('%Y', game_date) as year,
                    COUNT(*) as appearance_count
                FROM appearances 
                WHERE game_date IS NOT NULL
                GROUP BY strftime('%Y', game_date)
                ORDER BY year
            """)
            
            existing_appearances = cursor.fetchall()
            
            # Check total counts
            cursor.execute("SELECT COUNT(*) FROM pitches")
            total_pitches = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM appearances")
            total_appearances = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitchers")
            total_pitchers = cursor.fetchone()[0]
        
        summary = {
            'total_pitches': total_pitches,
            'total_appearances': total_appearances,
            'total_pitchers': total_pitchers,
            'pitches_by_year': {year: count for year, count, _ in existing_pitches},
            'appearances_by_year': {year: count for year, count in existing_appearances},
            'pitchers_by_year': {year: count for year, _, count in existing_pitches}
        }
        
        print(f"   📊 Current data summary:")
        print(f"      - Total pitches: {total_pitches:,}")
        print(f"      - Total appearances: {total_appearances:,}")
        print(f"      - Total pitchers: {total_pitchers}")
        
        for year in sorted(summary['pitches_by_year'].keys()):
            pitches = summary['pitches_by_year'][year]
            appearances = summary['appearances_by_year'].get(year, 0)
            pitchers = summary['pitchers_by_year'][year]
            print(f"      - {year}: {pitches:,} pitches, {appearances:,} appearances, {pitchers} pitchers")
        
        return summary
    
    def load_season_data(self, year: int, start_date: str, end_date: str, batch_size: int = 30) -> Dict:
        """Load data for a specific season with monthly batching."""
        print(f"📅 Loading {year} season data ({start_date} to {end_date})...")
        
        # Generate monthly batches for this season
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        monthly_batches = []
        current_date = start_dt
        
        while current_date <= end_dt:
            batch_start = current_date
            
            # Calculate next month's start date safely
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1, day=1)
            
            batch_end = min(next_month - timedelta(days=1), end_dt)
            
            monthly_batches.append({
                'start': batch_start.strftime('%Y-%m-%d'),
                'end': batch_end.strftime('%Y-%m-%d')
            })
            
            # Move to next month
            current_date = next_month
        
        print(f"   📊 Generated {len(monthly_batches)} monthly batches")
        
        season_stats = {
            'year': year,
            'total_pitches': 0,
            'total_appearances': 0,
            'unique_pitchers': set(),
            'batches_loaded': 0,
            'batches_skipped': 0,
            'errors': []
        }
        
        # Load each monthly batch
        for i, batch in enumerate(monthly_batches):
            print(f"   📦 Loading batch {i+1}/{len(monthly_batches)}: {batch['start']} to {batch['end']}")
            
            try:
                batch_stats = self._load_monthly_batch(year, batch['start'], batch['end'])
                
                season_stats['total_pitches'] += batch_stats['pitches_loaded']
                season_stats['total_appearances'] += batch_stats['appearances_loaded']
                season_stats['unique_pitchers'].update(batch_stats['pitchers_found'])
                season_stats['batches_loaded'] += 1
                
                print(f"      ✅ Loaded: {batch_stats['pitches_loaded']:,} pitches, {batch_stats['appearances_loaded']:,} appearances")
                
                # Rate limiting to be nice to MLB APIs
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Error loading batch {batch['start']}-{batch['end']}: {str(e)}"
                print(f"      ❌ {error_msg}")
                season_stats['errors'].append(error_msg)
                season_stats['batches_skipped'] += 1
        
        # Convert set to count
        season_stats['unique_pitchers'] = len(season_stats['unique_pitchers'])
        
        print(f"   🎉 {year} season complete:")
        print(f"      - {season_stats['total_pitches']:,} total pitches")
        print(f"      - {season_stats['total_appearances']:,} total appearances")
        print(f"      - {season_stats['unique_pitchers']} unique pitchers")
        print(f"      - {season_stats['batches_loaded']} batches loaded, {season_stats['batches_skipped']} skipped")
        
        return season_stats
    
    def _load_monthly_batch(self, year: int, start_date: str, end_date: str) -> Dict:
        """Load a single monthly batch of data."""
        
        # Check if we already have data for this period
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM pitches 
                WHERE game_date >= ? AND game_date <= ?
            """, (start_date, end_date))
            
            existing_pitches = cursor.fetchone()[0]
            
            if existing_pitches > 0:
                print(f"      ⏭️  Skipping batch (already have {existing_pitches:,} pitches)")
                return {
                    'pitches_loaded': 0,
                    'appearances_loaded': 0,
                    'pitchers_found': set()
                }
        
        # Load new data
        try:
            # Get Statcast data for this period
            statcast_df = self.integrator.get_statcast_data(start_date, end_date)
            
            if statcast_df.empty:
                return {
                    'pitches_loaded': 0,
                    'appearances_loaded': 0,
                    'pitchers_found': set()
                }
            
            # Process to pitch format
            pitches_df = self.integrator.process_statcast_to_pitches(statcast_df)
            
            if pitches_df.empty:
                return {
                    'pitches_loaded': 0,
                    'appearances_loaded': 0,
                    'pitchers_found': set()
                }
            
            # Aggregate to appearances
            appearances_df = self.integrator.aggregate_to_appearances(pitches_df)
            
            # Load to database with idempotent handling
            pitches_loaded = self._load_pitches_idempotent(pitches_df)
            appearances_loaded = self._load_appearances_idempotent(appearances_df)
            
            # Get unique pitchers from this batch
            pitchers_found = set(pitches_df['pitcher_id'].unique())
            
            return {
                'pitches_loaded': pitches_loaded,
                'appearances_loaded': appearances_loaded,
                'pitchers_found': pitchers_found
            }
            
        except Exception as e:
            print(f"      ❌ Error in batch: {str(e)}")
            raise
    
    def _load_pitches_idempotent(self, pitches_df: pd.DataFrame) -> int:
        """Load pitches with idempotent handling (no duplicates)."""
        if pitches_df.empty:
            return 0
        
        with get_db_session_context() as session:
            loaded_count = 0
            
            for _, row in pitches_df.iterrows():
                try:
                    # Check if pitch already exists
                    existing = session.query(Pitch).filter_by(
                        pitcher_id=int(row['pitcher_id']),
                        game_date=row['game_date'],
                        at_bat_id=int(row['at_bat_id']),
                        pitch_number=int(row['pitch_number'])
                    ).first()
                    
                    if not existing:
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
                        loaded_count += 1
                
                except Exception as e:
                    # Skip problematic rows
                    continue
            
            session.commit()
            return loaded_count
    
    def _load_appearances_idempotent(self, appearances_df: pd.DataFrame) -> int:
        """Load appearances with idempotent handling (no duplicates)."""
        if appearances_df.empty:
            return 0
        
        with get_db_session_context() as session:
            loaded_count = 0
            
            for _, row in appearances_df.iterrows():
                try:
                    # Check if appearance already exists
                    existing = session.query(Appearance).filter_by(
                        pitcher_id=int(row['pitcher_id']),
                        game_date=row['game_date']
                    ).first()
                    
                    if not existing:
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
                        loaded_count += 1
                
                except Exception as e:
                    # Skip problematic rows
                    continue
            
            session.commit()
            return loaded_count
    
    def generate_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        print("📊 Generating data quality report...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Overall counts
            cursor.execute("SELECT COUNT(*) FROM pitches")
            total_pitches = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM appearances")
            total_appearances = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitchers")
            total_pitchers = cursor.fetchone()[0]
            
            # Missing data analysis
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE release_speed IS NULL")
            null_velocity = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE release_spin_rate IS NULL")
            null_spin = cursor.fetchone()[0]
            
            # Data by year
            cursor.execute("""
                SELECT 
                    strftime('%Y', game_date) as year,
                    COUNT(*) as pitch_count,
                    COUNT(DISTINCT pitcher_id) as unique_pitchers,
                    AVG(release_speed) as avg_velocity,
                    AVG(release_spin_rate) as avg_spin
                FROM pitches 
                WHERE game_date IS NOT NULL
                GROUP BY strftime('%Y', game_date)
                ORDER BY year
            """)
            
            yearly_stats = cursor.fetchall()
            
            # Velocity distribution (using SQLite-compatible std dev calculation)
            cursor.execute("""
                SELECT 
                    MIN(release_speed) as min_vel,
                    MAX(release_speed) as max_vel,
                    AVG(release_speed) as avg_vel,
                    SQRT(AVG(release_speed * release_speed) - AVG(release_speed) * AVG(release_speed)) as std_vel
                FROM pitches 
                WHERE release_speed IS NOT NULL
            """)
            
            vel_stats = cursor.fetchone()
            
            # Spin rate distribution (using SQLite-compatible std dev calculation)
            cursor.execute("""
                SELECT 
                    MIN(release_spin_rate) as min_spin,
                    MAX(release_spin_rate) as max_spin,
                    AVG(release_spin_rate) as avg_spin,
                    SQRT(AVG(release_spin_rate * release_spin_rate) - AVG(release_spin_rate) * AVG(release_spin_rate)) as std_spin
                FROM pitches 
                WHERE release_spin_rate IS NOT NULL
            """)
            
            spin_stats = cursor.fetchone()
        
        # Calculate percentages
        null_velocity_pct = (null_velocity / total_pitches * 100) if total_pitches > 0 else 0
        null_spin_pct = (null_spin / total_pitches * 100) if total_pitches > 0 else 0
        
        report = {
            'summary': {
                'total_pitches': total_pitches,
                'total_appearances': total_appearances,
                'total_pitchers': total_pitchers,
                'null_velocity_count': null_velocity,
                'null_velocity_pct': null_velocity_pct,
                'null_spin_count': null_spin,
                'null_spin_pct': null_spin_pct
            },
            'yearly_stats': [
                {
                    'year': year,
                    'pitch_count': count,
                    'unique_pitchers': pitchers,
                    'avg_velocity': avg_vel,
                    'avg_spin': avg_spin
                }
                for year, count, pitchers, avg_vel, avg_spin in yearly_stats
            ],
            'velocity_stats': {
                'min': vel_stats[0],
                'max': vel_stats[1],
                'avg': vel_stats[2],
                'std': vel_stats[3]
            },
            'spin_stats': {
                'min': spin_stats[0],
                'max': spin_stats[1],
                'avg': spin_stats[2],
                'std': spin_stats[3]
            }
        }
        
        # Print summary
        print(f"   📈 Data Quality Summary:")
        print(f"      - Total pitches: {total_pitches:,}")
        print(f"      - Total appearances: {total_appearances:,}")
        print(f"      - Total pitchers: {total_pitchers}")
        print(f"      - Missing velocity: {null_velocity:,} ({null_velocity_pct:.1f}%)")
        print(f"      - Missing spin rate: {null_spin:,} ({null_spin_pct:.1f}%)")
        
        print(f"   📊 Velocity stats: {vel_stats[2]:.1f} ± {vel_stats[3]:.1f} MPH ({vel_stats[0]:.1f}-{vel_stats[1]:.1f})")
        print(f"   📊 Spin rate stats: {spin_stats[2]:.0f} ± {spin_stats[3]:.0f} RPM ({spin_stats[0]:.0f}-{spin_stats[1]:.0f})")
        
        return report
    
    def run_full_load(self, target_seasons: List[int] = None) -> Dict:
        """Run full multi-season data load."""
        if target_seasons is None:
            target_seasons = [2022, 2023, 2024]
        
        print("🚀 Starting Multi-Season MLB Data Load")
        print("=" * 60)
        print(f"📅 Target seasons: {target_seasons}")
        print()
        
        # Setup database indices
        self.setup_database_indices()
        
        # Check existing data
        existing_data = self.check_existing_data()
        
        # Load each season
        season_results = {}
        total_stats = {
            'total_pitches': 0,
            'total_appearances': 0,
            'total_pitchers': 0,
            'errors': []
        }
        
        for year in target_seasons:
            if year not in self.seasons:
                print(f"⚠️  Skipping {year} (not configured)")
                continue
            
            season_config = self.seasons[year]
            print(f"\n🎯 Loading {year} season...")
            
            try:
                season_stats = self.load_season_data(
                    year, 
                    season_config['start'], 
                    season_config['end']
                )
                
                season_results[year] = season_stats
                
                # Update totals
                total_stats['total_pitches'] += season_stats['total_pitches']
                total_stats['total_appearances'] += season_stats['total_appearances']
                total_stats['total_pitchers'] = max(total_stats['total_pitchers'], season_stats['unique_pitchers'])
                total_stats['errors'].extend(season_stats['errors'])
                
            except Exception as e:
                error_msg = f"Failed to load {year}: {str(e)}"
                print(f"❌ {error_msg}")
                total_stats['errors'].append(error_msg)
        
        # Generate final quality report
        print(f"\n📊 Generating final data quality report...")
        quality_report = self.generate_data_quality_report()
        
        # Print summary
        print(f"\n🎉 Multi-Season Load Complete!")
        print("=" * 60)
        print(f"📈 Total Results:")
        print(f"   - Pitches loaded: {total_stats['total_pitches']:,}")
        print(f"   - Appearances loaded: {total_stats['total_appearances']:,}")
        print(f"   - Unique pitchers: {total_stats['total_pitchers']}")
        print(f"   - Errors: {len(total_stats['errors'])}")
        
        if total_stats['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in total_stats['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
        
        return {
            'season_results': season_results,
            'total_stats': total_stats,
            'quality_report': quality_report
        }

if __name__ == "__main__":
    loader = MultiSeasonDataLoader()
    
    # Run full load for 2022-2024
    results = loader.run_full_load([2022, 2023, 2024])
    
    print(f"\n✅ Multi-season data load complete!")
    print(f"📊 Final database state:")
    print(f"   - Total pitches: {results['total_stats']['total_pitches']:,}")
    print(f"   - Total appearances: {results['total_stats']['total_appearances']:,}")
    print(f"   - Unique pitchers: {results['total_stats']['total_pitchers']}")
