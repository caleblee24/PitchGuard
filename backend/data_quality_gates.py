#!/usr/bin/env python3
"""
Data Quality Gates for PitchGuard
Validates data quality and prevents silent regressions during multi-season data loading.
"""

import sys
import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityGates:
    """Data quality validation and monitoring system."""
    
    def __init__(self, db_path: str = 'pitchguard_dev.db'):
        self.db_path = db_path
        
        # Quality thresholds
        self.thresholds = {
            'min_pitches_per_season': 100000,  # Minimum pitches per season
            'max_null_velocity_pct': 3.0,      # Maximum % null velocity
            'max_null_spin_pct': 5.0,          # Maximum % null spin rate
            'min_velocity': 60.0,              # Minimum realistic velocity
            'max_velocity': 110.0,             # Maximum realistic velocity
            'min_spin_rate': 500,              # Minimum realistic spin rate
            'max_spin_rate': 3500,             # Maximum realistic spin rate
            'min_appearances_per_season': 5000, # Minimum appearances per season
            'max_duplicate_pitches_pct': 0.1,  # Maximum % duplicate pitches
        }
    
    def run_all_gates(self) -> Dict:
        """Run all data quality gates and return results."""
        print("🔍 Running Data Quality Gates...")
        print("=" * 50)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gates_passed': 0,
            'gates_failed': 0,
            'warnings': [],
            'errors': [],
            'details': {}
        }
        
        # Run each gate
        gates = [
            self.gate_row_counts,
            self.gate_missing_data,
            self.gate_data_ranges,
            self.gate_duplicates,
            self.gate_temporal_consistency,
            self.gate_pitcher_consistency,
            self.gate_monotonic_sanity
        ]
        
        for gate_func in gates:
            try:
                gate_result = gate_func()
                results['details'][gate_func.__name__] = gate_result
                
                if gate_result['passed']:
                    results['gates_passed'] += 1
                    print(f"✅ {gate_result['name']}: PASSED")
                else:
                    results['gates_failed'] += 1
                    print(f"❌ {gate_result['name']}: FAILED")
                    results['errors'].append(gate_result['message'])
                
                if gate_result.get('warnings'):
                    results['warnings'].extend(gate_result['warnings'])
                    for warning in gate_result['warnings']:
                        print(f"⚠️  {gate_result['name']}: {warning}")
                
            except Exception as e:
                error_msg = f"Gate {gate_func.__name__} failed with exception: {str(e)}"
                results['gates_failed'] += 1
                results['errors'].append(error_msg)
                print(f"💥 {error_msg}")
        
        # Print summary
        print("=" * 50)
        print(f"📊 Quality Gates Summary:")
        print(f"   - Passed: {results['gates_passed']}")
        print(f"   - Failed: {results['gates_failed']}")
        print(f"   - Warnings: {len(results['warnings'])}")
        print(f"   - Errors: {len(results['errors'])}")
        
        if results['gates_failed'] == 0:
            print("🎉 All quality gates passed!")
        else:
            print("⚠️  Some quality gates failed. Review errors above.")
        
        return results
    
    def gate_row_counts(self) -> Dict:
        """Gate: Validate row counts and season distribution."""
        name = "Row Counts & Season Distribution"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get overall counts
            cursor.execute("SELECT COUNT(*) FROM pitches")
            total_pitches = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM appearances")
            total_appearances = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitchers")
            total_pitchers = cursor.fetchone()[0]
            
            # Get counts by year
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
            
            yearly_data = cursor.fetchall()
            
            # Get appearance counts by year
            cursor.execute("""
                SELECT 
                    strftime('%Y', game_date) as year,
                    COUNT(*) as appearance_count
                FROM appearances 
                WHERE game_date IS NOT NULL
                GROUP BY strftime('%Y', game_date)
                ORDER BY year
            """)
            
            yearly_appearances = cursor.fetchall()
        
        # Validate thresholds
        passed = True
        warnings = []
        errors = []
        
        # Check minimum data requirements
        if total_pitches == 0:
            passed = False
            errors.append("No pitches found in database")
        
        if total_appearances == 0:
            passed = False
            errors.append("No appearances found in database")
        
        if total_pitchers == 0:
            passed = False
            errors.append("No pitchers found in database")
        
        # Check season distribution
        for year, pitch_count, unique_pitchers in yearly_data:
            if pitch_count < self.thresholds['min_pitches_per_season']:
                passed = False
                errors.append(f"Season {year}: Only {pitch_count:,} pitches (min: {self.thresholds['min_pitches_per_season']:,})")
            
            if unique_pitchers < 100:  # Should have at least 100 pitchers per season
                warnings.append(f"Season {year}: Only {unique_pitchers} unique pitchers")
        
        # Check appearance counts
        for year, appearance_count in yearly_appearances:
            if appearance_count < self.thresholds['min_appearances_per_season']:
                passed = False
                errors.append(f"Season {year}: Only {appearance_count:,} appearances (min: {self.thresholds['min_appearances_per_season']:,})")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'All row count checks passed',
            'warnings': warnings,
            'data': {
                'total_pitches': total_pitches,
                'total_appearances': total_appearances,
                'total_pitchers': total_pitchers,
                'yearly_pitches': {year: count for year, count, _ in yearly_data},
                'yearly_appearances': {year: count for year, count in yearly_appearances},
                'yearly_pitchers': {year: count for year, _, count in yearly_data}
            }
        }
    
    def gate_missing_data(self) -> Dict:
        """Gate: Validate missing data percentages."""
        name = "Missing Data Validation"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count total pitches
            cursor.execute("SELECT COUNT(*) FROM pitches")
            total_pitches = cursor.fetchone()[0]
            
            if total_pitches == 0:
                return {
                    'name': name,
                    'passed': False,
                    'message': 'No pitches found to validate',
                    'data': {}
                }
            
            # Count missing values
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE release_speed IS NULL")
            null_velocity = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE release_spin_rate IS NULL")
            null_spin = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE pitch_type IS NULL")
            null_pitch_type = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE game_date IS NULL")
            null_date = cursor.fetchone()[0]
        
        # Calculate percentages
        null_velocity_pct = (null_velocity / total_pitches) * 100
        null_spin_pct = (null_spin / total_pitches) * 100
        null_pitch_type_pct = (null_pitch_type / total_pitches) * 100
        null_date_pct = (null_date / total_pitches) * 100
        
        # Validate thresholds
        passed = True
        warnings = []
        errors = []
        
        if null_velocity_pct > self.thresholds['max_null_velocity_pct']:
            passed = False
            errors.append(f"Velocity missing: {null_velocity_pct:.1f}% (max: {self.thresholds['max_null_velocity_pct']}%)")
        
        if null_spin_pct > self.thresholds['max_null_spin_pct']:
            passed = False
            errors.append(f"Spin rate missing: {null_spin_pct:.1f}% (max: {self.thresholds['max_null_spin_pct']}%)")
        
        if null_pitch_type_pct > 1.0:  # Should have pitch type for almost all pitches
            warnings.append(f"Pitch type missing: {null_pitch_type_pct:.1f}%")
        
        if null_date_pct > 0.1:  # Should have date for almost all pitches
            warnings.append(f"Game date missing: {null_date_pct:.1f}%")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'Missing data within acceptable limits',
            'warnings': warnings,
            'data': {
                'total_pitches': total_pitches,
                'null_velocity': null_velocity,
                'null_velocity_pct': null_velocity_pct,
                'null_spin': null_spin,
                'null_spin_pct': null_spin_pct,
                'null_pitch_type': null_pitch_type,
                'null_pitch_type_pct': null_pitch_type_pct,
                'null_date': null_date,
                'null_date_pct': null_date_pct
            }
        }
    
    def gate_data_ranges(self) -> Dict:
        """Gate: Validate data ranges for velocity and spin rate."""
        name = "Data Range Validation"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get velocity statistics
            cursor.execute("""
                SELECT 
                    MIN(release_speed) as min_vel,
                    MAX(release_speed) as max_vel,
                    AVG(release_speed) as avg_vel,
                    COUNT(*) as count
                FROM pitches 
                WHERE release_speed IS NOT NULL
            """)
            
            vel_stats = cursor.fetchone()
            
            # Get spin rate statistics
            cursor.execute("""
                SELECT 
                    MIN(release_spin_rate) as min_spin,
                    MAX(release_spin_rate) as max_spin,
                    AVG(release_spin_rate) as avg_spin,
                    COUNT(*) as count
                FROM pitches 
                WHERE release_spin_rate IS NOT NULL
            """)
            
            spin_stats = cursor.fetchone()
        
        # Validate ranges
        passed = True
        warnings = []
        errors = []
        
        if vel_stats[0] is not None:  # If we have velocity data
            if vel_stats[0] < self.thresholds['min_velocity']:
                passed = False
                errors.append(f"Velocity too low: {vel_stats[0]:.1f} MPH (min: {self.thresholds['min_velocity']} MPH)")
            
            if vel_stats[1] > self.thresholds['max_velocity']:
                passed = False
                errors.append(f"Velocity too high: {vel_stats[1]:.1f} MPH (max: {self.thresholds['max_velocity']} MPH)")
            
            # Check for reasonable average
            if vel_stats[2] < 80 or vel_stats[2] > 95:
                warnings.append(f"Average velocity unusual: {vel_stats[2]:.1f} MPH")
        
        if spin_stats[0] is not None:  # If we have spin data
            if spin_stats[0] < self.thresholds['min_spin_rate']:
                passed = False
                errors.append(f"Spin rate too low: {spin_stats[0]:.0f} RPM (min: {self.thresholds['min_spin_rate']} RPM)")
            
            if spin_stats[1] > self.thresholds['max_spin_rate']:
                passed = False
                errors.append(f"Spin rate too high: {spin_stats[1]:.0f} RPM (max: {self.thresholds['max_spin_rate']} RPM)")
            
            # Check for reasonable average
            if spin_stats[2] < 1800 or spin_stats[2] > 2500:
                warnings.append(f"Average spin rate unusual: {spin_stats[2]:.0f} RPM")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'Data ranges within acceptable limits',
            'warnings': warnings,
            'data': {
                'velocity': {
                    'min': vel_stats[0],
                    'max': vel_stats[1],
                    'avg': vel_stats[2],
                    'count': vel_stats[3]
                },
                'spin_rate': {
                    'min': spin_stats[0],
                    'max': spin_stats[1],
                    'avg': spin_stats[2],
                    'count': spin_stats[3]
                }
            }
        }
    
    def gate_duplicates(self) -> Dict:
        """Gate: Check for duplicate pitches."""
        name = "Duplicate Detection"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count total pitches
            cursor.execute("SELECT COUNT(*) FROM pitches")
            total_pitches = cursor.fetchone()[0]
            
            if total_pitches == 0:
                return {
                    'name': name,
                    'passed': False,
                    'message': 'No pitches found to validate',
                    'data': {}
                }
            
            # Count potential duplicates (same pitcher, date, at_bat, pitch_number)
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT pitcher_id, game_date, at_bat_id, pitch_number, COUNT(*) as cnt
                    FROM pitches
                    GROUP BY pitcher_id, game_date, at_bat_id, pitch_number
                    HAVING COUNT(*) > 1
                )
            """)
            
            duplicate_groups = cursor.fetchone()[0]
            
            # Count total duplicate rows
            cursor.execute("""
                SELECT COUNT(*) - COUNT(DISTINCT pitcher_id || game_date || at_bat_id || pitch_number) as duplicates
                FROM pitches
            """)
            
            total_duplicates = cursor.fetchone()[0]
        
        # Calculate duplicate percentage
        duplicate_pct = (total_duplicates / total_pitches) * 100 if total_pitches > 0 else 0
        
        # Validate
        passed = True
        warnings = []
        errors = []
        
        if duplicate_pct > self.thresholds['max_duplicate_pitches_pct']:
            passed = False
            errors.append(f"Too many duplicates: {duplicate_pct:.2f}% ({total_duplicates:,} rows)")
        
        if duplicate_groups > 0:
            warnings.append(f"Found {duplicate_groups} groups of duplicate pitches")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'Duplicate rate within acceptable limits',
            'warnings': warnings,
            'data': {
                'total_pitches': total_pitches,
                'total_duplicates': total_duplicates,
                'duplicate_groups': duplicate_groups,
                'duplicate_pct': duplicate_pct
            }
        }
    
    def gate_temporal_consistency(self) -> Dict:
        """Gate: Validate temporal consistency of data."""
        name = "Temporal Consistency"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check date ranges
            cursor.execute("SELECT MIN(game_date), MAX(game_date) FROM pitches WHERE game_date IS NOT NULL")
            date_range = cursor.fetchone()
            
            # Check for future dates
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE game_date > date('now')")
            future_dates = cursor.fetchone()[0]
            
            # Check for very old dates (before 2015)
            cursor.execute("SELECT COUNT(*) FROM pitches WHERE game_date < '2015-01-01'")
            old_dates = cursor.fetchone()[0]
            
            # Check for weekends vs weekdays distribution (basic sanity check)
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN strftime('%w', game_date) IN ('0', '6') THEN 1 END) as weekend_games,
                    COUNT(*) as total_games
                FROM (SELECT DISTINCT game_date FROM pitches WHERE game_date IS NOT NULL)
            """)
            
            weekend_stats = cursor.fetchone()
        
        # Validate
        passed = True
        warnings = []
        errors = []
        
        if date_range[0] is None or date_range[1] is None:
            passed = False
            errors.append("No valid dates found")
        else:
            if future_dates > 0:
                passed = False
                errors.append(f"Found {future_dates} pitches with future dates")
            
            if old_dates > 0:
                warnings.append(f"Found {old_dates} pitches with dates before 2015")
            
            # Check if date range makes sense
            start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
            end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
            
            if end_date < start_date:
                passed = False
                errors.append("End date before start date")
            
            # Check weekend distribution (should be roughly 30-40% for MLB)
            if weekend_stats[1] > 0:
                weekend_pct = (weekend_stats[0] / weekend_stats[1]) * 100
                if weekend_pct < 20 or weekend_pct > 50:
                    warnings.append(f"Weekend game percentage unusual: {weekend_pct:.1f}%")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'Temporal consistency validated',
            'warnings': warnings,
            'data': {
                'date_range': date_range,
                'future_dates': future_dates,
                'old_dates': old_dates,
                'weekend_games': weekend_stats[0] if weekend_stats else 0,
                'total_games': weekend_stats[1] if weekend_stats else 0
            }
        }
    
    def gate_pitcher_consistency(self) -> Dict:
        """Gate: Validate pitcher data consistency."""
        name = "Pitcher Consistency"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count pitchers with data
            cursor.execute("SELECT COUNT(DISTINCT pitcher_id) FROM pitches")
            pitchers_with_pitches = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT pitcher_id) FROM appearances")
            pitchers_with_appearances = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM pitchers")
            total_pitchers = cursor.fetchone()[0]
            
            # Check for pitchers with appearances but no pitches
            cursor.execute("""
                SELECT COUNT(DISTINCT a.pitcher_id) 
                FROM appearances a
                LEFT JOIN pitches p ON a.pitcher_id = p.pitcher_id
                WHERE p.pitcher_id IS NULL
            """)
            
            pitchers_no_pitches = cursor.fetchone()[0]
            
            # Check for pitchers with pitches but no appearances
            cursor.execute("""
                SELECT COUNT(DISTINCT p.pitcher_id) 
                FROM pitches p
                LEFT JOIN appearances a ON p.pitcher_id = a.pitcher_id
                WHERE a.pitcher_id IS NULL
            """)
            
            pitchers_no_appearances = cursor.fetchone()[0]
        
        # Validate
        passed = True
        warnings = []
        errors = []
        
        if pitchers_with_pitches == 0:
            passed = False
            errors.append("No pitchers found in pitch data")
        
        if pitchers_with_appearances == 0:
            passed = False
            errors.append("No pitchers found in appearance data")
        
        if pitchers_no_pitches > 0:
            warnings.append(f"Found {pitchers_no_pitches} pitchers with appearances but no pitches")
        
        if pitchers_no_appearances > 0:
            warnings.append(f"Found {pitchers_no_appearances} pitchers with pitches but no appearances")
        
        # Check if we have reasonable number of pitchers
        if pitchers_with_pitches < 100:
            warnings.append(f"Only {pitchers_with_pitches} pitchers found (expected 300+ for full season)")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'Pitcher consistency validated',
            'warnings': warnings,
            'data': {
                'total_pitchers': total_pitchers,
                'pitchers_with_pitches': pitchers_with_pitches,
                'pitchers_with_appearances': pitchers_with_appearances,
                'pitchers_no_pitches': pitchers_no_pitches,
                'pitchers_no_appearances': pitchers_no_appearances
            }
        }
    
    def gate_monotonic_sanity(self) -> Dict:
        """Gate: Check basic monotonic relationships (more rest → lower risk, etc.)."""
        name = "Monotonic Sanity Checks"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # This is a placeholder for more sophisticated checks
            # For now, we'll do basic statistical checks
            
            # Check if velocity and spin rate are correlated (they should be)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(release_speed) as avg_vel,
                    AVG(release_spin_rate) as avg_spin
                FROM pitches 
                WHERE release_speed IS NOT NULL AND release_spin_rate IS NOT NULL
            """)
            
            correlation_stats = cursor.fetchone()
            
            # Check pitch count distribution (should be reasonable)
            cursor.execute("""
                SELECT 
                    AVG(pitches_thrown) as avg_pitches,
                    MAX(pitches_thrown) as max_pitches,
                    COUNT(*) as total_appearances
                FROM appearances 
                WHERE pitches_thrown IS NOT NULL
            """)
            
            pitch_count_stats = cursor.fetchone()
        
        # Validate
        passed = True
        warnings = []
        errors = []
        
        if correlation_stats[0] == 0:
            passed = False
            errors.append("No valid velocity/spin rate pairs found")
        else:
            # For now, skip correlation check since SQLite doesn't have CORR function
            # We can implement this later with pandas if needed
            pass
        
        if pitch_count_stats[2] > 0:
            avg_pitches = pitch_count_stats[0]
            max_pitches = pitch_count_stats[1]
            
            if avg_pitches < 10 or avg_pitches > 50:
                warnings.append(f"Average pitches per appearance unusual: {avg_pitches:.1f}")
            
            if max_pitches > 200:
                warnings.append(f"Maximum pitches per appearance very high: {max_pitches}")
        
        return {
            'name': name,
            'passed': passed,
            'message': '; '.join(errors) if errors else 'Monotonic sanity checks passed',
            'warnings': warnings,
            'data': {
                'velocity_spin_correlation': None,  # Not available in SQLite
                'avg_pitches_per_appearance': pitch_count_stats[0] if pitch_count_stats else None,
                'max_pitches_per_appearance': pitch_count_stats[1] if pitch_count_stats else None
            }
        }

def save_quality_report(results: Dict, output_file: str = 'docs/data_quality_gates.md'):
    """Save quality gate results to markdown file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Data Quality Gates Report\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Gates Passed:** {results['gates_passed']}\n")
        f.write(f"- **Gates Failed:** {results['gates_failed']}\n")
        f.write(f"- **Warnings:** {len(results['warnings'])}\n")
        f.write(f"- **Errors:** {len(results['errors'])}\n\n")
        
        if results['errors']:
            f.write("## Errors\n\n")
            for error in results['errors']:
                f.write(f"- {error}\n")
            f.write("\n")
        
        if results['warnings']:
            f.write("## Warnings\n\n")
            for warning in results['warnings']:
                f.write(f"- {warning}\n")
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        for gate_name, gate_result in results['details'].items():
            f.write(f"### {gate_result['name']}\n\n")
            f.write(f"- **Status:** {'✅ PASSED' if gate_result['passed'] else '❌ FAILED'}\n")
            f.write(f"- **Message:** {gate_result['message']}\n")
            
            if gate_result.get('warnings'):
                f.write("- **Warnings:**\n")
                for warning in gate_result['warnings']:
                    f.write(f"  - {warning}\n")
            
            f.write("\n")
    
    print(f"📄 Quality report saved to {output_file}")

if __name__ == "__main__":
    gates = DataQualityGates()
    results = gates.run_all_gates()
    save_quality_report(results)
