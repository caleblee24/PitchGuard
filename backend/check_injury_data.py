#!/usr/bin/env python3
"""
Check collected injury data
"""

import sqlite3
import pandas as pd
from datetime import datetime

def check_injury_data():
    """Check the collected injury data"""
    conn = sqlite3.connect("pitchguard.db")
    
    # Get all injury records
    query = "SELECT * FROM injury_records ORDER BY created_at DESC LIMIT 20"
    df = pd.read_sql_query(query, conn)
    
    print("="*60)
    print("COLLECTED INJURY DATA ANALYSIS")
    print("="*60)
    
    print(f"\nTotal injuries in database: {len(df)}")
    
    if len(df) > 0:
        print(f"\nSample of collected injuries:")
        print(df[['pitcher_name', 'team', 'injury_type', 'injury_location', 'severity', 'source']].head(10))
        
        # Summary by source
        print(f"\nBy Source:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        # Summary by injury type
        print(f"\nBy Injury Type:")
        type_counts = df['injury_type'].value_counts()
        for injury_type, count in type_counts.items():
            print(f"  {injury_type}: {count}")
        
        # Summary by injury location
        print(f"\nBy Injury Location:")
        location_counts = df['injury_location'].value_counts()
        for location, count in location_counts.items():
            print(f"  {location}: {count}")
        
        # Summary by severity
        print(f"\nBy Severity:")
        severity_counts = df['severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
        
        # Check for pitchers specifically
        print(f"\nSample pitcher injuries:")
        pitcher_injuries = df[df['pitcher_name'].str.contains('|'.join(['P', 'pitcher']), case=False, na=False)]
        if len(pitcher_injuries) > 0:
            print(pitcher_injuries[['pitcher_name', 'injury_type', 'injury_location', 'source']].head(5))
        else:
            print("No specific pitcher injuries found - checking all records...")
            print(df[['pitcher_name', 'injury_type', 'injury_location', 'source']].head(5))
    
    conn.close()

if __name__ == "__main__":
    check_injury_data()
