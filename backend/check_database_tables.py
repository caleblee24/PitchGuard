#!/usr/bin/env python3
"""
Check Database Tables
Check what tables are available and their structure
"""

import sqlite3
import pandas as pd

def check_database_tables():
    """Check available tables and their structure"""
    conn = sqlite3.connect("pitchguard.db")
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("="*60)
    print("DATABASE TABLES CHECK")
    print("="*60)
    
    print(f"\n📋 Available Tables:")
    for table in tables:
        table_name = table[0]
        print(f"  - {table_name}")
        
        # Get table structure
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        print(f"    Columns:")
        for col in columns:
            col_name, col_type = col[1], col[2]
            print(f"      {col_name}: {col_type}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"    Row count: {count}")
        
        # Show sample data
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_data = cursor.fetchall()
            print(f"    Sample data:")
            for row in sample_data:
                print(f"      {row}")
        
        print()
    
    conn.close()

if __name__ == "__main__":
    check_database_tables()
