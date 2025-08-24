#!/usr/bin/env python3
"""
Fix database schema for enhanced injury scraper
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_database_schema():
    """Fix the database schema to include missing columns"""
    conn = sqlite3.connect("pitchguard.db")
    cursor = conn.cursor()
    
    try:
        # Check if raw_text column exists
        cursor.execute("PRAGMA table_info(injury_records)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'raw_text' not in columns:
            logger.info("Adding raw_text column to injury_records table")
            cursor.execute('''
                ALTER TABLE injury_records 
                ADD COLUMN raw_text TEXT
            ''')
            logger.info("raw_text column added successfully")
        else:
            logger.info("raw_text column already exists")
        
        # Check if confidence_score column exists
        if 'confidence_score' not in columns:
            logger.info("Adding confidence_score column to injury_records table")
            cursor.execute('''
                ALTER TABLE injury_records 
                ADD COLUMN confidence_score REAL DEFAULT 0.0
            ''')
            logger.info("confidence_score column added successfully")
        else:
            logger.info("confidence_score column already exists")
        
        # Check if data_quality column exists
        if 'data_quality' not in columns:
            logger.info("Adding data_quality column to injury_records table")
            cursor.execute('''
                ALTER TABLE injury_records 
                ADD COLUMN data_quality TEXT DEFAULT 'unknown'
            ''')
            logger.info("data_quality column added successfully")
        else:
            logger.info("data_quality column already exists")
        
        conn.commit()
        logger.info("Database schema updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating database schema: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database_schema()
