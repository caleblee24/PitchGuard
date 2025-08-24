#!/usr/bin/env python3
"""
MLB Injury Data Scraper
Collects real injury data from multiple sources for PitchGuard model training
"""

import requests
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import sqlite3
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InjuryRecord:
    """Data structure for injury records"""
    pitcher_id: str
    pitcher_name: str
    team: str
    injury_date: datetime
    injury_type: str
    injury_location: str
    severity: str
    expected_return: Optional[datetime]
    actual_return: Optional[datetime]
    source: str
    url: str
    notes: str

class MLBInjuryScraper:
    """Comprehensive MLB injury data scraper"""
    
    def __init__(self, db_path: str = "pitchguard.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize injury database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS injury_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pitcher_id TEXT,
                pitcher_name TEXT,
                team TEXT,
                injury_date DATE,
                injury_type TEXT,
                injury_location TEXT,
                severity TEXT,
                expected_return DATE,
                actual_return DATE,
                source TEXT,
                url TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_injury_pitcher_date 
            ON injury_records(pitcher_id, injury_date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_injury_type 
            ON injury_records(injury_type)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Injury database initialized")
    
    def scrape_mlb_com_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from MLB.com"""
        logger.info(f"Scraping MLB.com injuries from {start_date} to {end_date}")
        
        injuries = []
        base_url = "https://www.mlb.com/injuries"
        
        try:
            response = self.session.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find injury tables
            injury_tables = soup.find_all('table', class_='table')
            
            for table in injury_tables:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        try:
                            player_name = cells[0].get_text(strip=True)
                            team = cells[1].get_text(strip=True)
                            injury_info = cells[2].get_text(strip=True)
                            status = cells[3].get_text(strip=True)
                            
                            # Parse injury information
                            injury_type, injury_location = self._parse_injury_info(injury_info)
                            
                            # Extract dates
                            injury_date = self._extract_injury_date(status)
                            
                            injury = InjuryRecord(
                                pitcher_id=self._get_pitcher_id(player_name, team),
                                pitcher_name=player_name,
                                team=team,
                                injury_date=injury_date,
                                injury_type=injury_type,
                                injury_location=injury_location,
                                severity=self._classify_severity(status),
                                expected_return=None,
                                actual_return=None,
                                source="MLB.com",
                                url=base_url,
                                notes=status
                            )
                            
                            injuries.append(injury)
                            
                        except Exception as e:
                            logger.warning(f"Error parsing row: {e}")
                            continue
            
            logger.info(f"Scraped {len(injuries)} injuries from MLB.com")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping MLB.com: {e}")
            return []
    
    def scrape_espn_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from ESPN"""
        logger.info(f"Scraping ESPN injuries from {start_date} to {end_date}")
        
        injuries = []
        base_url = "https://www.espn.com/mlb/injuries"
        
        try:
            response = self.session.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find team injury sections
            team_sections = soup.find_all('div', class_='team-injuries')
            
            for section in team_sections:
                team_name = section.find('h3').get_text(strip=True) if section.find('h3') else "Unknown"
                
                injury_rows = section.find_all('tr', class_='player')
                
                for row in injury_rows:
                    try:
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            player_name = cells[0].get_text(strip=True)
                            position = cells[1].get_text(strip=True)
                            injury_info = cells[2].get_text(strip=True)
                            status = cells[3].get_text(strip=True)
                            
                            # Only include pitchers
                            if 'P' in position or 'pitcher' in position.lower():
                                injury_type, injury_location = self._parse_injury_info(injury_info)
                                injury_date = self._extract_injury_date(status)
                                
                                injury = InjuryRecord(
                                    pitcher_id=self._get_pitcher_id(player_name, team_name),
                                    pitcher_name=player_name,
                                    team=team_name,
                                    injury_date=injury_date,
                                    injury_type=injury_type,
                                    injury_location=injury_location,
                                    severity=self._classify_severity(status),
                                    expected_return=None,
                                    actual_return=None,
                                    source="ESPN",
                                    url=base_url,
                                    notes=status
                                )
                                
                                injuries.append(injury)
                    
                    except Exception as e:
                        logger.warning(f"Error parsing ESPN row: {e}")
                        continue
            
            logger.info(f"Scraped {len(injuries)} injuries from ESPN")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping ESPN: {e}")
            return []
    
    def scrape_rotowire_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from Rotowire"""
        logger.info(f"Scraping Rotowire injuries from {start_date} to {end_date}")
        
        injuries = []
        base_url = "https://www.rotowire.com/baseball/injuries.php"
        
        try:
            response = self.session.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find injury table
            injury_table = soup.find('table', class_='table')
            if injury_table:
                rows = injury_table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        try:
                            player_name = cells[0].get_text(strip=True)
                            team = cells[1].get_text(strip=True)
                            injury_info = cells[2].get_text(strip=True)
                            status = cells[3].get_text(strip=True)
                            date_info = cells[4].get_text(strip=True)
                            
                            injury_type, injury_location = self._parse_injury_info(injury_info)
                            injury_date = self._parse_date(date_info)
                            
                            injury = InjuryRecord(
                                pitcher_id=self._get_pitcher_id(player_name, team),
                                pitcher_name=player_name,
                                team=team,
                                injury_date=injury_date,
                                injury_type=injury_type,
                                injury_location=injury_location,
                                severity=self._classify_severity(status),
                                expected_return=None,
                                actual_return=None,
                                source="Rotowire",
                                url=base_url,
                                notes=status
                            )
                            
                            injuries.append(injury)
                            
                        except Exception as e:
                            logger.warning(f"Error parsing Rotowire row: {e}")
                            continue
            
            logger.info(f"Scraped {len(injuries)} injuries from Rotowire")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping Rotowire: {e}")
            return []
    
    def scrape_spotrac_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from Spotrac"""
        logger.info(f"Scraping Spotrac injuries from {start_date} to {end_date}")
        
        injuries = []
        base_url = "https://www.spotrac.com/mlb/disabled-list/"
        
        try:
            response = self.session.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find injury table
            injury_table = soup.find('table', class_='table')
            if injury_table:
                rows = injury_table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 6:
                        try:
                            player_name = cells[0].get_text(strip=True)
                            team = cells[1].get_text(strip=True)
                            injury_info = cells[2].get_text(strip=True)
                            status = cells[3].get_text(strip=True)
                            start_date_str = cells[4].get_text(strip=True)
                            end_date_str = cells[5].get_text(strip=True)
                            
                            injury_type, injury_location = self._parse_injury_info(injury_info)
                            injury_date = self._parse_date(start_date_str)
                            return_date = self._parse_date(end_date_str) if end_date_str != "TBD" else None
                            
                            injury = InjuryRecord(
                                pitcher_id=self._get_pitcher_id(player_name, team),
                                pitcher_name=player_name,
                                team=team,
                                injury_date=injury_date,
                                injury_type=injury_type,
                                injury_location=injury_location,
                                severity=self._classify_severity(status),
                                expected_return=return_date,
                                actual_return=None,
                                source="Spotrac",
                                url=base_url,
                                notes=status
                            )
                            
                            injuries.append(injury)
                            
                        except Exception as e:
                            logger.warning(f"Error parsing Spotrac row: {e}")
                            continue
            
            logger.info(f"Scraped {len(injuries)} injuries from Spotrac")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping Spotrac: {e}")
            return []
    
    def _parse_injury_info(self, injury_text: str) -> tuple:
        """Parse injury type and location from text"""
        injury_text = injury_text.lower()
        
        # Define injury mappings
        injury_types = {
            'elbow': ['elbow', 'ulnar', 'flexor', 'extensor'],
            'shoulder': ['shoulder', 'rotator cuff', 'labrum'],
            'knee': ['knee', 'acl', 'mcl', 'meniscus'],
            'back': ['back', 'spine', 'lumbar', 'thoracic'],
            'oblique': ['oblique', 'side'],
            'hamstring': ['hamstring', 'thigh'],
            'forearm': ['forearm', 'wrist', 'hand'],
            'ankle': ['ankle', 'foot'],
            'hip': ['hip', 'groin'],
            'other': ['illness', 'covid', 'personal']
        }
        
        injury_location = "unknown"
        for location, keywords in injury_types.items():
            if any(keyword in injury_text for keyword in keywords):
                injury_location = location
                break
        
        # Extract injury type
        injury_type = "unknown"
        if "strain" in injury_text:
            injury_type = "strain"
        elif "sprain" in injury_text:
            injury_type = "sprain"
        elif "tear" in injury_text:
            injury_type = "tear"
        elif "fracture" in injury_text:
            injury_type = "fracture"
        elif "surgery" in injury_text or "surgical" in injury_text:
            injury_type = "surgery"
        elif "inflammation" in injury_text:
            injury_type = "inflammation"
        elif "soreness" in injury_text:
            injury_type = "soreness"
        
        return injury_type, injury_location
    
    def _extract_injury_date(self, status_text: str) -> datetime:
        """Extract injury date from status text"""
        # Look for date patterns
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{1,2}/\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, status_text)
            if match:
                try:
                    date_str = match.group(1)
                    if len(date_str.split('/')) == 2:  # MM/DD format
                        date_str = f"{date_str}/2024"  # Assume current year
                    return datetime.strptime(date_str, '%m/%d/%Y')
                except:
                    continue
        
        # Default to current date if no date found
        return datetime.now()
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        if not date_str or date_str.lower() in ['tbd', 'unknown', '']:
            return datetime.now()
        
        try:
            # Try different date formats
            formats = ['%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y', '%B %d, %Y']
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
        except:
            pass
        
        return datetime.now()
    
    def _classify_severity(self, status_text: str) -> str:
        """Classify injury severity"""
        status_text = status_text.lower()
        
        if any(word in status_text for word in ['season', 'out for year', 'torn']):
            return "severe"
        elif any(word in status_text for word in ['60-day', 'long-term', 'surgery']):
            return "moderate"
        elif any(word in status_text for word in ['10-day', 'day-to-day', 'soreness']):
            return "mild"
        else:
            return "unknown"
    
    def _get_pitcher_id(self, player_name: str, team: str) -> str:
        """Get pitcher ID from database or generate one"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Try to find existing pitcher
        cursor.execute('''
            SELECT pitcher_id FROM pitchers 
            WHERE name = ? AND team = ?
        ''', (player_name, team))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            # Generate a temporary ID
            return f"temp_{hash(player_name + team) % 10000}"
    
    def save_injuries_to_db(self, injuries: List[InjuryRecord]):
        """Save injury records to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for injury in injuries:
            cursor.execute('''
                INSERT OR REPLACE INTO injury_records 
                (pitcher_id, pitcher_name, team, injury_date, injury_type, 
                 injury_location, severity, expected_return, actual_return, 
                 source, url, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                injury.pitcher_id, injury.pitcher_name, injury.team,
                injury.injury_date.strftime('%Y-%m-%d'), injury.injury_type,
                injury.injury_location, injury.severity,
                injury.expected_return.strftime('%Y-%m-%d') if injury.expected_return else None,
                injury.actual_return.strftime('%Y-%m-%d') if injury.actual_return else None,
                injury.source, injury.url, injury.notes
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(injuries)} injury records to database")
    
    def run_comprehensive_scrape(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31"):
        """Run comprehensive injury scraping from all sources"""
        logger.info("Starting comprehensive injury data scraping")
        
        all_injuries = []
        
        # Scrape from all sources
        sources = [
            self.scrape_mlb_com_injuries,
            self.scrape_espn_injuries,
            self.scrape_rotowire_injuries,
            self.scrape_spotrac_injuries
        ]
        
        for source_func in sources:
            try:
                injuries = source_func(start_date, end_date)
                all_injuries.extend(injuries)
                time.sleep(2)  # Be respectful to servers
            except Exception as e:
                logger.error(f"Error with {source_func.__name__}: {e}")
                continue
        
        # Remove duplicates
        unique_injuries = self._deduplicate_injuries(all_injuries)
        
        # Save to database
        self.save_injuries_to_db(unique_injuries)
        
        logger.info(f"Comprehensive scraping complete: {len(unique_injuries)} unique injuries collected")
        return unique_injuries
    
    def _deduplicate_injuries(self, injuries: List[InjuryRecord]) -> List[InjuryRecord]:
        """Remove duplicate injury records"""
        seen = set()
        unique_injuries = []
        
        for injury in injuries:
            # Create unique key based on pitcher, date, and injury type
            key = (injury.pitcher_name, injury.injury_date.strftime('%Y-%m-%d'), injury.injury_type)
            
            if key not in seen:
                seen.add(key)
                unique_injuries.append(injury)
        
        return unique_injuries
    
    def get_injury_summary(self) -> Dict:
        """Get summary statistics of collected injury data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total injuries
        cursor.execute('SELECT COUNT(*) FROM injury_records')
        total_injuries = cursor.fetchone()[0]
        
        # Injuries by type
        cursor.execute('''
            SELECT injury_type, COUNT(*) 
            FROM injury_records 
            GROUP BY injury_type 
            ORDER BY COUNT(*) DESC
        ''')
        by_type = dict(cursor.fetchall())
        
        # Injuries by location
        cursor.execute('''
            SELECT injury_location, COUNT(*) 
            FROM injury_records 
            GROUP BY injury_location 
            ORDER BY COUNT(*) DESC
        ''')
        by_location = dict(cursor.fetchall())
        
        # Injuries by severity
        cursor.execute('''
            SELECT severity, COUNT(*) 
            FROM injury_records 
            GROUP BY severity 
            ORDER BY COUNT(*) DESC
        ''')
        by_severity = dict(cursor.fetchall())
        
        # Injuries by source
        cursor.execute('''
            SELECT source, COUNT(*) 
            FROM injury_records 
            GROUP BY source 
            ORDER BY COUNT(*) DESC
        ''')
        by_source = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_injuries': total_injuries,
            'by_type': by_type,
            'by_location': by_location,
            'by_severity': by_severity,
            'by_source': by_source
        }

def main():
    """Main execution function"""
    scraper = MLBInjuryScraper()
    
    # Run comprehensive scraping
    injuries = scraper.run_comprehensive_scrape()
    
    # Print summary
    summary = scraper.get_injury_summary()
    print("\n" + "="*50)
    print("INJURY DATA SCRAPING SUMMARY")
    print("="*50)
    print(f"Total Injuries Collected: {summary['total_injuries']}")
    print(f"\nBy Source: {summary['by_source']}")
    print(f"\nBy Type: {summary['by_type']}")
    print(f"\nBy Location: {summary['by_location']}")
    print(f"\nBy Severity: {summary['by_severity']}")

if __name__ == "__main__":
    main()
