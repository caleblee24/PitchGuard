#!/usr/bin/env python3
"""
Practical MLB Injury Data Scraper
Focuses on publicly accessible data sources with robust error handling
"""

import requests
import pandas as pd
import time
import re
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import sqlite3
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
import hashlib
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InjuryRecord:
    """Injury record data structure"""
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
    confidence_score: float = 0.0
    data_quality: str = "unknown"

class PracticalInjuryScraper:
    """Practical MLB injury data scraper with focus on accessible sources"""
    
    def __init__(self, db_path: str = "pitchguard.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Rate limiting
        self.request_delay = 2.0
        self.last_request_time = 0
        
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
                confidence_score REAL DEFAULT 0.0,
                data_quality TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices
        indices = [
            'CREATE INDEX IF NOT EXISTS idx_injury_pitcher_date ON injury_records(pitcher_id, injury_date)',
            'CREATE INDEX IF NOT EXISTS idx_injury_type ON injury_records(injury_type)',
            'CREATE INDEX IF NOT EXISTS idx_injury_location ON injury_records(injury_location)',
            'CREATE INDEX IF NOT EXISTS idx_injury_severity ON injury_records(severity)',
            'CREATE INDEX IF NOT EXISTS idx_injury_source ON injury_records(source)'
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Injury database initialized")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def scrape_mlb_com_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from MLB.com injury page"""
        logger.info("Scraping MLB.com injuries")
        
        injuries = []
        
        try:
            # MLB.com injury page
            url = "https://www.mlb.com/injuries"
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury information in various formats
            # MLB.com structure can vary, so we'll try multiple approaches
            
            # Method 1: Look for injury tables
            injury_tables = soup.find_all('table')
            
            for table in injury_tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        try:
                            # Extract player info
                            player_cell = cells[0]
                            player_name = player_cell.get_text(strip=True)
                            
                            # Check if it's a pitcher (look for position info)
                            position_info = ""
                            if len(cells) > 1:
                                position_info = cells[1].get_text(strip=True)
                            
                            # Only include if it looks like a pitcher
                            if any(keyword in position_info.lower() for keyword in ['p', 'pitcher', 'sp', 'rp']) or not position_info:
                                injury_info = ""
                                if len(cells) > 2:
                                    injury_info = cells[2].get_text(strip=True)
                                
                                status_info = ""
                                if len(cells) > 3:
                                    status_info = cells[3].get_text(strip=True)
                                
                                if injury_info or status_info:
                                    injury_type, injury_location = self._parse_injury_info(injury_info + " " + status_info)
                                    injury_date = self._extract_injury_date(status_info)
                                    
                                    injury = InjuryRecord(
                                        pitcher_id=self._generate_pitcher_id(player_name),
                                        pitcher_name=player_name,
                                        team="Unknown",  # Will need to extract from context
                                        injury_date=injury_date,
                                        injury_type=injury_type,
                                        injury_location=injury_location,
                                        severity=self._classify_severity(status_info),
                                        expected_return=None,
                                        actual_return=None,
                                        source="MLB.com",
                                        url=url,
                                        notes=f"{injury_info} - {status_info}",
                                        confidence_score=0.6,
                                        data_quality="medium"
                                    )
                                    
                                    injuries.append(injury)
                        
                        except Exception as e:
                            logger.warning(f"Error parsing MLB.com row: {e}")
                            continue
            
            logger.info(f"Scraped {len(injuries)} injuries from MLB.com")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping MLB.com: {e}")
            return []
    
    def scrape_espn_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from ESPN MLB injuries page"""
        logger.info("Scraping ESPN injuries")
        
        injuries = []
        
        try:
            # ESPN MLB injuries page
            url = "https://www.espn.com/mlb/injuries"
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury information
            # ESPN typically has injury data in tables or structured divs
            
            # Method 1: Look for injury tables
            injury_tables = soup.find_all('table')
            
            for table in injury_tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        try:
                            player_name = cells[0].get_text(strip=True)
                            position = cells[1].get_text(strip=True)
                            injury_info = cells[2].get_text(strip=True)
                            status = cells[3].get_text(strip=True)
                            
                            # Only include pitchers
                            if 'P' in position or 'pitcher' in position.lower():
                                injury_type, injury_location = self._parse_injury_info(injury_info)
                                injury_date = self._extract_injury_date(status)
                                
                                injury = InjuryRecord(
                                    pitcher_id=self._generate_pitcher_id(player_name),
                                    pitcher_name=player_name,
                                    team="Unknown",
                                    injury_date=injury_date,
                                    injury_type=injury_type,
                                    injury_location=injury_location,
                                    severity=self._classify_severity(status),
                                    expected_return=None,
                                    actual_return=None,
                                    source="ESPN",
                                    url=url,
                                    notes=status,
                                    confidence_score=0.7,
                                    data_quality="medium"
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
    
    def scrape_rotowire_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from Rotowire"""
        logger.info("Scraping Rotowire injuries")
        
        injuries = []
        
        try:
            # Rotowire injury page
            url = "https://www.rotowire.com/baseball/injuries.php"
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury tables
            injury_tables = soup.find_all('table')
            
            for table in injury_tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        try:
                            player_name = cells[0].get_text(strip=True)
                            team = cells[1].get_text(strip=True)
                            injury_info = cells[2].get_text(strip=True)
                            status = cells[3].get_text(strip=True)
                            
                            # Check if it's a pitcher (might need to infer from context)
                            injury_type, injury_location = self._parse_injury_info(injury_info)
                            injury_date = self._extract_injury_date(status)
                            
                            injury = InjuryRecord(
                                pitcher_id=self._generate_pitcher_id(player_name),
                                pitcher_name=player_name,
                                team=team,
                                injury_date=injury_date,
                                injury_type=injury_type,
                                injury_location=injury_location,
                                severity=self._classify_severity(status),
                                expected_return=None,
                                actual_return=None,
                                source="Rotowire",
                                url=url,
                                notes=status,
                                confidence_score=0.6,
                                data_quality="medium"
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
    
    def scrape_spotrac_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from Spotrac"""
        logger.info("Scraping Spotrac injuries")
        
        injuries = []
        
        try:
            # Spotrac injury page
            url = "https://www.spotrac.com/mlb/disabled-list/"
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury tables
            injury_tables = soup.find_all('table')
            
            for table in injury_tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
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
                                pitcher_id=self._generate_pitcher_id(player_name),
                                pitcher_name=player_name,
                                team=team,
                                injury_date=injury_date,
                                injury_type=injury_type,
                                injury_location=injury_location,
                                severity=self._classify_severity(status),
                                expected_return=None,
                                actual_return=None,
                                source="Spotrac",
                                url=url,
                                notes=status,
                                confidence_score=0.7,
                                data_quality="medium"
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
    
    def scrape_fantasy_pros_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from FantasyPros"""
        logger.info("Scraping FantasyPros injuries")
        
        injuries = []
        
        try:
            # FantasyPros injury page (try different URL)
            url = "https://www.fantasypros.com/mlb/injuries"
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury information
            # FantasyPros might have different structure
            
            # Method 1: Look for injury tables
            injury_tables = soup.find_all('table')
            
            for table in injury_tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        try:
                            player_name = cells[0].get_text(strip=True)
                            position = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                            injury_info = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                            status = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                            
                            # Only include pitchers
                            if 'P' in position or 'pitcher' in position.lower() or not position:
                                injury_type, injury_location = self._parse_injury_info(injury_info)
                                injury_date = self._extract_injury_date(status)
                                
                                injury = InjuryRecord(
                                    pitcher_id=self._generate_pitcher_id(player_name),
                                    pitcher_name=player_name,
                                    team="Unknown",
                                    injury_date=injury_date,
                                    injury_type=injury_type,
                                    injury_location=injury_location,
                                    severity=self._classify_severity(status),
                                    expected_return=None,
                                    actual_return=None,
                                    source="FantasyPros",
                                    url=url,
                                    notes=status,
                                    confidence_score=0.5,
                                    data_quality="medium"
                                )
                                
                                injuries.append(injury)
                        
                        except Exception as e:
                            logger.warning(f"Error parsing FantasyPros row: {e}")
                            continue
            
            logger.info(f"Scraped {len(injuries)} injuries from FantasyPros")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping FantasyPros: {e}")
            return []
    
    def _parse_injury_info(self, injury_text: str) -> tuple:
        """Parse injury type and location from text"""
        injury_text = injury_text.lower()
        
        # Define injury mappings
        injury_types = {
            'elbow': ['elbow', 'ulnar', 'flexor', 'extensor', 'ucl', 'tcl'],
            'shoulder': ['shoulder', 'rotator cuff', 'labrum', 'ac joint', 'biceps'],
            'knee': ['knee', 'acl', 'mcl', 'meniscus', 'patella'],
            'back': ['back', 'spine', 'lumbar', 'thoracic', 'disc'],
            'oblique': ['oblique', 'side', 'intercostal'],
            'hamstring': ['hamstring', 'thigh', 'quadriceps'],
            'forearm': ['forearm', 'wrist', 'hand', 'finger', 'thumb'],
            'ankle': ['ankle', 'foot', 'achilles', 'heel'],
            'hip': ['hip', 'groin', 'adductor', 'abductor'],
            'other': ['illness', 'covid', 'personal', 'bereavement', 'paternity']
        }
        
        injury_location = "unknown"
        for location, keywords in injury_types.items():
            if any(keyword in injury_text for keyword in keywords):
                injury_location = location
                break
        
        # Extract injury type
        injury_type = "unknown"
        if any(word in injury_text for word in ['strain', 'pulled']):
            injury_type = "strain"
        elif any(word in injury_text for word in ['sprain', 'twisted']):
            injury_type = "sprain"
        elif any(word in injury_text for word in ['tear', 'rupture', 'torn']):
            injury_type = "tear"
        elif any(word in injury_text for word in ['fracture', 'broken', 'break']):
            injury_type = "fracture"
        elif any(word in injury_text for word in ['surgery', 'surgical', 'operation']):
            injury_type = "surgery"
        elif any(word in injury_text for word in ['inflammation', 'itis', 'tendonitis']):
            injury_type = "inflammation"
        elif any(word in injury_text for word in ['soreness', 'tightness', 'discomfort']):
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
        if not date_str or date_str.lower() in ['tbd', 'unknown', '', 'null']:
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
    
    def _generate_pitcher_id(self, player_name: str) -> str:
        """Generate a pitcher ID"""
        return f"temp_{hashlib.md5(player_name.encode()).hexdigest()[:8]}"
    
    def save_injuries_to_db(self, injuries: List[InjuryRecord]):
        """Save injury records to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for injury in injuries:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO injury_records 
                    (pitcher_id, pitcher_name, team, injury_date, injury_type, 
                     injury_location, severity, expected_return, actual_return, 
                     source, url, notes, confidence_score, data_quality, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    injury.pitcher_id, injury.pitcher_name, injury.team,
                    injury.injury_date.strftime('%Y-%m-%d'), injury.injury_type,
                    injury.injury_location, injury.severity,
                    injury.expected_return.strftime('%Y-%m-%d') if injury.expected_return else None,
                    injury.actual_return.strftime('%Y-%m-%d') if injury.actual_return else None,
                    injury.source, injury.url, injury.notes,
                    injury.confidence_score, injury.data_quality
                ))
                saved_count += 1
            except Exception as e:
                logger.warning(f"Error saving injury record: {e}")
                continue
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {saved_count} injury records to database")
    
    def run_comprehensive_scrape(self):
        """Run comprehensive injury scraping from all sources"""
        logger.info("Starting comprehensive injury data scraping")
        
        all_injuries = []
        
        # Define scraping functions
        scrape_functions = [
            self.scrape_mlb_com_injuries,
            self.scrape_espn_injuries,
            self.scrape_rotowire_injuries,
            self.scrape_spotrac_injuries,
            self.scrape_fantasy_pros_injuries
        ]
        
        # Run scraping sequentially to avoid overwhelming servers
        for func in scrape_functions:
            try:
                injuries = func()
                all_injuries.extend(injuries)
                logger.info(f"Completed {func.__name__}: {len(injuries)} injuries")
                time.sleep(1)  # Be respectful to servers
            except Exception as e:
                logger.error(f"Error with {func.__name__}: {e}")
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
            # Create unique key based on pitcher name and injury date
            key = (injury.pitcher_name.lower(), injury.injury_date.strftime('%Y-%m-%d'))
            
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
        
        # Injuries by various categories
        categories = ['injury_type', 'injury_location', 'severity', 'source', 'data_quality']
        summary = {'total_injuries': total_injuries}
        
        for category in categories:
            cursor.execute(f'''
                SELECT {category}, COUNT(*) 
                FROM injury_records 
                GROUP BY {category} 
                ORDER BY COUNT(*) DESC
            ''')
            summary[f'by_{category}'] = dict(cursor.fetchall())
        
        # Date range
        cursor.execute('''
            SELECT MIN(injury_date), MAX(injury_date) 
            FROM injury_records
        ''')
        date_range = cursor.fetchone()
        if date_range[0] and date_range[1]:
            summary['date_range'] = {
                'start': date_range[0],
                'end': date_range[1]
            }
        else:
            summary['date_range'] = {'start': None, 'end': None}
        
        conn.close()
        return summary

def main():
    """Main execution function"""
    scraper = PracticalInjuryScraper()
    
    # Run comprehensive scraping
    injuries = scraper.run_comprehensive_scrape()
    
    # Print summary
    summary = scraper.get_injury_summary()
    print("\n" + "="*50)
    print("PRACTICAL INJURY DATA SCRAPING SUMMARY")
    print("="*50)
    print(f"Total Injuries Collected: {summary['total_injuries']}")
    if summary['date_range']['start']:
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"\nBy Source: {summary['by_source']}")
    print(f"\nBy Type: {summary['by_type']}")
    print(f"\nBy Location: {summary['by_location']}")
    print(f"\nBy Severity: {summary['by_severity']}")

if __name__ == "__main__":
    main()
