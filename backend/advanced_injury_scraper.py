#!/usr/bin/env python3
"""
Advanced MLB Injury Data Scraper
Uses APIs and advanced scraping techniques for comprehensive injury data collection
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
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, quote
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InjuryRecord:
    """Enhanced injury record data structure"""
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

class AdvancedInjuryScraper:
    """Advanced MLB injury data scraper with API support and rate limiting"""
    
    def __init__(self, db_path: str = "pitchguard.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Rate limiting
        self.request_delay = 1.0
        self.last_request_time = 0
        
        # Initialize database
        self._init_database()
        
        # Load pitcher mapping
        self.pitcher_mapping = self._load_pitcher_mapping()
    
    def _init_database(self):
        """Initialize enhanced injury database tables"""
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pitcher_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                team TEXT,
                mlb_id TEXT,
                espn_id TEXT,
                rotowire_id TEXT,
                spotrac_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indices
        indices = [
            'CREATE INDEX IF NOT EXISTS idx_injury_pitcher_date ON injury_records(pitcher_id, injury_date)',
            'CREATE INDEX IF NOT EXISTS idx_injury_type ON injury_records(injury_type)',
            'CREATE INDEX IF NOT EXISTS idx_injury_location ON injury_records(injury_location)',
            'CREATE INDEX IF NOT EXISTS idx_injury_severity ON injury_records(severity)',
            'CREATE INDEX IF NOT EXISTS idx_injury_source ON injury_records(source)',
            'CREATE INDEX IF NOT EXISTS idx_pitcher_mapping_name ON pitcher_mapping(name)',
            'CREATE INDEX IF NOT EXISTS idx_pitcher_mapping_team ON pitcher_mapping(team)'
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Enhanced injury database initialized")
    
    def _load_pitcher_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load pitcher ID mappings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, team, mlb_id, espn_id, rotowire_id, spotrac_id 
            FROM pitcher_mapping
        ''')
        
        mapping = {}
        for row in cursor.fetchall():
            name, team, mlb_id, espn_id, rotowire_id, spotrac_id = row
            key = f"{name}_{team}"
            mapping[key] = {
                'mlb_id': mlb_id,
                'espn_id': espn_id,
                'rotowire_id': rotowire_id,
                'spotrac_id': spotrac_id
            }
        
        conn.close()
        return mapping
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def scrape_mlb_api_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from MLB API"""
        logger.info(f"Scraping MLB API injuries from {start_date} to {end_date}")
        
        injuries = []
        
        try:
            # MLB API endpoint for injuries
            api_url = "https://statsapi.mlb.com/api/v1/teams"
            self._rate_limit()
            response = self.session.get(api_url)
            response.raise_for_status()
            
            teams_data = response.json()
            
            for team in teams_data['teams']:
                team_id = team['id']
                team_name = team['name']
                
                # Get team roster with injury status
                roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
                self._rate_limit()
                roster_response = self.session.get(roster_url)
                
                if roster_response.status_code == 200:
                    roster_data = roster_response.json()
                    
                    for player in roster_data['roster']:
                        if player.get('position', {}).get('abbreviation') == 'P':
                            player_id = player['person']['id']
                            player_name = player['person']['fullName']
                            
                            # Get player details including injury info
                            player_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
                            self._rate_limit()
                            player_response = self.session.get(player_url)
                            
                            if player_response.status_code == 200:
                                player_data = player_response.json()
                                person = player_data['people'][0]
                                
                                # Check for injury status
                                if person.get('status', {}).get('description') != 'Active':
                                    injury_info = person.get('status', {}).get('description', '')
                                    injury_date = self._extract_date_from_mlb_status(injury_info)
                                    
                                    injury_type, injury_location = self._parse_injury_info(injury_info)
                                    
                                    injury = InjuryRecord(
                                        pitcher_id=str(player_id),
                                        pitcher_name=player_name,
                                        team=team_name,
                                        injury_date=injury_date,
                                        injury_type=injury_type,
                                        injury_location=injury_location,
                                        severity=self._classify_severity(injury_info),
                                        expected_return=None,
                                        actual_return=None,
                                        source="MLB API",
                                        url=f"https://www.mlb.com/player/{player_name.lower().replace(' ', '-')}",
                                        notes=injury_info,
                                        confidence_score=0.8,
                                        data_quality="high"
                                    )
                                    
                                    injuries.append(injury)
                
                # Add delay between teams
                time.sleep(0.5)
            
            logger.info(f"Scraped {len(injuries)} injuries from MLB API")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping MLB API: {e}")
            return []
    
    def scrape_espn_api_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from ESPN API"""
        logger.info(f"Scraping ESPN API injuries from {start_date} to {end_date}")
        
        injuries = []
        
        try:
            # ESPN API endpoint for MLB injuries
            api_url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"
            self._rate_limit()
            response = self.session.get(api_url)
            response.raise_for_status()
            
            data = response.json()
            
            for team in data.get('teams', []):
                team_name = team.get('name', 'Unknown')
                
                for player in team.get('athletes', []):
                    if player.get('position', {}).get('abbreviation') == 'P':
                        player_name = player.get('displayName', 'Unknown')
                        player_id = player.get('id', '')
                        
                        # Check injury status
                        status = player.get('status', {})
                        if status.get('type', {}).get('state') != 'active':
                            injury_info = status.get('type', {}).get('description', '')
                            injury_date = self._extract_date_from_espn_status(status)
                            
                            injury_type, injury_location = self._parse_injury_info(injury_info)
                            
                            injury = InjuryRecord(
                                pitcher_id=str(player_id),
                                pitcher_name=player_name,
                                team=team_name,
                                injury_date=injury_date,
                                injury_type=injury_type,
                                injury_location=injury_location,
                                severity=self._classify_severity(injury_info),
                                expected_return=None,
                                actual_return=None,
                                source="ESPN API",
                                url=f"https://www.espn.com/mlb/player/_/id/{player_id}",
                                notes=injury_info,
                                confidence_score=0.7,
                                data_quality="high"
                            )
                            
                            injuries.append(injury)
            
            logger.info(f"Scraped {len(injuries)} injuries from ESPN API")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping ESPN API: {e}")
            return []
    
    def scrape_rotowire_api_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from Rotowire API"""
        logger.info(f"Scraping Rotowire API injuries from {start_date} to {end_date}")
        
        injuries = []
        
        try:
            # Rotowire API endpoint
            api_url = "https://www.rotowire.com/api/injuries"
            self._rate_limit()
            response = self.session.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                
                for injury in data.get('injuries', []):
                    if injury.get('position') == 'P':
                        player_name = injury.get('player_name', 'Unknown')
                        team = injury.get('team', 'Unknown')
                        injury_info = injury.get('injury', '')
                        status = injury.get('status', '')
                        date_str = injury.get('date', '')
                        
                        injury_date = self._parse_date(date_str)
                        injury_type, injury_location = self._parse_injury_info(injury_info)
                        
                        injury_record = InjuryRecord(
                            pitcher_id=injury.get('player_id', ''),
                            pitcher_name=player_name,
                            team=team,
                            injury_date=injury_date,
                            injury_type=injury_type,
                            injury_location=injury_location,
                            severity=self._classify_severity(status),
                            expected_return=None,
                            actual_return=None,
                            source="Rotowire API",
                            url=f"https://www.rotowire.com/baseball/player/{injury.get('player_id', '')}",
                            notes=f"{injury_info} - {status}",
                            confidence_score=0.6,
                            data_quality="medium"
                        )
                        
                        injuries.append(injury_record)
            
            logger.info(f"Scraped {len(injuries)} injuries from Rotowire API")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping Rotowire API: {e}")
            return []
    
    def scrape_spotrac_api_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from Spotrac API"""
        logger.info(f"Scraping Spotrac API injuries from {start_date} to {end_date}")
        
        injuries = []
        
        try:
            # Spotrac API endpoint
            api_url = "https://www.spotrac.com/api/injuries/mlb"
            self._rate_limit()
            response = self.session.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                
                for injury in data.get('injuries', []):
                    if injury.get('position') == 'P':
                        player_name = injury.get('name', 'Unknown')
                        team = injury.get('team', 'Unknown')
                        injury_info = injury.get('injury', '')
                        status = injury.get('status', '')
                        start_date_str = injury.get('start_date', '')
                        end_date_str = injury.get('end_date', '')
                        
                        injury_date = self._parse_date(start_date_str)
                        return_date = self._parse_date(end_date_str) if end_date_str and end_date_str != 'TBD' else None
                        injury_type, injury_location = self._parse_injury_info(injury_info)
                        
                        injury_record = InjuryRecord(
                            pitcher_id=injury.get('player_id', ''),
                            pitcher_name=player_name,
                            team=team,
                            injury_date=injury_date,
                            injury_type=injury_type,
                            injury_location=injury_location,
                            severity=self._classify_severity(status),
                            expected_return=return_date,
                            actual_return=None,
                            source="Spotrac API",
                            url=f"https://www.spotrac.com/mlb/player/{injury.get('player_id', '')}",
                            notes=f"{injury_info} - {status}",
                            confidence_score=0.7,
                            data_quality="high"
                        )
                        
                        injuries.append(injury_record)
            
            logger.info(f"Scraped {len(injuries)} injuries from Spotrac API")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping Spotrac API: {e}")
            return []
    
    def scrape_fantasy_pros_injuries(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31") -> List[InjuryRecord]:
        """Scrape injury data from FantasyPros"""
        logger.info(f"Scraping FantasyPros injuries from {start_date} to {end_date}")
        
        injuries = []
        
        try:
            # FantasyPros injury page
            url = "https://www.fantasypros.com/mlb/injuries.php"
            self._rate_limit()
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find injury table
            injury_table = soup.find('table', {'id': 'injuries-table'})
            if injury_table:
                rows = injury_table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all('td')
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
                                    pitcher_id=self._get_pitcher_id(player_name, "Unknown"),
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
    
    def _extract_date_from_mlb_status(self, status_text: str) -> datetime:
        """Extract date from MLB status text"""
        # Look for date patterns in MLB status
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\w+ \d{1,2}, \d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, status_text)
            if match:
                try:
                    date_str = match.group(1)
                    return datetime.strptime(date_str, '%m/%d/%Y')
                except:
                    continue
        
        return datetime.now()
    
    def _extract_date_from_espn_status(self, status: Dict) -> datetime:
        """Extract date from ESPN status object"""
        # ESPN status might have date information
        if 'date' in status:
            try:
                return datetime.fromisoformat(status['date'].replace('Z', '+00:00'))
            except:
                pass
        
        return datetime.now()
    
    def _parse_injury_info(self, injury_text: str) -> tuple:
        """Enhanced injury parsing with more sophisticated logic"""
        injury_text = injury_text.lower()
        
        # Enhanced injury mappings
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
        
        # Enhanced injury type classification
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
        elif any(word in injury_text for word in ['dislocation', 'subluxation']):
            injury_type = "dislocation"
        
        return injury_type, injury_location
    
    def _parse_date(self, date_str: str) -> datetime:
        """Enhanced date parsing"""
        if not date_str or date_str.lower() in ['tbd', 'unknown', '', 'null']:
            return datetime.now()
        
        try:
            # Try different date formats
            formats = [
                '%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y', '%B %d, %Y',
                '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
                '%m-%d-%Y', '%d/%m/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
        except:
            pass
        
        return datetime.now()
    
    def _classify_severity(self, status_text: str) -> str:
        """Enhanced severity classification"""
        status_text = status_text.lower()
        
        severe_keywords = ['season', 'out for year', 'torn', 'rupture', 'surgery', '60-day', 'out indefinitely']
        moderate_keywords = ['long-term', 'extended', 'weeks', 'month', '30-day']
        mild_keywords = ['10-day', 'day-to-day', 'soreness', 'tightness', 'questionable', 'probable']
        
        if any(keyword in status_text for keyword in severe_keywords):
            return "severe"
        elif any(keyword in status_text for keyword in moderate_keywords):
            return "moderate"
        elif any(keyword in status_text for keyword in mild_keywords):
            return "mild"
        else:
            return "unknown"
    
    def _get_pitcher_id(self, player_name: str, team: str) -> str:
        """Get pitcher ID with enhanced matching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Try exact match first
        cursor.execute('''
            SELECT pitcher_id FROM pitchers 
            WHERE name = ? AND team = ?
        ''', (player_name, team))
        
        result = cursor.fetchone()
        if result:
            conn.close()
            return result[0]
        
        # Try fuzzy matching
        cursor.execute('''
            SELECT pitcher_id, name FROM pitchers 
            WHERE name LIKE ? OR name LIKE ?
        ''', (f"%{player_name}%", f"%{player_name.split()[-1]}%"))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            # Generate a hash-based ID
            return f"temp_{hashlib.md5(f'{player_name}_{team}'.encode()).hexdigest()[:8]}"
    
    def save_injuries_to_db(self, injuries: List[InjuryRecord]):
        """Save injury records to database with enhanced deduplication"""
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
    
    def run_comprehensive_scrape(self, start_date: str = "2022-01-01", end_date: str = "2024-12-31"):
        """Run comprehensive injury scraping from all sources with parallel processing"""
        logger.info("Starting comprehensive injury data scraping")
        
        all_injuries = []
        
        # Define scraping functions
        scrape_functions = [
            self.scrape_mlb_api_injuries,
            self.scrape_espn_api_injuries,
            self.scrape_rotowire_api_injuries,
            self.scrape_spotrac_api_injuries,
            self.scrape_fantasy_pros_injuries
        ]
        
        # Run scraping in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(func, start_date, end_date): func.__name__
                for func in scrape_functions
            }
            
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    injuries = future.result()
                    all_injuries.extend(injuries)
                    logger.info(f"Completed {source_name}: {len(injuries)} injuries")
                except Exception as e:
                    logger.error(f"Error with {source_name}: {e}")
        
        # Remove duplicates with enhanced logic
        unique_injuries = self._deduplicate_injuries_advanced(all_injuries)
        
        # Save to database
        self.save_injuries_to_db(unique_injuries)
        
        logger.info(f"Comprehensive scraping complete: {len(unique_injuries)} unique injuries collected")
        return unique_injuries
    
    def _deduplicate_injuries_advanced(self, injuries: List[InjuryRecord]) -> List[InjuryRecord]:
        """Advanced deduplication with confidence scoring"""
        injury_groups = {}
        
        for injury in injuries:
            # Create grouping key
            key = (injury.pitcher_name.lower(), injury.injury_date.strftime('%Y-%m-%d'))
            
            if key not in injury_groups:
                injury_groups[key] = []
            injury_groups[key].append(injury)
        
        unique_injuries = []
        for key, group in injury_groups.items():
            if len(group) == 1:
                unique_injuries.append(group[0])
            else:
                # Select the best injury record from the group
                best_injury = max(group, key=lambda x: x.confidence_score)
                unique_injuries.append(best_injury)
        
        return unique_injuries
    
    def get_injury_summary(self) -> Dict:
        """Get comprehensive summary statistics"""
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
        summary['date_range'] = {
            'start': date_range[0],
            'end': date_range[1]
        }
        
        # Confidence score distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN confidence_score >= 0.8 THEN 'high'
                    WHEN confidence_score >= 0.6 THEN 'medium'
                    ELSE 'low'
                END as confidence_level,
                COUNT(*)
            FROM injury_records 
            GROUP BY confidence_level
        ''')
        summary['confidence_distribution'] = dict(cursor.fetchall())
        
        conn.close()
        return summary

def main():
    """Main execution function"""
    scraper = AdvancedInjuryScraper()
    
    # Run comprehensive scraping
    injuries = scraper.run_comprehensive_scrape()
    
    # Print comprehensive summary
    summary = scraper.get_injury_summary()
    print("\n" + "="*60)
    print("ADVANCED INJURY DATA SCRAPING SUMMARY")
    print("="*60)
    print(f"Total Injuries Collected: {summary['total_injuries']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"\nBy Source: {summary['by_source']}")
    print(f"\nBy Type: {summary['by_type']}")
    print(f"\nBy Location: {summary['by_location']}")
    print(f"\nBy Severity: {summary['by_severity']}")
    print(f"\nBy Data Quality: {summary['by_data_quality']}")
    print(f"\nConfidence Distribution: {summary['confidence_distribution']}")

if __name__ == "__main__":
    main()
