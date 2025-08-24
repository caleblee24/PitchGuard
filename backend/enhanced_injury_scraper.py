#!/usr/bin/env python3
"""
Enhanced MLB Injury Data Scraper
Collects comprehensive injury data from multiple sources with improved parsing
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
from urllib.parse import urljoin, urlparse
import concurrent.futures

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
    raw_text: str = ""

class EnhancedInjuryScraper:
    """Enhanced MLB injury data scraper with multiple sources and improved parsing"""
    
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
        self.request_delay = 1.5
        self.last_request_time = 0
        
        # Initialize database
        self._init_database()
        
        # Enhanced injury parsing dictionaries
        self._init_parsing_dictionaries()
    
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
                raw_text TEXT,
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
            'CREATE INDEX IF NOT EXISTS idx_injury_source ON injury_records(source)',
            'CREATE INDEX IF NOT EXISTS idx_injury_confidence ON injury_records(confidence_score)'
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Enhanced injury database initialized")
    
    def _init_parsing_dictionaries(self):
        """Initialize enhanced parsing dictionaries"""
        # Comprehensive injury location mappings
        self.injury_locations = {
            'elbow': ['elbow', 'ulnar', 'flexor', 'extensor', 'ucl', 'tcl', 'medial', 'lateral', 'epicondyle'],
            'shoulder': ['shoulder', 'rotator cuff', 'labrum', 'ac joint', 'biceps', 'deltoid', 'supraspinatus', 'infraspinatus'],
            'knee': ['knee', 'acl', 'mcl', 'meniscus', 'patella', 'ligament', 'tendon', 'cartilage'],
            'back': ['back', 'spine', 'lumbar', 'thoracic', 'disc', 'herniated', 'bulging', 'sciatica'],
            'oblique': ['oblique', 'side', 'intercostal', 'rib', 'flank'],
            'hamstring': ['hamstring', 'thigh', 'quadriceps', 'adductor', 'abductor', 'groin'],
            'forearm': ['forearm', 'wrist', 'hand', 'finger', 'thumb', 'carpal', 'tunnel'],
            'ankle': ['ankle', 'foot', 'achilles', 'heel', 'plantar', 'fasciitis'],
            'hip': ['hip', 'groin', 'adductor', 'abductor', 'flexor', 'labrum'],
            'neck': ['neck', 'cervical', 'whiplash', 'strain'],
            'calf': ['calf', 'gastrocnemius', 'soleus', 'achilles'],
            'other': ['illness', 'covid', 'personal', 'bereavement', 'paternity', 'mental health']
        }
        
        # Enhanced injury type mappings
        self.injury_types = {
            'strain': ['strain', 'pulled', 'muscle strain', 'muscle pull', 'overstretched'],
            'sprain': ['sprain', 'twisted', 'ligament sprain', 'joint sprain'],
            'tear': ['tear', 'rupture', 'torn', 'complete tear', 'partial tear', 'full thickness'],
            'fracture': ['fracture', 'broken', 'break', 'stress fracture', 'hairline'],
            'surgery': ['surgery', 'surgical', 'operation', 'procedure', 'reconstruction'],
            'inflammation': ['inflammation', 'itis', 'tendonitis', 'bursitis', 'synovitis'],
            'soreness': ['soreness', 'tightness', 'discomfort', 'pain', 'tenderness'],
            'dislocation': ['dislocation', 'subluxation', 'popped out'],
            'contusion': ['contusion', 'bruise', 'hematoma'],
            'concussion': ['concussion', 'head injury', 'head trauma'],
            'infection': ['infection', 'bacterial', 'viral', 'fungal'],
            'other': ['unknown', 'undisclosed', 'personal', 'illness']
        }
        
        # Severity classification keywords
        self.severity_keywords = {
            'severe': ['season', 'out for year', 'torn', 'rupture', 'surgery', '60-day', 'out indefinitely', 'career ending'],
            'moderate': ['long-term', 'extended', 'weeks', 'month', '30-day', 'significant', 'major'],
            'mild': ['10-day', 'day-to-day', 'soreness', 'tightness', 'questionable', 'probable', 'minor']
        }
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def scrape_mlb_com_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from MLB.com with enhanced parsing"""
        logger.info("Scraping MLB.com injuries with enhanced parsing")
        
        injuries = []
        
        try:
            # Try multiple MLB.com injury pages
            urls = [
                "https://www.mlb.com/injuries",
                "https://www.mlb.com/transactions",
                "https://www.mlb.com/news"
            ]
            
            for url in urls:
                try:
                    self._rate_limit()
                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for injury information in various formats
                    injury_elements = self._find_injury_elements(soup)
                    
                    for element in injury_elements:
                        try:
                            injury = self._parse_injury_element(element, "MLB.com", url)
                            if injury:
                                injuries.append(injury)
                        except Exception as e:
                            logger.warning(f"Error parsing MLB.com element: {e}")
                            continue
                    
                    logger.info(f"Scraped {len(injury_elements)} potential injuries from {url}")
                    
                except Exception as e:
                    logger.warning(f"Error scraping {url}: {e}")
                    continue
            
            logger.info(f"Total MLB.com injuries: {len(injuries)}")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping MLB.com: {e}")
            return []
    
    def scrape_espn_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from ESPN with enhanced parsing"""
        logger.info("Scraping ESPN injuries with enhanced parsing")
        
        injuries = []
        
        try:
            # ESPN MLB injuries page
            url = "https://www.espn.com/mlb/injuries"
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury tables and structured data
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
                                injury = self._create_injury_record(
                                    player_name=player_name,
                                    team="Unknown",
                                    injury_info=injury_info,
                                    status=status,
                                    source="ESPN",
                                    url=url
                                )
                                if injury:
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
        """Scrape injury data from Rotowire with enhanced parsing"""
        logger.info("Scraping Rotowire injuries with enhanced parsing")
        
        injuries = []
        
        try:
            # Try multiple Rotowire URLs
            urls = [
                "https://www.rotowire.com/baseball/injuries.php",
                "https://www.rotowire.com/baseball/transactions.php",
                "https://www.rotowire.com/baseball/news.php"
            ]
            
            for url in urls:
                try:
                    self._rate_limit()
                    response = self.session.get(url, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for injury information
                    injury_elements = self._find_injury_elements(soup)
                    
                    for element in injury_elements:
                        try:
                            injury = self._parse_injury_element(element, "Rotowire", url)
                            if injury:
                                injuries.append(injury)
                        except Exception as e:
                            logger.warning(f"Error parsing Rotowire element: {e}")
                            continue
                    
                    logger.info(f"Scraped {len(injury_elements)} potential injuries from {url}")
                    
                except Exception as e:
                    logger.warning(f"Error scraping {url}: {e}")
                    continue
            
            logger.info(f"Total Rotowire injuries: {len(injuries)}")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping Rotowire: {e}")
            return []
    
    def scrape_spotrac_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from Spotrac with enhanced parsing"""
        logger.info("Scraping Spotrac injuries with enhanced parsing")
        
        injuries = []
        
        try:
            # Spotrac injury page
            url = "https://www.spotrac.com/mlb/disabled-list/"
            self._rate_limit()
            response = self.session.get(url, timeout=15)
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
                            
                            injury = self._create_injury_record(
                                player_name=player_name,
                                team=team,
                                injury_info=injury_info,
                                status=status,
                                date_info=date_info,
                                source="Spotrac",
                                url=url
                            )
                            if injury:
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
        """Scrape injury data from FantasyPros with enhanced parsing"""
        logger.info("Scraping FantasyPros injuries with enhanced parsing")
        
        injuries = []
        
        try:
            # FantasyPros injury page
            url = "https://www.fantasypros.com/mlb/injuries"
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury tables
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
                                injury = self._create_injury_record(
                                    player_name=player_name,
                                    team="Unknown",
                                    injury_info=injury_info,
                                    status=status,
                                    source="FantasyPros",
                                    url=url
                                )
                                if injury:
                                    injuries.append(injury)
                        
                        except Exception as e:
                            logger.warning(f"Error parsing FantasyPros row: {e}")
                            continue
            
            logger.info(f"Scraped {len(injuries)} injuries from FantasyPros")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping FantasyPros: {e}")
            return []
    
    def scrape_baseball_reference_injuries(self) -> List[InjuryRecord]:
        """Scrape injury data from Baseball Reference"""
        logger.info("Scraping Baseball Reference injuries")
        
        injuries = []
        
        try:
            # Baseball Reference injury page
            url = "https://www.baseball-reference.com/injuries/"
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for injury information
            injury_elements = self._find_injury_elements(soup)
            
            for element in injury_elements:
                try:
                    injury = self._parse_injury_element(element, "Baseball Reference", url)
                    if injury:
                        injuries.append(injury)
                except Exception as e:
                    logger.warning(f"Error parsing Baseball Reference element: {e}")
                    continue
            
            logger.info(f"Scraped {len(injuries)} injuries from Baseball Reference")
            return injuries
            
        except Exception as e:
            logger.error(f"Error scraping Baseball Reference: {e}")
            return []
    
    def _find_injury_elements(self, soup: BeautifulSoup) -> List:
        """Find injury-related elements in HTML"""
        elements = []
        
        # Look for various injury-related patterns
        patterns = [
            'injury', 'injured', 'disabled', 'out', 'surgery', 'recovery',
            'strain', 'sprain', 'tear', 'fracture', 'soreness'
        ]
        
        # Find elements containing injury-related text
        for pattern in patterns:
            elements.extend(soup.find_all(text=re.compile(pattern, re.IGNORECASE)))
        
        # Find tables with injury data
        elements.extend(soup.find_all('table'))
        
        # Find divs with injury information
        elements.extend(soup.find_all('div', class_=re.compile('injury|player|status', re.IGNORECASE)))
        
        return elements
    
    def _parse_injury_element(self, element, source: str, url: str) -> Optional[InjuryRecord]:
        """Parse injury information from HTML element"""
        try:
            # Extract text content
            text = element.get_text(strip=True) if hasattr(element, 'get_text') else str(element)
            
            # Look for player names and injury information
            # This is a simplified parser - would need more sophisticated NLP for production
            
            # Extract potential player name (simplified)
            lines = text.split('\n')
            player_name = ""
            injury_info = ""
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 2:
                    if not player_name and self._looks_like_name(line):
                        player_name = line
                    elif 'injury' in line.lower() or any(word in line.lower() for word in ['out', 'surgery', 'strain', 'sprain']):
                        injury_info = line
            
            if player_name and injury_info:
                return self._create_injury_record(
                    player_name=player_name,
                    team="Unknown",
                    injury_info=injury_info,
                    status=injury_info,
                    source=source,
                    url=url
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing injury element: {e}")
            return None
    
    def _looks_like_name(self, text: str) -> bool:
        """Check if text looks like a player name"""
        # Simple heuristic for player names
        if len(text) < 3 or len(text) > 50:
            return False
        
        # Check for typical name patterns
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', text):
            return True
        
        # Check for names with hyphens or periods
        if re.match(r'^[A-Z][a-z]+[-.][A-Z][a-z]+$', text):
            return True
        
        return False
    
    def _create_injury_record(self, player_name: str, team: str, injury_info: str, 
                            status: str, source: str, url: str, 
                            date_info: str = "") -> Optional[InjuryRecord]:
        """Create injury record with enhanced parsing"""
        try:
            # Enhanced injury parsing
            injury_type, injury_location = self._parse_injury_info_enhanced(injury_info + " " + status)
            injury_date = self._extract_injury_date_enhanced(status + " " + date_info)
            severity = self._classify_severity_enhanced(status)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                injury_info, status, injury_type, injury_location
            )
            
            # Only include if confidence is reasonable
            if confidence_score < 0.3:
                return None
            
            injury = InjuryRecord(
                pitcher_id=self._generate_pitcher_id(player_name),
                pitcher_name=player_name,
                team=team,
                injury_date=injury_date,
                injury_type=injury_type,
                injury_location=injury_location,
                severity=severity,
                expected_return=None,
                actual_return=None,
                source=source,
                url=url,
                notes=f"{injury_info} - {status}",
                confidence_score=confidence_score,
                data_quality="enhanced",
                raw_text=f"{injury_info} {status} {date_info}"
            )
            
            return injury
            
        except Exception as e:
            logger.warning(f"Error creating injury record: {e}")
            return None
    
    def _parse_injury_info_enhanced(self, injury_text: str) -> tuple:
        """Enhanced injury parsing with multiple strategies"""
        injury_text = injury_text.lower()
        
        # Strategy 1: Direct keyword matching
        injury_location = self._find_injury_location(injury_text)
        injury_type = self._find_injury_type(injury_text)
        
        # Strategy 2: Context-based inference
        if injury_location == "unknown":
            injury_location = self._infer_injury_location(injury_text)
        
        if injury_type == "unknown":
            injury_type = self._infer_injury_type(injury_text)
        
        # Strategy 3: Medical terminology matching
        if injury_location == "unknown":
            injury_location = self._match_medical_terms(injury_text)
        
        return injury_type, injury_location
    
    def _find_injury_location(self, text: str) -> str:
        """Find injury location using keyword matching"""
        for location, keywords in self.injury_locations.items():
            if any(keyword in text for keyword in keywords):
                return location
        return "unknown"
    
    def _find_injury_type(self, text: str) -> str:
        """Find injury type using keyword matching"""
        for injury_type, keywords in self.injury_types.items():
            if any(keyword in text for keyword in keywords):
                return injury_type
        return "unknown"
    
    def _infer_injury_location(self, text: str) -> str:
        """Infer injury location from context"""
        # Look for anatomical references
        if any(word in text for word in ['arm', 'throwing']):
            return "elbow"  # Most common for pitchers
        elif any(word in text for word in ['leg', 'running']):
            return "hamstring"
        elif any(word in text for word in ['back', 'spine']):
            return "back"
        
        return "unknown"
    
    def _infer_injury_type(self, text: str) -> str:
        """Infer injury type from context"""
        # Look for severity indicators
        if any(word in text for word in ['surgery', 'operation']):
            return "surgery"
        elif any(word in text for word in ['soreness', 'tightness']):
            return "soreness"
        elif any(word in text for word in ['strain', 'pulled']):
            return "strain"
        
        return "unknown"
    
    def _match_medical_terms(self, text: str) -> str:
        """Match medical terminology"""
        # Look for specific medical terms
        medical_terms = {
            'elbow': ['ulnar', 'flexor', 'extensor', 'epicondyle'],
            'shoulder': ['rotator', 'labrum', 'supraspinatus'],
            'knee': ['acl', 'mcl', 'meniscus', 'patella'],
            'back': ['herniated', 'bulging', 'sciatica']
        }
        
        for location, terms in medical_terms.items():
            if any(term in text for term in terms):
                return location
        
        return "unknown"
    
    def _extract_injury_date_enhanced(self, text: str) -> datetime:
        """Enhanced date extraction"""
        # Look for various date patterns
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{1,2}/\d{1,2})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(1)
                    if len(date_str.split('/')) == 2:  # MM/DD format
                        date_str = f"{date_str}/2024"  # Assume current year
                    return datetime.strptime(date_str, '%m/%d/%Y')
                except:
                    continue
        
        # Look for relative dates
        relative_patterns = {
            'yesterday': -1,
            'today': 0,
            'last week': -7,
            'this week': 0
        }
        
        for pattern, days in relative_patterns.items():
            if pattern in text.lower():
                return datetime.now() + timedelta(days=days)
        
        return datetime.now()
    
    def _classify_severity_enhanced(self, text: str) -> str:
        """Enhanced severity classification"""
        text = text.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text for keyword in keywords):
                return severity
        
        return "unknown"
    
    def _calculate_confidence_score(self, injury_info: str, status: str, 
                                  injury_type: str, injury_location: str) -> float:
        """Calculate confidence score for injury record"""
        score = 0.0
        
        # Base score for having injury information
        if injury_info.strip():
            score += 0.3
        
        if status.strip():
            score += 0.2
        
        # Bonus for specific injury classification
        if injury_type != "unknown":
            score += 0.2
        
        if injury_location != "unknown":
            score += 0.2
        
        # Bonus for severity classification
        if any(word in status.lower() for word in ['10-day', '15-day', '60-day', 'season']):
            score += 0.1
        
        # Penalty for very short or generic text
        if len(injury_info) < 10:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
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
                     source, url, notes, confidence_score, data_quality, raw_text, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    injury.pitcher_id, injury.pitcher_name, injury.team,
                    injury.injury_date.strftime('%Y-%m-%d'), injury.injury_type,
                    injury.injury_location, injury.severity,
                    injury.expected_return.strftime('%Y-%m-%d') if injury.expected_return else None,
                    injury.actual_return.strftime('%Y-%m-%d') if injury.actual_return else None,
                    injury.source, injury.url, injury.notes,
                    injury.confidence_score, injury.data_quality, injury.raw_text
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
        logger.info("Starting comprehensive injury data scraping with enhanced parsing")
        
        all_injuries = []
        
        # Define scraping functions
        scrape_functions = [
            self.scrape_mlb_com_injuries,
            self.scrape_espn_injuries,
            self.scrape_rotowire_injuries,
            self.scrape_spotrac_injuries,
            self.scrape_fantasy_pros_injuries,
            self.scrape_baseball_reference_injuries
        ]
        
        # Run scraping with parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                executor.submit(func): func.__name__
                for func in scrape_functions
            }
            
            for future in concurrent.futures.as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    injuries = future.result()
                    all_injuries.extend(injuries)
                    logger.info(f"Completed {source_name}: {len(injuries)} injuries")
                except Exception as e:
                    logger.error(f"Error with {source_name}: {e}")
        
        # Remove duplicates with enhanced logic
        unique_injuries = self._deduplicate_injuries_enhanced(all_injuries)
        
        # Save to database
        self.save_injuries_to_db(unique_injuries)
        
        logger.info(f"Comprehensive scraping complete: {len(unique_injuries)} unique injuries collected")
        return unique_injuries
    
    def _deduplicate_injuries_enhanced(self, injuries: List[InjuryRecord]) -> List[InjuryRecord]:
        """Enhanced deduplication with confidence scoring"""
        injury_groups = {}
        
        for injury in injuries:
            # Create grouping key based on pitcher name and date
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
    scraper = EnhancedInjuryScraper()
    
    # Run comprehensive scraping
    injuries = scraper.run_comprehensive_scrape()
    
    # Print comprehensive summary
    summary = scraper.get_injury_summary()
    print("\n" + "="*60)
    print("ENHANCED INJURY DATA SCRAPING SUMMARY")
    print("="*60)
    print(f"Total Injuries Collected: {summary['total_injuries']}")
    print(f"\nBy Source: {summary['by_source']}")
    print(f"\nBy Type: {summary['by_type']}")
    print(f"\nBy Location: {summary['by_location']}")
    print(f"\nBy Severity: {summary['by_severity']}")
    print(f"\nBy Data Quality: {summary['by_data_quality']}")
    print(f"\nConfidence Distribution: {summary['confidence_distribution']}")

if __name__ == "__main__":
    main()
