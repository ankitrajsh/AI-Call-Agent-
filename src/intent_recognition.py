"""
Intent Recognition Engine Module
Handles NLP and intent recognition for autonomous vehicle commands
"""
import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from rich.console import Console


class Intent(Enum):
    """Defined intents for the AI Call Agent"""
    GET_SPEED = "get_speed"
    SET_DESTINATION = "set_destination"
    STOP_VEHICLE = "stop_vehicle"
    EMERGENCY_STOP = "emergency_stop"
    SLOW_DOWN = "slow_down"
    SPEED_UP = "speed_up"
    GET_LOCATION = "get_location"
    GET_ETA = "get_eta"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class IntentRecognitionEngine:
    """Handles intent recognition from user input using rule-based NLU"""
    
    def __init__(self):
        self.console = Console()
        self.intent_patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[Intent, List[str]]:
        """
        Initialize intent recognition patterns
        
        Returns:
            Dictionary mapping intents to regex patterns
        """
        return {
            Intent.GET_SPEED: [
                r"what.{0,10}speed",
                r"how fast",
                r"current speed",
                r"speed.*now"
            ],
            Intent.SET_DESTINATION: [
                r"go to (.+)",
                r"take me to (.+)",
                r"destination (.+)",
                r"drive to (.+)",
                r"navigate to (.+)"
            ],
            Intent.STOP_VEHICLE: [
                r"stop",
                r"halt",
                r"pause",
                r"brake"
            ],
            Intent.EMERGENCY_STOP: [
                r"emergency.*stop",
                r"stop.*emergency",
                r"urgent.*stop",
                r"stop.*now",
                r"emergency.*brake"
            ],
            Intent.SLOW_DOWN: [
                r"slow.*down",
                r"reduce.*speed",
                r"drive.*slower",
                r"go.*slower"
            ],
            Intent.SPEED_UP: [
                r"speed.*up",
                r"go.*faster",
                r"increase.*speed",
                r"drive.*faster"
            ],
            Intent.GET_LOCATION: [
                r"where.*am.*i",
                r"current.*location",
                r"where.*are.*we",
                r"location.*now"
            ],
            Intent.GET_ETA: [
                r"how.*long",
                r"eta",
                r"arrival.*time",
                r"when.*arrive",
                r"time.*destination"
            ],
            Intent.GREETING: [
                r"hello",
                r"hi",
                r"hey",
                r"good.*morning",
                r"good.*afternoon",
                r"good.*evening"
            ]
        }
    
    def recognize_intent(self, text: str) -> Tuple[Intent, Optional[Dict[str, str]]]:
        """
        Recognize intent from user input text
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (recognized_intent, extracted_entities)
        """
        text_lower = text.lower().strip()
        
        # Check for emergency stop first (highest priority)
        if self._match_patterns(text_lower, self.intent_patterns[Intent.EMERGENCY_STOP]):
            return Intent.EMERGENCY_STOP, None
            
        # Check other intents
        for intent, patterns in self.intent_patterns.items():
            if intent == Intent.EMERGENCY_STOP:
                continue  # Already checked
                
            match = self._match_patterns_with_groups(text_lower, patterns)
            if match:
                entities = self._extract_entities(intent, match, text_lower)
                return intent, entities
        
        return Intent.UNKNOWN, None
    
    def _match_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _match_patterns_with_groups(self, text: str, patterns: List[str]) -> Optional[re.Match]:
        """Check patterns and return match object with groups"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match
        return None
    
    def _extract_entities(self, intent: Intent, match: re.Match, text: str) -> Optional[Dict[str, str]]:
        """
        Extract entities based on intent type
        
        Args:
            intent: Recognized intent
            match: Regex match object
            text: Original text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        if intent == Intent.SET_DESTINATION:
            if match.groups():
                destination = match.group(1).strip()
                # Clean up destination
                destination = self._clean_destination(destination)
                entities["destination"] = destination
        
        return entities if entities else None
    
    def _clean_destination(self, destination: str) -> str:
        """Clean and standardize destination names"""
        destination = destination.lower().strip()
        
        # Map common destination aliases
        destination_map = {
            "main gate": "Main Gate",
            "gate": "Main Gate",
            "entrance": "Main Gate",
            "hostel": "Hostel Circle",
            "hostel circle": "Hostel Circle",
            "hostels": "Hostel Circle",
            "dorms": "Hostel Circle",
            "dormitory": "Hostel Circle"
        }
        
        return destination_map.get(destination, destination.title())
    
    def get_intent_confidence(self, text: str, intent: Intent) -> float:
        """
        Get confidence score for a specific intent
        
        Args:
            text: Input text
            intent: Intent to check
            
        Returns:
            Confidence score between 0 and 1
        """
        if intent not in self.intent_patterns:
            return 0.0
            
        text_lower = text.lower()
        patterns = self.intent_patterns[intent]
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        return matches / len(patterns) if patterns else 0.0
