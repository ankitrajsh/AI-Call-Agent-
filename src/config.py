"""
Configuration file for AI Call Agent system
"""

# System Configuration
SYSTEM_CONFIG = {
    "name": "AI Call Agent",
    "version": "1.0.0",
    "description": "Autonomous Campus Shuttle AI Assistant"
}

# Speech-to-Text Configuration
STT_CONFIG = {
    "model_size": "base.en",  # Whisper model size
    "sample_rate": 16000,
    "channels": 1,
    "dtype": "int16"
}

# Text-to-Speech Configuration  
TTS_CONFIG = {
    "model_name": "suno/bark-small",
    "voice_preset": "v2/en_speaker_1",
    "device": "auto",  # auto, cuda, cpu
    "sample_rate": 24000
}

# Vehicle Configuration
VEHICLE_CONFIG = {
    "max_speed": 25.0,  # km/h
    "default_speed": 15.0,  # km/h
    "speed_increment": 5.0,  # km/h
    "emergency_stop_enabled": True
}

# Navigation Configuration
NAVIGATION_CONFIG = {
    "campus_locations": [
        "Main Gate",
        "Hostel Circle", 
        "Academic Block",
        "Library",
        "Cafeteria"
    ],
    "default_routes": {
        "Main Gate": {
            "Hostel Circle": 1200,  # meters
            "Academic Block": 800,
            "Library": 600
        },
        "Hostel Circle": {
            "Main Gate": 1200,
            "Academic Block": 500,
            "Cafeteria": 300
        }
    }
}

# Intent Recognition Configuration
INTENT_CONFIG = {
    "confidence_threshold": 0.7,
    "emergency_keywords": [
        "emergency", "urgent", "stop now", "help"
    ],
    "destination_aliases": {
        "gate": "Main Gate",
        "entrance": "Main Gate", 
        "hostel": "Hostel Circle",
        "dorms": "Hostel Circle",
        "academic": "Academic Block",
        "classes": "Academic Block"
    }
}

# UI Configuration
UI_CONFIG = {
    "theme": "dark",
    "emergency_button_color": "red",
    "control_button_color": "blue",
    "status_refresh_rate": 1.0,  # seconds
    "voice_button_enabled": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "ai_call_agent.log",
    "max_file_size": "10MB",
    "backup_count": 5
}
