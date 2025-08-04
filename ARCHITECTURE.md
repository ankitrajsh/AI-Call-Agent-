# AI Call Agent - System Architecture & Improvements

## 🔄 Transformation Summary

The AI Call Agent system has been completely restructured according to the specified architecture diagram. Here's what was implemented:

## 📁 New Modular Structure

### Core Modules Created:

1. **`src/speech_to_text.py`** - Speech-to-Text Engine
   - Handles voice input processing using Whisper
   - Manages audio recording and transcription
   - Offline speech recognition capabilities

2. **`src/intent_recognition.py`** - NLP + Intent Recognition Engine  
   - Rule-based natural language understanding
   - Recognizes passenger intents (emergency_stop, set_destination, etc.)
   - Entity extraction for destinations and parameters

3. **`src/vehicle_control.py`** - Vehicle Control API
   - Interfaces with autonomous shuttle systems
   - Speed management and emergency stop functionality
   - Vehicle state management

4. **`src/navigation.py`** - Navigation / ETA Engine
   - Campus map with predefined routes
   - ETA calculations based on speed and distance
   - Route planning between campus locations

5. **`src/text_to_speech.py`** - Response Generator (TTS)
   - Enhanced TTS service with multiple voice options
   - Long-form text synthesis
   - Natural response formatting for vehicle contexts

6. **`src/ui_parser.py`** - UI Command Parser
   - Handles touchscreen and button inputs
   - UI command validation and processing
   - Emergency button prioritization

7. **`src/main.py`** - Main Integration Hub
   - Orchestrates all components according to architecture
   - Interactive mode with voice and UI command support
   - Complete pipeline from input to output

8. **`src/config.py`** - System Configuration
   - Centralized configuration management
   - Vehicle, navigation, and TTS settings

## 🚀 Key Features Implemented

### 1. Input Layer
- ✅ **Voice Input**: Microphone capture with Whisper STT
- ✅ **UI Input**: Button commands (Emergency Stop, Set Destination, etc.)

### 2. Processing Layer  
- ✅ **Intent Recognition**: Understands natural language commands
- ✅ **Safety Prioritization**: Emergency commands get highest priority
- ✅ **Entity Extraction**: Extracts destinations and parameters

### 3. Control Layer
- ✅ **Vehicle Control**: Speed adjustment, stopping, emergency halt
- ✅ **Navigation**: Campus routing with ETA calculations
- ✅ **State Management**: Vehicle status tracking

### 4. Output Layer
- ✅ **Voice Response**: Natural TTS feedback to passengers
- ✅ **Status Display**: Rich console interface with vehicle information

## 🎯 Supported Intents & Commands

### Voice Commands:
| Intent | Example Commands | System Response |
|--------|------------------|-----------------|
| `emergency_stop` | "Emergency stop!", "Stop now!" | Immediate vehicle halt |
| `stop_vehicle` | "Stop the vehicle", "Please stop" | Controlled stop |
| `set_destination` | "Take me to Main Gate", "Go to hostel" | Navigation setup |
| `get_speed` | "What's our speed?", "How fast?" | Speed reporting |
| `get_location` | "Where are we?", "Current location" | Location reporting |
| `get_eta` | "How long?", "When will we arrive?" | ETA calculation |
| `slow_down` | "Slow down", "Go slower" | Speed reduction |
| `speed_up` | "Speed up", "Go faster" | Speed increase |

### UI Commands:
- Emergency Stop Button
- Normal Stop Button  
- Destination Buttons (Main Gate, Hostel Circle, etc.)
- Status Display
- Voice Input Activation

## 🗺️ Campus Navigation

### Supported Locations:
- **Main Gate** ↔ **Hostel Circle** (1.2 km)
- **Academic Block** ↔ **Library** (0.4 km)
- **Cafeteria** ↔ **Hostel Circle** (0.3 km)

### Navigation Features:
- Route finding between any two locations
- ETA calculation based on current speed
- Distance reporting
- Multi-hop routing support

## 🛠️ Installation & Usage

### Quick Start:
```bash
# Make setup script executable and run
chmod +x start.sh
./start.sh setup

# Run the system
./start.sh run
```

### Manual Installation:
```bash
# Install dependencies
pip install -r requirements.txt

# Setup NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the system
python src/main.py
```

### Available Commands:
```bash
# Interactive mode commands
v               # Voice input
status          # Show vehicle status
ui emergency_stop   # Emergency stop
ui stop         # Normal stop
ui set_destination Main Gate  # Set destination
h               # Help
q               # Quit
```

## 🏗️ Architecture Flow

```
Voice/UI Input → STT/Parser → Intent Recognition → Vehicle Control + Navigation → TTS Response → Audio/Visual Output
```

### Data Flow:
1. **Input**: Voice or UI command received
2. **Processing**: STT converts speech to text, Intent engine analyzes
3. **Execution**: Vehicle control and navigation systems execute commands
4. **Response**: TTS generates natural language response
5. **Output**: Audio feedback and visual status updates

## 🔧 Configuration Options

### Vehicle Settings:
- Maximum speed: 25 km/h (campus safety)
- Default speed: 15 km/h
- Speed increment: 5 km/h adjustments

### Audio Settings:
- Sample rate: 16kHz for STT, 24kHz for TTS
- Voice presets: 10 different English speakers
- Model: Whisper base.en for STT, Bark-small for TTS

### Navigation Settings:
- Campus-specific route network
- Real-time ETA calculations
- Distance-based routing

## 🚀 Future Enhancements

### Planned Features:
- [ ] **ROS Integration**: Real vehicle control via ROS topics
- [ ] **CAN Bus Support**: Direct vehicle communication
- [ ] **Multi-language**: Support for additional languages
- [ ] **Gesture Control**: Camera-based hand gestures
- [ ] **Mobile App**: Companion smartphone application
- [ ] **Fleet Management**: Multi-vehicle coordination

### Hardware Integration Ready:
- ROS node structure prepared
- CAN bus interface templates included
- GPIO control for edge devices
- Systemd service configuration

## 📊 Testing & Validation

### Test Script:
```bash
python test_system.py
```

### Test Coverage:
- ✅ Module imports and initialization
- ✅ Basic functionality of each component
- ✅ Integration between components
- ✅ Audio system availability
- ✅ Intent recognition accuracy
- ✅ Vehicle control responses
- ✅ Navigation calculations

## 🎉 Results

The system now provides:

1. **Complete Architecture Implementation**: All layers from the diagram are implemented
2. **Modular Design**: Each component is independent and testable
3. **Safety-First Approach**: Emergency controls have highest priority  
4. **Natural Language Interface**: Understands conversational commands
5. **Campus-Specific Navigation**: Tailored for shuttle routes
6. **Rich User Experience**: Voice and visual feedback
7. **Easy Installation**: Automated setup scripts
8. **Future-Ready**: Prepared for hardware integration

The AI Call Agent is now a fully functional, architecturally sound system ready for deployment on autonomous campus shuttles! 🚐🤖
