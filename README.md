# AI Call Agent for Autonomous Campus Shuttle 🚐🤖

An intelligent voice-controlled assistant for autonomous campus shuttle systems, providing natural language interaction for passenger convenience and safety.

## 🏗️ System Architecture

```
+---------------------------+            +------------------------------+
|   🎤 Voice Input (Mic)    |            |  💻 Touchscreen / UI Input    |
+---------------------------+            +------------------------------+
             |                                         |
             v                                         v
+---------------------------+      +------------------------------+
|  🧠 Speech-to-Text Engine  |<----->|         UI Command Parser    |
|   (e.g. Whisper, Vosk)    |      |  (Buttons like "Stop", etc.)  |
+---------------------------+      +------------------------------+
             |                                         |
             +---------------+-------------------------+
                             |
                             v
             +-----------------------------------+
             |  🧠 NLP + Intent Recognition Engine |
             |   (LLM/Ollama/GPT-4-mini/Rules)    |
             +-----------------------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
+---------------------------+     +-----------------------------+
|  🔌 Vehicle Control API     |     | 🧭 Navigation / ETA Engine   |
|  (ROS / REST / CAN Bridge) |     | (Campus Map Graph / GPS)    |
+---------------------------+     +-----------------------------+
              |                             |
              +-------------+---------------+
                            |
                            v
              +------------------------------+
              |  🔊 Response Generator         |
              |  (TTS engine + Visual UI)     |
              +------------------------------+
                            |
          +----------------+------------------+
          |                                   |
          v                                   v
+-------------------------+      +---------------------------+
| 🔊 Speaker Output (TTS)  |      |  📺 Display Output (UI)     |
+-------------------------+      +---------------------------+
```

## 🚀 Features

### 🎤 Voice Interface
- **Offline Speech Recognition**: Uses Whisper for real-time voice-to-text conversion
- **Natural Language Processing**: Understands passenger commands and intents
- **Voice Feedback**: Provides audio responses using advanced TTS

### 🚗 Vehicle Control
- **Speed Management**: Control vehicle speed with voice commands
- **Emergency Stop**: Immediate safety halt functionality
- **Destination Setting**: Voice-controlled navigation to campus locations

### 🗺️ Navigation System
- **Campus Mapping**: Pre-defined routes for campus shuttle service
- **ETA Calculations**: Real-time arrival time estimates
- **Route Optimization**: Efficient path planning between locations

### 🎯 Intent Recognition
- **Command Understanding**: Recognizes passenger intents from natural speech
- **Safety Prioritization**: Emergency commands get highest priority
- **Contextual Responses**: Intelligent responses based on vehicle state

## 📁 Project Structure

```
AI-Call-Agent/
├── src/                           # Main source code
│   ├── __init__.py
│   ├── main.py                    # Main application entry point
│   ├── config.py                  # Configuration settings
│   ├── speech_to_text.py          # Speech recognition module
│   ├── intent_recognition.py      # NLP and intent processing
│   ├── vehicle_control.py         # Vehicle control interface
│   ├── navigation.py              # Navigation and routing
│   ├── text_to_speech.py          # Speech synthesis
│   └── ui_parser.py               # UI command processing
├── app.py                         # Legacy entry point
├── assistant.py                   # Legacy assistant module
├── tts.py                         # Legacy TTS module
├── pyproject.toml                 # Poetry configuration
├── requirements.txt               # Pip requirements
├── Makefile                       # Build automation
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for faster TTS)
- Microphone and speakers
- Audio libraries (PortAudio/ALSA)

### Using Poetry (Recommended)
```bash
# Clone the repository
git clone https://github.com/ankitrajsh/AI-Call-Agent-.git
cd AI-Call-Agent-

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip
```bash
# Clone the repository
git clone https://github.com/ankitrajsh/AI-Call-Agent-.git
cd AI-Call-Agent-

# Create virtual environment
python -m venv ai-call-agent
source ai-call-agent/bin/activate  # On Windows: ai-call-agent\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Interactive Mode
```bash
# Start the AI Call Agent
python src/main.py

# Or use legacy entry points
python app.py
python assistant.py
```

### Available Commands

#### Voice Commands
- **"Take me to Main Gate"** - Set destination
- **"Stop the vehicle"** - Normal stop
- **"Emergency stop"** - Immediate halt
- **"Slow down"** - Reduce speed
- **"Speed up"** - Increase speed
- **"What's our current speed?"** - Get speed info
- **"Where are we?"** - Get location
- **"How long until we arrive?"** - Get ETA

#### UI Commands
```bash
# In interactive mode, use these commands:
v                    # Voice input
status              # Show vehicle status
ui emergency_stop   # Emergency stop via UI
ui stop             # Normal stop
ui set_destination Main Gate  # Set destination
h                   # Help
q                   # Quit
```

---

**Made with ❤️ for safer and smarter autonomous transportation**

## Usage

1. **Start the Assistant**: Run the script to start the assistant.
2. **Record Audio**: Press `Enter` to start recording your voice. Press `Enter` again to stop recording.
3. **Transcription**: The assistant transcribes your voice to text.
4. **AI Response**: The transcribed text is sent to an AI language model (Llama-2) to generate a response.
5. **Text-to-Speech**: The AI response is then converted to speech and played back to you.
6. **Repeat**: You can repeat the process by pressing `Enter` again.

## How It Works

1. **Recording Audio**: The assistant captures audio input using the `sounddevice` library in a separate thread.
2. **Transcription**: The audio is transcribed to text using the `Whisper` model from OpenAI.
3. **LLM Response**: The transcribed text is passed to a language model (Llama-2) using LangChain to generate a response.
4. **Playback**: The response is converted to speech using a text-to-speech service and played back to the user.

## Code Overview

- `record_audio(stop_event, data_queue)`: Captures audio data and adds it to a queue.
- `transcribe(audio_np)`: Transcribes audio data to text using Whisper.
- `get_llm_response(text)`: Generates an AI response using the Llama-2 language model.
- `play_audio(sample_rate, audio_array)`: Plays audio data using `sounddevice`.
- `main`: Handles the main loop for recording, transcribing, generating responses, and playing audio.

## Example

```bash
python assistant.py
```

Start the assistant and follow the prompts to record your voice and interact with the AI.

## Acknowledgments

- **Whisper**: Used for speech-to-text conversion.
- **LangChain**: Used to interact with the Llama-2 language model.
- **TextToSpeechService**: Custom service for text-to-speech conversion.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
