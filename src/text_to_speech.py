"""
Text-to-Speech Service Module
Enhanced TTS service with multiple engines and voice options
"""
import nltk
import torch
import warnings
import numpy as np
import sounddevice as sd
from typing import Tuple, Optional
from transformers import AutoProcessor, BarkModel
from rich.console import Console

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    """Enhanced Text-to-Speech service with multiple engine support"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the TTS service
        
        Args:
            device: Device to use for TTS model (cuda/cpu)
        """
        self.device = device
        self.console = Console()
        self.processor = None
        self.model = None
        self._initialize_bark_model()
        
    def _initialize_bark_model(self):
        """Initialize Bark TTS model"""
        try:
            self.processor = AutoProcessor.from_pretrained("suno/bark-small")
            self.model = BarkModel.from_pretrained("suno/bark-small")
            self.model.to(self.device)
            self.console.print(f"🔊 TTS Model loaded on {self.device}")
        except Exception as e:
            self.console.print(f"[red]TTS Model initialization error: {e}")
            
    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1") -> Tuple[int, np.ndarray]:
        """
        Synthesize speech from text using Bark TTS
        
        Args:
            text: Text to synthesize
            voice_preset: Voice preset to use
            
        Returns:
            Tuple of (sample_rate, audio_array)
        """
        try:
            if not self.model or not self.processor:
                raise Exception("TTS model not initialized")
                
            inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                audio_array = self.model.generate(**inputs, pad_token_id=10000)

            audio_array = audio_array.cpu().numpy().squeeze()
            sample_rate = self.model.generation_config.sample_rate
            
            return sample_rate, audio_array
            
        except Exception as e:
            self.console.print(f"[red]TTS Synthesis error: {e}")
            # Return silent audio as fallback
            return 24000, np.zeros(1000)

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_1") -> Tuple[int, np.ndarray]:
        """
        Synthesize long-form text by breaking into sentences
        
        Args:
            text: Long text to synthesize
            voice_preset: Voice preset to use
            
        Returns:
            Tuple of (sample_rate, audio_array)
        """
        try:
            pieces = []
            sentences = nltk.sent_tokenize(text)
            silence = np.zeros(int(0.25 * 24000))  # Default sample rate

            for sent in sentences:
                if len(sent.strip()) > 0:  # Skip empty sentences
                    sample_rate, audio_array = self.synthesize(sent, voice_preset)
                    pieces.extend([audio_array, silence.copy()])

            if pieces:
                return 24000, np.concatenate(pieces)  # Default sample rate
            else:
                return 24000, np.zeros(1000)
                
        except Exception as e:
            self.console.print(f"[red]Long-form TTS error: {e}")
            return 24000, np.zeros(1000)
    
    def play_audio(self, sample_rate: int, audio_array: np.ndarray) -> None:
        """
        Play audio through speakers
        
        Args:
            sample_rate: Audio sample rate
            audio_array: Audio data to play
        """
        try:
            sd.play(audio_array, sample_rate)
            sd.wait()
        except Exception as e:
            self.console.print(f"[red]Audio playback error: {e}")
    
    def synthesize_and_play(self, text: str, voice_preset: str = "v2/en_speaker_1") -> bool:
        """
        Synthesize text and play immediately
        
        Args:
            text: Text to synthesize and play
            voice_preset: Voice preset to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if len(text.strip()) == 0:
                return False
                
            # Choose synthesis method based on text length
            if len(text) > 100:  # Long text
                sample_rate, audio_array = self.long_form_synthesize(text, voice_preset)
            else:  # Short text
                sample_rate, audio_array = self.synthesize(text, voice_preset)
            
            self.play_audio(sample_rate, audio_array)
            return True
            
        except Exception as e:
            self.console.print(f"[red]TTS synthesis and play error: {e}")
            return False
    
    def get_available_voices(self) -> list:
        """Get list of available voice presets"""
        return [
            "v2/en_speaker_0",
            "v2/en_speaker_1", 
            "v2/en_speaker_2",
            "v2/en_speaker_3",
            "v2/en_speaker_4",
            "v2/en_speaker_5",
            "v2/en_speaker_6",
            "v2/en_speaker_7",
            "v2/en_speaker_8",
            "v2/en_speaker_9"
        ]
    
    def format_vehicle_response(self, response_type: str, data: dict) -> str:
        """
        Format vehicle responses for natural speech
        
        Args:
            response_type: Type of response (speed, location, eta, etc.)
            data: Data to include in response
            
        Returns:
            Formatted text for TTS
        """
        try:
            if response_type == "speed":
                speed = data.get("speed", 0)
                if speed == 0:
                    return "The vehicle is currently stopped."
                else:
                    return f"Current speed is {speed:.1f} kilometers per hour."
            
            elif response_type == "location":
                location = data.get("location", "unknown")
                return f"We are currently at {location}."
            
            elif response_type == "eta":
                eta_text = data.get("eta_text", "unknown")
                destination = data.get("destination", "destination")
                return f"Estimated arrival time to {destination} is {eta_text}."
            
            elif response_type == "destination_set":
                destination = data.get("destination", "destination")
                return f"Destination set to {destination}. Beginning route navigation."
            
            elif response_type == "emergency_stop":
                return "Emergency stop activated. All vehicle systems halted for safety."
            
            elif response_type == "stop":
                return "Vehicle stopped as requested."
            
            elif response_type == "speed_change":
                new_speed = data.get("new_speed", 0)
                return f"Speed adjusted to {new_speed:.1f} kilometers per hour."
            
            elif response_type == "error":
                message = data.get("message", "An error occurred")
                return f"I'm sorry, {message.lower()}"
            
            elif response_type == "greeting":
                return "Hello! I'm your AI assistant. How can I help you today?"
            
            else:
                return data.get("message", "Request processed.")
                
        except Exception as e:
            self.console.print(f"[red]Response formatting error: {e}")
            return "I'm sorry, I couldn't process that request."
