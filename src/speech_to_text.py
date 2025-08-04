"""
Speech-to-Text Engine Module
Handles voice input processing using Whisper for offline transcription
"""
import numpy as np
import whisper
import threading
import time
import sounddevice as sd
from queue import Queue
from typing import Optional
from rich.console import Console


class SpeechToTextEngine:
    """Handles speech-to-text conversion using Whisper model"""
    
    def __init__(self, model_size: str = "base.en"):
        """
        Initialize the Speech-to-Text engine
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.console = Console()
        self.model = whisper.load_model(model_size)
        self.is_recording = False
        
    def record_audio(self, stop_event: threading.Event, data_queue: Queue) -> None:
        """
        Captures audio data from microphone
        
        Args:
            stop_event: Event to signal recording stop
            data_queue: Queue to store audio data
        """
        def callback(indata, frames, time, status):
            if status:
                self.console.print(f"[red]Audio Error: {status}")
            data_queue.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=16000, 
                dtype="int16", 
                channels=1, 
                callback=callback
            ):
                self.is_recording = True
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            self.console.print(f"[red]Recording Error: {e}")
        finally:
            self.is_recording = False
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text using Whisper
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Transcribed text string
        """
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if audio_np.size == 0:
                return ""
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio_np, fp16=False)
            text = result["text"].strip()
            
            return text
        except Exception as e:
            self.console.print(f"[red]Transcription Error: {e}")
            return ""
    
    def start_voice_recording(self) -> tuple[str, bool]:
        """
        Start voice recording session
        
        Returns:
            Tuple of (transcribed_text, success_flag)
        """
        data_queue = Queue()
        stop_event = threading.Event()
        
        # Start recording thread
        recording_thread = threading.Thread(
            target=self.record_audio,
            args=(stop_event, data_queue)
        )
        recording_thread.start()
        
        # Wait for user input to stop recording
        self.console.print("[cyan]🎤 Recording... Press Enter to stop.")
        input()
        
        # Stop recording
        stop_event.set()
        recording_thread.join()
        
        # Process recorded audio
        audio_data = b"".join(list(data_queue.queue))
        
        if len(audio_data) > 0:
            with self.console.status("🧠 Transcribing...", spinner="earth"):
                text = self.transcribe_audio(audio_data)
            return text, True
        else:
            self.console.print("[red]No audio recorded. Please check microphone.")
            return "", False
