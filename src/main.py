"""
Main AI Call Agent System
Integrates all modules according to the specified architecture
"""
import time
import threading
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import our modules
from src.speech_to_text import SpeechToTextEngine
from .intent_recognition import IntentRecognitionEngine, Intent
from .llama_intent_recognition import LlamaIntentRecognitionEngine, create_intent_engine
from .vehicle_control import VehicleControlAPI
from .navigation import NavigationEngine
from .text_to_speech import TextToSpeechEngine
from .ui_parser import UIParser
from .config import Config


class AICallAgent:
    """Main AI Call Agent system integrating all components"""
    
    def __init__(self, llama_model_path=None, use_llama=True):
        """
        Initialize the AI Call Agent
        
        Args:
            llama_model_path: Path to fine-tuned Llama model
            use_llama: Whether to use Llama model for enhanced intent recognition
        """
        self.console = Console()
        self.llama_model_path = llama_model_path
        self.use_llama = use_llama
        self._initialize_components()
        self.running = False
        
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            self.console.print("🚀 Initializing AI Call Agent System...")
            
            # Initialize core components
            self.stt_engine = SpeechToTextEngine()
            
            # Initialize enhanced intent recognition
            self.intent_engine = create_intent_engine(self.llama_model_path, self.use_llama)
            
            self.vehicle_control = VehicleControlAPI()
            self.navigation = NavigationEngine()
            
            # Initialize optional components
            try:
                from .text_to_speech import TextToSpeechEngine
                self.tts_service = TextToSpeechEngine()
            except ImportError:
                self.console.print("[yellow]TTS service not available - using basic output")
                self.tts_service = None
                
            try:
                from .ui_parser import UIParser
                self.ui_parser = UIParser()
            except ImportError:
                self.console.print("[yellow]UI parser not available - using basic UI")
                self.ui_parser = None
            
            self.console.print("✅ All components initialized successfully!")
            
        except Exception as e:
            self.console.print(f"[red]❌ Initialization error: {e}")
            raise
    
    def process_voice_input(self) -> bool:
        """
        Process voice input through the complete pipeline
        
        Returns:
            True if processing was successful
        """
        try:
            # 1. Speech-to-Text
            self.console.print("\n🎤 [cyan]Starting voice input...[/cyan]")
            text, success = self.stt_engine.start_voice_recording()
            
            if not success or not text:
                self.console.print("[red]No valid speech detected.")
                return False
            
            self.console.print(f"🗣️  [yellow]You said:[/yellow] {text}")
            
            # 2. Intent Recognition
            intent, entities = self.intent_engine.recognize_intent(text)
            self.console.print(f"🧠 [cyan]Recognized intent:[/cyan] {intent.value}")
            
            if entities:
                self.console.print(f"📋 [cyan]Extracted entities:[/cyan] {entities}")
            
            # 3. Process Intent
            response = self._process_intent(intent, entities, text)
            
            # 4. Generate Response
            if response:
                self._generate_response(response)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Voice processing error: {e}")
            return False
    
    def process_ui_command(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process UI command input
        
        Args:
            command: UI command string
            parameters: Optional command parameters
            
        Returns:
            True if processing was successful
        """
        try:
            # Parse UI command
            parsed_command = self.ui_parser.parse_ui_command(command, parameters)
            
            if not parsed_command["success"]:
                self.console.print(f"[red]Invalid UI command: {parsed_command['error']}")
                return False
            
            ui_command = parsed_command["command"]
            cmd_params = parsed_command["parameters"]
            
            self.console.print(f"📱 [cyan]UI Command:[/cyan] {ui_command.value}")
            
            # Convert UI command to intent and process
            intent = self._ui_command_to_intent(ui_command)
            response = self._process_intent(intent, cmd_params, f"UI: {command}")
            
            if response:
                self._generate_response(response)
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]UI command processing error: {e}")
            return False
    
    def _generate_response(self, intent: Intent, entities: Optional[Dict[str, Any]], 
                          context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate natural language response using enhanced intent recognition
        
        Args:
            intent: Recognized intent
            entities: Extracted entities
            context: Additional context (vehicle status, etc.)
            
        Returns:
            Natural language response
        """
        # Get current vehicle context
        current_context = {
            "speed": f"{self.vehicle_control.get_speed():.1f}",
            "location": self.vehicle_control.get_location(),
            "status": self.vehicle_control.get_vehicle_status()
        }
        
        # Merge with provided context
        if context:
            current_context.update(context)
        
        # Use Llama model for response generation if available
        if hasattr(self.intent_engine, 'generate_response'):
            try:
                return self.intent_engine.generate_response(intent, entities, current_context)
            except Exception as e:
                self.console.print(f"[yellow]Fallback to basic response: {e}")
        
        # Fallback to basic responses
        return self._generate_basic_response(intent, entities, current_context)
    
    def _generate_basic_response(self, intent: Intent, entities: Optional[Dict[str, Any]], 
                               context: Dict[str, Any]) -> str:
        """Generate basic rule-based responses"""
        if intent == Intent.EMERGENCY_STOP:
            return "Emergency stop activated. Vehicle stopped immediately for safety."
        elif intent == Intent.STOP_VEHICLE:
            return "Bringing the vehicle to a controlled stop."
        elif intent == Intent.GET_SPEED:
            return f"Current speed is {context.get('speed', 'unknown')} km/h."
        elif intent == Intent.GET_LOCATION:
            return f"We are currently at {context.get('location', 'unknown location')}."
        elif intent == Intent.SET_DESTINATION:
            dest = entities.get('destination', 'unknown') if entities else 'unknown'
            return f"Setting destination to {dest}."
        elif intent == Intent.GET_ETA:
            eta = context.get('eta', 'calculating')
            return f"Estimated arrival time is {eta}."
        elif intent == Intent.SLOW_DOWN:
            return f"Reducing speed. Current speed: {context.get('speed', 'unknown')} km/h."
        elif intent == Intent.SPEED_UP:
            return f"Increasing speed. Current speed: {context.get('speed', 'unknown')} km/h."
        elif intent == Intent.GREETING:
            return "Hello! I'm your autonomous vehicle assistant. How can I help you?"
        else:
            return "I didn't understand that command. Please try again."

    def _ui_command_to_intent(self, ui_command) -> Intent:
        """Convert UI command to intent"""
        mapping = {
            UICommand.EMERGENCY_STOP: Intent.EMERGENCY_STOP,
            UICommand.STOP: Intent.STOP_VEHICLE,
            UICommand.SLOW_DOWN: Intent.SLOW_DOWN,
            UICommand.SPEED_UP: Intent.SPEED_UP,
            UICommand.SET_DESTINATION: Intent.SET_DESTINATION,
            UICommand.GET_STATUS: Intent.GET_SPEED,  # Default to speed status
            UICommand.GET_ETA: Intent.GET_ETA,
            UICommand.VOICE_INPUT: Intent.UNKNOWN
        }
        return mapping.get(ui_command, Intent.UNKNOWN)
    
    def _process_intent(self, intent: Intent, entities: Optional[Dict[str, Any]], original_text: str) -> Optional[Dict[str, Any]]:
        """
        Process recognized intent and execute appropriate actions
        
        Args:
            intent: Recognized intent
            entities: Extracted entities
            original_text: Original input text
            
        Returns:
            Response dictionary
        """
        try:
            if intent == Intent.EMERGENCY_STOP:
                result = self.vehicle_control.emergency_stop()
                return {
                    "type": "emergency_stop",
                    "data": result,
                    "intent": intent.value
                }
            
            elif intent == Intent.STOP_VEHICLE:
                result = self.vehicle_control.stop_vehicle()
                return {
                    "type": "stop",
                    "data": result,
                    "intent": intent.value
                }
            
            elif intent == Intent.SLOW_DOWN:
                result = self.vehicle_control.adjust_speed(-5.0)  # Reduce by 5 km/h
                return {
                    "type": "speed_change",
                    "data": result,
                    "intent": intent.value
                }
            
            elif intent == Intent.SPEED_UP:
                result = self.vehicle_control.adjust_speed(5.0)  # Increase by 5 km/h
                return {
                    "type": "speed_change",
                    "data": result,
                    "intent": intent.value
                }
            
            elif intent == Intent.SET_DESTINATION:
                if entities and "destination" in entities:
                    destination = entities["destination"]
                    
                    # Validate destination
                    if not self.navigation.is_valid_destination(destination):
                        return {
                            "type": "error",
                            "data": {"message": f"Invalid destination: {destination}"},
                            "intent": intent.value
                        }
                    
                    result = self.vehicle_control.set_destination(destination)
                    return {
                        "type": "destination_set",
                        "data": result,
                        "intent": intent.value
                    }
                else:
                    return {
                        "type": "error",
                        "data": {"message": "Please specify a destination"},
                        "intent": intent.value
                    }
            
            elif intent == Intent.GET_SPEED:
                speed = self.vehicle_control.get_speed()
                return {
                    "type": "speed",
                    "data": {"speed": speed},
                    "intent": intent.value
                }
            
            elif intent == Intent.GET_LOCATION:
                location = self.vehicle_control.get_location()
                return {
                    "type": "location",
                    "data": {"location": location},
                    "intent": intent.value
                }
            
            elif intent == Intent.GET_ETA:
                current_location = self.vehicle_control.get_location()
                status = self.vehicle_control.get_vehicle_status()
                destination = status.get("destination")
                
                if destination:
                    eta_info = self.navigation.calculate_eta(
                        current_location, 
                        destination, 
                        status["speed"]
                    )
                    
                    if eta_info:
                        return {
                            "type": "eta",
                            "data": eta_info,
                            "intent": intent.value
                        }
                    else:
                        return {
                            "type": "error",
                            "data": {"message": "Could not calculate ETA"},
                            "intent": intent.value
                        }
                else:
                    return {
                        "type": "error",
                        "data": {"message": "No destination set"},
                        "intent": intent.value
                    }
            
            elif intent == Intent.GREETING:
                return {
                    "type": "greeting",
                    "data": {"message": "Hello! I'm your AI assistant."},
                    "intent": intent.value
                }
            
            else:
                return {
                    "type": "error",
                    "data": {"message": "I didn't understand that command"},
                    "intent": intent.value
                }
                
        except Exception as e:
            self.console.print(f"[red]Intent processing error: {e}")
            return {
                "type": "error",
                "data": {"message": f"System error: {str(e)}"},
                "intent": intent.value if intent else "unknown"
            }
    
    def _generate_response(self, response: Dict[str, Any]):
        """
        Generate and output response using enhanced LLM capabilities
        
        Args:
            response: Response dictionary from intent processing
        """
        try:
            response_type = response["type"]
            data = response["data"]
            intent_str = response.get("intent", "unknown")
            
            # Convert intent string back to Intent enum
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN
            
            # Generate enhanced response text using Llama if available
            if hasattr(self.intent_engine, 'generate_response'):
                try:
                    # Prepare context from response data
                    context = {
                        "response_type": response_type,
                        **data
                    }
                    response_text = self.intent_engine.generate_response(intent, None, context)
                except Exception as e:
                    self.console.print(f"[yellow]Using fallback response: {e}")
                    response_text = self._format_basic_response(response_type, data)
            else:
                response_text = self._format_basic_response(response_type, data)
            
            # Display response
            self.console.print(f"🤖 [green]Assistant:[/green] {response_text}")
            
            # Synthesize and play audio response if TTS is available
            if self.tts_service:
                try:
                    with self.console.status("🔊 Generating speech...", spinner="earth"):
                        success = self.tts_service.synthesize_and_play(response_text)
                    
                    if not success:
                        self.console.print("[red]Failed to generate speech output")
                except Exception as e:
                    self.console.print(f"[yellow]TTS error: {e}")
                    
        except Exception as e:
            self.console.print(f"[red]Response generation error: {e}")
    
    def _format_basic_response(self, response_type: str, data: Dict[str, Any]) -> str:
        """Format basic response for fallback"""
        if response_type == "emergency_stop":
            return "Emergency stop activated immediately for safety."
        elif response_type == "stop":
            return "Vehicle stopped successfully."
        elif response_type == "speed_change":
            return f"Speed adjusted. Current speed: {data.get('speed', 'unknown')} km/h."
        elif response_type == "destination_set":
            return f"Destination set to {data.get('destination', 'unknown')}."
        elif response_type == "speed":
            return f"Current speed is {data.get('speed', 'unknown')} km/h."
        elif response_type == "location":
            return f"Current location: {data.get('location', 'unknown')}."
        elif response_type == "eta":
            return f"Estimated arrival: {data.get('eta', 'calculating')}."
        elif response_type == "error":
            return f"Error: {data.get('message', 'Unknown error occurred')}."
        else:
            return "Command processed successfully."
    
    def display_status(self):
        """Display current system status"""
        try:
            status = self.vehicle_control.get_vehicle_status()
            
            # Create status table
            table = Table(title="🚐 Vehicle Status", show_header=True, header_style="bold cyan")
            table.add_column("Property", style="yellow")
            table.add_column("Value", style="green")
            
            table.add_row("Speed", f"{status['speed']:.1f} km/h")
            table.add_row("Location", status['location'])
            table.add_row("Destination", status['destination'] or "Not set")
            table.add_row("State", status['state'].title())
            table.add_row("Emergency Stop", "Active" if status['emergency_stop'] else "Inactive")
            
            # Calculate ETA if destination is set
            if status['destination']:
                eta_info = self.navigation.calculate_eta(
                    status['location'], 
                    status['destination'], 
                    status['speed']
                )
                if eta_info:
                    table.add_row("ETA", eta_info['eta_text'])
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Status display error: {e}")
    
    def interactive_mode(self):
        """Run the system in interactive mode"""
        try:
            self.running = True
            
            # Welcome message
            welcome_panel = Panel(
                "🚐 AI Call Agent for Autonomous Campus Shuttle\n\n"
                "Available commands:\n"
                "• 'v' or 'voice' - Voice input\n"
                "• 's' or 'status' - Show vehicle status\n"
                "• 'ui <command>' - UI command (emergency_stop, stop, etc.)\n"
                "• 'q' or 'quit' - Exit system\n"
                "• 'h' or 'help' - Show this help",
                title="Welcome",
                title_align="center",
                border_style="cyan"
            )
            self.console.print(welcome_panel)
            
            while self.running:
                try:
                    # Get user input
                    user_input = self.console.input("\n[cyan]Command:[/cyan] ").strip().lower()
                    
                    if user_input in ['q', 'quit', 'exit']:
                        break
                    
                    elif user_input in ['h', 'help']:
                        self.console.print(welcome_panel)
                    
                    elif user_input in ['v', 'voice']:
                        self.process_voice_input()
                    
                    elif user_input in ['s', 'status']:
                        self.display_status()
                    
                    elif user_input.startswith('ui '):
                        # Parse UI command
                        ui_cmd = user_input[3:].strip()
                        if ' ' in ui_cmd:
                            cmd, params = ui_cmd.split(' ', 1)
                            # Simple parameter parsing for destination
                            if cmd == 'set_destination':
                                self.process_ui_command(cmd, {"destination": params.title()})
                            else:
                                self.process_ui_command(cmd)
                        else:
                            self.process_ui_command(ui_cmd)
                    
                    elif user_input:
                        self.console.print("[yellow]Unknown command. Type 'h' for help.")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}")
            
            self.console.print("\n[blue]👋 AI Call Agent shutting down...")
            
        except Exception as e:
            self.console.print(f"[red]Interactive mode error: {e}")
        finally:
            self.running = False


def main():
    """Main entry point"""
    try:
        agent = AICallAgent()
        agent.interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")


if __name__ == "__main__":
    main()
