"""
UI Command Parser Module
Handles touchscreen and button input processing
"""
from typing import Dict, Optional, Any
from enum import Enum
from rich.console import Console


class UICommand(Enum):
    """UI command types"""
    EMERGENCY_STOP = "emergency_stop"
    STOP = "stop"
    SLOW_DOWN = "slow_down"
    SPEED_UP = "speed_up"
    SET_DESTINATION = "set_destination"
    GET_STATUS = "get_status"
    GET_ETA = "get_eta"
    VOICE_INPUT = "voice_input"


class UICommandParser:
    """Handles UI command parsing and processing"""
    
    def __init__(self):
        self.console = Console()
        self.ui_commands = self._initialize_ui_commands()
        
    def _initialize_ui_commands(self) -> Dict[str, UICommand]:
        """Initialize UI command mappings"""
        return {
            "emergency_stop": UICommand.EMERGENCY_STOP,
            "stop": UICommand.STOP,
            "slow_down": UICommand.SLOW_DOWN,
            "speed_up": UICommand.SPEED_UP,
            "set_destination": UICommand.SET_DESTINATION,
            "status": UICommand.GET_STATUS,
            "eta": UICommand.GET_ETA,
            "voice": UICommand.VOICE_INPUT
        }
    
    def parse_ui_command(self, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse UI command input
        
        Args:
            command: Command string
            parameters: Optional command parameters
            
        Returns:
            Parsed command dictionary
        """
        try:
            command_lower = command.lower().strip()
            
            if command_lower in self.ui_commands:
                ui_command = self.ui_commands[command_lower]
                
                return {
                    "success": True,
                    "command": ui_command,
                    "parameters": parameters or {},
                    "source": "ui",
                    "raw_command": command
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown UI command: {command}",
                    "raw_command": command
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse UI command: {str(e)}",
                "raw_command": command
            }
    
    def create_ui_interface(self) -> Dict[str, Any]:
        """
        Create UI interface configuration
        
        Returns:
            UI interface configuration
        """
        return {
            "emergency_controls": {
                "emergency_stop": {
                    "label": "🚨 EMERGENCY STOP",
                    "color": "red",
                    "size": "large",
                    "priority": 1
                }
            },
            "vehicle_controls": {
                "stop": {
                    "label": "🛑 Stop",
                    "color": "orange",
                    "size": "medium",
                    "priority": 2
                },
                "slow_down": {
                    "label": "🐌 Slow Down",
                    "color": "yellow",
                    "size": "medium",
                    "priority": 3
                },
                "speed_up": {
                    "label": "🚀 Speed Up",
                    "color": "green",
                    "size": "medium",
                    "priority": 3
                }
            },
            "destination_controls": {
                "main_gate": {
                    "label": "🏫 Main Gate",
                    "command": "set_destination",
                    "parameters": {"destination": "Main Gate"},
                    "color": "blue",
                    "size": "medium"
                },
                "hostel_circle": {
                    "label": "🏠 Hostel Circle",
                    "command": "set_destination",
                    "parameters": {"destination": "Hostel Circle"},
                    "color": "blue",
                    "size": "medium"
                },
                "academic_block": {
                    "label": "🎓 Academic Block",
                    "command": "set_destination",
                    "parameters": {"destination": "Academic Block"},
                    "color": "blue",
                    "size": "medium"
                },
                "library": {
                    "label": "📚 Library",
                    "command": "set_destination",
                    "parameters": {"destination": "Library"},
                    "color": "blue",
                    "size": "medium"
                }
            },
            "info_controls": {
                "status": {
                    "label": "📊 Status",
                    "color": "gray",
                    "size": "small"
                },
                "eta": {
                    "label": "⏰ ETA",
                    "color": "gray",
                    "size": "small"
                }
            },
            "voice_control": {
                "voice_input": {
                    "label": "🎤 Voice Command",
                    "color": "purple",
                    "size": "large",
                    "priority": 1
                }
            }
        }
    
    def validate_command_parameters(self, command: UICommand, parameters: Dict[str, Any]) -> bool:
        """
        Validate command parameters
        
        Args:
            command: UI command
            parameters: Command parameters
            
        Returns:
            True if parameters are valid
        """
        try:
            if command == UICommand.SET_DESTINATION:
                return "destination" in parameters and isinstance(parameters["destination"], str)
            
            # Most commands don't require specific parameters
            return True
            
        except Exception:
            return False
    
    def get_command_help(self, command: str) -> str:
        """
        Get help text for a command
        
        Args:
            command: Command name
            
        Returns:
            Help text string
        """
        help_text = {
            "emergency_stop": "Immediately stops the vehicle for emergency situations",
            "stop": "Brings the vehicle to a controlled stop",
            "slow_down": "Reduces vehicle speed",
            "speed_up": "Increases vehicle speed",
            "set_destination": "Sets the vehicle destination",
            "status": "Shows current vehicle status",
            "eta": "Shows estimated time of arrival",
            "voice": "Activates voice command input"
        }
        
        return help_text.get(command.lower(), "No help available for this command")
    
    def get_available_commands(self) -> list:
        """Get list of available UI commands"""
        return list(self.ui_commands.keys())
