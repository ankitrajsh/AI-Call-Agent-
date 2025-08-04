"""
Vehicle Control API Module
Interfaces with autonomous shuttle system via ROS topics, CAN interface, or REST endpoints
"""
import time
import json
from typing import Dict, Optional, Any
from enum import Enum
from rich.console import Console


class VehicleState(Enum):
    """Vehicle operational states"""
    STOPPED = "stopped"
    MOVING = "moving"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE = "maintenance"


class VehicleControlAPI:
    """Handles vehicle control operations"""
    
    def __init__(self):
        self.console = Console()
        self.current_speed = 0.0  # km/h
        self.max_speed = 25.0  # km/h for campus shuttle
        self.current_location = "Main Gate"
        self.destination = None
        self.state = VehicleState.STOPPED
        self.emergency_stop_active = False
        
    def get_vehicle_status(self) -> Dict[str, Any]:
        """
        Get current vehicle status
        
        Returns:
            Dictionary containing vehicle status information
        """
        return {
            "speed": self.current_speed,
            "max_speed": self.max_speed,
            "location": self.current_location,
            "destination": self.destination,
            "state": self.state.value,
            "emergency_stop": self.emergency_stop_active,
            "timestamp": time.time()
        }
    
    def set_destination(self, destination: str) -> Dict[str, Any]:
        """
        Set vehicle destination
        
        Args:
            destination: Target destination
            
        Returns:
            Operation result
        """
        try:
            # Validate destination
            valid_destinations = ["Main Gate", "Hostel Circle"]
            if destination not in valid_destinations:
                return {
                    "success": False,
                    "message": f"Invalid destination. Valid options: {', '.join(valid_destinations)}",
                    "error_code": "INVALID_DESTINATION"
                }
            
            if self.emergency_stop_active:
                return {
                    "success": False,
                    "message": "Cannot set destination during emergency stop",
                    "error_code": "EMERGENCY_STOP_ACTIVE"
                }
            
            self.destination = destination
            self.console.print(f"🧭 Destination set to: {destination}")
            
            return {
                "success": True,
                "message": f"Destination set to {destination}",
                "destination": destination
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to set destination: {str(e)}",
                "error_code": "SYSTEM_ERROR"
            }
    
    def emergency_stop(self) -> Dict[str, Any]:
        """
        Activate emergency stop
        
        Returns:
            Operation result
        """
        try:
            self.emergency_stop_active = True
            self.current_speed = 0.0
            self.state = VehicleState.EMERGENCY_STOP
            
            self.console.print("🚨 EMERGENCY STOP ACTIVATED")
            
            return {
                "success": True,
                "message": "Emergency stop activated",
                "speed": self.current_speed,
                "state": self.state.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to activate emergency stop: {str(e)}",
                "error_code": "SYSTEM_ERROR"
            }
    
    def stop_vehicle(self) -> Dict[str, Any]:
        """
        Normal vehicle stop
        
        Returns:
            Operation result
        """
        try:
            if self.emergency_stop_active:
                return {
                    "success": False,
                    "message": "Vehicle in emergency stop mode. Use resume_from_emergency() first.",
                    "error_code": "EMERGENCY_STOP_ACTIVE"
                }
            
            self.current_speed = 0.0
            self.state = VehicleState.STOPPED
            
            self.console.print("🛑 Vehicle stopped")
            
            return {
                "success": True,
                "message": "Vehicle stopped successfully",
                "speed": self.current_speed,
                "state": self.state.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to stop vehicle: {str(e)}",
                "error_code": "SYSTEM_ERROR"
            }
    
    def adjust_speed(self, speed_change: float) -> Dict[str, Any]:
        """
        Adjust vehicle speed
        
        Args:
            speed_change: Speed change in km/h (positive to speed up, negative to slow down)
            
        Returns:
            Operation result
        """
        try:
            if self.emergency_stop_active:
                return {
                    "success": False,
                    "message": "Cannot adjust speed during emergency stop",
                    "error_code": "EMERGENCY_STOP_ACTIVE"
                }
            
            new_speed = max(0, min(self.max_speed, self.current_speed + speed_change))
            old_speed = self.current_speed
            self.current_speed = new_speed
            
            if new_speed > 0:
                self.state = VehicleState.MOVING
            else:
                self.state = VehicleState.STOPPED
            
            action = "increased" if speed_change > 0 else "decreased"
            self.console.print(f"🚗 Speed {action}: {old_speed:.1f} → {new_speed:.1f} km/h")
            
            return {
                "success": True,
                "message": f"Speed adjusted from {old_speed:.1f} to {new_speed:.1f} km/h",
                "old_speed": old_speed,
                "new_speed": new_speed,
                "state": self.state.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to adjust speed: {str(e)}",
                "error_code": "SYSTEM_ERROR"
            }
    
    def resume_from_emergency(self) -> Dict[str, Any]:
        """
        Resume normal operation from emergency stop
        
        Returns:
            Operation result
        """
        try:
            if not self.emergency_stop_active:
                return {
                    "success": False,
                    "message": "Vehicle is not in emergency stop mode",
                    "error_code": "NOT_IN_EMERGENCY"
                }
            
            self.emergency_stop_active = False
            self.state = VehicleState.STOPPED
            
            self.console.print("✅ Emergency stop deactivated. Vehicle ready for operation.")
            
            return {
                "success": True,
                "message": "Emergency stop deactivated. Vehicle ready for operation.",
                "state": self.state.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to resume from emergency: {str(e)}",
                "error_code": "SYSTEM_ERROR"
            }
    
    def get_speed(self) -> float:
        """Get current vehicle speed"""
        return self.current_speed
    
    def get_location(self) -> str:
        """Get current vehicle location"""
        return self.current_location
    
    def update_location(self, location: str) -> None:
        """Update current vehicle location (simulated GPS update)"""
        self.current_location = location
