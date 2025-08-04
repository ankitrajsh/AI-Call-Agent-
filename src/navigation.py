"""
Navigation Module
Handles campus navigation with predefined routes and ETA calculations
"""
import time
import math
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console


class RouteSegment:
    """Represents a segment of the route"""
    
    def __init__(self, start: str, end: str, distance: float, max_speed: float = 25.0):
        self.start = start
        self.end = end
        self.distance = distance  # in meters
        self.max_speed = max_speed  # in km/h
        
    def calculate_travel_time(self, speed: float) -> float:
        """Calculate travel time for this segment in seconds"""
        if speed <= 0:
            return float('inf')
        # Convert speed from km/h to m/s
        speed_ms = speed * (1000 / 3600)
        return self.distance / speed_ms


class NavigationEngine:
    """Handles navigation and ETA calculations for campus routes"""
    
    def __init__(self):
        self.console = Console()
        self.campus_map = self._initialize_campus_map()
        self.current_location = "Main Gate"
        self.current_destination = None
        
    def _initialize_campus_map(self) -> Dict[str, Dict[str, RouteSegment]]:
        """
        Initialize campus map with predefined routes
        
        Returns:
            Dictionary representing campus route network
        """
        # Campus route: Main Gate ↔ Hostel Circle
        # Approximate distances based on typical campus layout
        return {
            "Main Gate": {
                "Hostel Circle": RouteSegment("Main Gate", "Hostel Circle", 1200, 25.0),  # 1.2 km
                "Academic Block": RouteSegment("Main Gate", "Academic Block", 800, 25.0),   # 0.8 km
                "Library": RouteSegment("Main Gate", "Library", 600, 25.0)                 # 0.6 km
            },
            "Hostel Circle": {
                "Main Gate": RouteSegment("Hostel Circle", "Main Gate", 1200, 25.0),
                "Academic Block": RouteSegment("Hostel Circle", "Academic Block", 500, 25.0),
                "Cafeteria": RouteSegment("Hostel Circle", "Cafeteria", 300, 25.0)
            },
            "Academic Block": {
                "Main Gate": RouteSegment("Academic Block", "Main Gate", 800, 25.0),
                "Hostel Circle": RouteSegment("Academic Block", "Hostel Circle", 500, 25.0),
                "Library": RouteSegment("Academic Block", "Library", 400, 25.0)
            },
            "Library": {
                "Main Gate": RouteSegment("Library", "Main Gate", 600, 25.0),
                "Academic Block": RouteSegment("Library", "Academic Block", 400, 25.0),
                "Cafeteria": RouteSegment("Library", "Cafeteria", 350, 25.0)
            },
            "Cafeteria": {
                "Hostel Circle": RouteSegment("Cafeteria", "Hostel Circle", 300, 25.0),
                "Library": RouteSegment("Cafeteria", "Library", 350, 25.0)
            }
        }
    
    def find_route(self, start: str, destination: str) -> Optional[List[RouteSegment]]:
        """
        Find route between two locations
        
        Args:
            start: Starting location
            destination: Destination location
            
        Returns:
            List of route segments or None if no route found
        """
        if start not in self.campus_map or destination not in self.campus_map:
            return None
            
        # For simple campus, use direct routes or single hop
        if destination in self.campus_map[start]:
            return [self.campus_map[start][destination]]
        
        # Try to find route through intermediate points
        for intermediate in self.campus_map[start]:
            if destination in self.campus_map[intermediate]:
                return [
                    self.campus_map[start][intermediate],
                    self.campus_map[intermediate][destination]
                ]
        
        return None
    
    def calculate_eta(self, start: str, destination: str, current_speed: float) -> Optional[Dict[str, Any]]:
        """
        Calculate ETA for given route
        
        Args:
            start: Starting location
            destination: Destination location
            current_speed: Current vehicle speed in km/h
            
        Returns:
            Dictionary with ETA information
        """
        try:
            route = self.find_route(start, destination)
            if not route:
                return None
            
            total_distance = sum(segment.distance for segment in route)
            total_time = 0
            
            # Calculate time for each segment
            speed_to_use = max(current_speed, 10.0) if current_speed > 0 else 15.0  # Default speed
            
            for segment in route:
                segment_time = segment.calculate_travel_time(speed_to_use)
                total_time += segment_time
            
            # Convert to minutes
            eta_minutes = total_time / 60
            
            # Format ETA
            if eta_minutes < 1:
                eta_text = "Less than 1 minute"
            elif eta_minutes < 60:
                eta_text = f"{int(eta_minutes)} minute{'s' if int(eta_minutes) != 1 else ''}"
            else:
                hours = int(eta_minutes // 60)
                minutes = int(eta_minutes % 60)
                eta_text = f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
            
            return {
                "start": start,
                "destination": destination,
                "total_distance": total_distance,
                "total_time_seconds": total_time,
                "eta_minutes": eta_minutes,
                "eta_text": eta_text,
                "route_segments": len(route),
                "current_speed": speed_to_use
            }
            
        except Exception as e:
            self.console.print(f"[red]ETA calculation error: {e}")
            return None
    
    def get_nearby_locations(self, location: str) -> List[str]:
        """
        Get list of nearby locations from given location
        
        Args:
            location: Current location
            
        Returns:
            List of nearby location names
        """
        if location in self.campus_map:
            return list(self.campus_map[location].keys())
        return []
    
    def get_distance_to_destination(self, start: str, destination: str) -> Optional[float]:
        """
        Get total distance to destination
        
        Args:
            start: Starting location
            destination: Destination location
            
        Returns:
            Distance in meters or None if no route
        """
        route = self.find_route(start, destination)
        if route:
            return sum(segment.distance for segment in route)
        return None
    
    def update_location(self, new_location: str) -> bool:
        """
        Update current location
        
        Args:
            new_location: New location name
            
        Returns:
            True if location is valid, False otherwise
        """
        if new_location in self.campus_map:
            self.current_location = new_location
            return True
        return False
    
    def get_all_locations(self) -> List[str]:
        """Get list of all available locations"""
        return list(self.campus_map.keys())
    
    def is_valid_destination(self, destination: str) -> bool:
        """Check if destination is valid"""
        return destination in self.campus_map
    
    def get_route_info(self, start: str, destination: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed route information
        
        Args:
            start: Starting location
            destination: Destination location
            
        Returns:
            Dictionary with route details
        """
        route = self.find_route(start, destination)
        if not route:
            return None
        
        route_info = {
            "start": start,
            "destination": destination,
            "segments": [],
            "total_distance": 0,
            "waypoints": [start]
        }
        
        for segment in route:
            route_info["segments"].append({
                "from": segment.start,
                "to": segment.end,
                "distance": segment.distance,
                "max_speed": segment.max_speed
            })
            route_info["total_distance"] += segment.distance
            if segment.end not in route_info["waypoints"]:
                route_info["waypoints"].append(segment.end)
        
        return route_info
