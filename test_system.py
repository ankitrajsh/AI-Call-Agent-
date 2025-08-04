#!/usr/bin/env python3
"""
Test Script for AI Call Agent System
Run this to verify all components are working correctly
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print(" Testing module imports...")
    
    try:
        from src.speech_to_text import SpeechToTextEngine
        print("Speech-to-Text module imported")
    except Exception as e:
        print(f" Speech-to-Text import failed: {e}")
        return False
    
    try:
        from src.intent_recognition import IntentRecognitionEngine, Intent
        print(" Intent Recognition module imported")
    except Exception as e:
        print(f"Intent Recognition import failed: {e}")
        return False
    
    try:
        from src.vehicle_control import VehicleControlAPI
        print("Vehicle Control module imported")
    except Exception as e:
        print(f"Vehicle Control import failed: {e}")
        return False
    
    try:
        from src.navigation import NavigationEngine
        print("Navigation module imported")
    except Exception as e:
        print(f"Navigation import failed: {e}")
        return False
    
    try:
        from src.text_to_speech import TextToSpeechService
        print("Text-to-Speech module imported")
    except Exception as e:
        print(f"Text-to-Speech import failed: {e}")
        return False
    
    try:
        from src.ui_parser import UICommandParser
        print("UI Parser module imported")
    except Exception as e:
        print(f"UI Parser import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of each component"""
    print("\nTesting basic functionality...")
    
    try:
        # Test Intent Recognition
        from src.intent_recognition import IntentRecognitionEngine
        intent_engine = IntentRecognitionEngine()
        intent, entities = intent_engine.recognize_intent("stop the vehicle")
        print(f"Intent Recognition: '{intent}' from 'stop the vehicle'")
        
        # Test Vehicle Control
        from src.vehicle_control import VehicleControlAPI
        vehicle = VehicleControlAPI()
        status = vehicle.get_vehicle_status()
        print(f"Vehicle Control: Status retrieved - Speed: {status['speed']} km/h")
        
        # Test Navigation
        from src.navigation import NavigationEngine
        nav = NavigationEngine()
        route = nav.find_route("Main Gate", "Hostel Circle")
        print(f"Navigation: Route found with {len(route) if route else 0} segments")
        
        # Test UI Parser
        from src.ui_parser import UICommandParser
        ui = UICommandParser()
        parsed = ui.parse_ui_command("emergency_stop")
        print(f"UI Parser: Command parsed successfully")
        
        return True
        
    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False

def test_integration():
    """Test integration between components"""
    print("\nTesting component integration...")
    
    try:
        from src.main import AICallAgent
        
        # Initialize the main agent (this tests all components together)
        agent = AICallAgent()
        print("AI Call Agent initialized successfully")
        
        # Test a simulated voice command processing
        from src.intent_recognition import Intent
        response = agent._process_intent(Intent.GET_SPEED, None, "what's our speed")
        print(f"Intent processing: {response['type']}")
        
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

def test_audio_system():
    """Test audio system availability"""
    print("\nTesting audio system...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"Audio system: {len(devices)} audio devices found")
        return True
    except Exception as e:
        print(f"Audio system test failed: {e}")
        print("ℹThis is expected if audio libraries are not installed")
        return False

def main():
    """Run all tests"""
    print("AI Call Agent System Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Component Integration", test_integration),
        ("Audio System", test_audio_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready to use.")
        return 0
    else:
        print(" Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
