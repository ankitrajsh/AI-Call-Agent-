"""
AI Call Agent - Legacy Application Entry Point
This file provides backward compatibility while using the new modular system
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import AICallAgent

# Create global agent instance for compatibility
agent = AICallAgent()

def main():
    """Main application entry point"""
    try:
        print("🚀 Starting AI Call Agent...")
        agent.interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")


if __name__ == "__main__":
    main()
