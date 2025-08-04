"""
AI Call Agent - Assistant Module (Legacy)
This module provides backward compatibility for the assistant functionality
"""
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import AICallAgent

def main():
    """Assistant main entry point"""
    try:
        print("🤖 Starting AI Call Agent Assistant...")
        agent = AICallAgent()
        agent.interactive_mode()
    except KeyboardInterrupt:
        print("\n🛑 Assistant interrupted by user")
    except Exception as e:
        print(f"❌ Assistant error: {e}")


if __name__ == "__main__":
    main()
