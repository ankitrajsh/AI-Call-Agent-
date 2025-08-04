#!/bin/bash

# AI Call Agent - Quick Start Script
# This script helps set up and run the AI Call Agent system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}AI Call Agent:${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        return 0
    else
        print_error "Python 3 is not installed or not in PATH"
        return 1
    fi
}

# Check if virtual environment exists
check_venv() {
    if [ -d "ai-call-agent" ] || [ -d ".venv" ] || [ -n "$VIRTUAL_ENV" ]; then
        return 0
    else
        return 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    python3 -m venv ai-call-agent
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    if [ -d "ai-call-agent" ]; then
        source ai-call-agent/bin/activate
        print_success "Virtual environment activated"
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
        print_success "Virtual environment activated"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Check if Poetry is available
    if command -v poetry &> /dev/null; then
        print_status "Using Poetry for dependency management..."
        poetry install
    else
        print_status "Using pip for dependency management..."
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
    
    print_success "Dependencies installed"
}

# Download NLTK data
setup_nltk() {
    print_status "Setting up NLTK data..."
    python3 -c "import nltk; nltk.download('punkt', quiet=True); print('NLTK punkt downloaded')" || {
        print_warning "NLTK setup may have failed, but continuing..."
    }
}

# Check audio system
check_audio() {
    print_status "Checking audio system..."
    
    # Check if audio devices are available
    python3 -c "
import sys
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f'Found {len(devices)} audio devices')
    if len(devices) == 0:
        print('Warning: No audio devices found')
        sys.exit(1)
except ImportError:
    print('Warning: sounddevice not available')
    sys.exit(1)
except Exception as e:
    print(f'Warning: Audio check failed: {e}')
    sys.exit(1)
" && print_success "Audio system is ready" || print_warning "Audio system may not be properly configured"
}

# Run system test
run_test() {
    print_status "Running system test..."
    if python3 test_system.py; then
        print_success "System test passed"
        return 0
    else
        print_warning "System test completed with warnings"
        return 1
    fi
}

# Main setup function
setup() {
    echo
    echo "AI Call Agent - Quick Setup"
    echo "=============================="
    echo
    
    # Check requirements
    check_python || exit 1
    
    # Setup virtual environment
    if ! check_venv; then
        create_venv
    fi
    
    activate_venv
    install_dependencies
    setup_nltk
    check_audio
    
    echo
    print_success "Setup completed!"
    echo
    print_status "You can now run the AI Call Agent with:"
    echo "  make run              # Using Makefile"
    echo "  python src/main.py    # Direct execution"
    echo "  python app.py         # Legacy entry point"
    echo
}

# Run the application
run_app() {
    print_status "Starting AI Call Agent..."
    
    # Activate virtual environment if it exists
    if check_venv && [ -z "$VIRTUAL_ENV" ]; then
        activate_venv
    fi
    
    # Run the main application
    python3 src/main.py
}

# Show help
show_help() {
    echo "AI Call Agent - Quick Start Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  setup     - Set up the project (install dependencies, etc.)"
    echo "  run       - Run the AI Call Agent"
    echo "  test      - Run system tests"
    echo "  help      - Show this help message"
    echo
    echo "If no command is provided, setup will be run automatically."
}

# Main script logic
case "${1:-setup}" in
    "setup")
        setup
        ;;
    "run")
        run_app
        ;;
    "test")
        run_test
        ;;
    "help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
