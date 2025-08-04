# AI Call Agent - Makefile
# Build automation for AI Call Agent project

.PHONY: help install install-dev run run-legacy test clean lint format setup

# Color definitions
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
MAGENTA=\033[0;35m
CYAN=\033[0;36m
RESET=\033[0m

# Default target
help:
	@echo "🚐 AI Call Agent - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup          - Initial project setup"
	@echo "  install        - Install dependencies with poetry"
	@echo "  install-dev    - Install with development dependencies"
	@echo "  install-pip    - Install dependencies with pip"
	@echo ""
	@echo "Running:"
	@echo "  run            - Run the main AI Call Agent"
	@echo "  run-legacy     - Run legacy app.py"
	@echo "  run-assistant  - Run legacy assistant.py"
	@echo ""
	@echo "Development:"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linters"
	@echo "  format         - Format code"
	@echo "  clean          - Clean build artifacts"
	@echo ""
	@echo "System:"
	@echo "  check-audio    - Check audio system"
	@echo "  download-models - Download required models"

# Setup commands
setup:
	@echo "🚀 Setting up AI Call Agent..."
	python -c "import nltk; nltk.download('punkt')"
	@echo "✅ Setup complete!"

install:
	@echo "📦 Installing dependencies with Poetry..."
	poetry install
	@echo "✅ Installation complete!"

install-dev:
	@echo "📦 Installing with development dependencies..."
	poetry install --with dev
	@echo "✅ Development installation complete!"

install-pip:
	@echo "📦 Installing dependencies with pip..."
	pip install -r requirements.txt
	@echo "✅ Pip installation complete!"

# Running commands
run:
	@echo "🚐 Starting AI Call Agent..."
	python src/main.py

run-legacy:
	@echo "🚐 Starting legacy app..."
	python app.py

run-assistant:
	@echo "🤖 Starting legacy assistant..."
	python assistant.py

# Development commands
test:
	@echo "🧪 Running tests..."
	python -c "from src.main import AICallAgent; agent = AICallAgent(); print('✅ All components initialized successfully!')"

lint:
	@echo "🔍 Running basic code checks..."
	python -m py_compile src/*.py
	@echo "✅ Code syntax is valid!"

format:
	@echo "✨ Code formatting (manual review recommended)..."
	@echo "Consider using black and isort for automated formatting"

clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# System utilities
check-audio:
	@echo "🎤 Checking audio system..."
	python -c "import sounddevice as sd; print('Available audio devices:'); print(sd.query_devices())" || echo "⚠️  Audio check failed - ensure audio libraries are installed"

download-models:
	@echo "📥 Downloading required models..."
	python -c "import whisper; whisper.load_model('base.en'); print('✅ Whisper model ready')" || echo "⚠️  Whisper download failed"
	@echo "📥 Models download complete!"

hello:
	@echo "${MAGENTA}Hello, $$(whoami)!${RESET}"
	@echo "${GREEN}Current Time:${RESET}\t\t${YELLOW}$$(date)${RESET}"
	@echo "${GREEN}Working Directory:${RESET}\t${YELLOW}$$(pwd)${RESET}"
	@echo "${GREEN}Shell:${RESET}\t\t\t${YELLOW}$$(echo $$SHELL)${RESET}"
	@echo "${GREEN}Terminal:${RESET}\t\t${YELLOW}$$(echo $$TERM)${RESET}"


env:
	@echo "To activate the Poetry environment, run:"
	@echo "source $$(poetry env info --path)/bin/activate"

lint:
	@echo "Running linter..."
	@source $$(poetry env info --path)/bin/activate && pre-commit run --all-files
	@echo "Done."
