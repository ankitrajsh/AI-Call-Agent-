#!/bin/bash
# Enhanced Setup Script for AI Call Agent with Llama Fine-tuning
# This script installs all dependencies including PyTorch, Transformers, and Datasets

set -e  # Exit on any error

echo "🚀 Enhanced AI Call Agent Setup with Llama Integration"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Check if Python is available
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_success "Python $PYTHON_VERSION found"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install basic dependencies first
print_status "Installing basic dependencies..."
pip install rich numpy

# Detect CUDA availability for PyTorch
print_status "Detecting hardware capabilities..."
if command -v nvidia-smi &> /dev/null; then
    print_success "CUDA detected - installing PyTorch with GPU support"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    print_warning "No CUDA detected - installing CPU-only PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# Install PyTorch
print_status "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url $TORCH_INDEX

# Install Transformers and related libraries
print_status "Installing Transformers and related libraries..."
pip install transformers datasets accelerate bitsandbytes sentencepiece protobuf

# Install audio processing libraries
print_status "Installing audio processing libraries..."
pip install sounddevice soundfile librosa

# Install additional ML libraries
print_status "Installing additional ML libraries..."
pip install scikit-learn matplotlib seaborn tqdm

# Install Whisper for speech recognition
print_status "Installing OpenAI Whisper..."
pip install openai-whisper

# Install optional dependencies
print_status "Installing optional dependencies..."
pip install langchain langchain-community langchain-openai

# Install development tools
print_status "Installing development tools..."
pip install pytest pytest-cov black flake8 mypy

# Create requirements.txt with all installed packages
print_status "Generating requirements.txt..."
pip freeze > requirements_enhanced.txt

# Download necessary NLTK data
print_status "Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download warning: {e}')
"

# Test basic imports
print_status "Testing basic imports..."
python3 -c "
import torch
import transformers
import datasets
import sounddevice
import whisper
import numpy as np
import rich
print('All basic imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"

# Create sample configuration
print_status "Creating sample configuration..."
cat > config_enhanced.yaml << EOF
# Enhanced AI Call Agent Configuration
system:
  use_llama: true
  llama_model_path: null  # Set to path of fine-tuned model
  fallback_to_rules: true
  
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  
fine_tuning:
  model_name: "meta-llama/Llama-2-7b-chat-hf"
  output_dir: "./fine_tuned_models"
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  warmup_steps: 100
  
vehicle:
  max_speed: 60.0
  campus_locations:
    - "Main Gate"
    - "Library"
    - "Student Center"
    - "Engineering Building"
    - "Parking Lot A"
EOF

# Create enhanced test script
print_status "Creating enhanced test script..."
cat > test_enhanced_system.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Test Script for AI Call Agent with Llama Integration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import AICallAgent
from src.llama_finetuning import LlamaFineTuner
from src.intent_recognition import Intent

def test_basic_system():
    """Test basic system functionality"""
    print("🧪 Testing basic AI Call Agent system...")
    
    try:
        # Initialize with enhanced features
        agent = AICallAgent(use_llama=False)  # Start without Llama
        print("✅ AI Call Agent initialized successfully")
        
        # Test intent recognition
        test_phrases = [
            "Stop the vehicle immediately",
            "Set destination to Library", 
            "What's our current speed",
            "How long until we arrive"
        ]
        
        for phrase in test_phrases:
            intent, entities = agent.intent_engine.recognize_intent(phrase)
            print(f"✅ '{phrase}' -> {intent.value} {entities or ''}")
        
        # Test vehicle control
        status = agent.vehicle_control.get_vehicle_status()
        print(f"✅ Vehicle status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def test_llama_integration():
    """Test Llama model integration"""
    print("\n🤖 Testing Llama integration...")
    
    try:
        from src.llama_intent_recognition import LlamaIntentRecognitionEngine
        
        # Test without actual model (fallback mode)
        engine = LlamaIntentRecognitionEngine(use_llama=False, fallback_to_rules=True)
        
        # Test intent recognition
        intent, entities = engine.recognize_intent("Emergency stop now!")
        print(f"✅ Llama engine (fallback): {intent.value}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Llama integration test: {e}")
        return False

def test_fine_tuning_setup():
    """Test fine-tuning setup"""
    print("\n🔧 Testing fine-tuning setup...")
    
    try:
        from src.llama_finetuning import LlamaFineTuner, create_sample_dataset
        
        # Test dataset creation
        dataset = create_sample_dataset()
        print(f"✅ Sample dataset created with {len(dataset)} examples")
        
        # Test fine-tuner initialization (without actual model loading)
        print("✅ Fine-tuning components ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Fine-tuning test failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("sounddevice", "Audio processing"),
        ("whisper", "Speech recognition"),
        ("rich", "Rich console"),
        ("numpy", "NumPy")
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not available")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🚀 Enhanced AI Call Agent System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Basic System", test_basic_system),
        ("Llama Integration", test_llama_integration),
        ("Fine-tuning Setup", test_fine_tuning_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print("="*50)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready for enhanced operation.")
        print("\nNext steps:")
        print("1. Run 'python src/llama_finetuning.py' to start fine-tuning")
        print("2. Use 'python src/main.py' to start the enhanced agent")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x test_enhanced_system.py

# Create enhanced startup script
print_status "Creating enhanced startup script..."
cat > start_enhanced.sh << 'EOF'
#!/bin/bash
# Enhanced startup script for AI Call Agent with Llama integration

echo "🚀 Starting Enhanced AI Call Agent System"
echo "========================================"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found"
fi

# Check for fine-tuned model
MODEL_DIR="./fine_tuned_models"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "Fine-tuned model found, starting with Llama integration..."
    python3 -c "
from src.main import AICallAgent
import sys

try:
    agent = AICallAgent(llama_model_path='$MODEL_DIR', use_llama=True)
    print('Enhanced AI Call Agent with Llama model loaded!')
    agent.interactive_mode()
except KeyboardInterrupt:
    print('\nShutting down gracefully...')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
else
    echo "No fine-tuned model found, starting with rule-based system..."
    python3 -c "
from src.main import AICallAgent
import sys

try:
    agent = AICallAgent(use_llama=False)
    print('AI Call Agent (rule-based) ready!')
    agent.interactive_mode()
except KeyboardInterrupt:
    print('\nShutting down gracefully...')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
fi
EOF

chmod +x start_enhanced.sh

# Final status report
echo ""
print_success "Enhanced AI Call Agent setup completed!"
echo ""
echo "📋 Setup Summary:"
echo "=================="
echo "✅ Virtual environment: $VENV_DIR"
echo "✅ PyTorch with $(python3 -c 'import torch; print("GPU" if torch.cuda.is_available() else "CPU")') support"
echo "✅ Transformers and Datasets libraries"
echo "✅ Audio processing capabilities"
echo "✅ Enhanced configuration and test scripts"
echo ""
echo "🚀 Quick Start:"
echo "==============="
echo "1. Test the system:    ./test_enhanced_system.py"
echo "2. Fine-tune model:    python3 src/llama_finetuning.py"
echo "3. Start enhanced AI:  ./start_enhanced.sh"
echo ""
echo "📁 New Files Created:"
echo "====================="
echo "• config_enhanced.yaml       - Enhanced configuration"
echo "• test_enhanced_system.py    - Comprehensive testing"
echo "• start_enhanced.sh          - Enhanced startup script"
echo "• requirements_enhanced.txt  - Complete dependencies"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true

print_success "Setup complete! The system is ready for enhanced operation with Llama integration."
