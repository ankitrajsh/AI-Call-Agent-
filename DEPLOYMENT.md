# 🚀 AI Call Agent Deployment Guide

## Production Deployment for Autonomous Vehicles

This guide covers deploying the AI Call Agent system in production environments, including real vehicle integration, cloud deployment, and edge computing scenarios.

## 🏗️ Deployment Architecture

### 1. **Edge Deployment (In-Vehicle)**
```
┌─────────────────────────────────────────┐
│           Vehicle Edge Computer         │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  AI Agent   │  │  Local Models   │   │
│  │   (Main)    │  │  - Llama-2-7B   │   │
│  └─────────────┘  │  - Whisper      │   │
│                   │  - Bark TTS     │   │
│  ┌─────────────┐  └─────────────────┘   │
│  │ Vehicle CAN │                        │
│  │   Interface │  ┌─────────────────┐   │
│  └─────────────┘  │   Safety Layer  │   │
│                   │  - Emergency    │   │
│                   │  - Validation   │   │
│                   └─────────────────┘   │
└─────────────────────────────────────────┘
```

### 2. **Hybrid Cloud-Edge**
```
┌─────────────────┐    ┌─────────────────┐
│   Vehicle Edge  │────│   Cloud Backend │
│                 │    │                 │
│ • Local Safety  │    │ • Model Updates │
│ • Basic Commands│    │ • Analytics     │
│ • Emergency     │    │ • Monitoring    │
│                 │    │ • Fleet Mgmt    │
└─────────────────┘    └─────────────────┘
```

## 🔧 Hardware Requirements

### Minimum Requirements (CPU-only)
- **CPU**: 4-core Intel/AMD x64 processor
- **RAM**: 8GB DDR4
- **Storage**: 50GB SSD
- **Network**: WiFi 5 or Ethernet
- **Audio**: USB microphone + speakers

### Recommended (GPU-accelerated)
- **CPU**: 8-core Intel/AMD x64 processor
- **GPU**: NVIDIA RTX 3060 or better (8GB VRAM)
- **RAM**: 16GB DDR4 
- **Storage**: 100GB NVMe SSD
- **Network**: WiFi 6 + 5G/LTE backup
- **Audio**: High-quality DSP microphone array

### Production Vehicle Integration
- **Compute**: NVIDIA Jetson AGX Orin or similar
- **CAN Interface**: Vector CANoe or PEAK CAN
- **Safety**: Hardware watchdog timer
- **Redundancy**: Dual computing units
- **Power**: 12V/24V automotive power supply

## 🐳 Docker Deployment

### Basic Docker Setup
```dockerfile
# Dockerfile.production
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    portaudio19-dev \
    libsndfile1-dev \
    git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_enhanced.txt .
RUN pip3 install --no-cache-dir -r requirements_enhanced.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 aiagent && chown -R aiagent:aiagent /app
USER aiagent

# Expose port for API (if needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python3 -c "from src.main import AICallAgent; print('OK')" || exit 1

# Start the enhanced AI agent
CMD ["python3", "src/main.py", "--production"]
```

### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-call-agent:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: ai-call-agent
    restart: unless-stopped
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    volumes:
      - ./fine_tuned_models:/app/fine_tuned_models
      - ./logs:/app/logs
      - /dev/snd:/dev/snd:ro  # Audio devices
    devices:
      - /dev/nvidia0:/dev/nvidia0  # GPU access
    ports:
      - "8000:8000"
    networks:
      - ai-agent-network

  redis:
    image: redis:7-alpine
    container_name: ai-agent-cache
    restart: unless-stopped
    ports:
      - "6379:6379"
    networks:
      - ai-agent-network

  monitoring:
    image: grafana/grafana:latest
    container_name: ai-agent-monitoring
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - ai-agent-network

networks:
  ai-agent-network:
    driver: bridge
```

## ☸️ Kubernetes Deployment

### Kubernetes Manifests
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-call-agent
  labels:
    app: ai-call-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-call-agent
  template:
    metadata:
      labels:
        app: ai-call-agent
    spec:
      containers:
      - name: ai-call-agent
        image: ai-call-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /app/fine_tuned_models
        - name: log-storage
          mountPath: /app/logs
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ai-agent-models-pvc
      - name: log-storage
        persistentVolumeClaim:
          claimName: ai-agent-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ai-call-agent-service
spec:
  selector:
    app: ai-call-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔒 Security Configuration

### Production Security Checklist
- [ ] **Authentication**: API key or OAuth integration
- [ ] **Encryption**: TLS 1.3 for all communications
- [ ] **Network**: Firewall rules and VPN access
- [ ] **Container**: Non-root user, minimal base image
- [ ] **Secrets**: Environment variables for API keys
- [ ] **Monitoring**: Security event logging
- [ ] **Updates**: Automated security patches

### Security Environment Variables
```bash
# .env.production
AI_AGENT_API_KEY=your-secure-api-key
HUGGINGFACE_TOKEN=your-hf-token
OPENAI_API_KEY=your-openai-key
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## 🚗 Vehicle Integration

### CAN Bus Interface
```python
# src/vehicle_integration.py
"""
Real vehicle integration via CAN bus
"""
import can
import time
from typing import Optional, Dict, Any
from .vehicle_control import VehicleControlAPI

class RealVehicleInterface(VehicleControlAPI):
    """Real vehicle control via CAN bus"""
    
    def __init__(self, can_interface='can0', bitrate=500000):
        super().__init__()
        
        # Initialize CAN bus
        self.bus = can.interface.Bus(
            channel=can_interface,
            bustype='socketcan',
            bitrate=bitrate
        )
        
        # CAN message IDs (vehicle-specific)
        self.CAN_IDS = {
            'SPEED_REQUEST': 0x100,
            'EMERGENCY_STOP': 0x101,
            'DESTINATION_SET': 0x102,
            'STATUS_REQUEST': 0x103,
            'SPEED_RESPONSE': 0x200,
            'STATUS_RESPONSE': 0x201
        }
        
    def emergency_stop(self) -> Dict[str, Any]:
        """Send emergency stop command via CAN"""
        try:
            # Create emergency stop CAN message
            msg = can.Message(
                arbitration_id=self.CAN_IDS['EMERGENCY_STOP'],
                data=[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            # Send with high priority
            self.bus.send(msg, timeout=0.1)
            
            # Wait for acknowledgment
            response = self._wait_for_response(self.CAN_IDS['STATUS_RESPONSE'])
            
            return {
                "status": "emergency_stop_activated",
                "timestamp": time.time(),
                "can_response": response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"CAN emergency stop failed: {e}",
                "timestamp": time.time()
            }
    
    def adjust_speed(self, delta: float) -> Dict[str, Any]:
        """Adjust vehicle speed via CAN"""
        try:
            # Convert speed delta to CAN data
            speed_data = int(abs(delta) * 10)  # Scale for CAN
            direction = 0x01 if delta > 0 else 0x00
            
            msg = can.Message(
                arbitration_id=self.CAN_IDS['SPEED_REQUEST'],
                data=[direction, speed_data, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            self.bus.send(msg, timeout=0.1)
            
            # Get current speed after adjustment
            current_speed = self._get_current_speed_from_can()
            
            return {
                "status": "speed_adjusted",
                "new_speed": current_speed,
                "delta": delta,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Speed adjustment failed: {e}",
                "timestamp": time.time()
            }
    
    def _wait_for_response(self, expected_id: int, timeout: float = 1.0) -> Optional[Dict]:
        """Wait for CAN response message"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.bus.recv(timeout=0.1)
            if msg and msg.arbitration_id == expected_id:
                return {
                    "id": msg.arbitration_id,
                    "data": list(msg.data),
                    "timestamp": msg.timestamp
                }
        
        return None
    
    def _get_current_speed_from_can(self) -> float:
        """Get current speed from CAN bus"""
        try:
            # Request speed
            msg = can.Message(
                arbitration_id=self.CAN_IDS['STATUS_REQUEST'],
                data=[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            )
            self.bus.send(msg)
            
            # Wait for response
            response = self._wait_for_response(self.CAN_IDS['SPEED_RESPONSE'])
            
            if response:
                # Decode speed from CAN data (vehicle-specific)
                speed = (response['data'][0] << 8 | response['data'][1]) / 10.0
                return speed
            
            return 0.0
            
        except Exception:
            return 0.0
```

## 📊 Monitoring and Observability

### Prometheus Metrics
```python
# src/monitoring.py
"""
Production monitoring and metrics
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging
from functools import wraps

# Metrics
REQUEST_COUNT = Counter('ai_agent_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('ai_agent_request_duration_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('ai_agent_active_connections', 'Active connections')
MODEL_INFERENCE_TIME = Histogram('ai_agent_model_inference_seconds', 'Model inference time')
VEHICLE_COMMANDS = Counter('ai_agent_vehicle_commands_total', 'Vehicle commands', ['command_type'])

class MonitoringMixin:
    """Mixin to add monitoring to classes"""
    
    def __init__(self):
        # Start Prometheus metrics server
        start_http_server(8001)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def monitor_request(self, method='unknown', endpoint='unknown'):
        """Decorator to monitor requests"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.logger.error(f"Request failed: {e}")
                    raise
                finally:
                    REQUEST_LATENCY.observe(time.time() - start_time)
            
            return wrapper
        return decorator
    
    def monitor_vehicle_command(self, command_type):
        """Monitor vehicle commands"""
        VEHICLE_COMMANDS.labels(command_type=command_type).inc()
        self.logger.info(f"Vehicle command executed: {command_type}")
```

### Health Check Endpoint
```python
# src/health.py
"""
Health check and readiness endpoints
"""
from flask import Flask, jsonify
import threading
import time
from .main import AICallAgent

app = Flask(__name__)
health_status = {"status": "starting", "last_check": None}

def background_health_check(agent: AICallAgent):
    """Background health monitoring"""
    while True:
        try:
            # Test core components
            intent, _ = agent.intent_engine.recognize_intent("test")
            vehicle_status = agent.vehicle_control.get_vehicle_status()
            
            health_status.update({
                "status": "healthy",
                "last_check": time.time(),
                "components": {
                    "intent_engine": "ok",
                    "vehicle_control": "ok",
                    "tts_service": "ok" if agent.tts_service else "disabled"
                }
            })
            
        except Exception as e:
            health_status.update({
                "status": "unhealthy",
                "last_check": time.time(),
                "error": str(e)
            })
        
        time.sleep(30)  # Check every 30 seconds

@app.route('/health')
def health():
    """Health check endpoint"""
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code

@app.route('/ready')
def ready():
    """Readiness check endpoint"""
    is_ready = health_status["status"] in ["healthy", "starting"]
    status_code = 200 if is_ready else 503
    return jsonify({"ready": is_ready}), status_code

def start_health_server(agent: AICallAgent, port=8002):
    """Start health check server"""
    # Start background health check
    health_thread = threading.Thread(
        target=background_health_check, 
        args=(agent,), 
        daemon=True
    )
    health_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False)
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy AI Call Agent

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements_enhanced.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
        python test_enhanced_system.py
    
    - name: Lint code
      run: |
        flake8 src/ --max-line-length=100
        black --check src/

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ai-call-agent:${{ github.sha }} .
    
    - name: Run security scan
      run: |
        docker run --rm -v $(pwd):/app -w /app \
          securecodewarrior/docker-security-scan:latest \
          scan --image ai-call-agent:${{ github.sha }}

  deploy:
    needs: [test, build]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        # Deploy to your production environment
        echo "Deploying to production..."
```

## 📈 Performance Optimization

### Production Optimizations
```python
# src/optimizations.py
"""
Production performance optimizations
"""
import torch
import gc
from typing import Dict, Any

class ProductionOptimizer:
    """Optimizations for production deployment"""
    
    @staticmethod
    def optimize_model_for_inference(model):
        """Optimize model for faster inference"""
        # Enable inference mode
        model.eval()
        
        # Compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        # Enable mixed precision
        if torch.cuda.is_available():
            model = model.half()  # Convert to FP16
        
        return model
    
    @staticmethod
    def setup_memory_optimization():
        """Setup memory optimizations"""
        # Enable memory efficient attention
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Garbage collection settings
        gc.set_threshold(700, 10, 10)
    
    @staticmethod
    def optimize_batch_processing(batch_size: int = 4) -> Dict[str, Any]:
        """Optimize for batch processing"""
        return {
            "batch_size": batch_size,
            "dataloader_num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True
        }
```

## 🚦 Traffic Management

### Load Balancing Configuration
```nginx
# nginx.conf
upstream ai_call_agent {
    least_conn;
    server ai-call-agent-1:8000 weight=3;
    server ai-call-agent-2:8000 weight=3;
    server ai-call-agent-3:8000 weight=2 backup;
}

server {
    listen 80;
    server_name ai-call-agent.yourdomain.com;
    
    location / {
        proxy_pass http://ai_call_agent;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://ai_call_agent/health;
        access_log off;
    }
}
```

## 📋 Deployment Checklist

### Pre-deployment
- [ ] **Code Review**: All changes reviewed and approved
- [ ] **Testing**: Unit tests, integration tests pass
- [ ] **Security**: Security scan completed
- [ ] **Performance**: Load testing completed
- [ ] **Documentation**: Deployment docs updated

### Deployment
- [ ] **Backup**: Current system backed up
- [ ] **Environment**: Production environment prepared
- [ ] **Database**: Migrations applied (if applicable)
- [ ] **Models**: Fine-tuned models deployed
- [ ] **Configuration**: Production config verified

### Post-deployment
- [ ] **Health Check**: All health endpoints green
- [ ] **Monitoring**: Metrics and logs flowing
- [ ] **Performance**: Response times within SLA
- [ ] **Functionality**: Core features working
- [ ] **Rollback Plan**: Ready if needed

---

**Ready for production deployment? Follow this guide step by step!** 🚀🔧
