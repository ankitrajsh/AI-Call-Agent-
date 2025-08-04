"""
RESTful API Service for AI Call Agent
Provides HTTP endpoints for integration with external systems
"""
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import threading
import time
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our AI Call Agent components
from src.main import AICallAgent
from src.intent_recognition import Intent
from src.monitoring import MonitoringMixin, REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_CONNECTIONS

app = Flask(__name__)
CORS(app)  # Enable CORS for web integration

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

# Global AI agent instance
ai_agent: Optional[AICallAgent] = None
executor = ThreadPoolExecutor(max_workers=4)

# Session management
active_sessions = {}
session_timeout = 300  # 5 minutes

class APIService(MonitoringMixin):
    """API service for AI Call Agent"""
    
    def __init__(self, llama_model_path=None, use_llama=True):
        super().__init__()
        self.ai_agent = AICallAgent(llama_model_path, use_llama)
        self.logger = logging.getLogger('APIService')
        
    def process_voice_request(self, audio_data: bytes, session_id: str) -> Dict[str, Any]:
        """Process voice input asynchronously"""
        try:
            # Here you would implement actual audio processing
            # For now, we'll simulate with text processing
            
            # Placeholder: Convert audio to text (implement with Whisper)
            text = "Emergency stop now"  # Simulated transcription
            
            # Process with AI agent
            intent, entities = self.ai_agent.intent_engine.recognize_intent(text)
            response = self.ai_agent._process_intent(intent, entities, text)
            
            # Generate response text
            if hasattr(self.ai_agent.intent_engine, 'generate_response'):
                response_text = self.ai_agent.intent_engine.generate_response(
                    intent, entities, {"session_id": session_id}
                )
            else:
                response_text = self.ai_agent._format_basic_response(
                    response['type'], response['data']
                )
            
            return {
                "success": True,
                "session_id": session_id,
                "transcription": text,
                "intent": intent.value,
                "entities": entities,
                "response": response_text,
                "vehicle_status": self.ai_agent.vehicle_control.get_vehicle_status(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Voice processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Initialize API service
api_service = None

def init_api_service(llama_model_path=None, use_llama=True):
    """Initialize the API service"""
    global api_service
    api_service = APIService(llama_model_path, use_llama)
    return api_service

# Session management functions
def create_session() -> str:
    """Create a new session"""
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "created_at": time.time(),
        "last_activity": time.time(),
        "requests": 0
    }
    return session_id

def validate_session(session_id: str) -> bool:
    """Validate session and update activity"""
    if session_id not in active_sessions:
        return False
    
    session = active_sessions[session_id]
    current_time = time.time()
    
    # Check if session expired
    if current_time - session["last_activity"] > session_timeout:
        del active_sessions[session_id]
        return False
    
    # Update last activity
    session["last_activity"] = current_time
    session["requests"] += 1
    return True

def cleanup_sessions():
    """Clean up expired sessions"""
    current_time = time.time()
    expired_sessions = [
        sid for sid, session in active_sessions.items()
        if current_time - session["last_activity"] > session_timeout
    ]
    
    for sid in expired_sessions:
        del active_sessions[sid]

# API Routes

@app.route('/api/v1/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                "ai_agent": "ok" if api_service else "not_initialized",
                "sessions": len(active_sessions),
                "uptime": time.time()
            }
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/api/v1/session', methods=['POST'])
@limiter.limit("10 per minute")
def create_new_session():
    """Create a new conversation session"""
    try:
        session_id = create_session()
        ACTIVE_CONNECTIONS.inc()
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "expires_in": session_timeout,
            "timestamp": datetime.utcnow().isoformat()
        }), 201
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/session/<session_id>', methods=['DELETE'])
def end_session(session_id: str):
    """End a conversation session"""
    try:
        if session_id in active_sessions:
            del active_sessions[session_id]
            ACTIVE_CONNECTIONS.dec()
            
        return jsonify({
            "success": True,
            "message": "Session ended",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/chat', methods=['POST'])
@limiter.limit("20 per minute")
def chat():
    """Process text-based chat message"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"success": False, "error": "Message required"}), 400
        
        session_id = data.get('session_id')
        if not session_id or not validate_session(session_id):
            return jsonify({"success": False, "error": "Invalid or expired session"}), 401
        
        message = data['message']
        
        # Process with AI agent
        start_time = time.time()
        
        intent, entities = api_service.ai_agent.intent_engine.recognize_intent(message)
        response = api_service.ai_agent._process_intent(intent, entities, message)
        
        # Generate response text
        if hasattr(api_service.ai_agent.intent_engine, 'generate_response'):
            response_text = api_service.ai_agent.intent_engine.generate_response(
                intent, entities, {"session_id": session_id}
            )
        else:
            response_text = api_service.ai_agent._format_basic_response(
                response['type'], response['data']
            )
        
        # Monitor request
        REQUEST_COUNT.labels(method='POST', endpoint='/chat').inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "intent": intent.value,
            "entities": entities,
            "response": response_text,
            "vehicle_status": api_service.ai_agent.vehicle_control.get_vehicle_status(),
            "processing_time": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        api_service.logger.error(f"Chat processing error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/voice', methods=['POST'])
@limiter.limit("10 per minute")
def voice():
    """Process voice input"""
    try:
        # Check for audio file upload
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "Audio file required"}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id')
        
        if not session_id or not validate_session(session_id):
            return jsonify({"success": False, "error": "Invalid or expired session"}), 401
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Process asynchronously
        future = executor.submit(
            api_service.process_voice_request, 
            audio_data, 
            session_id
        )
        
        result = future.result(timeout=30)  # 30 second timeout
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        api_service.logger.error(f"Voice processing error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/vehicle/status', methods=['GET'])
def vehicle_status():
    """Get current vehicle status"""
    try:
        session_id = request.args.get('session_id')
        if session_id and not validate_session(session_id):
            return jsonify({"success": False, "error": "Invalid or expired session"}), 401
        
        status = api_service.ai_agent.vehicle_control.get_vehicle_status()
        location = api_service.ai_agent.vehicle_control.get_location()
        speed = api_service.ai_agent.vehicle_control.get_speed()
        
        return jsonify({
            "success": True,
            "vehicle": {
                "status": status,
                "location": location,
                "speed": speed,
                "destination": getattr(api_service.ai_agent.vehicle_control, 'current_destination', None)
            },
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/vehicle/emergency_stop', methods=['POST'])
@limiter.limit("5 per minute")
def emergency_stop():
    """Emergency stop endpoint"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') if data else None
        
        if session_id and not validate_session(session_id):
            return jsonify({"success": False, "error": "Invalid or expired session"}), 401
        
        # Execute emergency stop
        result = api_service.ai_agent.vehicle_control.emergency_stop()
        
        api_service.monitor_vehicle_command("emergency_stop")
        
        return jsonify({
            "success": True,
            "result": result,
            "message": "Emergency stop activated",
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        api_service.logger.error(f"Emergency stop error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/vehicle/destination', methods=['POST'])
def set_destination():
    """Set vehicle destination"""
    try:
        data = request.get_json()
        
        if not data or 'destination' not in data:
            return jsonify({"success": False, "error": "Destination required"}), 400
        
        session_id = data.get('session_id')
        if session_id and not validate_session(session_id):
            return jsonify({"success": False, "error": "Invalid or expired session"}), 401
        
        destination = data['destination']
        
        # Validate destination
        if not api_service.ai_agent.navigation.is_valid_destination(destination):
            return jsonify({
                "success": False, 
                "error": f"Invalid destination: {destination}"
            }), 400
        
        # Set destination
        result = api_service.ai_agent.vehicle_control.set_destination(destination)
        
        # Calculate ETA
        eta = api_service.ai_agent.navigation.calculate_eta(
            api_service.ai_agent.vehicle_control.get_location(),
            destination
        )
        
        api_service.monitor_vehicle_command("set_destination")
        
        return jsonify({
            "success": True,
            "result": result,
            "eta": eta,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/navigation/route', methods=['GET'])
def get_route():
    """Get route information"""
    try:
        session_id = request.args.get('session_id')
        if session_id and not validate_session(session_id):
            return jsonify({"success": False, "error": "Invalid or expired session"}), 401
        
        from_location = request.args.get('from', api_service.ai_agent.vehicle_control.get_location())
        to_location = request.args.get('to')
        
        if not to_location:
            return jsonify({"success": False, "error": "Destination required"}), 400
        
        # Find route
        route = api_service.ai_agent.navigation.find_route(from_location, to_location)
        eta = api_service.ai_agent.navigation.calculate_eta(from_location, to_location)
        
        return jsonify({
            "success": True,
            "route": route,
            "eta": eta,
            "distance": len(route) if route else 0,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/v1/metrics', methods=['GET'])
@limiter.exempt
def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/api/v1/sessions', methods=['GET'])
def list_sessions():
    """List active sessions (admin endpoint)"""
    try:
        # In production, add authentication/authorization here
        cleanup_sessions()  # Clean expired sessions
        
        sessions_info = []
        for sid, session in active_sessions.items():
            sessions_info.append({
                "session_id": sid,
                "created_at": datetime.fromtimestamp(session["created_at"]).isoformat(),
                "last_activity": datetime.fromtimestamp(session["last_activity"]).isoformat(),
                "requests": session["requests"]
            })
        
        return jsonify({
            "success": True,
            "total_sessions": len(active_sessions),
            "sessions": sessions_info,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# WebSocket support for real-time communication
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('connect')
    def handle_connect():
        session_id = create_session()
        join_room(session_id)
        emit('session_created', {'session_id': session_id})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        # Clean up session if needed
        pass
    
    @socketio.on('voice_message')
    def handle_voice_message(data):
        session_id = data.get('session_id')
        if not validate_session(session_id):
            emit('error', {'message': 'Invalid session'})
            return
        
        # Process voice message
        # Implementation would go here
        emit('voice_response', {'message': 'Voice processing not implemented yet'})
    
    @socketio.on('text_message')
    def handle_text_message(data):
        session_id = data.get('session_id')
        message = data.get('message')
        
        if not validate_session(session_id):
            emit('error', {'message': 'Invalid session'})
            return
        
        try:
            # Process with AI agent
            intent, entities = api_service.ai_agent.intent_engine.recognize_intent(message)
            response = api_service.ai_agent._process_intent(intent, entities, message)
            
            # Generate response text
            if hasattr(api_service.ai_agent.intent_engine, 'generate_response'):
                response_text = api_service.ai_agent.intent_engine.generate_response(
                    intent, entities, {"session_id": session_id}
                )
            else:
                response_text = api_service.ai_agent._format_basic_response(
                    response['type'], response['data']
                )
            
            emit('text_response', {
                'intent': intent.value,
                'response': response_text,
                'vehicle_status': api_service.ai_agent.vehicle_control.get_vehicle_status()
            })
            
        except Exception as e:
            emit('error', {'message': str(e)})

except ImportError:
    # SocketIO not available
    socketio = None

# Background task for session cleanup
def background_cleanup():
    """Background task to clean up expired sessions"""
    while True:
        cleanup_sessions()
        time.sleep(60)  # Clean every minute

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({"success": False, "error": "Rate limit exceeded"}), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

def create_app(llama_model_path=None, use_llama=True):
    """Create and configure the Flask app"""
    # Initialize API service
    init_api_service(llama_model_path, use_llama)
    
    # Start background cleanup task
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Call Agent API Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--llama-model', help='Path to fine-tuned Llama model')
    parser.add_argument('--no-llama', action='store_true', help='Disable Llama integration')
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(
        llama_model_path=args.llama_model,
        use_llama=not args.no_llama
    )
    
    # Run app
    if socketio:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)
