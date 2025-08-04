"""
Enhanced Intent Recognition with Llama Model Integration
Supports both rule-based and LLM-based intent recognition for autonomous vehicles
"""
import re
import json
import torch
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import existing intent recognition
from .intent_recognition import Intent, IntentRecognitionEngine as BaseIntentEngine


class LlamaIntentRecognitionEngine(BaseIntentEngine):
    """Enhanced intent recognition using fine-tuned Llama model"""
    
    def __init__(self, 
                 llama_model_path: Optional[str] = None,
                 use_llama: bool = True,
                 fallback_to_rules: bool = True):
        """
        Initialize enhanced intent recognition
        
        Args:
            llama_model_path: Path to fine-tuned Llama model
            use_llama: Whether to use Llama model for intent recognition
            fallback_to_rules: Whether to fallback to rule-based system if Llama fails
        """
        # Initialize base rule-based engine
        super().__init__()
        
        self.llama_model_path = llama_model_path
        self.use_llama = use_llama and TRANSFORMERS_AVAILABLE
        self.fallback_to_rules = fallback_to_rules
        
        # Llama model components
        self.llama_tokenizer = None
        self.llama_model = None
        
        # Load Llama model if specified
        if self.use_llama and llama_model_path:
            self._load_llama_model()
    
    def _load_llama_model(self):
        """Load the fine-tuned Llama model"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                print("Warning: transformers library not available. Using rule-based system.")
                self.use_llama = False
                return
                
            print(f"Loading fine-tuned Llama model from: {self.llama_model_path}")
            
            self.llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_model_path)
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.llama_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
                
            print("Llama model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Llama model: {e}")
            self.use_llama = False
    
    def _query_llama_model(self, text: str) -> str:
        """Query the fine-tuned Llama model for intent recognition"""
        if not self.use_llama or not self.llama_model:
            return ""
        
        try:
            # Create prompt for intent recognition
            prompt = f"""### Instruction: Analyze the following user input for an autonomous vehicle assistant and identify the intent. Respond with only the intent name and any extracted entities in JSON format.

User Input: {text}

Available Intents: emergency_stop, stop_vehicle, set_destination, get_speed, get_location, get_eta, slow_down, speed_up, greeting, unknown

### Response:"""
            
            # Tokenize input
            inputs = self.llama_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.llama_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.llama_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"Error querying Llama model: {e}")
            return ""
    
    def _parse_llama_response(self, response: str) -> Tuple[Intent, Optional[Dict[str, str]]]:
        """Parse Llama model response to extract intent and entities"""
        try:
            # Try to parse as JSON first
            if response.startswith('{') and response.endswith('}'):
                data = json.loads(response)
                intent_str = data.get('intent', 'unknown')
                entities = data.get('entities', {})
            else:
                # Try to extract intent from text response
                intent_str = "unknown"
                entities = {}
                
                # Look for intent keywords in response
                response_lower = response.lower()
                for intent in Intent:
                    if intent.value in response_lower:
                        intent_str = intent.value
                        break
                
                # Extract destination if mentioned
                dest_patterns = [
                    r"destination[:\s]+([^,.\n]+)",
                    r"going to[:\s]+([^,.\n]+)",
                    r"location[:\s]+([^,.\n]+)"
                ]
                for pattern in dest_patterns:
                    match = re.search(pattern, response_lower)
                    if match:
                        entities['destination'] = match.group(1).strip().title()
                        break
            
            # Convert string to Intent enum
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN
            
            return intent, entities if entities else None
            
        except Exception as e:
            print(f"Error parsing Llama response: {e}")
            return Intent.UNKNOWN, None
    
    def recognize_intent(self, text: str) -> Tuple[Intent, Optional[Dict[str, str]]]:
        """
        Enhanced intent recognition using Llama model with rule-based fallback
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (recognized_intent, extracted_entities)
        """
        intent = Intent.UNKNOWN
        entities = None
        
        # Try Llama model first
        if self.use_llama:
            try:
                llama_response = self._query_llama_model(text)
                if llama_response:
                    intent, entities = self._parse_llama_response(llama_response)
                    
                    # If we got a valid intent, return it
                    if intent != Intent.UNKNOWN:
                        return intent, entities
                        
            except Exception as e:
                print(f"Llama intent recognition failed: {e}")
        
        # Fallback to rule-based system
        if self.fallback_to_rules:
            intent, entities = super().recognize_intent(text)
        
        return intent, entities
    
    def get_intent_confidence(self, text: str, intent: Intent) -> float:
        """
        Get confidence score for a specific intent using both Llama and rules
        
        Args:
            text: Input text
            intent: Intent to check
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get rule-based confidence
        rule_confidence = super().get_intent_confidence(text, intent)
        
        # If Llama is available, combine scores
        if self.use_llama:
            try:
                llama_response = self._query_llama_model(text)
                predicted_intent, _ = self._parse_llama_response(llama_response)
                
                # If Llama predicted the same intent, increase confidence
                if predicted_intent == intent:
                    llama_confidence = 0.9  # High confidence from Llama
                    # Combine rule and Llama confidence
                    combined_confidence = (rule_confidence + llama_confidence) / 2
                    return min(combined_confidence, 1.0)
                else:
                    # Llama disagreed, reduce confidence slightly
                    return rule_confidence * 0.8
                    
            except Exception:
                pass
        
        return rule_confidence
    
    def generate_response(self, intent: Intent, entities: Optional[Dict[str, str]], 
                         context: Optional[Dict[str, str]] = None) -> str:
        """
        Generate natural language response using Llama model
        
        Args:
            intent: Recognized intent
            entities: Extracted entities
            context: Additional context (vehicle status, etc.)
            
        Returns:
            Natural language response
        """
        if not self.use_llama:
            return self._generate_rule_based_response(intent, entities, context)
        
        try:
            # Prepare context information
            context_str = ""
            if context:
                context_items = [f"{k}: {v}" for k, v in context.items()]
                context_str = f"Current context: {', '.join(context_items)}"
            
            entities_str = ""
            if entities:
                entity_items = [f"{k}: {v}" for k, v in entities.items()]
                entities_str = f"Entities: {', '.join(entity_items)}"
            
            # Create prompt for response generation
            prompt = f"""### Instruction: Generate a natural, helpful response for an autonomous vehicle assistant.

Intent: {intent.value}
{entities_str}
{context_str}

Generate a clear, concise response appropriate for passengers in an autonomous vehicle.

### Response:"""
            
            # Generate response using Llama
            inputs = self.llama_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.llama_tokenizer.eos_token_id
                )
            
            response = self.llama_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response
            response = response.split('\n')[0]  # Take first line
            response = response.strip('"\'')    # Remove quotes
            
            return response if response else self._generate_rule_based_response(intent, entities, context)
            
        except Exception as e:
            print(f"Error generating Llama response: {e}")
            return self._generate_rule_based_response(intent, entities, context)
    
    def _generate_rule_based_response(self, intent: Intent, entities: Optional[Dict[str, str]], 
                                    context: Optional[Dict[str, str]] = None) -> str:
        """Generate rule-based response as fallback"""
        responses = {
            Intent.EMERGENCY_STOP: "Emergency stop activated. Vehicle stopped immediately for safety.",
            Intent.STOP_VEHICLE: "Bringing the vehicle to a controlled stop.",
            Intent.GET_SPEED: f"Current speed is {context.get('speed', 'unknown')} km/h." if context else "Checking current speed.",
            Intent.GET_LOCATION: f"We are currently at {context.get('location', 'unknown location')}." if context else "Checking current location.",
            Intent.SET_DESTINATION: f"Setting destination to {entities.get('destination', 'unknown')}." if entities else "Please specify a destination.",
            Intent.GET_ETA: f"Estimated arrival time is {context.get('eta', 'calculating')}." if context else "Calculating estimated arrival time.",
            Intent.SLOW_DOWN: "Reducing vehicle speed.",
            Intent.SPEED_UP: "Increasing vehicle speed.",
            Intent.GREETING: "Hello! I'm your autonomous vehicle assistant. How can I help you?",
            Intent.UNKNOWN: "I didn't understand that command. Please try again."
        }
        
        return responses.get(intent, "I'm sorry, I couldn't process that request.")


# Factory function to create the appropriate intent engine
def create_intent_engine(llama_model_path: Optional[str] = None, 
                        use_llama: bool = True) -> Union[LlamaIntentRecognitionEngine, BaseIntentEngine]:
    """
    Factory function to create the appropriate intent recognition engine
    
    Args:
        llama_model_path: Path to fine-tuned Llama model
        use_llama: Whether to use Llama model
        
    Returns:
        Intent recognition engine instance
    """
    if use_llama and llama_model_path and Path(llama_model_path).exists():
        return LlamaIntentRecognitionEngine(llama_model_path, use_llama=True)
    elif use_llama:
        # Try to use Llama without custom model path
        return LlamaIntentRecognitionEngine(use_llama=False, fallback_to_rules=True)
    else:
        # Use rule-based system
        return BaseIntentEngine()
