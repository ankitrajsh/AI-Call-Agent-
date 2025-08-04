"""
Llama Model Fine-tuning Module for Autonomous Vehicle Assistant
This module handles fine-tuning Llama models on custom autonomous vehicle datasets
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousVehicleDataset(Dataset):
    """Dataset class for autonomous vehicle conversations and queries"""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the conversation for training
        if "instruction" in item and "response" in item:
            # Instruction-following format
            text = f"### Instruction: {item['instruction']}\n### Response: {item['response']}"
        elif "query" in item and "answer" in item:
            # Q&A format
            text = f"User: {item['query']}\nAssistant: {item['answer']}"
        else:
            # Generic conversation format
            text = f"{item.get('input', '')}\n{item.get('output', '')}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


class LlamaFineTuner:
    """Fine-tuning class for Llama models on autonomous vehicle data"""
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        output_dir: str = "./fine_tuned_llama_av",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load the base Llama model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_dataset(self, data_path: str) -> HFDataset:
        """Prepare the dataset for training"""
        logger.info(f"Loading dataset from: {data_path}")
        
        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        dataset = AutonomousVehicleDataset(data, self.tokenizer, self.max_length)
        
        # Convert to HuggingFace dataset
        hf_dataset = HFDataset.from_list([dataset[i] for i in range(len(dataset))])
        
        logger.info(f"Dataset prepared with {len(hf_dataset)} samples")
        return hf_dataset
    
    def fine_tune(
        self,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        """Fine-tune the Llama model"""
        logger.info("Starting fine-tuning process...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard logging
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Training started...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        logger.info(f"Fine-tuning completed! Model saved to: {self.output_dir}")
    
    def create_sample_dataset(self, output_path: str = "autonomous_vehicle_dataset.json"):
        """Create a sample dataset for autonomous vehicle fine-tuning"""
        
        sample_data = [
            {
                "instruction": "What should I do if a passenger requests an emergency stop?",
                "response": "Immediately activate the emergency stop system by setting the vehicle state to emergency stop, bringing the vehicle to a safe halt, and announcing the emergency stop to all passengers. Ensure all safety protocols are followed."
            },
            {
                "instruction": "How do I set the destination to the Main Gate?",
                "response": "To set the destination to Main Gate, use the vehicle control API to update the destination parameter. Validate the destination is in the approved campus locations list, then update the navigation system with the new route."
            },
            {
                "instruction": "A passenger asks about the current vehicle speed.",
                "response": "Check the vehicle control system status and report the current speed in km/h. For example: 'Our current speed is 15.5 kilometers per hour.'"
            },
            {
                "instruction": "How do I calculate the ETA to Hostel Circle?",
                "response": "Use the navigation engine to calculate ETA by getting the current location, destination distance, and current vehicle speed. The formula is: ETA = distance / speed, then format the result in minutes or hours as appropriate."
            },
            {
                "instruction": "What are the safety protocols for autonomous vehicle operation?",
                "response": "Key safety protocols include: 1) Emergency stop capability with highest priority, 2) Speed limits not exceeding 25 km/h on campus, 3) Continuous monitoring of passenger requests, 4) Immediate response to safety commands, 5) Clear audio feedback for all actions."
            },
            {
                "instruction": "How should the vehicle respond to 'slow down' command?",
                "response": "Reduce the vehicle speed by the configured increment (typically 5 km/h). Announce the speed change to passengers, ensure the new speed doesn't go below 0, and update the vehicle status accordingly."
            },
            {
                "instruction": "What campus locations are available for navigation?",
                "response": "Available campus locations include: Main Gate, Hostel Circle, Academic Block, Library, and Cafeteria. These are pre-configured safe destinations with established routes."
            },
            {
                "instruction": "How do I handle an unrecognized voice command?",
                "response": "For unrecognized commands, politely ask the passenger to repeat or clarify their request. Provide examples of valid commands like 'stop the vehicle', 'go to main gate', or 'what's our current speed'."
            },
            {
                "instruction": "What information should be included in vehicle status reports?",
                "response": "Vehicle status should include: current speed, location, destination (if set), vehicle state (stopped/moving/emergency), and estimated time of arrival if applicable."
            },
            {
                "instruction": "How do I ensure passenger safety during operation?",
                "response": "Maintain safe speeds, respond immediately to emergency commands, provide clear audio feedback, monitor vehicle systems continuously, and prioritize safety over convenience in all decisions."
            }
        ]
        
        # Add more conversational examples
        conversation_data = [
            {
                "query": "Take me to the library please",
                "answer": "Setting destination to Library. Calculating route from your current location. Estimated arrival time is 2 minutes."
            },
            {
                "query": "How fast are we going?",
                "answer": "We are currently traveling at 18 kilometers per hour."
            },
            {
                "query": "Stop the vehicle",
                "answer": "Bringing the vehicle to a controlled stop. We are now stopped."
            },
            {
                "query": "Where are we right now?",
                "answer": "We are currently at the Academic Block."
            },
            {
                "query": "Emergency stop now!",
                "answer": "Emergency stop activated. All vehicle systems halted immediately for safety."
            }
        ]
        
        # Combine all data
        all_data = sample_data + conversation_data
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sample dataset created with {len(all_data)} examples: {output_path}")
        return output_path


def main():
    """Example usage of the fine-tuning system"""
    # Initialize fine-tuner
    fine_tuner = LlamaFineTuner(
        model_name="meta-llama/Llama-2-7b-chat-hf",  # or use a smaller model for testing
        output_dir="./fine_tuned_llama_av"
    )
    
    # Create sample dataset
    dataset_path = fine_tuner.create_sample_dataset()
    
    # Load model and tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Prepare dataset
    train_dataset = fine_tuner.prepare_dataset(dataset_path)
    
    # Split dataset (80% train, 20% eval)
    split_dataset = train_dataset.train_test_split(test_size=0.2)
    train_ds = split_dataset['train']
    eval_ds = split_dataset['test']
    
    # Fine-tune the model
    fine_tuner.fine_tune(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        num_epochs=3,
        learning_rate=2e-5,
        batch_size=2,  # Small batch size for limited memory
        gradient_accumulation_steps=8
    )
    
    print("Fine-tuning completed!")


if __name__ == "__main__":
    main()
