import sys
import json
import time
import jsonlines
import os
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Constants
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "lora_adapters"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

class CommandLineAgent:
    def __init__(self):
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()

    def generate_response(self, user_input: str) -> str:
        """Generate a response for the given user input"""
        prompt = f"<|system|>You are a helpful command-line assistant. Provide a step-by-step plan. If the first step is a command, prefix it with 'CMD:'.</s><|user|>{user_input}</s><|assistant|>"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("<|assistant|>")[-1].strip()
        return response

    def log_interaction(self, user_input: str, response: str, executed_cmd: str = None):
        """Log the interaction to trace.jsonl"""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "response": response,
            "executed_command": executed_cmd
        }
        
        with jsonlines.open("logs/trace.jsonl", mode="a") as writer:
            writer.write(log_entry)

    def process_input(self, user_input: str):
        """Process user input and handle command execution"""
        # Generate response
        response = self.generate_response(user_input)
        
        # Check if first line is a command
        lines = response.split("\n")
        executed_cmd = None
        
        if lines and lines[0].startswith("CMD:"):
            cmd = lines[0][4:].strip()
            print(f"\nDry run command: {cmd}\n")
            executed_cmd = cmd
        
        # Print full response
        print("\nStep-by-step plan:")
        print(response)
        
        # Log interaction
        self.log_interaction(user_input, response, executed_cmd)

def main():
    if len(sys.argv) != 2:
        print("Usage: python agent.py \"<natural language command>\"")
        sys.exit(1)
    
    user_input = sys.argv[1]
    agent = CommandLineAgent()
    agent.process_input(user_input)

if __name__ == "__main__":
    main() 