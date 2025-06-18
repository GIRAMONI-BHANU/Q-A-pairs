import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import numpy as np

# Ensure required directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Model configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LEARNING_RATE = 3e-4
BATCH_SIZE = 4
EPOCHS = 1

def prepare_qa_data():
    """
    Prepare Q&A pairs for training.
    Returns a list of formatted examples.
    """
    # Load data from qa_pairs.json
    with open("data/qa_pairs.json", "r") as f:
        qa_pairs = json.load(f)
    
    formatted_data = []
    for qa in qa_pairs:
        # Format: "<|system|>You are a helpful command-line assistant.</s><|user|>{question}</s><|assistant|>{answer}</s>"
        formatted = f"<|system|>You are a helpful command-line assistant.</s><|user|>{qa['question']}</s><|assistant|>{qa['answer']}</s>"
        formatted_data.append({"text": formatted})
    
    return formatted_data

def create_bnb_config():
    """Create BitsAndBytes configuration for 4-bit quantization"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

def create_peft_config():
    """Create LoRA configuration"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare training data
    train_data = prepare_qa_data()
    dataset = Dataset.from_list(train_data)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        num_proc=1
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=create_bnb_config(),
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, create_peft_config())
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="lora_adapters",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train model
    trainer.train()
    
    # Save trained adapters
    model.save_pretrained("lora_adapters")

if __name__ == "__main__":
    main() 