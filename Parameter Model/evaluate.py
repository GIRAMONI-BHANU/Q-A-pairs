import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
from typing import List, Dict
import json
import os

# Constants
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "lora_adapters"
TEST_PROMPTS = [
    "Create a new Git branch and switch to it",
    "How do I compress multiple files into a tar.gz archive?",
    "Search for all Python files containing the word 'import'",
    "Set up a new Python project with virtual environment",
    "Show disk usage for all directories, sorted by size"
]
EDGE_CASES = [
    "Delete all files recursively without confirmation",
    "Find files modified in the last 24 hours and copy to backup"
]

class ModelEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model,
            ADAPTER_PATH
        )
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        
    def generate_response(self, model, prompt: str) -> str:
        formatted_prompt = f"<|system|>You are a helpful command-line assistant. Provide a step-by-step plan. If the first step is a command, prefix it with 'CMD:'.</s><|user|>{prompt}</s><|assistant|>"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()
    
    def evaluate_prompts(self) -> Dict:
        results = {
            "test_prompts": [],
            "edge_cases": [],
            "metrics": {
                "avg_bleu": 0.0,
                "avg_rouge_l": 0.0
            }
        }
        
        # Evaluate test prompts
        for prompt in TEST_PROMPTS:
            base_response = self.generate_response(self.base_model, prompt)
            ft_response = self.generate_response(self.fine_tuned_model, prompt)
            
            bleu_score = self.bleu_metric.compute(
                predictions=[ft_response],
                references=[base_response]
            )["bleu"]
            
            rouge_score = self.rouge_metric.compute(
                predictions=[ft_response],
                references=[base_response]
            )["rougeL"]
            
            results["test_prompts"].append({
                "prompt": prompt,
                "base_response": base_response,
                "ft_response": ft_response,
                "bleu": bleu_score,
                "rouge_l": rouge_score
            })
        
        # Evaluate edge cases
        for prompt in EDGE_CASES:
            base_response = self.generate_response(self.base_model, prompt)
            ft_response = self.generate_response(self.fine_tuned_model, prompt)
            
            results["edge_cases"].append({
                "prompt": prompt,
                "base_response": base_response,
                "ft_response": ft_response
            })
        
        # Calculate average metrics
        bleu_scores = [r["bleu"] for r in results["test_prompts"]]
        rouge_scores = [r["rouge_l"] for r in results["test_prompts"]]
        results["metrics"]["avg_bleu"] = sum(bleu_scores) / len(bleu_scores)
        results["metrics"]["avg_rouge_l"] = sum(rouge_scores) / len(rouge_scores)
        
        return results

    def update_eval_files(self, results: Dict):
        # Update static evaluation
        with open("eval/eval_static.md", "r") as f:
            static_content = f.read()
        
        for result in results["test_prompts"]:
            static_content = static_content.replace(
                f'### {result["prompt"]}\n**Base Model:**\n```\n[Response will be filled after evaluation]\n```',
                f'### {result["prompt"]}\n**Base Model:**\n```\n{result["base_response"]}\n```'
            )
            static_content = static_content.replace(
                f'**Fine-tuned Model:**\n```\n[Response will be filled after evaluation]\n```\n\n**BLEU Score:** [TBD]\n**ROUGE-L Score:** [TBD]',
                f'**Fine-tuned Model:**\n```\n{result["ft_response"]}\n```\n\n**BLEU Score:** {result["bleu"]:.4f}\n**ROUGE-L Score:** {result["rouge_l"]:.4f}'
            )
        
        for result in results["edge_cases"]:
            static_content = static_content.replace(
                f'### {result["prompt"]}\n**Base Model:**\n```\n[Response will be filled after evaluation]\n```',
                f'### {result["prompt"]}\n**Base Model:**\n```\n{result["base_response"]}\n```'
            )
            static_content = static_content.replace(
                f'**Fine-tuned Model:**\n```\n[Response will be filled after evaluation]\n```',
                f'**Fine-tuned Model:**\n```\n{result["ft_response"]}\n```'
            )
        
        with open("eval/eval_static.md", "w") as f:
            f.write(static_content)
        
        # Update dynamic evaluation
        with open("eval/eval_dynamic.md", "r") as f:
            dynamic_content = f.read()
        
        for result in results["test_prompts"] + results["edge_cases"]:
            cmd = None
            for line in result["ft_response"].split("\n"):
                if line.startswith("CMD:"):
                    cmd = line[4:].strip()
                    break
            
            if cmd:
                dynamic_content = dynamic_content.replace(
                    f'### {result["prompt"]}\n**Command Generated:** [TBD]',
                    f'### {result["prompt"]}\n**Command Generated:** {cmd}'
                )
        
        with open("eval/eval_dynamic.md", "w") as f:
            f.write(dynamic_content)

def main():
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_prompts()
    evaluator.update_eval_files(results)
    
    # Print summary
    print(f"Evaluation completed!")
    print(f"Average BLEU Score: {results['metrics']['avg_bleu']:.4f}")
    print(f"Average ROUGE-L Score: {results['metrics']['avg_rouge_l']:.4f}")

if __name__ == "__main__":
    main() 