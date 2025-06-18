# Command Line Assistant - Technical Task

This project implements a command-line assistant that uses a fine-tuned language model to help with shell commands.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── data/
│   └── qa_pairs.json
├── agent.py
├── train.py
├── eval/
│   ├── eval_static.md
│   └── eval_dynamic.md
├── logs/
│   └── trace.jsonl
└── report.md
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare the model:
```bash
python train.py
```

## Usage

Run the agent with a natural language command:
```bash
python agent.py "Create a new Git branch and switch to it"
```

## Evaluation

- Static evaluation results are in `eval/eval_static.md`
- Dynamic evaluation results are in `eval/eval_dynamic.md`
- Full project report is in `report.md`

## Notes

- The model uses LoRA fine-tuning on TinyLlama-1.1B
- Training data consists of 150+ curated command-line Q&A pairs
- All outputs are logged to `logs/trace.jsonl` 