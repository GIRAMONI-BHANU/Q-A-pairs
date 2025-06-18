#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Train model
echo "Training model..."
python train.py

# Run evaluation
echo "Running evaluation..."
python evaluate.py

# Test agent
echo "Testing agent with sample command..."
python agent.py "Create a new Git branch and switch to it"

echo "Pipeline completed! Check eval/ directory for results." 