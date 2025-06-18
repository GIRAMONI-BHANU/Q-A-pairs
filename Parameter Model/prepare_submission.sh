#!/bin/bash

# Create submission directory
echo "Creating submission directory..."
mkdir -p submission

# Copy required files
echo "Copying files to submission directory..."
cp README.md submission/
cp requirements.txt submission/
cp -r data submission/
cp train.py submission/
cp agent.py submission/
cp evaluate.py submission/
cp run_pipeline.sh submission/
cp -r eval submission/
cp report.md submission/

# Create submission checklist
echo "Creating submission checklist..."
cat > submission/CHECKLIST.md << EOL
# Submission Checklist

## Required Files
- [ ] README.md (Project documentation and setup instructions)
- [ ] requirements.txt (Python dependencies)
- [ ] data/qa_pairs.json (≥150 Q&A pairs)
- [ ] train.py (Model fine-tuning script)
- [ ] agent.py (CLI agent implementation)
- [ ] evaluate.py (Evaluation script)
- [ ] run_pipeline.sh (End-to-end pipeline script)
- [ ] eval/eval_static.md (Static evaluation results)
- [ ] eval/eval_dynamic.md (Dynamic evaluation results)
- [ ] report.md (Project report)
- [ ] demo.mp4 (≤5 min demo video)

## Verification Steps
- [ ] Run pipeline successfully: \`./run_pipeline.sh\`
- [ ] Count Q&A pairs (should be ≥150): \`python -c "import json; print(len(json.load(open('data/qa_pairs.json'))))"\`
- [ ] Check evaluation results in eval/ directory
- [ ] Test agent with sample command: \`python agent.py "Create a new Git branch and switch to it"\`
- [ ] Review report.md for completeness
- [ ] Verify demo video length (≤5 min)
- [ ] Check no repository or public links are included
- [ ] Test in fresh environment

## Submission Instructions
1. Record demo video and save as demo.mp4
2. Add demo.mp4 to submission directory
3. Create ZIP file: \`zip -r submission.zip submission/\`
4. Email to hr@fenrir-security.com
   - Subject: "AI/ML Internship Technical Task Submission – [Your Name]"
   - Attach: submission.zip
   - Deadline: 10 PM IST, Wednesday 18 June 2025
EOL

echo "Submission directory prepared. Please:"
echo "1. Record your demo video"
echo "2. Save it as 'submission/demo.mp4'"
echo "3. Review submission/CHECKLIST.md"
echo "4. Create ZIP file: zip -r submission.zip submission/"

# Run the complete pipeline
./run_pipeline.sh

# Test the agent
python agent.py "Create a new Git branch and switch to it"

# Count Q&A pairs
python -c "import json; print(len(json.load(open('data/qa_pairs.json'))))" 