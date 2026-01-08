#!/bin/bash

cd /gpfs0/bgu-ataitler/users/harelgam/AlphaSteer || exit

echo "======================================================="
echo "STEP 1: Safety Evaluation (LlamaGuard)"
echo "Checking: Jailbroken, GCG, AutoDAN, PAIR, AIM, Renellm"
echo "======================================================="

# רשימת קבצי ההתקפה לבדיקה
declare -a attack_files=("jailbroken" "gcg" "autodan" "pair" "aim" "renellm" "cipher")

for model in "${attack_files[@]}"; do
    FILE="data/responses/llama3.1/${model}_llama3.1_results.json"
    if [ -f "$FILE" ]; then
        echo ">>> Evaluating Safety for: $model"
        # שימוש בטוקן מהסביבה באופן אוטומטי
        python evaluation/jailbreak_llamaguard.py --input-file "$FILE" --max-model-len 4096
    else
        echo "WARNING: File not found: $FILE"
    fi
done

echo "======================================================="
echo "STEP 2: Math Capability Evaluation"
echo "Checking: GSM8K, Math"
echo "======================================================="

python evaluation/math_eval.py --input_file data/responses/llama3.1/gsm8k_llama3.1_results.json
python evaluation/math_eval.py --input_file data/responses/llama3.1/math_llama3.1_results.json

echo "======================================================="
echo "STEP 3: General Quality & Usability"
echo "Checking: XSTest, Alpaca"
echo "======================================================="

python evaluation/xstest_local.py --input_file data/responses/llama3.1/xstest_llama3.1_results.json
python evaluation/alpaca_local.py --input_file data/responses/llama3.1/alpaca_eval_llama3.1_results.json

echo "======================================================="
echo "DONE! All evaluations finished."
echo "Check folders for files ending in *_eval.json or *_v.json"
echo "======================================================="
