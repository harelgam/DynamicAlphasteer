import json
import argparse
import os
import torch
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# שימוש במודל Qwen-2.5-7B כשופט חיצוני ואובייקטיבי
JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# הפרומפט להשוואה זוגית (Pairwise)
PAIRWISE_PROMPT = """You are a helpful assistant that acts as an impartial judge.
<|im_start|>user
Select the output (a) or (b) that best matches the given instruction. Choose your preferred output, which can be subjective. Your answer should ONLY contain: Output (a) or Output (b). Here's an example:

# Example:
## Instruction:
Give a description of the following job: "ophthalmologist"

## Output (a):
An ophthalmologist is a medical doctor who specializes in the diagnosis and treatment of eye diseases and conditions.

## Output (b):
An ophthalmologist is a medical doctor who pokes and prods at your eyes while asking you to read letters from a chart.

## Which is best, Output (a) or Output (b)?
Output (a)

# Task:
Now is the real task, do not explain your answer, just say Output (a) or Output (b).

## Instruction:
{instruction}

## Output (a):
{output_1}

## Output (b):
{output_2}

## Which is best, Output (a) or Output (b)?
<|im_end|>
<|im_start|>assistant
"""

def load_judge_model(model_name=JUDGE_MODEL_ID, device="cuda:0"):
    logging.info(f"Loading Judge Model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype="auto", # יטען ב-bfloat16 אם נתמך
            trust_remote_code=True
        )
        return tokenizer, model
    except Exception as e:
        logging.error(f"Failed to load model {model_name}. Make sure you have internet access to download it.")
        raise e

def judge_pair(tokenizer, model, instruction, response_test, response_baseline):
    """
    מבצע שיפוט: האם response_test טוב יותר מ-response_baseline?
    מחזיר 1.0 לניצחון, 0.0 להפסד.
    """
    # החלפה אקראית (Swapping) למניעת הטיית מיקום
    is_swapped = random.random() > 0.5
    
    if is_swapped:
        output_a, output_b = response_baseline, response_test
    else:
        output_a, output_b = response_test, response_baseline

    # בניית הפרומפט
    # הערה: Qwen משתמש ב-Chat Template, אבל כאן בנינו פרומפט ידני שמתאים לרוב המודלים.
    # אם רוצים לדייק עם הטמפלייט של Qwen, אפשר להשתמש ב-tokenizer.apply_chat_template,
    # אבל הפרומפט למעלה תפור למבנה של AlpacaEval ו-Qwen מבין אותו היטב.
    prompt = PAIRWISE_PROMPT.replace("{instruction}", instruction)\
                            .replace("{output_1}", output_a)\
                            .replace("{output_2}", output_b)
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            temperature=0.0, # דטרמיניסטי
            do_sample=False
        )
    
    new_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
    judge_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    winner = None
    if "Output (a)" in judge_output:
        winner = "a"
    elif "Output (b)" in judge_output:
        winner = "b"
    
    if winner is None:
        logging.debug(f"Judge was unclear: {judge_output}")
        return 0.5 # תיקו במקרה של חוסר בהירות

    # פענוח המנצח האמיתי (ביטול ה-Swap)
    # אם לא הוחלף: a=test. אם a ניצח -> אנחנו ניצחנו.
    # אם הוחלף: b=test. אם b ניצח -> אנחנו ניצחנו.
    if (winner == "a" and not is_swapped) or (winner == "b" and is_swapped):
        return 1.0
    return 0.0

def process_file(args):
    # שם קובץ פלט מעודכן
    output_path = args.input_file.replace(".json", "_alpaca_qwen_eval.json")
    
    if os.path.exists(output_path):
        logging.info(f"Output already exists: {output_path}, skipping.")
        return

    tokenizer, model = load_judge_model()
    
    with open(args.input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("data", [])

    if not items:
        logging.warning("File is empty.")
        return

    # זיהוי עמודות
    all_keys = items[0].keys()
    response_keys = [k for k in all_keys if "response_strength" in k]
    
    # מציאת עמודת הבייסליין (ללא היגוי) באופן אוטומטי
    baseline_key = next((k for k in response_keys if "0.0" in k), None)
    
    if not baseline_key:
        logging.error("Could not find a baseline column (strength 0.0). Cannot perform self-comparison.")
        return
    
    logging.info(f"Baseline for comparison: {baseline_key}")

    wins = {k: 0 for k in response_keys if k != baseline_key}
    total = 0

    for item in tqdm(items, desc="Evaluating with Qwen"):
        instruction = item.get("instruction") or item.get("prompt") or item.get("query")
        baseline_response = item.get(baseline_key, "")
        
        for key in response_keys:
            if key == baseline_key: continue
            
            # בדיקה אם כבר חושב
            eval_key = f"win_rate_vs_base_{key}"
            if eval_key in item: continue
            
            target_response = item.get(key, "")
            
            score = judge_pair(tokenizer, model, instruction, target_response, baseline_response)
            
            item[eval_key] = score
            wins[key] += score
        
        total += 1

    # הדפסת סיכום
    if total > 0:
        logging.info(f"=== Results (Win Rate vs {baseline_key}) ===")
        for key, win_count in wins.items():
            rate = (win_count / total) * 100
            logging.info(f"Strength {key}: {rate:.1f}%")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    process_file(args)