#!/usr/bin/env python3
"""
AlpacaEval for Dynamic AlphaSteer (Pairwise Win-Rate Version)
Evaluates win-rate of dynamic responses against baseline (strength 0.0).
"""
import json
import argparse
import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Judge Model
JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# PAIRWISE PROMPT
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
            torch_dtype="auto",
            trust_remote_code=True
        )
        return tokenizer, model
    except Exception as e:
        logging.error(f"Failed to load model {model_name}")
        raise e

def judge_pair(tokenizer, model, instruction, response_test, response_baseline):
    """
    Compares test response vs baseline.
    Returns 1.0 if test wins, 0.0 if baseline wins.
    """
    # Swap to avoid position bias
    is_swapped = random.random() > 0.5
    
    if is_swapped:
        output_a, output_b = response_baseline, response_test
    else:
        output_a, output_b = response_test, response_baseline

    prompt = PAIRWISE_PROMPT.replace("{instruction}", instruction)\
                            .replace("{output_1}", output_a)\
                            .replace("{output_2}", output_b)
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            temperature=0.0,
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
        return 0.5 # Tie/Error

    # Resolve winner
    # If not swapped: a=test. If a wins -> test wins (1.0)
    # If swapped: b=test. If b wins -> test wins (1.0)
    if (winner == "a" and not is_swapped) or (winner == "b" and is_swapped):
        return 1.0
    return 0.0

def process_file(args):
    # התיקון שביקשת: חזרה לשם הקובץ המקורי
    output_path = args.input_file.replace(".json", "_alpaca_dynamic_eval.json")
    
    if os.path.exists(output_path):
        logging.info(f"Output already exists: {output_path}, skipping.")
        return

    tokenizer, model = load_judge_model(device=args.device)
    
    with open(args.input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("data", [])

    if not items:
        logging.warning("File is empty.")
        return

    # 1. Locate Baseline Column (strength 0.0)
    all_keys = items[0].keys()
    baseline_key = next((k for k in all_keys if "response_strength" in k and "0.0" in k), None)
    
    if not baseline_key:
        logging.error("No baseline column found (looking for 'response_strength' and '0.0'). Cannot compare.")
        logging.info("Available keys: " + str(list(all_keys)))
        return
    
    # 2. Locate Dynamic Column
    dynamic_key = "response_dynamic"
    if dynamic_key not in items[0]:
        logging.error(f"No '{dynamic_key}' column found. Run generation first.")
        return

    logging.info(f"Comparing '{dynamic_key}' vs Baseline '{baseline_key}'")
    
    wins = 0
    total = 0
    
    for item in tqdm(items, desc="Evaluating Win Rate"):
        instruction = item.get("instruction") or item.get("prompt") or item.get("query")
        
        resp_dynamic = item.get(dynamic_key, "")
        resp_baseline = item.get(baseline_key, "")
        
        if not resp_dynamic or not resp_baseline:
            continue

        if "win_dynamic_vs_baseline" in item:
            wins += item["win_dynamic_vs_baseline"]
            total += 1
            continue
        
        score = judge_pair(tokenizer, model, instruction, resp_dynamic, resp_baseline)
        
        item["win_dynamic_vs_baseline"] = score
        wins += score
        total += 1
    
    if total > 0:
        win_rate = (wins / total) * 100
        logging.info(f"=== Final Results ===")
        logging.info(f"Dynamic Model Win Rate: {win_rate:.2f}% (vs {baseline_key})")
        logging.info(f"Total examples: {total}")
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    process_file(args)