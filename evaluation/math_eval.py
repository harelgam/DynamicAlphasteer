import json
import argparse
import re
import os
from tqdm import tqdm

def extract_answer(text):
    # מנסה למצוא פורמט \boxed{15}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match: return match.group(1).strip()
    # מנסה למצוא את המספר האחרון בטקסט
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers: return numbers[-1]
    return None

def normalize(ans):
    if not ans: return ""
    return str(ans).lower().replace(',', '').replace(' ', '').replace('$', '').strip()

def process_file(args):
    output_path = args.input_file.replace(".json", "_eval.json")
    if os.path.exists(output_path):
        print(f"Skipping {output_path}")
        return

    with open(args.input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    items = data if isinstance(data, list) else data.get("data", [])
    
    for item in tqdm(items, desc=f"Evaluating Math {os.path.basename(args.input_file)}"):
        # מציאת התשובה הנכונה מהדאטה
        ground_truth = item.get("answer") or item.get("solution") or item.get("target")
        if "boxed" in str(ground_truth):
            ground_truth = extract_answer(str(ground_truth))
        
        clean_gt = normalize(ground_truth)
        
        # בדיקת כל רמות החוזק
        for key in list(item.keys()):
            if "response_strength" in key and "eval" not in key:
                model_ans = extract_answer(item.get(key, ""))
                clean_model = normalize(model_ans)
                
                # 1 = נכון, 0 = לא נכון
                score = 1 if (clean_gt and clean_model and clean_gt == clean_model) else 0
                item[f"eval_{key}"] = score

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    process_file(args)