import json
import argparse
import re
import os
from tqdm import tqdm

def extract_answer(text):
    if not text: return None
    # 1. מנסה למצוא פורמט \boxed{15} (סטנדרט ב-GSM8K)
    match = re.search(r'\\boxed\{([^}]+)\}', str(text))
    if match: return match.group(1).strip()
    
    # 2. מנסה למצוא פורמט #### 15 (סטנדרט נוסף)
    match_hash = re.search(r'####\s*(-?\d+\.?\d*)', str(text))
    if match_hash: return match_hash.group(1).strip()

    # 3. מנסה למצוא את המספר האחרון בטקסט (גיבוי)
    numbers = re.findall(r'-?\d+\.?\d*', str(text).replace(',', ''))
    if numbers: return numbers[-1]
    
    return None

def normalize(ans):
    if not ans: return ""
    # מנקה רווחים, פסיקים, דולרים, והופך למחרוזת פשוטה להשוואה
    return str(ans).lower().replace(',', '').replace(' ', '').replace('$', '').strip()

def process_file(args):
    # קובע את שם קובץ הפלט
    # אם הקלט כבר מסתיים ב-_eval, הוא ידרוס אותו. אחרת ייצור חדש.
    if args.input_file.endswith("_eval.json"):
        output_path = args.input_file
    else:
        output_path = args.input_file.replace(".json", "_eval.json")

    print(f"Processing: {args.input_file} -> {output_path}")

    with open(args.input_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    items = data if isinstance(data, list) else data.get("data", [])
    
    total_correct = 0
    total_samples = 0

    for item in tqdm(items, desc=f"Evaluating Math"):
        # מציאת התשובה הנכונה (Ground Truth)
        ground_truth = item.get("answer") or item.get("solution") or item.get("target")
        
        # חילוץ המספר הנקי מתוך ה-Ground Truth
        gt_extracted = extract_answer(ground_truth)
        clean_gt = normalize(gt_extracted)
        
        # מעבר על כל המפתחות באובייקט כדי למצוא תשובות של מודל
        for key in list(item.keys()):
            # --- התיקון המרכזי כאן ---
            # בודק אם המפתח מתחיל ב-response (תופס גם dynamic וגם strength)
            if key.startswith("response") and "eval" not in key:
                
                model_raw = item.get(key, "")
                model_extracted = extract_answer(model_raw)
                clean_model = normalize(model_extracted)
                
                # השוואה בינארית (1 או 0)
                # מוודאים שגם ה-GT וגם המודל לא ריקים לפני השוואה
                if clean_gt and clean_model and clean_gt == clean_model:
                    score = 1
                else:
                    score = 0
                
                # שמירת הציון בתוך האובייקט
                item[f"eval_{key}"] = score
                
                # מעקב סטטיסטי (רק עבור המפתח הראשי אם יש כמה)
                if "dynamic" in key or "strength" in key: 
                    total_correct += score
                    total_samples += 1

    # חישוב דיוק סופי להצגה
    if total_samples > 0:
        accuracy = (total_correct / total_samples) * 100
        print(f"\nResults for {os.path.basename(args.input_file)}:")
        print(f"Total Samples: {total_samples}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo scorable responses found.")

    # שמירת הקובץ המעודכן
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(items, f, indent=4) # שומר כ-list שטוח במקום dict מורכב אם היה
    
    print(f"Saved evaluated file to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    process_file(args)