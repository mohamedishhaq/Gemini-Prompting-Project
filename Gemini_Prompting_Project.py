import os
import json
import csv
from pathlib import Path
from datetime import datetime
import google.generativeai as genai

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA7wqgOGl5T8BoUXH0j0GqHuAscFfMbo84")
genai.configure(api_key=GEMINI_API_KEY)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results_task1"
OUTPUT_DIR.mkdir(exist_ok=True)
pr


# ---------------- PROMPT HELPERS ----------------
def build_zero_shot_prompt(task):
    return f"Question: {task['input']}\nAnswer:"


def build_few_shot_prompt(task, examples):
    prompt = "Here are some examples:\n"
    for ex in examples:
        prompt += f"Q: {ex['input']}\nA: {ex['output']}\n\n"
    prompt += f"Now answer this:\nQ: {task['input']}\nA:"
    return prompt


def build_cot_prompt(task):
    return f"Question: {task['input']}\nLet's think step by step."


# ---------------- RUN MODEL ----------------
def call_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "Error: No response"
    except Exception as e:
        return f"Error: {str(e)}"


def run_for_task(task, pool, idx):
    result = {"task_id": idx + 1}

    # Zero-shot
    zs = build_zero_shot_prompt(task)
    result["zero_shot"] = call_gemini(zs)

    # Few-shot (first 2 examples)
    try:
        fs = build_few_shot_prompt(task, pool[1:3])
        result["few_shot"] = call_gemini(fs)
    except Exception as e:
        result["few_shot"] = f"Error: {str(e)}"

    # Chain of Thought
    cot = build_cot_prompt(task)
    result["cot"] = call_gemini(cot)

    return result


# ---------------- SAVE HELPERS ----------------
def save_json(results, filename):
    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_csv(results, filename):
    keys = ["task_id", "zero_shot", "few_shot", "cot"]
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Example dataset (replace with your actual one)
    dataset = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is 2+2?", "output": "4"},
        {"input": "Who wrote Hamlet?", "output": "Shakespeare"},
    ]

    all_results = []
    for i, task in enumerate(dataset):
        print(f"⚡ Running task {i+1} ...")
        res = run_for_task(task, dataset, i)
        all_results.append(res)

    # timestamped filenames
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    json_file = f"task1_{timestamp}.json"
    csv_file = f"task1_{timestamp}.csv"

    # save outputs
    save_json(all_results, json_file)
    save_csv(all_results, csv_file)

    print(f"✅ Results saved in {OUTPUT_DIR}/{json_file} and {OUTPUT_DIR}/{csv_file}")
