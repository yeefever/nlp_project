from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
from tqdm import tqdm

math_dataset = load_dataset("lighteval/MATH", split="test")

output_file = 'math_output_simple.jsonl'
with open(output_file, "w") as f:
    for idx, example in tqdm(enumerate(math_dataset)):
        # if idx > 50:
        #   break
        problem = example["problem"]
        ground_truth = example.get("solution", "N/A")
        result = {
            "id": idx,
            "problem": problem,
            "ground_truth": ground_truth,
            "model_output": 42
        }
        f.write(json.dumps(result) + "\n")