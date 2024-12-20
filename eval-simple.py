import json
from tqdm import tqdm

def extract_boxed_number(text):
    start = text.find("\\boxed{")
    if start == -1:
        return None
    end = text.find("}", start)
    if end == -1:
        return None
    try:
        return int(text[start + 7:end])
    except ValueError:
        return None

def evaluate(file_path):
    total = 0
    correct = 0

    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Evaluating"):
            data = json.loads(line)
            ground_truth = data.get("ground_truth", "")
            model_output = data.get("model_output")

            # Extract the boxed number from ground_truth
            boxed_number = extract_boxed_number(ground_truth)

            # Compare with model_output
            if boxed_number is not None:
                total += 1
                if boxed_number == model_output:
                    correct += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Correct: {correct}")

if __name__ == "__main__":
    file_path = "math_output_simple.jsonl"
    evaluate(file_path)
