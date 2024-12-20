# Evaluation Metric for MATH Dataset

This document describes the evaluation metric used to assess the performance of language models on the MATH dataset. It provides a formal definition of the accuracy metric and instructions on how to run the evaluation script.

## Accuracy Metric Definition

**Accuracy** is calculated based on the model's ability to produce the correct final answer to mathematical problems in the MATH dataset. The evaluation process involves the following steps for each problem in the test set:

1. **Tokenization and Decoding**:
   - Both the model's predicted output and the ground truth solution are tokenized and decoded into text strings, removing any special tokens or padding.

2. **Extraction of Final Answers**:
   - From the decoded texts, the final answers are extracted by searching for the last occurrence of the LaTeX expression enclosed within `\boxed{...}`.
   - This pattern is chosen because, in the MATH dataset, the final answers are typically formatted within `\boxed{...}` to denote the solution.

3. **Comparison**:
   - The extracted answers from the model's prediction (`pred_answer`) and the ground truth (`label_answer`) are compared for exact string equality.

4. **Accuracy Calculation**:
   - The number of correct predictions is summed over all problems.
   - **Accuracy** is computed as:

     \[
     \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Problems}}
     \]

## Running the Evaluation Script

The evaluation script `eval.py` assesses a Hugging Face language model's performance on the MATH dataset using the accuracy metric defined above.

### Prerequisites

- Python 3.x
- Required Python packages:
  - `torch`
  - `transformers`
  - `datasets`
  - `numpy`
  - `re`

### Usage

```bash
python eval.py --model <model_name_or_path> [--batch_size <batch_size>] [--max_length <max_length>] [--should_pad]
```

#### Arguments

- `--model`: **(Required)** The name or path of the Hugging Face model to evaluate.
- `--batch_size`: *(Optional)* The batch size to use during evaluation. Default is `1`.
- `--max_length`: *(Optional)* The maximum sequence length for tokenization. Default is `1300`.
- `--should_pad`: *(Optional)* If set, the inputs will be padded to `max_length`.

#### Example

Evaluate the `meta-llama/Llama-3.2-1B` model with a batch size of `2`:

```bash
python eval.py --model meta-llama/Llama-3.2-1B --batch_size 2
```

#### Sample Output

```plaintext
Final accuracy: 0.215
```

- **Preds type and Labels type**: Information about the data types and shapes of the predictions and labels arrays.
- **Final accuracy**: The computed accuracy of the model on the MATH test set.

## Understanding the Output

- The **Final accuracy** indicates the proportion of problems for which the model's predicted answer exactly matches the ground truth answer extracted from the `\boxed{...}` expression.
- An accuracy of `0.215` means that the model correctly answered 21.5% of the problems in the test set.

## Notes

- The evaluation focuses on the correctness of the final answer, not on the reasoning or intermediate steps.
- Ensure that the model and tokenizer are compatible, especially when working with specialized models like LLaMA.
- The script uses the `LlamaTokenizer` for models based on LLaMA architecture. Adjust the tokenizer if using a different model type.

## Troubleshooting

- **Memory Errors**: If you encounter out-of-memory errors, try reducing the `--batch_size` or `--max_length`.
- **Tokenizer Issues**: If the outputs are garbled or contain unexpected characters, verify that the correct tokenizer is being used for your model.


### Simple Evaluation

Just takes in the output of the simple jsonlines file and checks using regex if the boxed answer is the same as model "output". 

Run python eval-simple.py after simple-baseline.py