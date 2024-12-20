# Math Problem Solver with Few-Shot Learning

This folder contains two implementations for fine-tuning Large Language Models on mathematical problems using the MATH dataset: one with few-shot examples and one without.

## Prerequisites
- pip install transformers datasets peft accelerate torch setproctitle


## Files
- `Few Shot.py`: Implementation with few-shot learning examples
- `peft.py`: Implementation without few-shot examples

## Usage

### Basic Usage

Without few-shot examples
```python peft.py --model meta-llama/Llama-3.2-1B --epochs 1 --batch_size 32```
With few-shot examples
```python "Few Shot.py" --model meta-llama/Llama-3.2-1B --epochs 1 --batch_size 32```


### Key Arguments

- `--model`: HuggingFace model to use (default: "meta-llama/Llama-3.2-1B")
- `--epochs`: Number of training epochs (default: 1)
- `--batch_size`: Batch size for training (default: 32)
- `--max_length`: Maximum sequence length (default: 512)

## Key Features

1. **Few-Shot Learning**: The `Few Shot.py` implementation includes predefined examples to guide the model's learning
2. **PEFT Support**: Both implementations support Parameter-Efficient Fine-Tuning methods (LoRA, IA³)
3. **Metrics**: Tracks accuracy using boxed answers in mathematical notation


## Notes

1. Make sure you have sufficient GPU memory for your chosen model and batch size
2. The few-shot implementation may require more memory due to longer sequences
3. Both implementations support the same PEFT methods and training options