### Strong baseline

This file contains a script to fine-tune transformer-based language models on the MATH dataset. It includes features for tokenization, fine-tuning, and evaluation with an emphasis on modularity and experimentation.

### Features

- Dataset Support: Prepares and tokenizes the MATH dataset for training and evaluation.\
- Hugging Face Integration: Utilizes Hugging Face's transformers library for model loading and training.
- Accuracy Evaluation: Measures accuracy based on mathematical solutions extracted from model outputs.
- Command-Line Interface: Configurable parameters for training via command-line arguments.
- Logging: Logs training metrics and memory usage for reproducibility.

### Usage
Command-Line Arguments
Run the script with customizable parameters:

**python fine_tune_math.py**

- task: Task name (math by default).
- model: Hugging Face model name or path.
- epochs: Number of training epochs.
- batch_size: Training batch size.
- max_length: Maximum sequence length for tokenization.
- should_pad: Enables padding to the maximum sequence length.
#### Example usage

python fine_tune_math.py --model "meta-llama/Llama-2-7b-hf" --lora

### Outputs
Outputs
Logs: Saved in logs/, containing training details and memory usage.\
Metrics: Final accuracy and memory consumption are reported.\
Sample Output: After training, the script decodes a test input to demonstrate the model's output.