import os
import sys
import argparse
import torch
import time
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import re
import numpy as np

set_seed(0)

def tokenize_math(
    *,
    tokenizer,
    validation_set,
    **kwargs,
):
    assert validation_set in [
        "validation",
        "test",
    ], "Please enter a valid `validation_set`"

    if validation_set == "validation":
        split = [
            "train[:500]",
            "train[500:2000]",
        ]
    else:
        split = [
            "train",
            "test",
        ]

    train_set, eval_set = load_dataset(
        "lighteval/MATH",
        "all",
        split=split,
    )

    def tokenize_function(example):
        """Tokenize and prepare inputs and labels for evaluation."""
        IGNORE_INDEX = -100  # Define the ignore index for labels

        sources = example["problem"]
        targets = example["solution"]

        example_text = sources + " Full Response: " + targets
        modified_source = sources + " Full Response: "

        # Tokenize concatenated examples
        examples_tokenized = tokenizer(
            example_text,
            return_tensors="pt",
            **kwargs,
        )

        # Tokenize sources to obtain source lengths
        sources_tokenized = tokenizer(
            modified_source,
            return_tensors="pt",
            **kwargs,
        )

        input_ids = examples_tokenized["input_ids"][0]
        labels = input_ids.clone()

        # Compute lengths of source sequences (excluding padding tokens)
        source_len = sources_tokenized["input_ids"][0].ne(tokenizer.pad_token_id).sum()

        # Set labels corresponding to source tokens to IGNORE_INDEX
        labels[:source_len] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}

    train_dataset = train_set.map(
        tokenize_function,
        batched=False,
        remove_columns=train_set.column_names,
    )
    eval_dataset = eval_set.map(
        tokenize_function,
        batched=False,
        remove_columns=eval_set.column_names,
    )

    return train_dataset, eval_dataset

def evaluate_math_hf(
    *,
    auto_model,
    tokenizer,
    tokenized_eval,
    batch_size,
):
    """
    Evaluates the model on the MATH dataset.
    """
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=auto_model,
        label_pad_token_id=-100,
        padding="longest",
    )

    # Evaluation arguments
    output_dir = os.path.join("evaluation_output")
    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    pattern = r"\\boxed\{(.*?)\}"

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Ensure preds and labels are numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Replace -100s used for padding as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Pattern matching for metrics
        correct = 0
        total = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            total += 1
            pred_matches = re.findall(pattern, pred)
            label_matches = re.findall(pattern, label)
            if len(pred_matches) > 0 and len(label_matches) > 0:
                pred_answer = pred_matches[-1]
                label_answer = label_matches[-1]
                if pred_answer == label_answer:
                    correct += 1
        torch.cuda.empty_cache()
        return {"accuracy": correct / total if total > 0 else 0}

    def preprocess_logits_for_metrics(logits, labels):
        """
        Convert logits to predicted token IDs for metric computation.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids

    # Make trainer for evaluation
    trainer = Trainer(
        model=auto_model,
        args=eval_args,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    eval_result = trainer.evaluate()
    final_score = eval_result["eval_accuracy"]
    return final_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face model on the MATH dataset.")

    parser.add_argument("--model", type=str, required=True, help="Path or name of the Hugging Face model to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size to use during evaluation")
    parser.add_argument("--max_length", type=int, default=1300, help="The max length for tokenizing the dataset")
    parser.add_argument("--should_pad", action="store_true", help="Whether to pad inputs to max length")

    cli = parser.parse_args()

    set_seed(0)

    # Use LlamaTokenizer for LLaMA models
    tokenizer = AutoTokenizer.from_pretrained(
        cli.model,
        padding_side="left",
        use_fast=False,
    )
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        need_resize = True
    else:
        need_resize = False

    auto_config = AutoConfig.from_pretrained(cli.model)
    auto_model = AutoModelForCausalLM.from_pretrained(
        cli.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=auto_config,
    )

    # Ensure the model's pad_token_id matches the tokenizer's
    auto_model.config.pad_token_id = tokenizer.pad_token_id

    if need_resize:
        auto_model.resize_token_embeddings(len(tokenizer))

    _, tokenized_eval = tokenize_math(
        tokenizer=tokenizer,
        validation_set="test",
        max_length=cli.max_length,
        padding=cli.should_pad,
        truncation=True,
    )

    accuracy = evaluate_math_hf(
        auto_model=auto_model,
        tokenizer=tokenizer,
        tokenized_eval=tokenized_eval,
        batch_size=cli.batch_size,
    )

    print(f"Final accuracy: {accuracy}")
