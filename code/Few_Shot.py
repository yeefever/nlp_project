import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import sys
import logging
import argparse

import torch
import time
from datasets import Dataset, load_dataset
from peft import IA3Config, LoraConfig, TaskType, get_peft_model
from setproctitle import setproctitle
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from accelerate import Accelerator
import re
import numpy as np

set_seed(0)
def tokenize_math_with_few_shot(
    *,
    tokenizer,
    validation_set,
    examples,
    **kwargs,
):
    assert validation_set in [
        "validation",
        "test",
    ], "please enter a valid `validation_set`"

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

    # Prepare the few-shot prompt
    few_shot_prompt = "\n\n".join(
        [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples]
    ) + "\n\n"

    def tokenize_function(example):
        """Tokenize and prepare inputs and labels for training with few-shot examples."""
        IGNORE_INDEX = -100  # Define the ignore index for labels

        sources = example["problem"]
        targets = example["solution"]

        # Combine the few-shot examples with the current input
        example_combined = few_shot_prompt + f"Question: {sources}\nAnswer: {targets}"
        modified_source = few_shot_prompt + f"Question: {sources}\nAnswer:"

        # Tokenize concatenated examples
        examples_tokenized = tokenizer(
            example_combined,
            return_tensors="pt",
            **kwargs,
        )

        # Tokenize sources to obtain source lengths
        sources_tokenized = tokenizer(
            modified_source,
            return_tensors="pt",
            **kwargs,
        )

        input_ids = examples_tokenized["input_ids"][-1]
        labels = input_ids.clone()

        # Compute lengths of source sequences (excluding padding tokens)
        source_len = sources_tokenized["input_ids"][-1].ne(tokenizer.pad_token_id).sum()

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
        remove_columns=train_set.column_names,
    )

    return train_dataset, eval_dataset


def finetune_math_hf(
    *,
    auto_model,
    tokenizer,
    tokenized_train,
    tokenized_eval,
    epochs,
    batch_size,
    learning_rate,
    train_head,
    use_multi_lr,
    full_parameter=False,
):
    """
    Fine-tunes an abstract transformer on a specified task.
    """
    if full_parameter:
        for param in auto_model.parameters():
            param.requires_grad = True
        print("full parameter ready")
    # else:
    #     mark_adapters_as_trainable(auto_model)
    if train_head:
        for name, param in auto_model.named_parameters():
            if "score" in name:
                param.requires_grad = True

    auto_model = Accelerator().prepare(auto_model)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=auto_model,
        label_pad_token_id=-100,
        padding="longest",
    )

    # Training arguments
    output_dir = os.path.join("testing")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if epochs > 0 else "no",
        learning_rate=learning_rate,
        weight_decay=0.06,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        optim="paged_adamw_32bit",
    )

    optimizers = (None, None)

    pattern = r"\\boxed\{(.*?)\}"

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        print(f"Preds type: {type(preds)}, Preds shape: {preds.shape}")
        print(f"Labels type: {type(labels)}, Labels shape: {labels.shape}")

        # Replace -100s used for padding as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        correct = 0
        total = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            total += 1
            pred_matches = re.findall(pattern, pred)
            label_matches = re.findall(pattern, label)
            if len(pred_matches) > 0 and len(label_matches) > 0:
                pred = pred_matches[-1]
                label = label_matches[-1]
                if pred == label:
                    correct += 1
        torch.cuda.empty_cache()

        return {"accuracy": correct / total}

    def preprocess_logits_for_metrics(logits, labels):
        """
        This function processes the logits to extract predicted token IDs.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    # Make trainer
    trainer = Trainer(
        model=auto_model,
        args=training_args,
        train_dataset=tokenized_train if epochs > 0 else None,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if epochs > 0:
        trainer.train()
        history = trainer.state.log_history

        final_score = max(
            history,
            key=lambda i: (
                i.get("eval_accuracy", -1)
            ),
        )["eval_accuracy"]
    else:
        eval_result = trainer.evaluate()
        final_score = eval_result["eval_accuracy"]
        history = eval_result

    return final_score, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line arguments for experiments")

    parser.add_argument("--task", type=str, default="math", help="LLM task to optimize over")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Huggingface model to use")
    parser.add_argument("--epochs", type=int, default=0, help="Number of epochs to train model for")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use when training the model")
    parser.add_argument("--max_length", type=int, default=512, help="The max length for tokenizing the dataset")
    parser.add_argument("--should_pad", action="store_true", help="Whether to pad inputs to max length")
    parser.add_argument("--train_head", action="store_true", help="Whether to train a classification head on the model")
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model")
    parser.add_argument("--prune", action="store_true", help="Whether to prune the model")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5, help="Sparsity ratio for pruning (if enabled)")
    parser.add_argument("--structured", action="store_true", help="Whether to do structured pruning")
    parser.add_argument("--cosine", action="store_true", help="Whether to use cosine similarity")
    parser.add_argument("--model_dir", type=str, default="", help="Directory to save or load the model")
    parser.add_argument("--sampler", type=str, default="tpe", help="Sampler type for hyperparameter optimization")
    parser.add_argument("--topk", type=int, default=3, help="Top-k results to consider")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--ia3", action="store_true")
    parser.add_argument("--full", action="store_true")

    cli = parser.parse_args()

    setproctitle(f"CLAM Memory Consumption, {cli.model} {cli.task}")

    technique = "none"
    if cli.lora:
        technique = "lora"
    if cli.ia3:
        technique = "ia3"
    if cli.full:
        technique = "full"

    model_name = cli.model.split("/")[-1]
    log_name = (
        "memory_consumption"
        f"_model_[{model_name}]"
        f"_task_[synthetic]"
        f"_batch_size_[{cli.batch_size}]"
        f"_method_[{technique}]"
    )
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.out")

    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logger.setLevel(logging.INFO)

    logger.info(cli)

    tokenizer = AutoTokenizer.from_pretrained(
        cli.model,
        padding_side="left",
        use_fast=True,
    )
    auto_config = AutoConfig.from_pretrained(cli.model)
    auto_model = AutoModelForCausalLM.from_pretrained(
        cli.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=auto_config,
        ignore_mismatched_sizes=True,
    )

    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        need_resize = True
    else:
        need_resize = False
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            need_resize = True

    if need_resize:
        auto_model.resize_token_embeddings(len(tokenizer))

    if cli.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=8,
            target_modules=[
                "down_proj"
            ],
        )
        auto_model = get_peft_model(auto_model, lora_config)
        logger.info("Applied Hugging Face PEFT LoRA")

    if cli.ia3:
        ia3_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "down_proj", "up_proj"
            ],
            feedforward_modules=["up_proj", "down_proj"],
        )
        auto_model = get_peft_model(auto_model, ia3_config)
        logger.info("Applied Hugging Face PEFT IA3")

    start_time = time.time()

    few_shot_examples = [
        {
            "question": "A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?",
            "answer": "Let $x$ be the number of band members in each row for the original formation, when two are left over. Then we can write two equations from the given information: $$rx+2=m$$ $$(r-2)(x+1)=m$$ Setting these equal, we find: $$rx+2=(r-2)(x+1)=rx-2x+r-2$$ $$2=-2x+r-2$$ $$4=r-2x$$ We know that the band has less than 100 members. Based on the first equation, we must have $rx$ less than 98. We can guess and check some values of $r$ and $x$ in the last equation. If $r=18$, then $x=7$, and $rx=126$ which is too big. If $r=16$, then $x=6$, and $rx=96$, which is less than 98. Checking back in the second formation, we see that $(16-2)(6+1)=14\cdot 7=98$ as it should. This is the best we can do, so the largest number of members the band could have is $\boxed{98}$.",
        },
        {
            "question": "What is the degree of the polynomial $(4 + 5x^3 + 100 + 2\\pi x^4 + \\sqrt{10}x^4 + 9)$?",
            "answer": "This polynomial is not written in standard form. However, we don't need to write it in standard form, nor do we need to pay attention to the coefficients. We just look for the exponents on $x$. We have an $x^4$ term and no other term of higher degree, so $\boxed{4}$ is the degree of the polynomial.",
        },
        {
            "question": "Evaluate $\\left\\lceil3\\left(6-\\frac12\\right)\\right\\rceil$.",
            "answer": "Firstly, $3\left(6-\frac12\right)=18-1-\frac12=17-\frac12$. Because $0\le\frac12<1$, we have $\left\lceil17-\frac12\right\rceil=\boxed{17}$.",
        },
    ]

    tokenized_train, tokenized_eval = tokenize_math_with_few_shot(
        tokenizer=tokenizer,
        validation_set="test",
        examples=few_shot_examples,
        max_length=cli.max_length,
    )

    result, history = finetune_math_hf(
        auto_model=auto_model,
        tokenizer=tokenizer,
        tokenized_train=tokenized_train,
        tokenized_eval=tokenized_eval,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        learning_rate=1e-4,
        train_head=True,
        use_multi_lr=False,
        full_parameter=cli.full,
    )

    train_time = time.time() - start_time

    max_memories = [
        torch.cuda.max_memory_allocated(device) / 1024**3
        for device in range(torch.cuda.device_count())
    ]
    logger.info("Max memory usage (GB): %s", sum(max_memories))
    logger.info("Final accuracy: %s", result)
    logger.info(history)

    sys.exit(0)
