import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import gc
from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from accelerate import Accelerator


def match_submodules(model: nn.Module, key:str) -> List[str]:
        return [module for module, _ in model.named_modules() if key in module]

def get_submodule(model: nn.Module, module_name:str):
    return model.get_submodule(module_name)

#takes a model and replaces the submodule at module_path with the new_module
def replace_submodule(model: nn.Module, module_path: str, new_module):
    p = module_path.split('.')
    submod = model
    for part in p[:-1]:
        submod = submod._modules.get(part)
        if submod is None:
            raise ValueError()

    if p[-1] in submod._modules:
        submod._modules[p[-1]] = new_module
    else:
        raise ValueError()

#replaces all submodules which matches on any string in match_on according to adapter_fn
def inject_adapter(model: nn.Module, match_on: List[str], adapter_fn):
    for group_string in match_on:
      for module in match_submodules(model, group_string):
        og_mod = get_submodule(model, module)
        og_device = next(og_mod.parameters()).device #get device of old module
        replace_submodule(model, module, adapter_fn(og_mod).to(og_device))
        
        
        
def finetune(
        self,
        train_dataset,
        epochs=1,
        learning_rate=1e-4,
        add_eos_token=False,
        add_bos_token=False,
        prune=False
    ):
        tokenizer = self.get_tokenizer(add_eos_token, add_bos_token)

        def tokenize_function(examples):
            result = tokenizer(examples["text"], padding="max_length", truncation=True)
            result["labels"] = result["input_ids"].copy()
            # TODO: does this break Llama?
            result["labels"] = [
                [-100 if id == tokenizer.pad_token_id else id for id in i]
                for i in result["labels"]
            ]
            return result

        test_dataset = {"text": train_dataset["text"][:10]}
        train_dataset = Dataset.from_dict(train_dataset)
        test_dataset = Dataset.from_dict(test_dataset)
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer)
        self.auto_model = Accelerator().prepare(self.auto_model)
        
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=0.01,
            num_train_epochs=epochs,
            per_device_train_batch_size=64,
        )
        
        trainer = Trainer(
                model=self.auto_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_test_dataset,
                data_collator=data_collator,
            )

        trainer.train()