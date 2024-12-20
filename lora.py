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


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LoRAAdapter(nn.Module, LoRALayer):
    def __init__(
        self,
        existing_layer: nn.Module,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        **kwargs
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.existing_layer = existing_layer
        if r > 0:
            self.lora_A = nn.Parameter(self.existing_layer.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.existing_layer.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            #different from the paper, but that is okay
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        if self.r == 0:
          return self.existing_layer(x)
        else:
          result = self.existing_layer(x)
          result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
          return result
      
      
      
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
          p.requires_grad = False
        else:
          p.requires_grad = True