# trainer.py
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Union, Any
import numpy as np

def get_device():
    """Helper function to determine the appropriate device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cpu")

class MultitaskDataset(Dataset):
    """
    Custom dataset for handling multiple tasks (text classification and sentiment analysis).
    
    This dataset alternates between two tasks during training, providing a balanced
    mix of examples from both tasks. It handles tokenization and preparation of
    inputs for the model.
    
    Args:
        task_a_dataset: Dataset for the first task (text classification)
        sentiment_dataset: Dataset for the sentiment analysis task
        tokenizer: Tokenizer instance for processing text inputs
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 128.
    
    Attributes:
        task_a_dataset: The text classification dataset
        sentiment_dataset: The sentiment analysis dataset
        tokenizer: The tokenizer instance
        max_length (int): Maximum sequence length
        device (torch.device): The device to use for computations
    """
    def __init__(self, task_a_dataset, sentiment_dataset, tokenizer, max_length=128):
        self.task_a_dataset = task_a_dataset
        self.sentiment_dataset = sentiment_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = get_device()
        
        # Set tokenizer parameters
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        
    def __len__(self):
        """Returns the length of the dataset (the minimum of both task datasets)."""
        return min(len(self.task_a_dataset), len(self.sentiment_dataset))
    
    def __getitem__(self, idx):
        """
        Retrieves a single example from either task randomly.
        
        Args:
            idx (int): Index of the example to retrieve
            
        Returns:
            dict: Dictionary containing:
                - input_ids: Tokenized and padded input text
                - attention_mask: Attention mask for the input
                - labels: Task labels
                - task_name: Name of the selected task
        """
        # Randomly choose task
        if np.random.random() < 0.5:
            text = self.task_a_dataset[idx]["text"]
            labels = self.task_a_dataset[idx]["labels"]
            task_name = "task_a"
        else:
            text = self.sentiment_dataset[idx]["text"]
            labels = self.sentiment_dataset[idx]["labels"]
            task_name = "sentiment"
            
        # Tokenize with explicit padding and truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),
            'task_name': task_name
        }


class MultitaskCollator:
    """
    Collator for batching multitask examples with proper padding.
    
    This collator ensures all sequences in a batch are properly padded to the same length
    and handles the combination of examples from different tasks.
    
    Args:
        tokenizer: Tokenizer instance for handling padding
    
    Attributes:
        tokenizer: The tokenizer instance
        device (torch.device): The device to use for computations
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.device = get_device()

    def __call__(self, examples):
        """Collates a list of examples into a properly formatted batch.
        
        Args:
            examples (list): List of dictionaries containing example data
            
        Returns:
            dict: Batched inputs containing:
                - input_ids: Padded input sequences
                - attention_mask: Attention masks for padded sequences
                - labels: Batched labels
                - task_name: Name of the task for this batch
        """
        max_length = max(ex['input_ids'].size(0) for ex in examples)
        
        batch = {
            'input_ids': torch.stack([
                torch.nn.functional.pad(
                    ex['input_ids'],
                    (0, max_length - ex['input_ids'].size(0)),
                    value=self.tokenizer.pad_token_id
                ) for ex in examples
            ]),
            'attention_mask': torch.stack([
                torch.nn.functional.pad(
                    ex['attention_mask'],
                    (0, max_length - ex['attention_mask'].size(0)),
                    value=0
                ) for ex in examples
            ]),
            'labels': torch.stack([ex['labels'] for ex in examples]),
            'task_name': examples[0]['task_name']
        }
        return batch

class LayerwiseLearningRateTrainer(Trainer):
    """Custom trainer implementing layer-wise learning rates and multitask handling.
    
    This trainer extends the Hugging Face Trainer to support:
    1. Different learning rates for different layers
    2. Custom loss computation for multiple tasks
    3. Task-specific prediction handling
    4. Proper input preparation for multitask models
    """
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation handling for multi-task setup
        """
        # Create a new dict with only the required inputs for forward pass
        forward_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'task_name': inputs['task_name']
        }
        
        # Get outputs from model
        outputs = model(**forward_inputs)
        logits = outputs['logits']
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        labels = inputs['labels']
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_outputs:
            outputs['loss'] = loss
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys = None
    ):
        """
        Custom prediction step that properly handles our model's input requirements
        """
        # Prepare inputs
        inputs = self._prepare_inputs(inputs)
        
        # Create forward inputs without labels
        forward_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'task_name': inputs['task_name']
        }
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(**forward_inputs)
            logits = outputs['logits']
            
            # Compute loss if needed
            if prediction_loss_only:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
                return (loss, None, None)
            
            return (None, logits, inputs['labels'])

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs before passing to model
        """
        prepared_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared_inputs[k] = v.to(self.args.device)
            else:
                prepared_inputs[k] = v
        return prepared_inputs

    def create_optimizer(self):
        """
        Create optimizer with layer-wise learning rates
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            
            optimizer_grouped_parameters = [
                # Encoder layers
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if "encoder" in n and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if "encoder" in n and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
                # Task heads
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if ("task_a_head" in n or "sentiment_head" in n) and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 5
                }
            ]
            
            # Filter out empty groups
            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters 
                if len(group["params"]) > 0
            ]
            
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
            )
        
        return self.optimizer