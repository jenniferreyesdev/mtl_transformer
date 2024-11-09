# train.py
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from transformers import Trainer, TrainingArguments

from trainer import get_device
from trainer import MultitaskDataset
from trainer import LayerwiseLearningRateTrainer
from trainer import MultitaskCollator
from model import MultitaskTransformer
from SentenceTransformerTest import test_sentence_transformer


# Define the callback class before using it
class SaveCheckpointCallback(TrainerCallback):
    """
    Custom callback to ensure checkpoints are properly saved during training.
    
    This callback enforces saving checkpoints at both epoch end and specified step intervals,
    providing additional logging and verification of the saving process.
    
    Inherits from:
        TrainerCallback: Base callback class from transformers library
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        """Ensures checkpoint saving at the end of each epoch."""
        control.should_save = True
        return control
        
    def on_step_end(self, args, state, control, **kwargs):
        """Monitors and logs checkpoint saving at specified steps."""
        if state.global_step % args.save_steps == 0:
            print(f"Saving checkpoint at step {state.global_step}")


def main():
    # Test Sentence Transformer model with a few sentences to showcase embeddings.abs
    test_sentence_transformer()

    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    def tokenize_function(examples):
        """
        Tokenizes text examples using the RoBERTa tokenizer.
    
        Args:
            examples (dict): Dictionary containing text examples under 'text' key
            
        Returns:
            dict: Tokenized examples with input_ids, attention_mask, etc.
        """
        return tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    # Load datasets
    # Task A: Text Classification
    dataset_task_a = load_dataset("ag_news", trust_remote_code=True)
    # Task B: Sentiment Analysis
    dataset_sentiment = load_dataset("imdb", trust_remote_code=True)

    def preprocess_imdb(examples):
        """
        Convert IMDB's binary labels (0: negative, 1: positive)
        to three classes (0: negative, 1: neutral, 2: positive)

        Args:
        examples (dict): Dictionary containing text and binary labels
        
        Returns:
        dict: Processed examples with three-class labels
        """
        texts = examples["text"]
        labels = examples["label"]
        
        neutral_indicators = [
            "but ", "however ", "although ", "though ",
            "mixed ", "average ", "mediocre ", "moderate "
        ]
        
        new_labels = []
        for text, label in zip(texts, labels):
            text_lower = text.lower()
            has_mixed = any(indicator in text_lower for indicator in neutral_indicators)
            
            if has_mixed:
                new_labels.append(1)  # neutral
            else:
                new_labels.append(0 if label == 0 else 2)
        
        examples["labels"] = new_labels
        return examples

    def preprocess_ag_news(examples):
        """
        Prepares AG News dataset by adjusting label indexing. 
        - Adjust labels to start from 0 instead of 1

        Args:
        examples (dict): Dictionary containing AG News examples with 1-based labels
        
        Returns:
        dict: Processed examples with 0-based labels
        """
        examples["labels"] = [label - 1 for label in examples["label"]]
        return examples

    # Process datasets
    dataset_task_a = dataset_task_a.map(
        preprocess_ag_news,
        batched=True,
        desc="Processing AG News dataset"
    )

    dataset_sentiment = dataset_sentiment.map(
        preprocess_imdb,
        batched=True,
        desc="Converting to three-class sentiment"
    )

    def balance_dataset(dataset):
        """
        Balance the dataset to have roughly equal numbers of each class
        """
        # Get unique labels
        all_labels = set(dataset["labels"])
        grouped = {label: [] for label in all_labels}
        
        # Group by label
        for i, example in enumerate(dataset):
            grouped[example["labels"]].append(i)
        
        # Find size of smallest group
        min_size = min(len(group) for group in grouped.values())
        
        # Subsample each group
        balanced_indices = []
        for group in grouped.values():
            balanced_indices.extend(np.random.choice(group, min_size, replace=False))
        
        return dataset.select(balanced_indices)

    # Create validation split for AG News (it only comes with train/test)
    train_test_split = dataset_task_a["train"].train_test_split(test_size=0.1, seed=42)
    dataset_task_a = DatasetDict({
        "train": train_test_split["train"],
        "validation": train_test_split["test"],
        "test": dataset_task_a["test"]
    })

    # Balance both datasets
    dataset_task_a_balanced = DatasetDict({
        "train": balance_dataset(dataset_task_a["train"]),
        "validation": balance_dataset(dataset_task_a["validation"]),
        "test": balance_dataset(dataset_task_a["test"])
    })

    dataset_sentiment_balanced = DatasetDict({
        "train": balance_dataset(dataset_sentiment["train"]),
        "test": balance_dataset(dataset_sentiment["test"])
    })

    # Split IMDB test into validation and test
    dataset_sentiment_final = DatasetDict({
        "train": dataset_sentiment_balanced["train"],
        "validation": dataset_sentiment_balanced["test"].select(range(len(dataset_sentiment_balanced["test"]) // 2)),
        "test": dataset_sentiment_balanced["test"].select(range(len(dataset_sentiment_balanced["test"]) // 2, len(dataset_sentiment_balanced["test"])))
    })
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Create multi-task datasets
    train_dataset = MultitaskDataset(
        dataset_task_a_balanced["train"],
        dataset_sentiment_final["train"],
        tokenizer=tokenizer
    )
    
    eval_dataset = MultitaskDataset(
        dataset_task_a_balanced["validation"],
        dataset_sentiment_final["validation"],
        tokenizer=tokenizer
    )
    
    # Initialize model
    model = MultitaskTransformer(
        model_name="roberta-base",
        num_classes_task_a=4,
        sentiment_classes=3,
        freeze_backbone=False,
        freeze_task_a=False,
        freeze_sentiment=False
    ).to(device)
    
    # Training arguments 
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,

        logging_dir="./logs",
        logging_steps=100,

        evaluation_strategy="steps",
        eval_steps=500,

        # Checkpoint saving configuration, intervals, and total saves
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        load_best_model_at_end=True,
        metric_for_best_model="loss",
        remove_unused_columns=False,
        greater_is_better=False,

        # Add gradient clipping
        max_grad_norm=1.0,
        # Add warmup steps
        warmup_steps=500,

        # Progress display
        report_to="none",
        # Device settings
        no_cuda=device.type == "cpu",
        use_cpu=device.type == "cpu",
        # Save safeguards
        save_safetensors=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = LayerwiseLearningRateTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultitaskCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            SaveCheckpointCallback()
        ]
    )
     
    try:
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print("Saving final model...")
        trainer.save_model("./results/final_model")
        
        # Verify saves
        if not (Path("./results/final_model/pytorch_model.bin").exists() or 
                Path("./results/final_model/model.safetensors").exists()):
            print("Warning: Final model save may have failed!")
            
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save model even if training fails
        try:
            trainer.save_model("./results/emergency_save")
        except:
            print("Emergency save failed!")
        raise
    
    print("Training completed!")
    
    # Check final saves
    final_path = Path("./results/final_model")
    if final_path.exists():
        print(f"Final model saved at: {final_path}")
        print("Saved files:", list(final_path.glob("*")))
    else:
        print("Warning: Final model directory not found!")

    # Evaluate
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()