# model.py
import torch
import torch.nn as nn
from transformers import RobertaModel, AutoConfig, AutoModel
from typing import Dict, List, Optional, Tuple

class SentenceTransformer(nn.Module):
    """
    A transformer-based sentence encoder that produces fixed-size embeddings.
    
    This module incorporates a pre-trained transformer model (default: RoBERTa) with 
    an added dropout layer for regularization. It utilizes the CLS token embedding 
    to represent the sentence.
    
    Args: 
            model_name (str): Name of pretrained model to load (default:'roberta-based')
            freeze_backbone (bool): Boolean value (default: False), if True freezes the encoder backbone parameters, making them non-trainable
    """
    def __init__(self, model_name: str = "roberta-base", freeze_backbone: bool = False):
        """
         Initialize the SentenceTransformer model
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model

        Args:
            input_ids (torch.Tensor): Tensor of token ids, shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor of attention masks, shape (batch_size, sequence_length)
            
        Returns:
            torch.Tensor: Sentence embeddings, shape (batch_size, hidden_size)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.dropout(cls_output)

class MultitaskTransformer(nn.Module):
    """A multi-task transformer model that handles both task A and sentiment classification.
    
    This model uses a shared sentence encoder backbone with task-specific classification heads.
    Each head can be optionally frozen for transfer learning scenarios.
    
    Args:
        model_name (str): Name of the pre-trained model to use (default: "roberta-base")
        num_classes_task_a (int): Number of classes for task A (default: 4)
        sentiment_classes (int): Number of sentiment classes (default: 3)
        freeze_backbone (bool): If True, freezes the transformer backbone (default: False)
        freeze_task_a (bool): If True, freezes the task A classification head (default: False)
        freeze_sentiment (bool): If True, freezes the sentiment classification head (default: False)
    
    Attributes:
        sentence_transformer: The shared sentence encoder
        task_a_head: Classification head for task A
        sentiment_head: Classification head for sentiment analysis
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes_task_a: int = 4,
        sentiment_classes: int = 3,
        freeze_backbone: bool = False,
        freeze_task_a: bool = False,
        freeze_sentiment: bool = False
    ):
        super().__init__()
        
        # Sentence encoder
        self.sentence_transformer = SentenceTransformer(model_name, freeze_backbone)
        hidden_size = self.sentence_transformer.encoder.config.hidden_size
        
        # Task A head
        self.task_a_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes_task_a)
        )
        
        # Sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, sentiment_classes)
        )
        
        if freeze_task_a:
            for param in self.task_a_head.parameters():
                param.requires_grad = False
                
        if freeze_sentiment:
            for param in self.sentiment_head.parameters():
                param.requires_grad = False
    
    def forward(self, task_name: str, input_ids, attention_mask):
        """Forward pass of the multi-task model.
        
        Args:
            task_name (str): Name of the task to perform ("task_a" or "sentiment")
            input_ids (torch.Tensor): Tensor of token ids, shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Tensor of attention masks, shape (batch_size, sequence_length)
            
        Returns:
            dict: Dictionary containing 'logits' key with the model predictions
                 Shape of logits depends on the task:
                 - task_a: (batch_size, num_classes_task_a)
                 - sentiment: (batch_size, sentiment_classes)
                 
        Raises:
            ValueError: If task_name is not one of ["task_a", "sentiment"]
        """
        # Get embeddings from the sentence transformer
        embeddings = self.sentence_transformer(input_ids, attention_mask)
        
        # Route to appropriate task head
        if task_name == "task_a":
            logits = self.task_a_head(embeddings)
        elif task_name == "sentiment":
            logits = self.sentiment_head(embeddings)
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        return {'logits': logits}