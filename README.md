# Multi-Task Transformer Architecture and Training Strategy

## Quick Start with Docker

```bash
# Build the CPU image
docker build -t mtl_transformer .

# Run the container
docker run mtl_transformer
```

## 1. Sentence Transformer Architecture Decisions

The `SentenceTransformer` class implements several key architectural choices beyond the transformer backbone:

```python
class SentenceTransformer(nn.Module):
    def __init__(self, model_name: str = "roberta-base", freeze_backbone: bool = False):
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.dropout(cls_output)
```

### Key Design Decisions:

1. **CLS Token Selection**
   - Uses first token (`[:,0,:]`) of last hidden state
   - Advantages over alternatives:
     - More efficient than mean/max pooling
     - Maintains sentence-level context
     - Single vector representation for classification
     - Consistent with BERT-style pretraining

2. **Dropout Implementation**
   - Strategic 0.1 dropout rate after CLS token
   - Rationale:
     - Prevents overfitting on sentence representations
     - Applied post-pooling for efficient computation
     - Maintains consistent regularization across tasks

## 2. Multi-Task Architecture Adaptations

The architecture incorporates several changes to support multi-task learning:

```python
class MultitaskTransformer(nn.Module):
    def __init__(self, model_name: str, num_classes_task_a: int = 4,
                 sentiment_classes: int = 3):
        # Shared backbone
        self.sentence_transformer = SentenceTransformer(model_name, freeze_backbone)
        hidden_size = self.sentence_transformer.encoder.config.hidden_size
        
        # Task-specific heads with identical architecture
        self.task_a_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes_task_a)
        )
```

### Key Adaptations:

1. **Shared Feature Extractor**
   - Common sentence transformer backbone
   - Unified representation learning
   - Parameter efficient design

2. **Parallel Task Heads**
   - Identical architectures for consistency
   - Task-specific output dimensions
   - Independent dropout for task regularization

3. **Dynamic Task Routing**
```python
def forward(self, task_name: str, input_ids, attention_mask):
    embeddings = self.sentence_transformer(input_ids, attention_mask)
    if task_name == "task_a":
        logits = self.task_a_head(embeddings)
    elif task_name == "sentiment":
        logits = self.sentiment_head(embeddings)
```

## 3. Training Scenarios Analysis

### Scenario 1: Full Network Frozen
```python
model = MultitaskTransformer(
    freeze_backbone=True,
    freeze_task_a=True,
    freeze_sentiment=True
)
```
**Implications:**
- Model becomes fixed feature extractor
- No parameter updates during training
- Fastest inference time
- Limited task adaptation

**Recommended When:**
- Very limited computational resources
- Tasks highly similar to pretraining
- Need for rapid deployment

### Scenario 2: Frozen Backbone Only
```python
model = MultitaskTransformer(
    freeze_backbone=True,
    freeze_task_a=False,
    freeze_sentiment=False
)
```
**Benefits:**
- Preserves pretrained knowledge
- Allows task-specific adaptation
- Reduced training parameters
- Memory efficient

**Optimal For:**
- Limited training data
- Resource constraints
- Similar domain to pretraining

### Scenario 3: Single Frozen Head
```python
model = MultitaskTransformer(
    freeze_backbone=False,
    freeze_task_a=True,  # or freeze_sentiment=True
    freeze_sentiment=False
)
```
**Advantages:**
- Enables knowledge transfer between tasks
- Maintains performance on frozen task
- Allows backbone adaptation
- Efficient incremental learning

## 4. Transfer Learning Strategy

Our code structure supports flexible transfer learning approaches. Let's analyze how to effectively implement transfer learning for general use cases:

### 1. Pre-trained Model Selection

Looking at our model initialization:
```python
model = MultitaskTransformer(
    model_name="roberta-base",
    num_classes_task_a=4,
    sentiment_classes=3,
    freeze_backbone=False
)
```

**Model Selection Criteria:**

1. **RoBERTa-base Advantages**
   - 125M parameters (balanced size)
   - Strong performance across diverse tasks
   - Robust pretrained representations
   - Memory efficient compared to larger models

2. **Alternative Options**
   - BERT-base: If memory is constrained
   - DeBERTa: For higher accuracy requirements
   - DistilRoBERTa: For speed-critical applications

3. **Selection Factors**
   ```python
   self.encoder = AutoModel.from_pretrained(model_name)
   hidden_size = self.encoder.config.hidden_size  # Adapts to model size
   ```
   - Target task complexity
   - Available computational resources
   - Dataset size
   - Inference speed requirements

### 2. Layer Freezing Strategy

Our code implements flexible freezing capabilities:
```python
class SentenceTransformer(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = False):
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
```

**Progressive Unfreezing Approach:**

1. **Initial Phase**
   ```python
   # Start with frozen backbone, train only heads
   model = MultitaskTransformer(
       freeze_backbone=True,
       freeze_task_a=False,
       freeze_sentiment=False
   )
   ```

2. **Middle Phase**
   ```python
   # Selectively unfreeze upper layers
   def unfreeze_upper_layers(model):
       for name, param in model.named_parameters():
           if "encoder.layer" in name:
               layer_num = int(name.split(".")[2])
               # Unfreeze only layers 9-12
               param.requires_grad = layer_num >= 9
   ```

3. **Final Phase**
   ```python
   # Gradually unfreeze with differentiated learning rates
   optimizer_grouped_parameters = [
       {
           "params": [p for n, p in model.named_parameters() 
                     if "encoder" in n and "layer" in n],
           "lr": base_lr,
           "weight_decay": 0.01
       },
       {
           "params": [p for n, p in model.named_parameters() 
                     if "head" in n],
           "lr": base_lr * 5
       }
   ]
   ```

### 3. Rationale for Layer Choices

1. **Lower Transformer Layers (0-4)**
   - **Strategy**: Keep frozen longest
   - **Rationale**:
     - Capture universal language patterns
     - Most transferable across tasks
     - Least task-specific
     - Stable gradient flow

2. **Middle Layers (5-8)**
   - **Strategy**: Selectively unfreeze
   - **Rationale**:
     - Balance general and specific features
     - Moderate adaptation needs
     - Cross-task feature sharing
     - Controlled parameter updates

3. **Upper Layers (9-12)**
   - **Strategy**: Unfreeze early
   - **Rationale**:
     - Most task-specific
     - Needs maximum adaptation
     - Less risk of catastrophic forgetting
     - Direct impact on task performance

4. **Task-Specific Heads**
   ```python
   self.task_a_head = nn.Sequential(
       nn.Linear(hidden_size, hidden_size),
       nn.ReLU(),
       nn.Dropout(0.1),
       nn.Linear(hidden_size, num_classes_task_a)
   )
   ```
   - **Strategy**: Always trainable
   - **Rationale**:
     - Completely new components
     - Task-specific architecture
     - Random initialization
     - Needs full training

### Implementation Best Practices

```python
class TransferLearningTrainer(LayerwiseLearningRateTrainer):
    def training_step(self, model, inputs, **kwargs):
        # Implement dynamic freezing based on training progress
        current_epoch = self.state.epoch
        
        if current_epoch < 1:
            # Initial phase: only train heads
            freeze_all_except_heads(model)
        elif current_epoch < 2:
            # Middle phase: unfreeze upper layers
            unfreeze_upper_layers(model)
        else:
            # Final phase: full fine-tuning
            unfreeze_all_layers(model)
            
        return super().training_step(model, inputs, **kwargs)
```

Key benefits of this approach:
1. Controlled adaptation to new tasks
2. Efficient use of pretrained knowledge
3. Prevention of catastrophic forgetting
4. Optimized training resource usage
5. Flexible adaptation to different task types

This transfer learning strategy can be adjusted based on:
- Dataset size
- Task similarity to pretraining
- Available computational resources
- Required model performance
- Training time constraints


## 5. Learning Rate Strategy

### Layer-wise Learning Rate Implementation
```python
optimizer_grouped_parameters = [
    # Encoder layers
    {
        "params": [p for n, p in self.model.named_parameters() 
                  if "encoder" in n],
        "lr": self.args.learning_rate
    },
    # Task heads
    {
        "params": [p for n, p in self.model.named_parameters() 
                  if ("task_a_head" in n or "sentiment_head" in n)],
        "lr": self.args.learning_rate * 5
    }
]
```

### Learning Rate Choices

1. **Base Rate (2e-5)**
   - Default learning rate for encoder
   - Balanced between stability and adaptation
   - Prevents catastrophic forgetting

2. **Task Head Rate (1e-4)**
   - 5x higher than base rate
   - Rationale:
     - New parameters need faster adaptation
     - Random initialization requires higher learning rate
     - Task-specific features need rapid development

### Benefits in Multi-task Setting

1. **Gradient Management**
   ```python
   training_args = TrainingArguments(
       learning_rate=2e-5,
       weight_decay=0.01,
       max_grad_norm=1.0,
       warmup_steps=500
   )
   ```
   - Different learning rates balance task gradients
   - Prevents dominant task interference
   - Maintains stable updates across tasks

2. **Knowledge Transfer**
   - Controlled feature adaptation
   - Preserves useful pretrained knowledge
   - Enables task-specific optimization

3. **Training Stability**
   - Gradient clipping prevents explosion
   - Warmup steps for stable initialization
   - Weight decay for regularization

4. **Resource Efficiency**
   ```python
   per_device_train_batch_size=8,  # Reduced batch size
   per_device_eval_batch_size=8
   ```
   - Memory-efficient batch sizes
   - Focused parameter updates
   - Balanced computational resources

### Multi-Task Specific Benefits

1. **Task Interference Management**
   ```python
   def __getitem__(self, idx):
       if np.random.random() < 0.5:
           task_name = "task_a"
       else:
           task_name = "sentiment"
   ```
   - Balanced task sampling
   - Independent gradient scaling
   - Controlled feature sharing

2. **Adaptive Optimization**
   - Task-specific learning rates
   - Shared feature development
   - Efficient knowledge transfer
