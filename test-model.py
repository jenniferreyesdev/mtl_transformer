import torch
from transformers import AutoTokenizer, AutoModel
from model import MultitaskTransformer
from pathlib import Path
from safetensors.torch import load_file

def load_model(model_path):
    """Load the trained multitask model"""
    print(f"Loading model from {model_path}...")
    
    # Initialize model architecture
    model = MultitaskTransformer(
        model_name="roberta-base",
        num_classes_task_a=4,
        sentiment_classes=3
    )
    
    # Determine which checkpoint file exists and load accordingly
    if Path(model_path + "/pytorch_model.bin").exists():
        print("Loading PyTorch checkpoint...")
        state_dict = torch.load(model_path + "/pytorch_model.bin", map_location='cpu')
    elif Path(model_path + "/model.safetensors").exists():
        print("Loading safetensors checkpoint...")
        try:
            state_dict = load_file(model_path + "/model.safetensors")
        except Exception as e:
            print(f"Error loading safetensors: {e}")
            raise
    else:
        raise FileNotFoundError(f"No checkpoint found in {model_path}")
    
    # Load the state dict and handle any missing keys
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        raise
    
    model.eval()
    return model

def predict(model, tokenizer, text, task_name):
    """Make prediction for a given text and task"""
    # Tokenize input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            task_name=task_name,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        
        # Get prediction and confidence
        confidence, pred_idx = torch.max(probs, dim=1)
        
        # Map prediction to class label
        if task_name == "task_a":
            classes = ["World", "Sports", "Business", "Sci/Tech"]
        else:  # sentiment
            classes = ["Negative", "Neutral", "Positive"]
            
        prediction = classes[pred_idx.item()]
        confidence = confidence.item()
        
        return prediction, confidence, dict(zip(classes, probs[0].tolist()))

def main():
    # Load model and tokenizer
    model_path = "./results/final_model"
    
    # Initialize tokenizer with proper settings
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test samples
    news_samples = [
        "Apple's new iPhone sales break previous records in Asian markets",
        "Scientists discover new species of deep-sea creatures in Pacific",
        "Manchester United signs new striker for record transfer fee",
        "Global stock markets show signs of recovery after recent decline"
    ]
    
    sentiment_samples = [
        "This movie was absolutely fantastic, I loved every minute of it!",
        "The product works fine but the customer service needs improvement",
        "Worst experience ever, complete waste of money and time",
        "While it has some flaws, overall I found it quite enjoyable"
    ]
    
    # Test news classification
    print("\nTesting News Classification:")
    print("-" * 60)
    for text in news_samples:
        prediction, confidence, all_probs = predict(model, tokenizer, text, "task_a")
        print(f"\nText: {text}")
        print(f"Category: {prediction} (Confidence: {confidence:.2%})")
        print("All probabilities:")
        for category, prob in all_probs.items():
            print(f"  {category}: {prob:.2%}")
    
    # Test sentiment analysis
    print("\nTesting Sentiment Analysis:")
    print("-" * 60)
    for text in sentiment_samples:
        prediction, confidence, all_probs = predict(model, tokenizer, text, "sentiment")
        print(f"\nText: {text}")
        print(f"Sentiment: {prediction} (Confidence: {confidence:.2%})")
        print("All probabilities:")
        for sentiment, prob in all_probs.items():
            print(f"  {sentiment}: {prob:.2%}")

if __name__ == "__main__":
    main()
