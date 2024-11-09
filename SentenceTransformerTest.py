import torch
from transformers import AutoTokenizer
from model import SentenceTransformer
import numpy as np

def test_sentence_transformer():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = SentenceTransformer("roberta-base").to(device)
    model.eval()

    # Sample sentences
    sentences = [
        "The weather is beautiful today.",
        "I love programming in Python.",
        "The weather is terrible today.",
        "Machine learning is fascinating.",
    ]

    # Tokenize sentences
    encodings = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move tensors to device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Get embeddings
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)

    # Convert to numpy for easier analysis
    embeddings_np = embeddings.cpu().numpy()

    # Print embedding information
    print(f"\nEmbedding shape: {embeddings_np.shape}")
    print(f"Each sentence is represented by a {embeddings_np.shape[1]}-dimensional vector")

    # Calculate and print cosine similarities between sentences
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    print("\nCosine similarities between sentences:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(embeddings_np[i], embeddings_np[j])
            print(f"\nSentence {i+1} vs Sentence {j+1}")
            print(f"'{sentences[i]}' vs '{sentences[j]}'")
            print(f"Similarity: {sim:.4f}")

    # Print first few dimensions of each embedding
    print("\nFirst 5 dimensions of each embedding:")
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: '{sentence}'")
        print(f"Embedding start: {embeddings_np[i][:5]}")

