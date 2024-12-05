from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def get_sentiment_scores(texts, batch_size=32):
    """
    Extract sentiment scores for a list of texts using FinBERT.

    Args:
        texts (list): A list of strings (texts) to analyze.
        batch_size (int): Number of texts to process in a batch.

    Returns:
        list: A list of sentiment scores (continuous values) for each text.
               Positive sentiment: Positive score.
               Neutral sentiment: 0 score.
               Negative sentiment: Negative score.
    """
    scores = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()  # Convert logits to probabilities
        
        # Compute sentiment scores
        for probs in probabilities:
            # Positive score: positive prob - negative prob
            score = probs[0] - probs[2]
            scores.append(score)
    
    return scores

# Example usage
if __name__ == "__main__":
    # Example text inputs
    sample_texts = [
        "The company's earnings report exceeded expectations.",
        "The market remains uncertain due to ongoing economic concerns.",
        "The stock price fell sharply after the announcement."
    ]
    
    # Extract sentiment scores
    sentiment_scores = get_sentiment_scores(sample_texts)
    
    # Output sentiment scores
    for text, score in zip(sample_texts, sentiment_scores):
        print(f"Text: {text}")
        print(f"Sentiment Score: {score}\n")