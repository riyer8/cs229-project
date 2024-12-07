import pandas as pd
import numpy as np
import re
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Import Reuters and Bloomberg news articles into Pandas dataframes
reuters_parquet_path = "data/financial-news-data/reuters_data.parquet.gzip"
df_reuters = pd.read_parquet(reuters_parquet_path)
bloomberg_parquet_path = "data/financial-news-data/bloomberg_data.parquet.gzip"
df_bloomberg = pd.read_parquet(bloomberg_parquet_path)

# Import news headlines into a Pandas dataframe
partner_headlines_path = "data/financial-news-data/raw_partner_headlines.csv"
df_partner_headlines = pd.read_csv(partner_headlines_path)
df_partner_headlines = df_partner_headlines.drop(columns=["Unnamed: 0"])
print(df_partner_headlines.columns)
print(df_partner_headlines.head())

# Clean and pre-process text
def preprocess_text(article):
    """Clean and preprocess text for FinBERT."""
    article = article.lower() # Convert to lowercase
    article = re.sub(r'http\S+|www\S+|https\S+', '', article, flags=re.MULTILINE) # Remove URLs
    article = re.sub(r'<.*?>', '', article) # Remove HTML
    article = re.sub(r'[^a-zA-Z0-9\s]', '', article) # Remove special characters and punctuation
    article = re.sub(r'\d+', '', article) # Remove numbers (optional, depending on the use case)
    article = re.sub(r'\s+', ' ', article).strip() # Remove extra whitespace
    return article

# Tokenize text
def tokenize_text(articles, tokenizer):
    """
    Tokenize preprocessed text using FinBERT tokenizer for a batch.
    
    Parameters:
    articles (pd.Series or list): The preprocessed text of the news articles.
    tokenizer: The FinBERT tokenizer instance.
    
    Returns:
    dict: Tokenized representation ready for FinBERT.
    """
    return tokenizer(list(articles), truncation=True, padding=True, return_tensors="pt")

# Compute sentiment scores
def compute_sentiment_scores(tokens, model):
    """Compute sentiment scores for a batch of tokens."""
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1).numpy()
    
    # Compute numerical sentiment scores: positive=1, neutral=0, negative=-1
    sentiment_scores = probabilities[:, 2] - probabilities[:, 0]
    return sentiment_scores

if __name__ == '__main__':
    finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    # Apply cleaning to your text column
    df_bloomberg['cleaned_text'] = df_bloomberg['Article'].apply(preprocess_text)
    df_reuters['cleaned_text'] = df_reuters['Article'].apply(preprocess_text)

    df_bloomberg['cleaned_text'] = df_bloomberg['cleaned_text'].fillna("").astype(str)
    df_reuters['cleaned_text'] = df_reuters['cleaned_text'].fillna("").astype(str)
    df_partner_headlines['cleaned_text'] = df_partner_headlines['headline'].fillna("").astype(str)

    # Tokenize the cleaned text
    bloomberg_tokens = tokenize_text(df_bloomberg['cleaned_text'], finbert_tokenizer)
    reuters_tokens = tokenize_text(df_reuters['cleaned_text'], finbert_tokenizer)
    headlines_tokens = tokenize_text(df_partner_headlines['headline'], finbert_tokenizer)

    # Compute numerical sentiment scores
    df_bloomberg['numerical_sentiment_score'] = compute_sentiment_scores(tokens, finbert_model)
    df_reuters['numerical_sentiment_score'] = compute_sentiment_scores(tokens, finbert_model)
    df_partner_headlines['numerical_sentiment_score'] = compute_sentiment_scores(tokens, finbert_model)

    # Ensure the 'date' column is in datetime format
    df_reuters['Date'] = pd.to_datetime(df_reuters['Date']).dt.date
    df_bloomberg['Date'] = pd.to_datetime(df_bloomberg['Date']).dt.date
    df_partner_headlines['Date'] = pd.to_datetime(df_partner_headlines['date']).dt.date

    # Aggregate sentiment scores by day
    bloomberg_daily_sentiment_scores = df_bloomberg.groupby('Date')['numerical_sentiment_score'].mean()
    reuters_daily_sentiment_scores = df_reuters.groupby('Date')['numerical_sentiment_score'].mean()
    headilnes_daily_sentiment_scores = df_headlines.groupby('Date')['numerical_sentiment_score'].mean()

    # Combine news sources
    combined_daily_sentiment = pd.DataFrame({
        'bloomberg': bloomberg_daily,
        'reuters': reuters_daily,
        'headlines': headlines_daily
    })

    # Compute overall daily average sentiment
    combined_daily_sentiment['average_sentiment_score'] = combined_daily_sentiment.mean(axis=1)

    # Reset index to create a proper DataFrame
    combined_daily_sentiment.reset_index(inplace=True)
    combined_daily_sentiment.rename(columns={'index': 'date'}, inplace=True)

    # Save results to a CSV
    combined_daily_sentiment.to_csv("daily_numerical_sentiment_scores.csv", index=False)
    print("Daily numerical sentiment scores saved successfully!")

     # Plot the average sentiment score over time
    import matplotlib.pyplot as plt

    plt.plot(combined_daily_sentiment['Date'], combined_daily_sentiment['average_sentiment_score'], marker='o')
    plt.title("Daily Average Sentiment Score")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()