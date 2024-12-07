import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Import Reuters and Bloomberg news articles into Pandas dataframes
reuters_parquet_path = "data/financial-news-data/reuters_data.parquet.gzip"
df_reuters = pd.read_parquet(reuters_parquet_path)
print(df_reuters.columns)
print("Data loaded successfully!")
print(f"Here's a preview of the data:\n{df_reuters.head()}")
bloomberg_parquet_path = "data/financial-news-data/bloomberg_data.parquet.gzip"
df_bloomberg = pd.read_parquet(bloomberg_parquet_path)
print(df_reuters.columns)
print("Data loaded successfully!")
print(f"Here's a preview of the data:\n{df_bloomberg.head()}")

# Import news headlines into a Pandas dataframe
partner_headlines_path = "data/archive/raw_partner_headlines.csv"
df_partner_headlines = pd.read_csv(partner_headlines_path)
df_partner_headlines = df_partner.drop(columns=["Unnamed: 0"])
print(df_partner_headlines.columns)
print(df_partner_headlines.head())

# Clean and pre-process text
def preprocess_text(text):
    """Clean and preprocess text for FinBERT."""
    if not isinstance(text, str):  # Handle non-string entries
        return ""
    text = text.lower()
    text = "".join([char if char.isalnum() or char.isspace() else " " for char in text])
    text = " ".join(text.split())  # Remove extra spaces
    return text

# Apply cleaning to your text column
df_bloomberg['cleaned_text'] = df_bloomberg['text'].apply(preprocess_text)
df_reuters['cleaned_text'] = df_reuters['text'].apply(preprocess_text)

# Load FinBERT from HuggingFace
finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

print("FinBERT model and tokenizer loaded successfully!")

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

# Tokenize the cleaned text
tokens = tokenize_texts(df_bloomberg['cleaned_text'], finbert_tokenizer)

# Compute numerical sentiment scores
df_bloomberg['numerical_sentiment_score'] = compute_sentiment_scores(tokens, finbert_model)

# Ensure the 'date' column is in datetime format
df_bloomberg['date'] = pd.to_datetime(df_bloomberg['date']).dt.date

# Aggregate sentiment scores by day
daily_sentiment_scores = df_bloomberg.groupby('date')['numerical_sentiment_score'].mean()

# Convert to a DataFrame for easy handling
daily_sentiment_df = daily_sentiment_scores.reset_index()
daily_sentiment_df.columns = ['date', 'average_sentiment_score']

# Preview the results
print(daily_sentiment_df.head())

# Save results to a CSV
daily_sentiment_df.to_csv("daily_numerical_sentiment_scores.csv", index=False)
print("Daily numerical sentiment scores saved successfully!")

# Plot the average sentiment score over time
import matplotlib.pyplot as plt

plt.plot(daily_sentiment_df['date'], daily_sentiment_df['average_sentiment_score'], marker='o')
plt.title("Daily Average Sentiment Score")
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()