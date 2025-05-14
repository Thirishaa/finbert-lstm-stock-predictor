from flask import Flask, render_template, request, jsonify
import os
import json
import torch
import re
import string
import datetime
import pandas as pd
import requests
import nltk
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch.nn as nn

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# === Clean text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# === Define BertMultiTask model ===
class BertMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        score = self.regressor(pooled_output).squeeze(-1)
        return logits, score, pooled_output

MODEL_PATH = r"D:\\MAIN PROJECT PHASE 2\\DATASETS\\finbert\\finbert_weights_balanced_ver1\\checkpoint-363"
TOKENIZER_PATH = r"D:\\MAIN PROJECT PHASE 2\\DATASETS\\finbert\\finbert_weights_balanced_ver1"
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
config = BertConfig.from_pretrained(TOKENIZER_PATH)
model = BertMultiTask.from_pretrained(MODEL_PATH, config=config)
model.eval()

import datetime
import requests
import pandas as pd

def fetch_yahoo_news(ticker, api_token):
    try:
        url = f"https://eodhd.com/api/news?s={ticker}&api_token={api_token}&fmt=json"
        response = requests.get(url)
        response.raise_for_status()
        news = response.json()
        df = pd.DataFrame(news)[['title', 'content', 'date']].dropna()

        dates_to_check = [
            (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(5)
        ]

        collected_news = pd.DataFrame(columns=['title', 'content', 'date'])
        used_dates = []

        for date in dates_to_check:
            daily_news = df[df['date'].str.startswith(date)]
            if not daily_news.empty:
                available = 10 - len(collected_news)
                collected_news = pd.concat([collected_news, daily_news.head(available)], ignore_index=True)
                used_dates.append(date)
            if len(collected_news) >= 10:
                break

        return collected_news.reset_index(drop=True), " + ".join(used_dates)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame(), ""


# === Predict sentiment and get embeddings ===
def predict_sentiment_and_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits, score, pooled_output = model(**inputs)
        probs = softmax(logits, dim=1).squeeze()
        return {
            "positive": probs[2].item(),
            "neutral": probs[0].item(),
            "negative": probs[1].item(),
            "embedding": pooled_output.squeeze().tolist()
        }

# === Adjusted Sentiment Assignment ===
def assign_sentiment(score):
    if score >= 0.8:
        return "Positive"
    elif score <= 0.3:
        return "Negative"
    else:
        return "Neutral"

# === Analyze sentiment for the day ===
def analyze_day(ticker, api_token):
    df, used_date = fetch_yahoo_news(ticker, api_token)
    if df.empty:
        return None, None, None

    ticker = ticker.upper()
    df = df[df['title'].str.contains(ticker, case=False, na=False)]
    if df.empty:
        return None, None, None

    df = df.head(3)
    results = []
    embeddings = []

    for idx, row in df.iterrows():
        combined = clean_text(row['title'] + " " + row['content'])
        result = predict_sentiment_and_embedding(combined)
        sentiment_score = result["positive"] - result["negative"]
        sentiment = assign_sentiment(sentiment_score)

        results.append({
            "title": row['title'],
            "positive": result["positive"],
            "neutral": result["neutral"],
            "negative": result["negative"],
            "sentiment": sentiment
        })
        embeddings.append(result["embedding"])

    result_df = pd.DataFrame(results)
    avg_embedding = torch.tensor(embeddings).mean(dim=0).tolist()
    total_pos = result_df['positive'].sum()
    total_neg = result_df['negative'].sum()
    day_sentiment = "Positive" if total_pos > total_neg else "Negative"
    rep_row = result_df.loc[result_df['positive'].idxmax()] if day_sentiment == "Positive" else result_df.loc[result_df['negative'].idxmax()]
    summary = {
        "date": used_date,
        "day_sentiment": day_sentiment,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "representative_title": rep_row['title'],
        "representative_score": rep_row['positive'] if day_sentiment == "Positive" else rep_row['negative']
    }

    return result_df, summary, avg_embedding

import matplotlib.pyplot as plt

def plot_sentiment_scores(df, date, ticker):
    labels = [f'News {i+1}' for i in range(len(df))]
    x = range(len(df))
    plt.figure(figsize=(12, 6))
    plt.bar(x, df['positive'], color='green', label='Positive')
    plt.bar(x, df['neutral'], bottom=df['positive'], color='gray', label='Neutral')
    plt.bar(x, df['negative'], bottom=df['positive'] + df['neutral'], color='red', label='Negative')
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Score")
    plt.title(f"Sentiment Scores for {date}")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_filename = f"{ticker}_sentiment_plot.png"
    plot_dir = os.path.join("static", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return plot_filename

