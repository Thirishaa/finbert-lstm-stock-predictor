import os
import torch
import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertConfig
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertPreTrainedModel, BertModel
from torch import nn

# === Define FinBERT MultiTask Model ===
class BertMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        score = self.regressor(pooled_output).squeeze(-1)
        return logits, score, pooled_output

# === Config ===
TICKERS = ["TSLA"]
NEWSAPI_KEY = "your newsapi key"
N_DAYS = 30
SAVE_DIR = r"D:\finbert lstm front end\fusion model"
MODEL_PATH = r"D:\MAIN PROJECT PHASE 2\DATASETS\finbert\finbert_weights_balanced_ver1\checkpoint-363"
TOKENIZER_PATH = r"D:\MAIN PROJECT PHASE 2\DATASETS\finbert\finbert_weights_balanced_ver1"

# === Load model and tokenizer ===
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
config = BertConfig.from_pretrained(TOKENIZER_PATH)
model = BertMultiTask.from_pretrained(MODEL_PATH, config=config)
model.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

# === Fetch news from NewsAPI ===
def fetch_news(ticker, date_str):
    query = f"{ticker} stock OR {ticker} shares OR alphabet inc"
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from={date_str}&to={date_str}&sortBy=publishedAt"
        f"&pageSize=100&apiKey={NEWSAPI_KEY}"
    )
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        articles = data.get("articles", [])
        return [{"title": a["title"], "content": a.get("description", "")} for a in articles]
    except Exception as e:
        print(f"[ERROR] Fetching NewsAPI news for {ticker} on {date_str}: {e}")
        return []

# === Embedding extraction ===
def extract_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        logits, _, pooled_output = model(**inputs)
        probs = softmax(logits, dim=1).squeeze()
        return pooled_output.squeeze().numpy(), probs.numpy()

# === Backfill loop ===
def backfill_embeddings(ticker):
    print(f"\n📦 Backfilling FinBERT embeddings for {ticker}")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=N_DAYS)
    results = []

    for i in tqdm(range(N_DAYS)):
        day = start_date + timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        news_items = fetch_news(ticker, day_str)

        print(f"{day_str} → {len(news_items)} news articles")

        if not news_items:
            continue

        pooled_list = []
        scores = []

        for item in news_items:
            title = item.get("title", "")
            content = item.get("content", "")

            # Skip if content is None
            if content is None:
                continue

            full_text = (title + " " + content).strip()
            if not full_text:
                continue

            pooled, probs = extract_embedding(full_text)
            sentiment_score = probs[2] - probs[1]  # pos - neg
            pooled_list.append(pooled)
            scores.append((probs, sentiment_score))


        if not pooled_list:
            continue

        pooled_avg = sum(pooled_list) / len(pooled_list)
        probs_avg = sum(s[0] for s in scores) / len(scores)
        score_avg = sum(s[1] for s in scores) / len(scores)

        row = {
            "date": day_str,
            "sentiment_score": score_avg,
            "prob_pos": probs_avg[2],
            "prob_neg": probs_avg[1],
            "prob_neu": probs_avg[0]
        }

        for j, val in enumerate(pooled_avg):
            row[f"emb_{j}"] = val

        results.append(row)

    df = pd.DataFrame(results)
    save_path = os.path.join(SAVE_DIR, f"{ticker}_finbert_embeddings.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ Saved embeddings to {save_path}")

# === Run for all tickers ===
if __name__ == "__main__":
    for ticker in TICKERS:
        backfill_embeddings(ticker)
