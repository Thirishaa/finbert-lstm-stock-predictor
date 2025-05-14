📈 FinBERT-LSTM Stock Prediction System
This project predicts stock price movements by combining real-time sentiment analysis from financial news headlines using a fine-tuned FinBERT model with stock price forecasting using LSTM. A multimodal fusion model integrates both outputs for enhanced accuracy.

The system includes:

🔍 Real-time financial news retrieval

🧠 Fine-tuned FinBERT for sentiment classification + regression

📊 LSTM for historical stock price prediction

🔗 Fusion model combining LLM + LSTM embeddings

🌐 Flask-based web interface for interaction

📚 Datasets Used
1. Sentiment Dataset
Source: FIQA-2018 Shared Task

Contains: Financial headlines, sentiment labels (positive/neutral/negative), and sentiment scores

2. Stock Price Dataset
Source: Yahoo Finance

Tool: Downloaded using yfinance

Data Format: OHLCV for each ticker from 2023 onward

Model Overview
Fine-Tuned FinBERT (LLM):
Multi-task FinBERT model fine-tuned using dataset for sentiment classification (3 classes) and score regression on financial headlines.

LSTM Model:
Trained on OHLCV stock data (2023–2025) to predict daily stock trends for 160 companies.

Fusion Model:
Combines 768D FinBERT embeddings and LSTM outputs using a fully connected neural classifier for final stock movement prediction.