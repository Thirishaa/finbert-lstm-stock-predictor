📈 FinBERT-LSTM Stock Prediction System
Overview
The FinBERT-LSTM Stock Prediction System leverages a hybrid approach by combining real-time sentiment analysis of financial news headlines and historical stock price forecasting. By fine-tuning the FinBERT model for sentiment analysis and using LSTM for time-series forecasting, the system predicts stock price movements with enhanced accuracy. A multimodal fusion model integrates both the sentiment embeddings and stock price forecasts to produce the final predictions.

This project involves:
 * Real-time financial news retrieval
 * Sentiment analysis using a fine-tuned FinBERT model
 * Stock price prediction using LSTM
 * Multimodal fusion of sentiment and price prediction outputs
 * Flask-based web interface for interaction

🧰 Tools & Technologies
 * FinBERT: Fine-tuned transformer-based model for sentiment analysis.
 * LSTM (Long Short-Term Memory): Deep learning model for time-series forecasting.
 * Flask: Web framework to serve the prediction model via API.
 * yfinance: Python library to download stock price data from Yahoo Finance.
 * PyTorch: Deep learning framework for training and inference of the models.
 * Pandas & NumPy: Data manipulation and numerical computation.
 * Matplotlib: Data visualization.
 * scikit-learn: Machine learning utilities.

📚 Datasets Used
1. Sentiment Dataset
Source: FIQA-2018 Shared Task
Contents:
 * Financial headlines
 * Sentiment labels (positive, neutral, negative)
 * Sentiment scores

2. Stock Price Dataset
Source: Yahoo Finance
Tool: yfinance
Data Format: OHLCV (Open, High, Low, Close, Volume) data for each ticker from 2023 onward.

⚙️ Model Overview
Fine-Tuned FinBERT (LLM)
Model Type: Transformer-based model (BERT architecture)

Task:
 * Sentiment classification: Classifies financial headlines into three categories: positive, neutral, or negative.
 * Regression: Predicts sentiment score from the financial headlines.
 * Training: The model is fine-tuned on the financial sentiment dataset using both sentiment classification and regression tasks.

LSTM Model
 * Data Used: OHLCV stock data for 160 companies from 2023–2025.
 * Task: Forecasts daily stock price movements and trends using time-series data.

Architecture:

A sequence-to-sequence model that predicts the next day's price trends based on historical data.

Fusion Model
Purpose: Combines the outputs of the sentiment analysis (FinBERT) and stock price forecasting (LSTM) models.

Method:

 * Uses a fully connected neural network that takes as input both the 768-dimensional embeddings from FinBERT and the LSTM outputs.
 * The fusion model makes the final prediction of stock price movement (up/down/neutral).

🌐 Web Interface
 * The system is packaged in a Flask-based web interface, allowing users to interact with the model by:
 * Entering company Ticker for sentiment analysis.
 * Retrieving stock price predictions for selected companies.
 * Viewing the combined results of sentiment and price movement predictions.

