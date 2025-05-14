from flask import Flask, render_template, request, jsonify, url_for
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.finbert_module import analyze_day, plot_sentiment_scores

app = Flask(__name__)

# Define output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "finbert_embeddings")
PLOT_DIR = os.path.join(BASE_DIR, "static", "finbert_plots")
LSTM_PLOT_DIR = os.path.join(BASE_DIR, "static", "lstm_plots")
LSTM_EMBEDDINGS_DIR = os.path.join(BASE_DIR, "lstm_embeddings")

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LSTM_PLOT_DIR, exist_ok=True)
os.makedirs(LSTM_EMBEDDINGS_DIR, exist_ok=True)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()
        api_token = "your eodhd api key"   # Replace with your actual token
        
        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400
        
        result_df, summary, avg_embedding = analyze_day(ticker, api_token)
        if result_df is None:
            return "No relevant news found for today."
        
        # Save JSON
        save_path = os.path.join(OUTPUT_DIR, f"{ticker}_sentiment_output.json")
        with open(save_path, "w") as f:
            json.dump({
                "summary": summary,
                "results": result_df.to_dict(orient="records"),
                "embedding": avg_embedding
            }, f, indent=4)
        
        # Save CSV
        metadata = {
            "ticker": ticker,
            "date": summary['date'],
            "sentiment_score": summary['total_positive'] - summary['total_negative'],
            "prob_pos": summary['total_positive'],
            "prob_neg": summary['total_negative'],
            "prob_neu": 1.0 - (summary['total_positive'] + summary['total_negative'])
        }
        emb_dict = {f"embedding_{i+1}": val for i, val in enumerate(avg_embedding)}
        combined = {**metadata, **emb_dict}
        csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_full_sentiment_embedding.csv")
        pd.DataFrame([combined]).to_csv(csv_path, index=False)
        
        # Generate plot
        sentiment_plot_filename = plot_sentiment_scores(result_df, summary['date'], ticker)
        
        return render_template("finbert_results.html",
                              ticker=ticker,
                              summary=summary,
                              news_data=result_df.to_dict(orient='records'),
                              sentiment_plot=sentiment_plot_filename)
    else:
        return render_template('index.html')

def load_and_preprocess_data(ticker):
    """Load and preprocess stock data for LSTM prediction"""
    try:
        # Path should be adjusted to your actual data file location
        file_path = os.path.join(BASE_DIR, "DATA", f"{ticker}_training_stock_2023_2025.csv")
        
        if not os.path.exists(file_path):
            # Fall back to TSLA if specific ticker data not found
            file_path = os.path.join(BASE_DIR, "DATA", "TSLA_training_stock_2023_2025.csv")
            print(f"Data for {ticker} not found, using TSLA data instead")
        
        df = pd.read_csv(file_path)
        
        # Process date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
        
        # Ensure we have a 'close' column
        close_columns = ['close', 'Close', 'Adj Close']
        for col in close_columns:
            if col in df.columns:
                if col != 'close':
                    df['close'] = df[col]
                break
        
        if 'close' not in df.columns:
            raise ValueError("No closing price column found in data")
        
        # Make sure 'close' is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None

def engineer_features(df):
    """Create features for LSTM model"""
    data = df[['close']].copy()
    
    # Create basic features
    data['MA5'] = data['close'].rolling(window=5).mean()
    data['MA10'] = data['close'].rolling(window=10).mean()
    data['Price_Change'] = data['close'].diff()
    data['Volatility_5d'] = data['close'].rolling(window=5).std()
    
    # Add time features if we have date index
    if isinstance(df.index, pd.DatetimeIndex):
        data['sin_day'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
        data['cos_day'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    
    # Drop rows with NaN
    data.dropna(inplace=True)
    
    return data

def prepare_sequences(features, target, time_steps=5):
    """Scale data and create sequences for LSTM"""
    # Normalize features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(scaled_features, scaled_target, time_steps)
    
    return X, y, scaler_features, scaler_target, scaled_features, scaled_target

def create_sequences(X, y, time_steps):
    """Create sequences for LSTM model"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape):
    """Build a simple LSTM model"""
    model = Sequential()
    model.add(LSTM(units=50,
                   return_sequences=True,
                   input_shape=input_shape,
                   kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=30,
                   return_sequences=False,
                   kernel_regularizer=l1_l2(l1=0, l2=1e-5)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_future(model, last_sequence, scaler_features, scaler_target, days_to_predict=30):
    """Forecast future stock prices using trained model"""
    # Make a copy of the last sequence
    curr_sequence = np.copy(last_sequence)
    forecasted_prices = []
    
    for _ in range(days_to_predict):
        # Reshape for prediction
        curr_sequence_reshaped = curr_sequence.reshape(1, curr_sequence.shape[0], curr_sequence.shape[1])
        
        # Predict next day
        next_day_scaled = model.predict(curr_sequence_reshaped)
        
        # Inverse transform
        next_day_price = scaler_target.inverse_transform(next_day_scaled)[0, 0]
        forecasted_prices.append(next_day_price)
        
        # Update sequence for next prediction
        last_features = curr_sequence[-1].copy()
        last_features[0] = next_day_scaled[0, 0]
        
        # Roll the window forward
        curr_sequence = np.roll(curr_sequence, -1, axis=0)
        curr_sequence[-1] = last_features
    
    return forecasted_prices

def plot_forecast(df, forecasted_prices, ticker):
    """Generate plot of historical and forecasted prices"""
    # Get last 30 days of historical data
    last_known_prices = df['close'].values[-30:]
    
    # Create date range for forecast
    if isinstance(df.index, pd.DatetimeIndex):
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecasted_prices))
    else:
        last_idx = len(df)
        forecast_dates = range(last_idx, last_idx + len(forecasted_prices))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    if isinstance(df.index, pd.DatetimeIndex):
        plt.plot(df.index[-30:], last_known_prices, 'b-', label='Historical Prices')
        plt.plot(forecast_dates, forecasted_prices, 'r--', label='Forecasted Prices')
        plt.xticks(rotation=45)
    else:
        plt.plot(range(len(last_known_prices)), last_known_prices, 'b-', label='Historical Prices')
        plt.plot(range(len(last_known_prices), len(last_known_prices) + len(forecasted_prices)), 
                 forecasted_prices, 'r--', label='Forecasted Prices')
    
    # Add forecast start line
    if isinstance(df.index, pd.DatetimeIndex):
        plt.axvline(x=df.index[-1], color='green', linestyle='--', linewidth=2, label='Forecast Start')
    else:
        plt.axvline(x=len(last_known_prices)-1, color='green', linestyle='--', linewidth=2, label='Forecast Start')
    
    # Calculate confidence intervals (simple approximation)
    std_dev = np.std(last_known_prices[-30:]) if len(last_known_prices) >= 30 else np.std(last_known_prices)
    upper_bound = np.array(forecasted_prices) + 2 * std_dev
    lower_bound = np.array(forecasted_prices) - 2 * std_dev
    
    # Plot confidence intervals
    plt.fill_between(forecast_dates, lower_bound, upper_bound, color='red', alpha=0.1, label='95% Confidence Interval')
    
    plt.title(f'{ticker} - Stock Price Forecast')
    plt.xlabel('Date' if isinstance(df.index, pd.DatetimeIndex) else 'Time Step')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory exists (using the specific path requested)
    output_dir = LSTM_PLOT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot with the requested naming format
    plot_filename = f'{ticker}_FORECAST.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

def calculate_risk_metrics(historical_prices, forecasted_prices):
    """Calculate risk metrics and volatility"""
    # Historical volatility (standard deviation of returns)
    historical_returns = np.diff(historical_prices) / historical_prices[:-1]
    hist_volatility = np.std(historical_returns) * np.sqrt(252)  # Annualized
    
    # Value at Risk (VaR)
    confidence_level = 0.95
    var_95 = np.percentile(historical_returns, (1 - confidence_level) * 100) * forecasted_prices[0]
    
    # Maximum potential loss
    forecast_std = np.std(historical_returns) * np.array(forecasted_prices[:-1])
    max_loss = abs(min(np.diff(forecasted_prices) - 2 * forecast_std)) if len(forecasted_prices) > 1 else 0
    
    # Maximum Drawdown in forecast
    if len(forecasted_prices) > 1:
        forecast_returns = np.diff(forecasted_prices) / forecasted_prices[:-1]
        cumulative_returns = np.cumprod(1 + forecast_returns)
        max_drawdown = 1 - min(cumulative_returns) if len(cumulative_returns) > 0 else 0
    else:
        max_drawdown = 0
    
    # Risk assessment
    risk_level = "Low"
    if hist_volatility > 0.3:
        risk_level = "High"
    elif hist_volatility > 0.15:
        risk_level = "Medium"
    
    return {
        'volatility': hist_volatility * 100,  # Convert to percentage
        'var_95': abs(var_95),
        'max_loss': max_loss,
        'max_drawdown': max_drawdown * 100,  # Convert to percentage
        'risk_level': risk_level
    }

def generate_trading_signals(df, forecasted_prices):
    """Generate trading signals based on forecast"""
    # Current price is the last historical price
    current_price = df['close'].iloc[-1]
    
    # Calculate short-term and medium-term forecasts
    short_term = forecasted_prices[4] if len(forecasted_prices) > 4 else forecasted_prices[-1]
    medium_term = forecasted_prices[-1]
    
    # Calculate percent changes
    short_term_change = (short_term - current_price) / current_price * 100
    medium_term_change = (medium_term - current_price) / current_price * 100
    
    # Determine signals
    signal = "HOLD"
    confidence = "Medium"
    
    if short_term_change > 3 and medium_term_change > 5:
        signal = "STRONG BUY"
        confidence = "High" if medium_term_change > 10 else "Medium"
    elif short_term_change > 1 and medium_term_change > 0:
        signal = "BUY"
        confidence = "Medium"
    elif short_term_change < -3 and medium_term_change < -5:
        signal = "STRONG SELL"
        confidence = "High" if medium_term_change < -10 else "Medium"
    elif short_term_change < -1 and medium_term_change < 0:
        signal = "SELL"
        confidence = "Medium"
    
    # Check for volatility in historical data
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    high_volatility = volatility > 0.4
    
    # Generate recommendation
    if signal == "STRONG BUY":
        recommendation = f"Consider buying with a target price of ${medium_term:.2f}"
        stop_loss = f"Suggested stop loss: ${current_price * 0.95:.2f} (-5%)"
    elif signal == "BUY":
        recommendation = f"Consider buying with a target price of ${medium_term:.2f}"
        stop_loss = f"Suggested stop loss: ${current_price * 0.97:.2f} (-3%)"
    elif signal == "STRONG SELL":
        recommendation = f"Consider selling with a target price of ${medium_term:.2f}"
        stop_loss = ""
    elif signal == "SELL":
        recommendation = f"Consider reducing position"
        stop_loss = ""
    else:
        recommendation = f"Monitor for better entry or exit points"
        stop_loss = ""
    
    return {
        'current_price': current_price,
        'short_term_price': short_term,
        'short_term_change': short_term_change,
        'medium_term_price': medium_term,
        'medium_term_change': medium_term_change,
        'signal': signal,
        'confidence': confidence,
        'high_volatility': high_volatility,
        'recommendation': recommendation,
        'stop_loss': stop_loss
    }

def save_lstm_embeddings(model, X, df, ticker):
    """Extract and save LSTM embeddings (use only final hidden state)"""
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import LSTM

        lstm_layer = None
        for layer in model.layers:
            if isinstance(layer, LSTM):
                lstm_layer = layer
        if lstm_layer is None:
            print("No LSTM layer found")
            return None

        intermediate_model = Model(inputs=model.input, outputs=lstm_layer.output)
        embeddings = intermediate_model.predict(X)

        # FIX: Only keep the last timestep's output (if 3D)
        if len(embeddings.shape) == 3:
            embeddings = embeddings[:, -1, :]  # shape: (samples, features)

        # Now convert to DataFrame (will be 15 columns if 15 units)
        embedding_df = pd.DataFrame(embeddings)

        if df is not None and isinstance(df.index, pd.DatetimeIndex):
            embedding_df['date'] = df.index[-len(embeddings):]

        output_path = os.path.join(LSTM_EMBEDDING_DIR, f"{ticker}_lstm_embeddings.csv")
        embedding_df.to_csv(output_path, index=False)

        print(f"\n✅ Saved LSTM embeddings to CSV at: {output_path}")
        return output_path

    except Exception as e:
        print(f"[ERROR] Failed to extract/save LSTM embeddings: {e}")
        traceback.print_exc()
        return None


    
@app.route('/analyze_lstm', methods=['POST'])
def analyze_lstm():
    try:
        ticker = request.form.get('ticker').upper()
        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400

        df = load_and_preprocess_data(ticker)
        if df is None:
            return render_template("lstm_error.html", ticker=ticker, error="Could not load data for this ticker")

        data = engineer_features(df)
        target = data['close']
        features = data.drop('close', axis=1)
        time_steps = 5
        X, y, scaler_features, scaler_target, scaled_features, scaled_target = prepare_sequences(features, target, time_steps)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

        model_metrics = {}
        if len(X_test) > 0 and len(y_test) > 0:
            y_pred = model.predict(X_test)
            y_test_inv = scaler_target.inverse_transform(y_test)
            y_pred_inv = scaler_target.inverse_transform(y_pred)
            model_metrics = {
                'mae': mean_absolute_error(y_test_inv, y_pred_inv),
                'mse': mean_squared_error(y_test_inv, y_pred_inv),
                'rmse': np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
                'r2': r2_score(y_test_inv, y_pred_inv),
                'tracking_error': np.std(y_test_inv - y_pred_inv),
                'direction_accuracy': np.mean((np.diff(y_test_inv.flatten()) > 0) == (np.diff(y_pred_inv.flatten()) > 0))
            }

        forecast_days = 30
        forecasted_prices = forecast_future(model, X[-1], scaler_features, scaler_target, forecast_days)
        plot_filename = plot_forecast(df, forecasted_prices, ticker)
        risk_metrics = calculate_risk_metrics(df['close'].values[-90:], forecasted_prices)
        trading_signals = generate_trading_signals(df, forecasted_prices)

        technical_indicators = {}
        if len(df) >= 200:
            try:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                macd = (ema12 - ema26).iloc[-1]
                low_14 = df['close'].rolling(window=14).min()
                high_14 = df['close'].rolling(window=14).max()
                k = 100 * ((df['close'] - low_14) / (high_14 - low_14)).iloc[-1]
                ma_50 = df['close'].rolling(window=50).mean().iloc[-1]
                ma_200 = df['close'].rolling(window=200).mean().iloc[-1]
                ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                std_20 = df['close'].rolling(window=20).std().iloc[-1]
                upper_band = ma_20 + (std_20 * 2)
                lower_band = ma_20 - (std_20 * 2)
                rsi_signal = "neutral"
                if rsi > 70: rsi_signal = "negative"
                elif rsi < 30: rsi_signal = "positive"
                macd_signal = "positive" if macd > 0 else "negative"
                stoch_signal = "neutral"
                if k > 80: stoch_signal = "negative"
                elif k < 20: stoch_signal = "positive"
                ma_50_signal = "positive" if df['close'].iloc[-1] > ma_50 else "negative"
                ma_200_signal = "positive" if df['close'].iloc[-1] > ma_200 else "negative"
                current_price = df['close'].iloc[-1]
                bollinger_signal = "neutral"
                if current_price > upper_band: bollinger_signal = "negative"
                elif current_price < lower_band: bollinger_signal = "positive"
                technical_indicators = {
                    'rsi': rsi,
                    'rsi_signal': rsi_signal,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'stoch': k,
                    'stoch_signal': stoch_signal,
                    'ma_50': ma_50,
                    'ma_50_signal': ma_50_signal,
                    'ma_200': ma_200,
                    'ma_200_signal': ma_200_signal,
                    'bollinger_upper': upper_band,
                    'bollinger_lower': lower_band,
                    'bollinger_signal': bollinger_signal
                }
            except Exception as e:
                print(f"Error calculating technical indicators: {e}")
                traceback.print_exc()

        embeddings_path = save_lstm_embeddings(model, X, df, ticker)

        return render_template("lstm_results.html",
                              ticker=ticker,
                              forecast_plot=f"{ticker}_FORECAST.png",
                              risk_metrics=risk_metrics,
                              trading_signals=trading_signals,
                              technical_indicators=technical_indicators,
                              model_metrics=model_metrics,
                              embeddings_path=embeddings_path,
                              now=datetime.now)  # FIXED

    except Exception as e:
        traceback.print_exc()
        return render_template("lstm_error.html", ticker=ticker if 'ticker' in locals() else "Unknown", error=str(e))

if __name__ == "__main__":
    app.run(debug=True , use_reloader=False)