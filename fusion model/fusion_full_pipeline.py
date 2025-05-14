import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def merge_embeddings(ticker, lstm_dir, finbert_dir, output_dir):
    """
    Merge LSTM and FinBERT embeddings for a given ticker
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    lstm_path = os.path.join(lstm_dir, f"{ticker}_lstm_embeddings.csv")
    finbert_path = os.path.join(finbert_dir, f"{ticker}_finbert_embeddings.csv")
    output_path = os.path.join(output_dir, f"{ticker}_merged_embeddings.csv")
    
    print(f"Processing {ticker}...")
    print(f"LSTM file: {lstm_path}")
    print(f"FinBERT file: {finbert_path}")
    
    # Load data
    lstm_df = pd.read_csv(lstm_path)
    finbert_df = pd.read_csv(finbert_path)
    
    # Convert date columns to datetime
    lstm_df['date'] = pd.to_datetime(lstm_df['date'])
    finbert_df['date'] = pd.to_datetime(finbert_df['date'])
    
    # Merge on matching dates
    merged_df = pd.merge(lstm_df, finbert_df, on='date', how='inner')
    
    # Save merged dataframe
    merged_df.to_csv(output_path, index=False)
    print(f"Merged embeddings saved to: {output_path}")
    
    return output_path, merged_df

def add_price_target(ticker, merged_data_path, price_data_path, output_dir):
    """
    Add stock price data and create target columns
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load merged embeddings
    df = pd.read_csv(merged_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load stock price data
    price_df = pd.read_csv(price_data_path)
    
    # Check date format and convert
    # Try different date formats if needed
    try:
        price_df['date'] = pd.to_datetime(price_df['Date'], dayfirst=True)
    except:
        try:
            price_df['date'] = pd.to_datetime(price_df['Date'])
        except:
            print("Error: Could not parse date column in price data. Please check format.")
            return None
    
    # Select relevant columns
    if 'close' in price_df.columns:
        price_df = price_df[['date', 'close']]
    elif 'Close' in price_df.columns:
        price_df = price_df[['date', 'Close']]
        price_df.rename(columns={'Close': 'close'}, inplace=True)
    else:
        print("Error: Could not find close/Close column in price data.")
        return None
    
    # Merge embeddings with price data
    df = df.merge(price_df, on='date', how='left')
    
    # Check for missing values after merge
    if df['close'].isna().any():
        print(f"Warning: {df['close'].isna().sum()} rows have missing close prices after merge.")
        # Fill missing values or drop rows
        df = df.dropna(subset=['close'])
    
    # Create target column (1 if price increases next day, 0 otherwise)
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    
    # Drop temporary column
    df.drop(columns=['next_close'], inplace=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"{ticker}_with_price_target.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset with price and target saved to: {output_path}")
    
    return output_path, df

def train_model(ticker, data_path, model_dir, epochs=50, batch_size=32):
    """
    Train a neural network model on the merged data
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Drop non-feature columns
    X = df.drop(columns=['date', 'close', 'target'])
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for future predictions
    scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Build neural network
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train with early stopping
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate model
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_dir, f"{ticker}_training_history.png")
    plt.savefig(plot_path)
    
    # Save model
    model_path = os.path.join(model_dir, f"{ticker}_prediction_model.h5")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature names for future predictions
    feature_names = X.columns.tolist()
    feature_path = os.path.join(model_dir, f"{ticker}_feature_names.txt")
    with open(feature_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    return {
        'model_path': model_path,
        'scaler_path': scaler_path,
        'feature_names': feature_names,
        'accuracy': history.history['val_accuracy'][-1],
        'plot_path': plot_path
    }

def prepare_new_data(lstm_path, finbert_path, feature_names_path):
    """
    Prepare new data for prediction by combining the latest LSTM and FinBERT embeddings
    """
    # Load the feature names
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Load LSTM data and get the latest row
    lstm_df = pd.read_csv(lstm_path)
    lstm_cols = [col for col in lstm_df.columns if col != 'date']
    lstm_last_row = lstm_df.iloc[-1][lstm_cols]
    
    # Load FinBERT data and get the latest row
    finbert_df = pd.read_csv(finbert_path)
    finbert_cols = [col for col in finbert_df.columns if col != 'date']
    finbert_last_row = finbert_df.iloc[-1][finbert_cols]
    
    # Combine them into a single row
    combined_row = pd.concat([lstm_last_row, finbert_last_row])
    final_df = combined_row.to_frame().T
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(final_df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feature in missing_features:
            final_df[feature] = 0  # Fill with zeros as default
    
    # Select only the required features in the correct order
    final_df = final_df[feature_names]
    
    return final_df

def make_prediction(ticker, lstm_path, finbert_path, model_dir):
    """
    Make prediction using the latest data
    """
    # File paths
    model_path = os.path.join(model_dir, f"{ticker}_prediction_model.h5")
    scaler_path = os.path.join(model_dir, f"{ticker}_scaler.pkl")
    feature_names_path = os.path.join(model_dir, f"{ticker}_feature_names.txt")
    
    # Check if all required files exist
    for path in [model_path, scaler_path, feature_names_path]:
        if not os.path.exists(path):
            print(f"Error: Required file {path} not found.")
            return None
    
    # Load model
    model = load_model(model_path)
    
    # Load scaler
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare input data
    input_df = prepare_new_data(lstm_path, finbert_path, feature_names_path)
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    binary_prediction = (prediction > 0.5).astype(int)[0][0]
    probability = prediction[0][0]
    
    result = {
        'ticker': ticker,
        'prediction': binary_prediction,
        'probability': float(probability),
        'prediction_text': "UP" if binary_prediction == 1 else "DOWN",
        'date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    print(f"\nPrediction for {ticker}:")
    print(f"Direction: {'UP ↑' if binary_prediction == 1 else 'DOWN ↓'}")
    print(f"Confidence: {probability:.4f}")
    
    return result

def run_full_pipeline(ticker, lstm_dir, finbert_dir, price_data_path, output_dir, model_dir):
    """
    Run the full pipeline: merge, add target, train, predict
    """
    # Step 1: Merge embeddings
    merged_path, _ = merge_embeddings(ticker, lstm_dir, finbert_dir, output_dir)
    
    # Step 2: Add price and target
    data_path, _ = add_price_target(ticker, merged_path, price_data_path, output_dir)
    
    # Step 3: Train model
    model_info = train_model(ticker, data_path, model_dir)
    
    return {
        'ticker': ticker,
        'merged_path': merged_path,
        'data_path': data_path,
        'model_path': model_info['model_path'],
        'accuracy': model_info['accuracy']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Prediction Fusion Model Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], required=True, 
                        help='Mode: train, predict, or full pipeline')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    parser.add_argument('--lstm_dir', default='lstm_embeddings', help='Directory with LSTM embeddings')
    parser.add_argument('--finbert_dir', default='finbert_embeddings', help='Directory with FinBERT embeddings')
    parser.add_argument('--price_data', help='Path to stock price CSV file')
    parser.add_argument('--output_dir', default='merged_embeddings', help='Output directory for merged data')
    parser.add_argument('--model_dir', default='trained_models', help='Directory for trained models')
    parser.add_argument('--lstm_file', help='Path to LSTM embeddings file (for prediction mode)')
    parser.add_argument('--finbert_file', help='Path to FinBERT embeddings file (for prediction mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'full':
        if not args.price_data:
            print("Error: --price_data is required for train and full modes")
            exit(1)
    
    if args.mode == 'predict':
        if not (args.lstm_file and args.finbert_file):
            print("Error: --lstm_file and --finbert_file are required for predict mode")
            exit(1)
    
    # Execute based on mode
    if args.mode == 'full':
        result = run_full_pipeline(
            args.ticker, 
            args.lstm_dir, 
            args.finbert_dir, 
            args.price_data, 
            args.output_dir, 
            args.model_dir
        )
        print("\nFull pipeline completed successfully!")
        print(f"Model accuracy: {result['accuracy']:.4f}")
    
    elif args.mode == 'train':
        # Merge embeddings
        merged_path, _ = merge_embeddings(args.ticker, args.lstm_dir, args.finbert_dir, args.output_dir)
        
        # Add price and target
        data_path, _ = add_price_target(args.ticker, merged_path, args.price_data, args.output_dir)
        
        # Train model
        model_info = train_model(args.ticker, data_path, args.model_dir)
        print(f"\nTraining completed successfully!")
        print(f"Model accuracy: {model_info['accuracy']:.4f}")
    
    elif args.mode == 'predict':
        result = make_prediction(args.ticker, args.lstm_file, args.finbert_file, args.model_dir)
        print("\nPrediction completed successfully!")