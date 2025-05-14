import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model


def prepare_new_data(ticker, feature_names_path):
    """
    Prepare new data for prediction by combining the latest LSTM and FinBERT embeddings
    
    This function handles specific folder structures and different naming conventions
    between input files and feature names.
    """
    # Define paths based on ticker
    lstm_path = os.path.join(r"D:\finbert_lstm_stock_prediction_system\lstm_embeddings", f"{ticker}_lstm_embeddings.csv")
    finbert_path = os.path.join(r"D:\finbert_lstm_stock_prediction_system\sentiment_results", f"{ticker}_full_sentiment_embedding.csv")
    
    print(f"Loading LSTM embeddings from: {lstm_path}")
    print(f"Loading FinBERT embeddings from: {finbert_path}")
    
    # Check if files exist
    if not os.path.exists(lstm_path):
        print(f"Error: LSTM file not found at {lstm_path}")
        return None
    if not os.path.exists(finbert_path):
        print(f"Error: FinBERT file not found at {finbert_path}")
        return None
    
    # Load the feature names used during training
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Load LSTM data
    lstm_df = pd.read_csv(lstm_path)
    # Remove any date or ticker columns
    lstm_df = lstm_df.drop([col for col in lstm_df.columns if col.lower() in ['date', 'ticker']], axis=1, errors='ignore')
    # Get all remaining columns - these should be our embeddings
    lstm_cols = list(lstm_df.columns)
    
    if not lstm_cols:
        print("Error: No columns left in LSTM data after removing date/ticker")
        return None
    
    lstm_last_row = lstm_df.iloc[-1]
    print(f"LSTM columns found: {len(lstm_cols)}")
    
    # Load FinBERT data
    finbert_df = pd.read_csv(finbert_path)
    # Remove any date or ticker columns
    finbert_df = finbert_df.drop([col for col in finbert_df.columns if col.lower() in ['date', 'ticker']], axis=1, errors='ignore')
    # Get all remaining columns - these should be our embeddings
    finbert_cols = list(finbert_df.columns)
    
    if not finbert_cols:
        print("Error: No columns left in FinBERT data after removing date/ticker")
        return None
    
    finbert_last_row = finbert_df.iloc[-1]
    print(f"FinBERT columns found: {len(finbert_cols)}")
    
    # Create a mapping between input column patterns and feature names
    # This will handle differences like "embedding_X" vs "emb_X"
    combined_data = {}
    
    # Initialize all feature values to zero
    for feature_name in feature_names:
        combined_data[feature_name] = 0
    
    # Function to map input column names to feature names
    def map_column_to_feature(col_name):
        # Handle different naming patterns
        if col_name.isdigit():  # LSTM columns might be just numbers
            feature_idx = int(col_name)
            matching_features = [f for f in feature_names if f.isdigit() and int(f) == feature_idx]
            if matching_features:
                return matching_features[0]
        
        # For embedding_X format in input maps to emb_X in features
        if col_name.startswith('embedding_'):
            num = col_name.split('_')[1]
            matching_features = [f for f in feature_names if f == f"emb_{num}"]
            if matching_features:
                return matching_features[0]
        
        # Direct match (emb_X in both)
        if col_name in feature_names:
            return col_name
        
        # For sentiment scores and probabilities
        if col_name in ['sentiment_score', 'prob_pos', 'prob_neg', 'prob_neu']:
            if col_name in feature_names:
                return col_name
        
        # No match found
        return None
    
    # Map LSTM data to features
    for col in lstm_cols:
        feature = map_column_to_feature(col)
        if feature:
            combined_data[feature] = lstm_last_row[col]
    
    # Map FinBERT data to features
    for col in finbert_cols:
        feature = map_column_to_feature(col)
        if feature:
            combined_data[feature] = finbert_last_row[col]
    
    # If we still have unmapped features, try to infer relationships based on position
    unmapped_features = [f for f in feature_names if combined_data[f] == 0]
    if unmapped_features:
        print(f"Warning: {len(unmapped_features)} features couldn't be mapped by name")
        
        # Map LSTM numeric columns to the first features
        if any(col.isdigit() for col in lstm_cols) and any(f.isdigit() for f in unmapped_features):
            lstm_numeric = sorted([int(col) for col in lstm_cols if col.isdigit()])
            features_numeric = sorted([int(f) for f in unmapped_features if f.isdigit()])
            
            for i, col_idx in enumerate(lstm_numeric):
                if i < len(features_numeric):
                    combined_data[str(features_numeric[i])] = lstm_last_row[str(col_idx)]
        
        # Map emb_X features to embedding_X columns for FinBERT
        emb_features = [f for f in unmapped_features if f.startswith('emb_')]
        embedding_cols = [col for col in finbert_cols if col.startswith('embedding_')]
        
        if emb_features and embedding_cols:
            # Sort by the number in the name
            emb_features.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else float('inf'))
            embedding_cols.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else float('inf'))
            
            for i, feature in enumerate(emb_features):
                if i < len(embedding_cols):
                    combined_data[feature] = finbert_last_row[embedding_cols[i]]
    
    # Create final DataFrame and ensure correct order
    final_df = pd.DataFrame([combined_data])
    final_df = final_df[feature_names]
    
    # Check for any features that still have zero values (potential missing mappings)
    zero_features = [f for f in feature_names if final_df[f].iloc[0] == 0]
    if len(zero_features) > 0:
        print(f"Warning: {len(zero_features)} features have zero values after mapping")
    
    return final_df

def make_prediction(ticker, model_dir=r"D:\finbert_lstm_stock_prediction_system\fusion model\models"):
    """
    Make prediction using the latest data for the given ticker
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        model_dir: Directory containing the trained model files (defaults to fusion model/models dir)
        
    Returns:
        Dictionary with prediction results or None if error occurs
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
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Load scaler
    print(f"Loading scaler from: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Prepare input data using the last rows from both embedding files
    input_df = prepare_new_data(ticker, feature_names_path)
    
    if input_df is None:
        print("Error: Failed to prepare input data.")
        return None
    
    # Scale input - ensure we're only scaling numeric data
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        print(f"Error during scaling: {e}")
        print("Input data columns:")
        print(input_df.columns)
        print("Input data types:")
        print(input_df.dtypes)
        return None
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(input_scaled, verbose=0)  # Suppress verbose output
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
    print(f"Confidence: {probability:.4f} ({probability:.2%})")
    
    # Save prediction to CSV
    results_dir = os.path.join(os.path.dirname(model_dir), "predictions")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{ticker}_prediction_{pd.Timestamp.now().strftime('%Y%m%d')}.csv")
    
    pd.DataFrame([result]).to_csv(results_path, index=False)
    print(f"Prediction saved to: {results_path}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--model_dir', default=r"D:\finbert_lstm_stock_prediction_system\fusion model\models", 
                        help='Directory containing trained model')
    
    args = parser.parse_args()
    
    # If ticker is not provided through command line, prompt user
    ticker = args.ticker
    if not ticker:
        ticker = input("Enter stock ticker symbol (e.g., TSLA): ").strip().upper()
        if not ticker:
            print("Error: Ticker symbol is required.")
            sys.exit(1)
    
    print(f"\nMaking prediction for {ticker}...")
    result = make_prediction(ticker, args.model_dir)
    
    if result:
        print("\nPrediction completed successfully!")
        sys.exit(0)
    else:
        print("\nPrediction failed!")
        sys.exit(1)