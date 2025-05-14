import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Get script base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "DATA")

# Create DATA folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define ticker to company name mapping (10 examples)
ticker_to_company = {
    'TSLA': 'Tesla, Inc',
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'MSFT': 'Microsoft Corporation',
    'META': 'Meta Platforms, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix, Inc.',
    'INTC': 'Intel Corporation',
    'JPM': 'JPMorgan Chase & Co.'
}

# Display options to user
print("Choose a company to download stock data:")
for i, (ticker, company) in enumerate(ticker_to_company.items(), 1):
    print(f"{i}. {ticker} - {company}")

# Get user input
choice = input("Enter the number of the company (1-10): ")

try:
    choice = int(choice)
    if 1 <= choice <= len(ticker_to_company):
        selected_ticker = list(ticker_to_company.keys())[choice - 1]
        selected_company = ticker_to_company[selected_ticker]
    else:
        raise ValueError("Invalid selection.")
except ValueError as e:
    print(f"❌ Invalid input: {e}")
    exit()

# Set date range
start_date = "2023-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"\n⏳ Fetching data for {selected_ticker} - {selected_company} from {start_date} to {end_date}...")

try:
    df = yf.download(selected_ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Ticker'] = selected_ticker
    df['Company_Name'] = selected_company

    # Rename columns
    df.rename(columns={
        'Open': 'open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'close',
        'Volume': 'Volume',
        'Date': 'Date'
    }, inplace=True)

    # Filter only required columns
    df = df[['Date', 'Ticker', 'Company_Name', 'close', 'High', 'Low', 'open', 'Volume']]

    # Save to CSV
    filename = f"{selected_ticker}_training_stock_2023_2025.csv"
    save_path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(save_path, index=False)

    print(f"✅ Data saved to {save_path}")

except Exception as e:
    print(f"❌ Failed to fetch or save data: {e}")
