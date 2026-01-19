import yfinance as yf
import pandas as pd
import os


def download_data(
    ticker: str = "^GSPC",
    start_date: str = "2010-01-01",
    end_date: str = None,
    output_dir: str = "./data",
    output_filename: str = "SP500.csv",
) -> pd.DataFrame:
    """
    Download S&P 500 data from yfinance and save it in the format compatible with Time-Series-Library.

    Args:
        ticker: Stock ticker symbol (default: ^GSPC for S&P 500 index)
        start_date: Start date for data download (format: YYYY-MM-DD)
        end_date: End date for data download (default: None, uses today's date)
        output_dir: Directory to save the output file
        output_filename: Name of the output CSV file

    Returns:
        DataFrame with the downloaded data
    """
    # Download data from yfinance
    print(f"Downloading {ticker} data from {start_date} to {end_date or 'today'}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    # Reset index to make date a column
    df = df.reset_index()

    # Rename columns to match the format in data/ directory
    # Format: date, open, high, low, close, volume (lowercase)
    df.columns = df.columns.get_level_values(
        0
    )  # Handle multi-level columns from yfinance
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
        }
    )

    # Convert date to string format (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Select and reorder columns to match the expected format
    # The format follows Dataset_Custom which expects: date, features..., target
    columns_to_keep = ["date", "open", "high", "low", "close"]
    df = df[columns_to_keep]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    return df


if __name__ == "__main__":
    df = download_data(
        ticker="^NDX",
        start_date="2013-12-01",
        end_date="2023-12-31",
        output_dir="data",
        output_filename="NDX.csv",
    )
    print("\nSample data:")
    print(df.head())
