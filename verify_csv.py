import polars as pl
import os

# Test Trade CSV
trade_file = r"m:\prosperity_4\dataviz\trades_round_0_day_-1.csv"
print(f"Testing {trade_file} with separator=';'")
try:
    df = pl.read_csv(trade_file, separator=";")
    print(f"Success! Columns: {df.columns}")
    print(df.head(2))
except Exception as e:
    print(f"Failed with comma: {e}")

# Test Price CSV
price_file = r"m:\prosperity_4\dataviz\prices_round_0_day_-1.csv"
print(f"\nTesting {price_file} with separator=';'")
try:
    df = pl.read_csv(price_file, separator=";")
    print(f"Success! Columns: {df.columns}")
    print(df.head(2))
except Exception as e:
    print(f"Failed with semicolon: {e}")
