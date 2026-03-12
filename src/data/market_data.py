import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ── constants ──────────────────────────────────────────────────────────────────
TRAIN_START  = "2010-01-01"
TRAIN_END    = "2019-12-31"   # training window
COVID_START  = "2020-01-01"   # held-out test set (never seen during training)
COVID_END    = "2020-12-31"

AVAILABLE_ASSETS = ["AAPL", "SPY", "TSLA", "GLD", "BTC-USD"]


# ── downloader ─────────────────────────────────────────────────────────────────
def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted OHLCV data for a list of tickers.
    Returns a MultiIndex DataFrame: columns = (OHLCV field, ticker)
    """
    print(f"Downloading {tickers} from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # if single ticker yfinance drops the ticker level — restore it
    if isinstance(raw.columns, pd.Index) and not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    raw.dropna(how="all", inplace=True)
    print(f"  Downloaded {len(raw)} trading days.")
    return raw


# ── feature engineering ────────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Given a MultiIndex OHLCV dataframe and a ticker, compute a feature DataFrame:
        - daily return
        - 5-day & 20-day rolling volatility
        - 10-day & 30-day SMA ratio  (price / SMA — momentum proxy)
        - RSI-14
        - normalised volume
    Returns a clean DataFrame with no NaN rows.
    """
    close  = df["Close"][ticker]
    volume = df["Volume"][ticker]

    feat = pd.DataFrame(index=df.index)

    # returns & volatility
    feat["return"]    = close.pct_change()
    feat["vol_5d"]    = feat["return"].rolling(5).std()
    feat["vol_20d"]   = feat["return"].rolling(20).std()

    # momentum
    feat["sma10_ratio"] = close / close.rolling(10).mean()
    feat["sma30_ratio"] = close / close.rolling(30).mean()

    # RSI
    feat["rsi14"] = _rsi(close, 14)

    # volume (z-score normalised)
    feat["volume_norm"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)

    # raw close kept for reward calculation in the environment
    feat["close"] = close

    feat.dropna(inplace=True)
    return feat


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ── portfolio builder ──────────────────────────────────────────────────────────
def build_portfolio_data(
    tickers: List[str],
    weights: Dict[str, float],
    split: str = "train"          # "train" | "covid"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """
    Downloads and processes data for all tickers in the portfolio.
    Returns:
        features_dict : { ticker: feature_DataFrame }
        weights       : normalised weight dict
    """
    start, end = (TRAIN_START, TRAIN_END) if split == "train" else (COVID_START, COVID_END)
    raw = download_data(tickers, start, end)

    features_dict = {}
    for t in tickers:
        features_dict[t] = compute_features(raw, t)

    # normalise weights so they sum to 1
    total = sum(weights.values())
    norm_weights = {t: w / total for t, w in weights.items()}

    return features_dict, norm_weights


# ── quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}

    feats, w = build_portfolio_data(tickers, weights, split="train")

    for ticker, df in feats.items():
        print(f"\n{ticker}  shape={df.shape}")
        print(df.tail(3))

    print("\nNormalised weights:", w)