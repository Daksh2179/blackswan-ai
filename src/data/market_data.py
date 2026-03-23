import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# constants
TRAIN_START = "2010-01-01"
TRAIN_END   = "2019-12-31"
COVID_START = "2020-01-01"
COVID_END   = "2020-12-31"

AVAILABLE_ASSETS = ["AAPL", "SPY", "TSLA", "GLD", "BTC-USD"]


# downloader
def download_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted OHLCV data for a list of tickers."""
    print(f"Downloading {tickers} from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.Index) and not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    raw.dropna(how="all", inplace=True)
    print(f"  Downloaded {len(raw)} trading days.")
    return raw


# feature engineering
def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute technical features for a single ticker from a MultiIndex OHLCV dataframe."""
    close  = df["Close"][ticker]
    volume = df["Volume"][ticker]

    feat = pd.DataFrame(index=df.index)

    feat["return"]      = close.pct_change()
    feat["vol_5d"]      = feat["return"].rolling(5).std()
    feat["vol_20d"]     = feat["return"].rolling(20).std()
    feat["sma10_ratio"] = close / close.rolling(10).mean()
    feat["sma30_ratio"] = close / close.rolling(30).mean()
    feat["rsi14"]       = _rsi(close, 14)
    feat["volume_norm"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)
    feat["close"]       = close

    feat.dropna(inplace=True)
    return feat


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# shares-based portfolio builder
def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch the latest available closing price for each ticker.
    Returns a dict of { ticker: price }.
    """
    prices = {}
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).fast_info
            prices[ticker] = float(data.last_price)
        except Exception:
            # fallback: download last 5 days and take most recent close
            fallback = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if not fallback.empty:
                prices[ticker] = float(fallback["Close"].iloc[-1])
            else:
                prices[ticker] = 1.0
                print(f"Warning: could not fetch price for {ticker}, defaulting to 1.0")
    return prices


def shares_to_weights(
    shares: Dict[str, float],
    prices: Dict[str, float],
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Convert share counts to portfolio weights based on current prices.

    Args:
        shares : { ticker: number_of_shares }
        prices : { ticker: current_price }

    Returns:
        weights         : { ticker: allocation_weight }  (sums to 1)
        total_value     : total portfolio value in dollars
        position_values : { ticker: dollar_value }
    """
    position_values = {t: shares[t] * prices[t] for t in shares}
    total_value     = sum(position_values.values())

    if total_value == 0:
        raise ValueError("Total portfolio value is zero. Check share counts and prices.")

    weights = {t: v / total_value for t, v in position_values.items()}
    return weights, total_value, position_values


def build_portfolio_from_shares(
    shares: Dict[str, float],
    split: str = "train",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float], float]:
    """
    Full pipeline: takes user share counts, fetches live prices, computes weights,
    downloads historical data, and returns everything needed to initialize the trainer.

    Args:
        shares : { ticker: number_of_shares }  e.g. {"AAPL": 10, "SPY": 5}
        split  : "train" or "covid"

    Returns:
        features_dict   : { ticker: feature_DataFrame }
        weights         : normalized allocation weights
        total_value     : total portfolio value in dollars (used as initial_capital)
    """
    tickers = list(shares.keys())
    start, end = (TRAIN_START, TRAIN_END) if split == "train" else (COVID_START, COVID_END)

    # fetch live prices to compute weights and initial capital
    print("Fetching current prices...")
    prices = get_current_prices(tickers)
    for t, p in prices.items():
        print(f"  {t}: ${p:,.2f}")

    weights, total_value, position_values = shares_to_weights(shares, prices)

    print(f"\nPortfolio value: ${total_value:,.2f}")
    for t in tickers:
        print(f"  {t}: {shares[t]} shares x ${prices[t]:,.2f} = ${position_values[t]:,.2f} ({weights[t]:.1%})")

    # download historical data
    raw = download_data(tickers, start, end)

    features_dict = {}
    for t in tickers:
        features_dict[t] = compute_features(raw, t)

    return features_dict, weights, total_value


# legacy weight-based builder (kept for internal testing)
def build_portfolio_data(
    tickers: List[str],
    weights: Dict[str, float],
    split: str = "train",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    """Weight-based portfolio builder. Used internally for sanity checks."""
    start, end = (TRAIN_START, TRAIN_END) if split == "train" else (COVID_START, COVID_END)
    raw = download_data(tickers, start, end)

    features_dict = {}
    for t in tickers:
        features_dict[t] = compute_features(raw, t)

    total = sum(weights.values())
    norm_weights = {t: w / total for t, w in weights.items()}
    return features_dict, norm_weights


# sanity check
if __name__ == "__main__":
    # simulate user entering share counts
    shares = {
        "AAPL": 10,
        "SPY":  5,
    }

    feats, weights, total_value = build_portfolio_from_shares(shares, split="train")

    print(f"\nInitial capital : ${total_value:,.2f}")
    print(f"Weights         : {weights}")
    for ticker, df in feats.items():
        print(f"\n{ticker}  shape={df.shape}")
        print(df.tail(3))