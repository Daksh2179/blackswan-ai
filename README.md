# BlackSwan AI
> Work in progress

Adversarial stress-testing system for trading strategies using multi-agent reinforcement learning and LSTM-based market scenario generation.

---

## What It Does

Most trading strategies are validated through backtesting on historical data. BlackSwan AI takes a different approach — it stress-tests your strategy against worst-case market conditions before you deploy it.

The system is a two-player adversarial game:

- **Market Adversary** — an LSTM neural network that generates extreme market scenarios (flash crashes, volatility spikes, regime changes) designed to break your trading strategy
- **Trading Agent** — a PPO reinforcement learning agent that learns to survive and adapt under adversarial conditions

Both agents train simultaneously in a minimax loop. The adversary finds weaknesses, the trading agent adapts. This co-evolution produces a strategy that is robust to conditions it has never seen before.

---

## Project Structure

```
blackswan-ai/
├── src/
│   ├── data/
│   │   └── market_data.py        # yfinance data pipeline and feature engineering
│   ├── environment/
│   │   └── trading_env.py        # custom Gymnasium trading environment
│   ├── agents/
│   │   └── adversary.py          # LSTM market adversary
│   ├── training/
│   │   └── trainer.py            # minimax training loop and curve generation
│   └── utils/
│       └── metrics.py            # Sharpe, drawdown, comparison table, failure modes
└── tests/
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/blackswan-ai.git
cd blackswan-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Running the Backend

Each module has a sanity check you can run independently:

```bash
# test data pipeline
python src/data/market_data.py

# test trading environment
python src/environment/trading_env.py

# test market adversary
python src/agents/adversary.py

# run full minimax training and see metrics
python src/utils/metrics.py
```

---

## Current Status

Backend implemented and working:

- [x] Data pipeline with feature engineering (returns, volatility, RSI, SMA ratios, volume)
- [x] Custom Gymnasium trading environment with adversarial price injection
- [x] LSTM market adversary with four scenario types
- [x] Minimax training loop with PPO trading agent
- [x] Performance metrics and failure mode analysis
- [ ] Shares-based portfolio input
- [ ] Strategy reward shaping (Momentum, Mean Reversion, Trend Following)
- [ ] Training intensity presets
- [ ] Streamlit UI
- [ ] COVID crash held-out test

---

## Tech Stack

| Component | Library |
|---|---|
| RL Trading Agent | Stable-Baselines3 (PPO) |
| Market Adversary | PyTorch LSTM |
| Trading Environment | Gymnasium |
| Data | yfinance + pandas |
| UI (upcoming) | Streamlit + Plotly |
| Backend API (upcoming) | FastAPI |
