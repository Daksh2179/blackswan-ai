# BlackSwan AI
> Adversarial stress-testing for trading strategies

BlackSwan AI puts your trading strategy into a simulated battle against an AI adversary whose only job is to find and exploit its weaknesses. The result is a hardened version of your strategy that has been stress-tested against conditions it has never seen before.

Built with reinforcement learning, adversarial game theory, and LSTM-based market scenario generation.

---

## Screenshots

<img width="1919" height="827" alt="Battle Screen" src="https://github.com/user-attachments/assets/e3495c42-893e-4151-bf14-369cff5e9d85" />
<img width="1919" height="824" alt="Results" src="https://github.com/user-attachments/assets/25bdb45c-a6b4-4884-8df4-79fe517160b2" />

---

## How It Works

The system is a two-player adversarial game:

**Market Adversary** — an LSTM neural network that generates extreme market scenarios (flash crashes, volatility spikes, regime changes) designed to break your trading strategy. It gets smarter every round.

**Trading Agent** — a PPO reinforcement learning agent that learns to survive and adapt under adversarial conditions. Its policy is shaped by the user's chosen strategy type.

Both agents train simultaneously in a minimax loop. The adversary finds weaknesses, the trading agent adapts. After training, three portfolio curves are compared on the same price window:

- **Baseline** — naive agent on clean prices (standard backtest)
- **Attacked** — naive agent hit by adversarial event unprepared
- **Hardened** — adversarially trained agent facing the same event

---

## Features

- Dynamic portfolio builder — search any stock or asset by name or ticker, set share counts, live prices fetched automatically
- Four strategy types with reward shaping — Momentum, Mean Reversion, Trend Following, Pure RL
- Three risk profiles — Conservative, Moderate, Aggressive
- Four adversarial scenario types — Flash Crash, Volatility Spike, Regime Change, Full Adversarial
- Three training intensity levels — Light (8 rounds), Standard (20 rounds), Deep (40 rounds)
- Live battle feed — round by round score tracker, adversary vs your strategy
- Portfolio performance chart with adversarial event marker
- Before vs after metrics — Sharpe ratio, max drawdown, return, volatility, Calmar ratio
- Transaction costs — 0.1% turnover fee applied at each rebalancing step
- AI analysis via Groq LLaMA 3.3 70B — training summary, failure mode explanation, strategy recommendation
- Live Q&A — ask anything about your results in plain English
- 2020 COVID crash validation — test against real held-out black swan data the system never saw during training

```
blackswan-ai/
├── app.py                        # Streamlit UI entry point
├── assets/
│   └── style.css                 # custom dark theme CSS
├── src/
│   ├── data/
│   │   └── market_data.py        # yfinance data pipeline, feature engineering, ticker search
│   ├── environment/
│   │   └── trading_env.py        # custom Gymnasium trading environment
│   ├── agents/
│   │   └── adversary.py          # LSTM market adversary
│   ├── training/
│   │   └── trainer.py            # minimax training loop, intensity and risk presets
│   └── utils/
│       ├── metrics.py            # Sharpe, drawdown, Calmar, comparison table, failure modes
│       └── groq_summary.py       # Groq API integration for AI summaries and Q&A
└── tests/
```

## Setup

```bash
git clone https://github.com/Daksh2179/blackswan-ai.git
cd blackswan-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the root with your Groq API key:
GROQ_API_KEY=your_groq_api_key_here

Get a free Groq API key at [console.groq.com](https://console.groq.com). No credit card required.

---

## Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

**How to use:**
1. Build your portfolio — search any stock or asset by name or ticker, enter share counts
2. Choose your trading strategy — Momentum, Mean Reversion, Trend Following, or Pure RL
3. Set your risk profile and stress scenario
4. Choose training intensity and optionally enable the 2020 COVID crash validation
5. Hit Start Battle and watch the adversarial training unfold live
6. Review results — portfolio curves, metrics comparison, failure modes, and AI analysis
7. Ask questions about your results in plain English using the Q&A panel

---

## Tech Stack

| Component | Technology |
|---|---|
| RL Trading Agent | Stable-Baselines3 PPO |
| Market Adversary | PyTorch LSTM |
| Trading Environment | Custom Gymnasium |
| Data | yfinance + pandas |
| AI Summaries | Groq API — LLaMA 3.3 70B |
| UI | Streamlit + Plotly |
| Language | Python 3.12 |
