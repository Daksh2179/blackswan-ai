import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class MarketAdversary(nn.Module):
    """
    LSTM-based adversarial agent that generates worst-case market price sequences.

    Given a seed price sequence, the adversary outputs a perturbed version
    designed to maximise the trading agent's losses.

    The adversary is parameterised by scenario type:
        - flash_crash    : sudden sharp drop then partial recovery
        - volatility     : amplified noise with no directional bias
        - regime_change  : gradual trend reversal
        - adversarial    : fully learned, no constraints (hardest)
    """

    def __init__(
        self,
        n_assets: int,
        seq_len: int,
        hidden_size: int  = 64,
        num_layers: int   = 2,
        scenario: str     = "adversarial",
        lr: float         = 1e-3,
    ):
        super().__init__()

        self.n_assets   = n_assets
        self.seq_len    = seq_len
        self.scenario   = scenario

        # LSTM takes per-asset log returns as input
        self.lstm = nn.LSTM(
            input_size   = n_assets,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = 0.1 if num_layers > 1 else 0.0,
        )

        # output head: predict a perturbation for each asset at each timestep
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, n_assets),
            nn.Tanh(),   # perturbations bounded in [-1, 1]
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, seed_returns: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seed_returns: (batch, seq_len, n_assets) — historical log returns as seed
        Returns:
            perturbations: (batch, seq_len, n_assets) — additive perturbation to returns
        """
        lstm_out, _ = self.lstm(seed_returns)       # (batch, seq_len, hidden)
        perturbation = self.output_head(lstm_out)   # (batch, seq_len, n_assets)
        return perturbation

    def generate_scenario(
        self,
        seed_prices: np.ndarray,   # (seq_len, n_assets) real historical prices
        intensity: float = 1.0,    # how extreme the scenario is, 0.0 to 2.0
    ) -> np.ndarray:
        """
        Generate adversarial price sequence.
        Returns perturbed prices of shape (seq_len, n_assets).
        """
        seq_len, n_assets = seed_prices.shape

        # compute log returns from seed prices
        log_returns = np.zeros_like(seed_prices)
        log_returns[1:] = np.log(seed_prices[1:] / (seed_prices[:-1] + 1e-9))

        if self.scenario == "flash_crash":
            perturbed = self._flash_crash(seed_prices, intensity)
        elif self.scenario == "volatility":
            perturbed = self._volatility_spike(seed_prices, intensity)
        elif self.scenario == "regime_change":
            perturbed = self._regime_change(seed_prices, intensity)
        else:
            # fully adversarial: use LSTM to generate perturbation
            perturbed = self._lstm_scenario(seed_prices, log_returns, intensity)

        return perturbed.astype(np.float32)

    # scenario implementations

    def _flash_crash(self, prices: np.ndarray, intensity: float) -> np.ndarray:
        """
        Sharp drop starting early in the window so the agent feels the full impact.
        Crash begins at step 5, bottoms out by step 15, partial recovery after.
        """
        out = prices.copy().astype(float)
        crash_start = 5
        crash_end   = 15
        recovery    = min(crash_end + 20, len(prices))

        drop = 0.20 * intensity  # up to 20% drop at peak intensity
        for i in range(crash_start, crash_end):
            progress = (i - crash_start + 1) / (crash_end - crash_start)
            out[i]   = out[i] * (1 - drop * progress)
        for i in range(crash_end, recovery):
            # 40% bounce back only
            progress = (i - crash_end) / max(1, recovery - crash_end)
            out[i]   = out[i] * (1 + drop * 0.4 * progress)
        return out

    def _volatility_spike(self, prices: np.ndarray, intensity: float) -> np.ndarray:
        """Amplify noise without changing the overall trend."""
        out = prices.copy().astype(float)
        noise = np.random.normal(0, 0.02 * intensity, size=prices.shape)
        # apply cumulatively so prices stay connected
        out = out * np.exp(np.cumsum(noise, axis=0))
        return out

    def _regime_change(self, prices: np.ndarray, intensity: float) -> np.ndarray:
        """Reverse the trend gradually from the midpoint."""
        out = prices.copy().astype(float)
        mid = len(prices) // 2
        for i in range(mid, len(prices)):
            # introduce a downward drift proportional to intensity
            drift = 1 - (0.003 * intensity * (i - mid))
            out[i] = out[i] * max(drift, 0.5)  # floor at 50% of original
        return out

    def _lstm_scenario(
        self,
        prices: np.ndarray,
        log_returns: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Use the trained LSTM to generate a learned perturbation."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(log_returns).unsqueeze(0)  # (1, seq_len, n_assets)
            perturbation = self.forward(x).squeeze(0).numpy()  # (seq_len, n_assets)

        # scale perturbation by intensity and apply to prices
        scaled = perturbation * intensity * 0.05
        perturbed_returns = log_returns + scaled
        # reconstruct prices from perturbed returns
        out = np.zeros_like(prices)
        out[0] = prices[0]
        for i in range(1, len(prices)):
            out[i] = out[i - 1] * np.exp(perturbed_returns[i])
        return out

    def compute_adversarial_loss(self, trading_agent_reward: torch.Tensor) -> torch.Tensor:
        """
        Adversary wants to minimise the trading agent's reward (zero-sum).
        """
        return trading_agent_reward  # adversary maximises agent's loss = minimise reward

    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()


# sanity check
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.market_data import build_portfolio_data

    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}
    feats, w = build_portfolio_data(tickers, weights, split="train")

    # sample a 60-day window matching the adversary window size
    close_aapl  = feats["AAPL"]["close"].values[:60]
    close_spy   = feats["SPY"]["close"].values[:60]
    seed_prices = np.stack([close_aapl, close_spy], axis=1)  # (60, 2)

    print(f"Seed prices shape : {seed_prices.shape}")
    print(f"Seed AAPL range   : ${seed_prices[:,0].min():.2f} - ${seed_prices[:,0].max():.2f}")
    print()

    adversary = MarketAdversary(n_assets=2, seq_len=60)

    for scenario in ["flash_crash", "volatility", "regime_change", "adversarial"]:
        adversary.scenario = scenario
        perturbed  = adversary.generate_scenario(seed_prices, intensity=1.0)
        pct_change = (perturbed[-1] / perturbed[0] - 1) * 100
        worst_day  = ((perturbed[1:] - perturbed[:-1]) / (perturbed[:-1] + 1e-9)).min(axis=0) * 100
        print(f"{scenario:<15s} | final AAPL: ${perturbed[-1,0]:.2f} | total: {pct_change[0]:+.1f}% | worst single day: {worst_day[0]:+.2f}%")