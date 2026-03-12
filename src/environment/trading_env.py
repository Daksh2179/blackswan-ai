import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple


class TradingEnvironment(gym.Env):
    """
    A Gymnasium-compatible trading environment for a multi-asset portfolio.

    Observation:
        - For each asset: [return, vol_5d, vol_20d, sma10_ratio, sma30_ratio, rsi14, volume_norm]
        - Current portfolio weights (one per asset)
        - Current drawdown
        Total obs dim = n_assets * 7 + n_assets + 1

    Actions:
        Continuous vector of length n_assets in [-1, 1].
        Softmax applied so weights always sum to 1 (long-only).

    Reward:
        Step return in percent minus drawdown penalty.
        Episode is exactly episode_len steps so the adversarial window
        (injected in the second half) is always significant.
    """

    metadata = {"render_modes": []}

    FEATURE_COLS = ["return", "vol_5d", "vol_20d", "sma10_ratio", "sma30_ratio", "rsi14", "volume_norm"]

    def __init__(
        self,
        features_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        initial_capital: float       = 100_000.0,
        max_drawdown_tol: float      = 0.20,
        position_sizing: str         = "moderate",
        trading_horizon: str         = "daily",
        episode_len: int             = 120,
        adversarial_prices: Optional[np.ndarray] = None,
        adversarial_start: Optional[int]         = None,
    ):
        super().__init__()

        self.tickers          = list(features_dict.keys())
        self.n_assets         = len(self.tickers)
        self.features_dict    = features_dict
        self.initial_capital  = initial_capital
        self.max_drawdown_tol = max_drawdown_tol
        self.trading_horizon  = trading_horizon
        self.episode_len      = episode_len
        self.size_scalar      = {"conservative": 0.5, "moderate": 1.0, "aggressive": 1.5}[position_sizing]

        self.dates, self.feature_array, self.close_array = self._align_data()
        self.n_total_steps = len(self.dates)
        self.stride        = 5 if trading_horizon == "weekly" else 1

        self.adversarial_prices = adversarial_prices
        self.adversarial_start  = adversarial_start

        # episode start index — randomised each reset
        self.episode_start = 0

        obs_dim = self.n_assets * len(self.FEATURE_COLS) + self.n_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        total = sum(weights.values())
        self.init_weights = np.array([weights[t] / total for t in self.tickers], dtype=np.float32)

        self._reset_state()

    # internal helpers

    def _align_data(self):
        common_idx = None
        for df in self.features_dict.values():
            common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
        feat_list, close_list = [], []
        for t in self.tickers:
            df = self.features_dict[t].loc[common_idx]
            feat_list.append(df[self.FEATURE_COLS].values)
            close_list.append(df["close"].values)
        feature_array = np.stack(feat_list, axis=1).astype(np.float32)
        close_array   = np.stack(close_list, axis=1).astype(np.float32)
        return common_idx, feature_array, close_array

    def _reset_state(self):
        # pick a random starting point so each episode is a different 120-day window
        max_start          = self.n_total_steps - self.episode_len - 1
        self.episode_start = np.random.randint(0, max(1, max_start))
        self.current_step  = 0           # relative to episode_start
        self.portfolio_val = self.initial_capital
        self.peak_val      = self.initial_capital
        self.weights       = self.init_weights.copy()
        self.history       = []
        self.done          = False

    def _abs_step(self) -> int:
        """Absolute index into the full price array."""
        return self.episode_start + self.current_step

    def _get_obs(self) -> np.ndarray:
        abs_idx      = min(self._abs_step(), self.n_total_steps - 1)
        mkt_features = self.feature_array[abs_idx].flatten()
        drawdown     = (self.peak_val - self.portfolio_val) / (self.peak_val + 1e-9)
        return np.concatenate([mkt_features, self.weights, [drawdown]]).astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        scaled = action * self.size_scalar
        exp    = np.exp(scaled - scaled.max())
        return (exp / exp.sum()).astype(np.float32)

    def _get_returns(self) -> np.ndarray:
        """
        Returns for the current step.
        If the current relative step falls inside the adversarial window, use
        adversarial prices. Otherwise use real historical prices.
        """
        adv   = self.adversarial_prices
        a_start = self.adversarial_start   # relative step index

        if adv is not None and a_start is not None:
            adv_len = len(adv)
            rel     = self.current_step
            if a_start < rel <= a_start + adv_len:
                i      = rel - a_start
                p_now  = adv[min(i, adv_len - 1)]
                p_prev = adv[max(i - 1, 0)]
                return (p_now - p_prev) / (p_prev + 1e-9)

        abs_now  = min(self._abs_step(), self.n_total_steps - 1)
        abs_prev = max(abs_now - 1, 0)
        return (self.close_array[abs_now] - self.close_array[abs_prev]) / (self.close_array[abs_prev] + 1e-9)

    # gym interface

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        self.weights  = self._apply_action(action)
        asset_returns = self._get_returns()

        port_return        = float(np.dot(self.weights, asset_returns))
        self.portfolio_val *= (1 + port_return)
        self.peak_val       = max(self.peak_val, self.portfolio_val)
        drawdown            = (self.peak_val - self.portfolio_val) / (self.peak_val + 1e-9)

        self.history.append(self.portfolio_val)
        self.current_step += self.stride

        reward     = port_return * 100
        dd_penalty = max(0.0, drawdown - self.max_drawdown_tol) * 200
        reward    -= dd_penalty

        terminated = (self.current_step >= self.episode_len) or (drawdown > 0.99)
        truncated  = False
        self.done  = terminated

        info = {
            "portfolio_value": self.portfolio_val,
            "drawdown":        drawdown,
            "port_return":     port_return,
            "weights":         self.weights.tolist(),
            "step":            self.current_step,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def inject_adversarial_prices(self, prices: Optional[np.ndarray], start_step: Optional[int] = None):
        """Inject adversarial price window. start_step is relative to episode start."""
        self.adversarial_prices = prices
        self.adversarial_start  = start_step

    def get_portfolio_history(self) -> List[float]:
        return self.history.copy()


# sanity check
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.market_data import build_portfolio_data

    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}
    feats, w = build_portfolio_data(tickers, weights, split="train")

    env = TradingEnvironment(
        features_dict    = feats,
        weights          = w,
        initial_capital  = 100_000,
        max_drawdown_tol = 0.20,
        position_sizing  = "moderate",
        trading_horizon  = "daily",
        episode_len      = 120,
    )

    obs, _ = env.reset()
    print(f"Obs shape    : {obs.shape}")
    print(f"Episode len  : {env.episode_len} days")
    print(f"Episode start: day {env.episode_start} of {env.n_total_steps}")

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    start = env.initial_capital
    end   = info["portfolio_value"]
    print(f"Start value  : ${start:,.2f}")
    print(f"End value    : ${end:,.2f}  ({(end/start - 1)*100:+.2f}%)")
    print(f"Max drawdown : {info['drawdown']:.2%}")