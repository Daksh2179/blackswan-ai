import numpy as np
import torch
from stable_baselines3 import PPO
from typing import Dict, List, Optional
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.trading_env import TradingEnvironment
from src.agents.adversary import MarketAdversary


# training intensity presets
INTENSITY_PRESETS = {
    "light":    {"n_rounds": 8,  "agent_steps": 256},
    "standard": {"n_rounds": 20, "agent_steps": 512},
    "deep":     {"n_rounds": 40, "agent_steps": 1024},
}

# risk profile presets — combines position sizing and max drawdown tolerance
RISK_PRESETS = {
    "conservative": {"position_sizing": "conservative", "max_drawdown_tol": 0.10},
    "moderate":     {"position_sizing": "moderate",     "max_drawdown_tol": 0.20},
    "aggressive":   {"position_sizing": "aggressive",   "max_drawdown_tol": 0.35},
}


class MinimaxTrainer:
    """
    Alternating minimax training loop.

    Each episode is exactly 120 trading days.
    The adversarial window (60 days) is injected at the midpoint so the agent
    always experiences the stress event and must respond.
    """

    SCENARIO_ROTATION = ["flash_crash", "volatility", "regime_change", "adversarial"]

    def __init__(
        self,
        features_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        initial_capital: float  = 100_000.0,
        scenario: str           = "adversarial",
        strategy: str           = "pure_rl",
        risk_profile: str       = "moderate",
        trading_horizon: str    = "daily",
        intensity: str          = "light",
        episode_len: int        = 120,
        window_size: int        = 60,
        adv_intensity: float    = 1.5,
    ):
        # store all params first before anything else is instantiated
        self.features_dict   = features_dict
        self.weights         = weights
        self.initial_capital = initial_capital
        self.scenario        = scenario
        self.strategy        = strategy
        self.trading_horizon = trading_horizon
        self.adv_intensity   = adv_intensity
        self.episode_len     = episode_len
        self.window_size     = window_size
        self.tickers         = list(features_dict.keys())
        self.n_assets        = len(self.tickers)
        self.adv_inject_step = episode_len // 2

        # unpack risk profile
        risk = RISK_PRESETS[risk_profile]
        self.position_sizing  = risk["position_sizing"]
        self.max_drawdown_tol = risk["max_drawdown_tol"]

        # unpack intensity
        preset = INTENSITY_PRESETS[intensity]
        self.n_rounds    = preset["n_rounds"]
        self.agent_steps = preset["agent_steps"]

        self.close_array, self.dates = self._get_close_array()
        self.n_total_steps = len(self.dates)

        self.env = TradingEnvironment(
            features_dict    = features_dict,
            weights          = weights,
            initial_capital  = initial_capital,
            max_drawdown_tol = self.max_drawdown_tol,
            position_sizing  = self.position_sizing,
            trading_horizon  = trading_horizon,
            episode_len      = episode_len,
            strategy         = self.strategy,
        )

        self.agent = PPO(
            "MlpPolicy",
            self.env,
            verbose       = 0,
            learning_rate = 3e-4,
            n_steps       = min(self.agent_steps, 128),
            batch_size    = 64,
            n_epochs      = 5,
        )

        self.adversary = MarketAdversary(
            n_assets    = self.n_assets,
            seq_len     = window_size,
            scenario    = scenario,
            hidden_size = 64,
            num_layers  = 2,
        )

        self.history = {
            "round":           [],
            "scenario_label":  [],
            "agent_reward":    [],
            "adversary_loss":  [],
            "portfolio_value": [],
            "drawdown":        [],
            "scenario_drop":   [],
        }

    def _get_close_array(self):
        common_idx = None
        for df in self.features_dict.values():
            common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
        closes = np.stack(
            [self.features_dict[t].loc[common_idx, "close"].values for t in self.tickers],
            axis=1
        ).astype(np.float32)
        return closes, common_idx

    def _sample_seed_prices(self) -> np.ndarray:
        max_start = self.n_total_steps - self.window_size - 1
        start     = np.random.randint(0, max(1, max_start))
        return self.close_array[start: start + self.window_size]

    def _run_eval_episode(self, adv_prices: np.ndarray) -> tuple:
        self.env.inject_adversarial_prices(adv_prices, start_step=self.adv_inject_step)
        obs, _ = self.env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = self.agent.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward, info, self.env.get_portfolio_history()

    def _compute_metrics(self, portfolio_history: List[float]) -> dict:
        vals    = np.array(portfolio_history)
        returns = np.diff(vals) / (vals[:-1] + 1e-9)
        sharpe  = (returns.mean() / (max(returns.std(), 1e-4))) * np.sqrt(252)
        peak    = np.maximum.accumulate(vals)
        max_dd  = float(((peak - vals) / (peak + 1e-9)).max())
        return {
            "sharpe":           round(float(sharpe), 3),
            "max_drawdown":     round(max_dd, 4),
            "final_return_pct": round(float((vals[-1] / vals[0] - 1) * 100), 2),
        }

    def train(self, progress_callback=None):
        print(f"Starting minimax training | {self.n_rounds} rounds | scenario={self.scenario} | strategy={self.strategy}")
        print(f"Assets: {self.tickers} | Episode: {self.episode_len}d | Adv window: {self.window_size}d injected @ step {self.adv_inject_step}")
        print(f"Risk: {self.position_sizing} | Max DD tolerance: {self.max_drawdown_tol:.0%} | Intensity: {self.n_rounds} rounds x {self.agent_steps} steps\n")

        for round_idx in range(self.n_rounds):
            if self.scenario == "adversarial":
                current_scenario = self.SCENARIO_ROTATION[round_idx % len(self.SCENARIO_ROTATION)]
                self.adversary.scenario = current_scenario
            else:
                current_scenario = self.scenario

            seed_prices = self._sample_seed_prices()
            adv_prices  = self.adversary.generate_scenario(seed_prices, intensity=self.adv_intensity)

            self.env.inject_adversarial_prices(adv_prices, start_step=self.adv_inject_step)
            self.env.reset()
            self.agent.learn(total_timesteps=self.agent_steps, reset_num_timesteps=False)

            agent_reward, info, port_history = self._run_eval_episode(adv_prices)
            metrics = self._compute_metrics(port_history)

            adv_loss_val = 0.0
            if current_scenario == "adversarial":
                seed_t = torch.FloatTensor(
                    np.log(seed_prices[1:] / (seed_prices[:-1] + 1e-9) + 1e-9)
                )
                seed_t       = torch.cat([seed_t, seed_t[-1:]], dim=0).unsqueeze(0)
                perturbation = self.adversary(seed_t)
                adv_loss     = -torch.sign(torch.tensor(agent_reward)) * perturbation.abs().mean()
                self.adversary.update(adv_loss)
                adv_loss_val = float(adv_loss.item())

            scenario_drop = float((adv_prices[-1].mean() / adv_prices[0].mean() - 1) * 100)
            end_val       = info["portfolio_value"]
            pnl_pct       = (end_val / self.initial_capital - 1) * 100

            self.history["round"].append(round_idx + 1)
            self.history["scenario_label"].append(current_scenario)
            self.history["agent_reward"].append(agent_reward)
            self.history["adversary_loss"].append(adv_loss_val)
            self.history["portfolio_value"].append(end_val)
            self.history["drawdown"].append(metrics["max_drawdown"])
            self.history["scenario_drop"].append(scenario_drop)

            print(
                f"Round {round_idx+1:3d}/{self.n_rounds} [{current_scenario:<14s}] | "
                f"Return: {pnl_pct:+6.1f}% | "
                f"Max DD: {metrics['max_drawdown']:.2%} | "
                f"Sharpe: {metrics['sharpe']:+.3f} | "
                f"Scenario drop: {scenario_drop:+.1f}%"
            )

            if progress_callback:
                progress_callback(round_idx + 1, self.n_rounds, {
                    "scenario":        current_scenario,
                    "agent_reward":    agent_reward,
                    "portfolio_value": end_val,
                    "drawdown":        metrics["max_drawdown"],
                    "sharpe":          metrics["sharpe"],
                    "scenario_drop":   scenario_drop,
                })

        print("\nTraining complete.")
        return self.history

    def get_three_curves(self) -> Dict:
        """
        Generate three portfolio value curves on the same fixed price window.

        baseline : naive agent on clean prices
        attacked : naive agent under adversarial attack
        hardened : adversarially trained agent under the same attack
        """
        seed_prices = self._sample_seed_prices()

        # force scenario to be harmful
        for _ in range(10):
            adv_prices = self.adversary.generate_scenario(seed_prices, intensity=self.adv_intensity)
            if (adv_prices[-1].mean() / adv_prices[0].mean()) - 1 < -0.03:
                break
        else:
            adv_prices = self.adversary._flash_crash(seed_prices, intensity=self.adv_intensity)

        max_start   = self.n_total_steps - self.episode_len - 1
        fixed_start = int(np.random.randint(0, max(1, max_start)))

        naive_env = TradingEnvironment(
            features_dict    = self.features_dict,
            weights          = self.weights,
            initial_capital  = self.initial_capital,
            max_drawdown_tol = self.max_drawdown_tol,
            position_sizing  = self.position_sizing,
            trading_horizon  = self.trading_horizon,
            episode_len      = self.episode_len,
            strategy         = self.strategy,
        )
        naive_agent = PPO(
            "MlpPolicy", naive_env, verbose=0,
            learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=5,
        )
        naive_agent.learn(total_timesteps=self.n_rounds * self.agent_steps)

        def run_curve(agent, env, inject):
            env.set_episode_start(fixed_start)
            if inject:
                env.inject_adversarial_prices(adv_prices, start_step=self.adv_inject_step)
            else:
                env.inject_adversarial_prices(None, start_step=None)
            obs, _ = env.reset()
            vals = [self.initial_capital]
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                vals.append(info["portfolio_value"])
                done = terminated or truncated
            return vals, env.get_portfolio_history()

        baseline_vals, baseline_hist = run_curve(naive_agent, naive_env, inject=False)
        attacked_vals, attacked_hist = run_curve(naive_agent, naive_env, inject=True)
        hardened_vals, hardened_hist = run_curve(self.agent, self.env, inject=True)

        return {
            "baseline":         baseline_vals,
            "attacked":         attacked_vals,
            "hardened":         hardened_vals,
            "baseline_metrics": self._compute_metrics(baseline_hist),
            "attacked_metrics": self._compute_metrics(attacked_hist),
            "hardened_metrics": self._compute_metrics(hardened_hist),
            "adv_prices":       adv_prices.tolist(),
            "seed_prices":      seed_prices.tolist(),
            "inj_step":         self.adv_inject_step,
        }


# sanity check
if __name__ == "__main__":
    from src.data.market_data import build_portfolio_data

    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}
    feats, w = build_portfolio_data(tickers, weights, split="train")

    trainer = MinimaxTrainer(
        features_dict   = feats,
        weights         = w,
        scenario        = "adversarial",
        strategy        = "momentum",
        risk_profile    = "moderate",
        trading_horizon = "daily",
        intensity       = "light",
    )

    print(f"Risk profile  : moderate → position_sizing={trainer.position_sizing}, max_dd={trainer.max_drawdown_tol:.0%}")
    print(f"Intensity     : light → n_rounds={trainer.n_rounds}, agent_steps={trainer.agent_steps}")
    print(f"Strategy      : {trainer.strategy}\n")

    history = trainer.train()

    print("\nGenerating three curves...")
    curves = trainer.get_three_curves()
    print(f"\nBaseline : final=${curves['baseline'][-1]:,.2f}  metrics={curves['baseline_metrics']}")
    print(f"Attacked : final=${curves['attacked'][-1]:,.2f}  metrics={curves['attacked_metrics']}")
    print(f"Hardened : final=${curves['hardened'][-1]:,.2f}  metrics={curves['hardened_metrics']}")