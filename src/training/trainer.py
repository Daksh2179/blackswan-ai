import numpy as np
import torch
from stable_baselines3 import PPO
from typing import Dict, List, Optional
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.environment.trading_env import TradingEnvironment
from src.agents.adversary import MarketAdversary


class MinimaxTrainer:
    """
    Alternating minimax training loop.

    Each episode is exactly 120 trading days (6 months).
    The adversarial window (60 days) is injected in the second half of the
    episode so the agent always experiences the stress event and must respond.

    Each round:
      1. Sample a random 120-day episode window from training data.
      2. Adversary generates a 60-day stress scenario.
      3. Inject it at step 60 (exactly the second half of the episode).
      4. PPO trains for agent_steps inside this episode.
      5. Evaluate and update the adversary.
    """

    SCENARIO_ROTATION = ["flash_crash", "volatility", "regime_change", "adversarial"]

    def __init__(
        self,
        features_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        scenario: str            = "adversarial",
        position_sizing: str     = "moderate",
        trading_horizon: str     = "daily",
        max_drawdown_tol: float  = 0.20,
        initial_capital: float   = 100_000.0,
        episode_len: int         = 120,
        window_size: int         = 60,
        n_rounds: int            = 20,
        agent_steps: int         = 512,
        adv_intensity: float     = 1.0,
    ):
        self.features_dict    = features_dict
        self.weights          = weights
        self.scenario         = scenario
        self.position_sizing  = position_sizing
        self.trading_horizon  = trading_horizon
        self.max_drawdown_tol = max_drawdown_tol
        self.initial_capital  = initial_capital
        self.episode_len      = episode_len
        self.window_size      = window_size
        self.n_rounds         = n_rounds
        self.agent_steps      = agent_steps
        self.adv_intensity    = adv_intensity

        # adversary always injected at the halfway point of the episode
        self.adv_inject_step  = episode_len // 2

        self.tickers  = list(features_dict.keys())
        self.n_assets = len(self.tickers)

        self.close_array, self.dates = self._get_close_array()
        self.n_total_steps = len(self.dates)

        self.env = TradingEnvironment(
            features_dict    = features_dict,
            weights          = weights,
            initial_capital  = initial_capital,
            max_drawdown_tol = max_drawdown_tol,
            position_sizing  = position_sizing,
            trading_horizon  = trading_horizon,
            episode_len      = episode_len,
        )

        self.agent = PPO(
            "MlpPolicy",
            self.env,
            verbose       = 0,
            learning_rate = 3e-4,
            n_steps       = min(agent_steps, 128),
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
        """Sample a 60-day price window to seed the adversary."""
        max_start = self.n_total_steps - self.window_size - 1
        start     = np.random.randint(0, max_start)
        return self.close_array[start: start + self.window_size]

    def _run_eval_episode(self, adv_prices: np.ndarray) -> tuple:
        """Evaluate current agent for one full episode. No learning."""
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
        """Compute Sharpe ratio, max drawdown, and final return from a value curve."""
        vals    = np.array(portfolio_history)
        returns = np.diff(vals) / (vals[:-1] + 1e-9)

        sharpe      = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        peak        = np.maximum.accumulate(vals)
        drawdowns   = (peak - vals) / (peak + 1e-9)
        max_dd      = float(drawdowns.max())
        final_ret   = float((vals[-1] / vals[0] - 1) * 100)

        return {"sharpe": float(sharpe), "max_drawdown": max_dd, "final_return_pct": final_ret}

    def train(self, progress_callback=None):
        print(f"Starting minimax training | {self.n_rounds} rounds | scenario={self.scenario}")
        print(f"Assets: {self.tickers} | Episode: {self.episode_len}d | Adv window: {self.window_size}d injected @ step {self.adv_inject_step}")
        print(f"Agent steps/round: {self.agent_steps}\n")

        for round_idx in range(self.n_rounds):

            if self.scenario == "adversarial":
                current_scenario = self.SCENARIO_ROTATION[round_idx % len(self.SCENARIO_ROTATION)]
                self.adversary.scenario = current_scenario
            else:
                current_scenario = self.scenario

            # 1. sample seed prices and generate adversarial scenario
            seed_prices = self._sample_seed_prices()
            adv_prices  = self.adversary.generate_scenario(seed_prices, intensity=self.adv_intensity)

            # 2. train agent inside adversarial episode
            self.env.inject_adversarial_prices(adv_prices, start_step=self.adv_inject_step)
            self.env.reset()
            self.agent.learn(total_timesteps=self.agent_steps, reset_num_timesteps=False)

            # 3. evaluate
            agent_reward, info, port_history = self._run_eval_episode(adv_prices)
            metrics = self._compute_metrics(port_history)

            # 4. update adversary on adversarial rounds
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
        Generate three portfolio value curves for the final output plot.

        All three curves run on the SAME fixed 120-day price window and the
        SAME adversarial injection so results are directly comparable.

        baseline : naive agent on clean prices (normal backtest, no stress)
        attacked : naive agent hit by adversarial event unprepared
        hardened : adversarially trained agent faces the same event
        """
        seed_prices = self._sample_seed_prices()

        # fix 2: force adversarial scenario to always be harmful (net negative)
        # regenerate until we get a scenario that actually drops prices
        for _ in range(10):
            adv_prices = self.adversary.generate_scenario(seed_prices, intensity=self.adv_intensity)
            scenario_return = (adv_prices[-1].mean() / adv_prices[0].mean()) - 1
            if scenario_return < -0.03:   # must be at least 3% net negative
                break
        else:
            # fallback: force a flash crash manually if LSTM keeps generating up moves
            adv_prices = self.adversary._flash_crash(seed_prices, intensity=self.adv_intensity)

        # fix 1: pin the episode start so all three curves use the same price window
        max_start   = self.n_total_steps - self.episode_len - 1
        fixed_start = int(np.random.randint(0, max(1, max_start)))

        # fix 3: use a fixed seed for the naive agent so evaluation is deterministic
        naive_env = TradingEnvironment(
            features_dict    = self.features_dict,
            weights          = self.weights,
            initial_capital  = self.initial_capital,
            max_drawdown_tol = self.max_drawdown_tol,
            position_sizing  = self.position_sizing,
            trading_horizon  = self.trading_horizon,
            episode_len      = self.episode_len,
        )
        naive_agent = PPO(
            "MlpPolicy", naive_env, verbose=0,
            learning_rate=3e-4, n_steps=128, batch_size=64, n_epochs=5,
        )
        naive_agent.learn(total_timesteps=self.n_rounds * self.agent_steps)

        def run_curve(agent, env, inject, seed=42):
            if inject:
                env.inject_adversarial_prices(adv_prices, start_step=self.adv_inject_step)
            else:
                env.inject_adversarial_prices(None, start_step=None)
            # fix 1: force same episode start for all three curves
            obs, _ = env.reset(seed=seed)
            env.episode_start = fixed_start   # override random start
            env.current_step  = 0
            obs = env._get_obs()
            vals = [self.initial_capital]
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                vals.append(info["portfolio_value"])
                done = terminated or truncated
            return vals, env.get_portfolio_history()

        # baseline: naive agent, clean prices, fixed window
        baseline_vals, baseline_hist = run_curve(naive_agent, naive_env, inject=False)

        # attacked: naive agent, adversarial prices, same fixed window
        attacked_vals, attacked_hist = run_curve(naive_agent, naive_env, inject=True)

        # hardened: trained agent, adversarial prices, same fixed window
        hardened_vals, hardened_hist = run_curve(self.agent, self.env, inject=True)

        return {
            "baseline":       baseline_vals,
            "attacked":       attacked_vals,
            "hardened":       hardened_vals,
            "baseline_metrics": self._compute_metrics(baseline_hist),
            "attacked_metrics": self._compute_metrics(attacked_hist),
            "hardened_metrics": self._compute_metrics(hardened_hist),
            "adv_prices":     adv_prices.tolist(),
            "seed_prices":    seed_prices.tolist(),
            "inj_step":       self.adv_inject_step,
        }


# sanity check
if __name__ == "__main__":
    from src.data.market_data import build_portfolio_data

    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}
    feats, w = build_portfolio_data(tickers, weights, split="train")

    trainer = MinimaxTrainer(
        features_dict    = feats,
        weights          = w,
        scenario         = "adversarial",
        position_sizing  = "moderate",
        trading_horizon  = "daily",
        max_drawdown_tol = 0.20,
        episode_len      = 120,
        window_size      = 60,
        n_rounds         = 8,
        agent_steps      = 256,
    )

    history = trainer.train()

    print("\nGenerating three curves for output plot...")
    curves = trainer.get_three_curves()
    print(f"\nBaseline  : final=${curves['baseline'][-1]:,.2f}  metrics={curves['baseline_metrics']}")
    print(f"Attacked  : final=${curves['attacked'][-1]:,.2f}  metrics={curves['attacked_metrics']}")
    print(f"Hardened  : final=${curves['hardened'][-1]:,.2f}  metrics={curves['hardened_metrics']}")