import numpy as np
from typing import List, Dict


def compute_metrics(portfolio_history: List[float], risk_free_rate: float = 0.02) -> Dict:
    """
    Compute full performance metrics from a portfolio value curve.

    Args:
        portfolio_history : list of portfolio values at each step
        risk_free_rate    : annualised risk-free rate (default 2%)

    Returns dict with:
        final_return_pct  : total return over the episode in percent
        sharpe            : annualised Sharpe ratio
        max_drawdown      : maximum drawdown as a fraction (0.15 = 15%)
        avg_daily_return  : mean daily return in percent
        volatility        : annualised volatility of daily returns
        calmar            : return / max_drawdown (reward per unit of tail risk)
    """
    vals = np.array(portfolio_history, dtype=float)
    if len(vals) < 2:
        return _empty_metrics()

    returns = np.diff(vals) / (vals[:-1] + 1e-9)

    final_return  = float((vals[-1] / vals[0] - 1) * 100)
    avg_ret       = float(returns.mean())
    vol           = float(returns.std())
    daily_rf      = risk_free_rate / 252
    sharpe        = float(((avg_ret - daily_rf) / (max(vol, 1e-4))) * np.sqrt(252))
    max_dd        = float(_max_drawdown(vals))
    calmar        = float((avg_ret * 252) / (max(max_dd, 1e-4)))
    annualised_vol = float(vol * np.sqrt(252) * 100)

    return {
        "final_return_pct": round(final_return, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown":     round(max_dd, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_daily_return": round(avg_ret * 100, 4),
        "volatility_ann":   round(annualised_vol, 2),
        "calmar":           round(calmar, 3),
    }


def _max_drawdown(vals: np.ndarray) -> float:
    peak      = np.maximum.accumulate(vals)
    drawdowns = (peak - vals) / (peak + 1e-9)
    return float(drawdowns.max())


def _empty_metrics() -> Dict:
    return {
        "final_return_pct": 0.0,
        "sharpe":           0.0,
        "max_drawdown":     0.0,
        "max_drawdown_pct": 0.0,
        "avg_daily_return": 0.0,
        "volatility_ann":   0.0,
        "calmar":           0.0,
    }


def build_comparison_table(
    baseline_history: List[float],
    attacked_history: List[float],
    hardened_history: List[float],
) -> List[Dict]:
    """
    Build the before/after comparison table for the UI.
    Returns a list of row dicts ready for display.
    """
    b = compute_metrics(baseline_history)
    a = compute_metrics(attacked_history)
    h = compute_metrics(hardened_history)

    def delta(hardened_val, attacked_val, higher_is_better=True):
        diff = hardened_val - attacked_val
        if higher_is_better:
            tag = "improved" if diff > 0 else "worse"
        else:
            tag = "improved" if diff < 0 else "worse"
        return round(diff, 3), tag

    rows = []

    ret_delta, ret_tag = delta(h["final_return_pct"], a["final_return_pct"])
    rows.append({
        "metric":   "Total Return (%)",
        "baseline": f"{b['final_return_pct']:+.2f}%",
        "attacked": f"{a['final_return_pct']:+.2f}%",
        "hardened": f"{h['final_return_pct']:+.2f}%",
        "delta":    f"{ret_delta:+.2f}%",
        "status":   ret_tag,
    })

    sh_delta, sh_tag = delta(h["sharpe"], a["sharpe"])
    rows.append({
        "metric":   "Sharpe Ratio",
        "baseline": f"{b['sharpe']:+.3f}",
        "attacked": f"{a['sharpe']:+.3f}",
        "hardened": f"{h['sharpe']:+.3f}",
        "delta":    f"{sh_delta:+.3f}",
        "status":   sh_tag,
    })

    dd_delta, dd_tag = delta(h["max_drawdown_pct"], a["max_drawdown_pct"], higher_is_better=False)
    rows.append({
        "metric":   "Max Drawdown (%)",
        "baseline": f"{b['max_drawdown_pct']:.2f}%",
        "attacked": f"{a['max_drawdown_pct']:.2f}%",
        "hardened": f"{h['max_drawdown_pct']:.2f}%",
        "delta":    f"{dd_delta:+.2f}%",
        "status":   dd_tag,
    })

    vol_delta, vol_tag = delta(h["volatility_ann"], a["volatility_ann"], higher_is_better=False)
    rows.append({
        "metric":   "Annualised Volatility (%)",
        "baseline": f"{b['volatility_ann']:.2f}%",
        "attacked": f"{a['volatility_ann']:.2f}%",
        "hardened": f"{h['volatility_ann']:.2f}%",
        "delta":    f"{vol_delta:+.2f}%",
        "status":   vol_tag,
    })

    cal_delta, cal_tag = delta(h["calmar"], a["calmar"])
    rows.append({
        "metric":   "Calmar Ratio",
        "baseline": f"{b['calmar']:+.3f}",
        "attacked": f"{a['calmar']:+.3f}",
        "hardened": f"{h['calmar']:+.3f}",
        "delta":    f"{cal_delta:+.3f}",
        "status":   cal_tag,
    })

    return rows


def identify_failure_modes(
    scenario_history: List[str],
    drawdown_history: List[float],
    reward_history:   List[float],
) -> List[Dict]:
    """
    Analyse training history to identify which scenario types caused the most damage.
    Returns a ranked list of failure modes for display.
    """
    scenario_stats: Dict[str, Dict] = {}

    for scenario, dd, reward in zip(scenario_history, drawdown_history, reward_history):
        if scenario not in scenario_stats:
            scenario_stats[scenario] = {"drawdowns": [], "rewards": []}
        scenario_stats[scenario]["drawdowns"].append(dd)
        scenario_stats[scenario]["rewards"].append(reward)

    failure_modes = []
    for scenario, stats in scenario_stats.items():
        avg_dd     = np.mean(stats["drawdowns"])
        avg_reward = np.mean(stats["rewards"])
        worst_dd   = np.max(stats["drawdowns"])
        failure_modes.append({
            "scenario":    scenario,
            "avg_drawdown_pct": round(avg_dd * 100, 2),
            "worst_drawdown_pct": round(worst_dd * 100, 2),
            "avg_reward":  round(avg_reward, 3),
            "n_rounds":    len(stats["drawdowns"]),
            "threat_level": "HIGH" if avg_dd > 0.20 else "MEDIUM" if avg_dd > 0.10 else "LOW",
        })

    failure_modes.sort(key=lambda x: x["avg_drawdown_pct"], reverse=True)
    return failure_modes


# sanity check
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.training.trainer import MinimaxTrainer
    from src.data.market_data import build_portfolio_data

    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}
    feats, w = build_portfolio_data(tickers, weights, split="train")

    trainer = MinimaxTrainer(
        features_dict = feats,
        weights       = w,
        n_rounds      = 12,
        agent_steps   = 256,
        adv_intensity = 1.5,
    )
    trainer.train()
    curves = trainer.get_three_curves()

    baseline = curves["baseline"]
    attacked = curves["attacked"]
    hardened = curves["hardened"]
    initial_capital = trainer.initial_capital

    print("=== Metrics ===")
    for label, hist in [("Baseline", baseline), ("Attacked", attacked), ("Hardened", hardened)]:
        m = compute_metrics(hist)
        print(f"{label:10s}: return={m['final_return_pct']:+.2f}%  sharpe={m['sharpe']:+.3f}  maxDD={m['max_drawdown_pct']:.2f}%")

    print("\n=== Comparison Table ===")
    table = build_comparison_table(baseline, attacked, hardened)
    for row in table:
        print(f"{row['metric']:<28s} | baseline={row['baseline']:>8s} | attacked={row['attacked']:>8s} | hardened={row['hardened']:>8s} | delta={row['delta']:>8s} | {row['status']}")

    print("\n=== Failure Modes ===")
    modes = identify_failure_modes(
        trainer.history["scenario_label"],
        trainer.history["drawdown"],
        trainer.history["agent_reward"],
    )
    for m in modes:
        print(f"{m['scenario']:<15s} | avg DD={m['avg_drawdown_pct']:.1f}% | worst DD={m['worst_drawdown_pct']:.1f}% | threat={m['threat_level']}")