import os
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


def _call_groq(prompt: str, max_tokens: int = 400) -> str:
    """Base function to call Groq API and return response text."""
    try:
        response = client.chat.completions.create(
            model    = MODEL,
            messages = [{"role": "user", "content": prompt}],
            max_tokens = max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {str(e)}"


def generate_training_summary(
    strategy: str,
    tickers: List[str],
    risk_profile: str,
    scenario: str,
    history: Dict,
    curves: Dict,
    initial_capital: float = 0.0,
) -> str:
    b = curves["baseline_metrics"]
    a = curves["attacked_metrics"]
    h = curves["hardened_metrics"]

    b_final = initial_capital * (1 + b['final_return_pct'] / 100) if initial_capital else 0
    a_final = initial_capital * (1 + a['final_return_pct'] / 100) if initial_capital else 0
    h_final = initial_capital * (1 + h['final_return_pct'] / 100) if initial_capital else 0

    worst_scenario = max(
        set(history["scenario_label"]),
        key=lambda s: history["drawdown"][
            [i for i, x in enumerate(history["scenario_label"]) if x == s][0]
        ]
    )

    prompt = f"""
You are a financial AI assistant explaining adversarial trading strategy stress test results.
Write a punchy, clear summary for a non-technical user. 4-5 sentences. No bullet points.
Sound like a smart analyst narrating a fight — honest, direct, slightly dramatic where appropriate.

Portfolio: {', '.join(tickers)}
Strategy: {strategy}
Risk profile: {risk_profile}
Stress scenario: {scenario}
Most dangerous scenario found: {worst_scenario}
Initial capital: ${initial_capital:,.2f}

Results:
- Baseline (no stress): {b['final_return_pct']:+.1f}% return → ${b_final:,.2f}, Sharpe {b['sharpe']:.2f}, max drawdown {b['max_drawdown']*100:.1f}%
- Attacked (unprepared): {a['final_return_pct']:+.1f}% return → ${a_final:,.2f}, Sharpe {a['sharpe']:.2f}, max drawdown {a['max_drawdown']*100:.1f}%
- Hardened (after training): {h['final_return_pct']:+.1f}% return → ${h_final:,.2f}, Sharpe {h['sharpe']:.2f}, max drawdown {h['max_drawdown']*100:.1f}%

Mention the actual dollar amounts. Explain what happened in plain English. Be direct about whether adversarial training helped.
Only reference the exact numbers above, do not invent figures.
"""
    return _call_groq(prompt)


def generate_failure_mode_explanation(
    strategy: str,
    failure_modes: List[Dict],
) -> str:
    """
    Explain in plain English why the top failure modes are dangerous
    for this specific strategy.
    """
    top = failure_modes[:2]  # focus on top 2 threats
    threats = "\n".join(
        f"- {m['scenario']}: average drawdown {m['avg_drawdown_pct']:.1f}%, threat level {m['threat_level']}"
        for m in top
    )

    prompt = f"""
You are a financial AI assistant. Explain in 3-4 plain English sentences why the following market scenarios
are dangerous for a {strategy} trading strategy. Be specific about why this strategy type struggles
in these conditions. No bullet points, no technical jargon.

Top threats identified:
{threats}
"""
    return _call_groq(prompt)


def generate_strategy_recommendation(
    strategy: str,
    risk_profile: str,
    curves: Dict,
    failure_modes: List[Dict],
) -> str:
    """
    Give a practical recommendation based on the stress test results.
    Suggests whether the user should adjust their strategy or risk profile.
    """
    h = curves["hardened_metrics"]
    a = curves["attacked_metrics"]
    top_threat = failure_modes[0]["scenario"] if failure_modes else "unknown"

    improvement = h["final_return_pct"] - a["final_return_pct"]
    dd_change   = h["max_drawdown"] - a["max_drawdown"]

    prompt = f"""
You are a financial AI assistant giving practical advice after a stress test.
Write 3-4 sentences of actionable advice for a trader using a {strategy} strategy with a {risk_profile} risk profile.

Stress test results:
- Biggest threat to this strategy: {top_threat}
- Return improvement after hardening: {improvement:+.1f}%
- Drawdown change after hardening: {dd_change*100:+.1f}%
- Hardened strategy final return: {h['final_return_pct']:+.1f}%
- Hardened strategy max drawdown: {h['max_drawdown']*100:.1f}%

Give specific practical advice. Should they change their risk profile? Reduce exposure to certain assets?
Consider a different strategy? Keep it friendly and actionable, no bullet points.
Important: only reference the exact numbers provided above, do not invent or estimate any figures.
"""
    return _call_groq(prompt)


def generate_qa_response(
    question: str,
    strategy: str,
    tickers: List[str],
    history: Dict,
    curves: Dict,
    failure_modes: List[Dict],
    initial_capital: float = 0.0,
) -> str:
    b = curves["baseline_metrics"]
    a = curves["attacked_metrics"]
    h = curves["hardened_metrics"]

    b_final = initial_capital * (1 + b['final_return_pct'] / 100) if initial_capital else 0
    a_final = initial_capital * (1 + a['final_return_pct'] / 100) if initial_capital else 0
    h_final = initial_capital * (1 + h['final_return_pct'] / 100) if initial_capital else 0

    context = f"""
Portfolio: {', '.join(tickers)}
Strategy: {strategy}
Initial capital: ${initial_capital:,.2f}
Training rounds: {len(history['round'])}
Scenarios tested: {', '.join(set(history['scenario_label']))}

Results:
- Baseline (no attack): {b['final_return_pct']:+.1f}% → ${b_final:,.2f}, Sharpe {b['sharpe']:.2f}, max drawdown {b['max_drawdown']*100:.1f}%
- Attacked (unprepared): {a['final_return_pct']:+.1f}% → ${a_final:,.2f}, Sharpe {a['sharpe']:.2f}, max drawdown {a['max_drawdown']*100:.1f}%
- Hardened (trained): {h['final_return_pct']:+.1f}% → ${h_final:,.2f}, Sharpe {h['sharpe']:.2f}, max drawdown {h['max_drawdown']*100:.1f}%

Top failure modes:
{chr(10).join(f"- {m['scenario']}: avg drawdown {m['avg_drawdown_pct']:.1f}%" for m in failure_modes[:3])}
"""

    prompt = f"""
You are a financial AI assistant. Answer the following question about a trading strategy stress test.
Be concise, clear, and direct. Use plain English and actual dollar amounts where relevant. 2-3 sentences max.
Only reference the exact numbers provided in the context, do not invent figures.

Context:
{context}

User question: {question}
"""
    return _call_groq(prompt, max_tokens=200)


# sanity check
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.market_data import build_portfolio_data
    from src.training.trainer import MinimaxTrainer
    from src.utils.metrics import identify_failure_modes

    tickers = ["AAPL", "SPY"]
    weights = {"AAPL": 0.6, "SPY": 0.4}
    feats, w = build_portfolio_data(tickers, weights, split="train")

    trainer = MinimaxTrainer(
        features_dict   = feats,
        weights         = w,
        scenario        = "adversarial",
        strategy        = "momentum",
        risk_profile    = "moderate",
        intensity       = "light",
    )
    history = trainer.train()
    curves  = trainer.get_three_curves()

    failure_modes = identify_failure_modes(
        history["scenario_label"],
        history["drawdown"],
        history["agent_reward"],
    )

    print("\n=== Training Summary ===")
    print(generate_training_summary("momentum", tickers, "moderate", "adversarial", history, curves))

    print("\n=== Failure Mode Explanation ===")
    print(generate_failure_mode_explanation("momentum", failure_modes))

    print("\n=== Strategy Recommendation ===")
    print(generate_strategy_recommendation("momentum", "moderate", curves, failure_modes))

    print("\n=== Q&A Test ===")
    print(generate_qa_response(
        "Why did my strategy struggle during flash crashes?",
        "momentum", tickers, history, curves, failure_modes
    ))