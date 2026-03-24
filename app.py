import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import sys, os

sys.path.append(os.path.dirname(__file__))

from src.data.market_data import build_portfolio_from_shares, get_current_prices, search_ticker
from src.training.trainer import MinimaxTrainer, INTENSITY_PRESETS, RISK_PRESETS
from src.utils.metrics import build_comparison_table, identify_failure_modes, compute_metrics
from src.utils.groq_summary import (
    generate_training_summary,
    generate_failure_mode_explanation,
    generate_strategy_recommendation,
    generate_qa_response,
)


# page config
st.set_page_config(
    page_title     = "BlackSwan AI",
    page_icon      = "🦢",
    layout         = "wide",
    initial_sidebar_state = "collapsed",
)


# load CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# session state init
def init_state():
    defaults = {
        "screen":       "setup",
        "setup_step":   1,
        "shares":       {},
        "prices":       {},
        "strategy":     "momentum",
        "risk_profile": "moderate",
        "horizon":      "daily",
        "scenario":     "adversarial",
        "intensity":    "light",
        "covid_test":   False,
        "trainer":      None,
        "history":      None,
        "curves":       None,
        "failure_modes": None,
        "groq_summary":  None,
        "groq_failure":  None,
        "groq_rec":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# header
def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🦢 BlackSwan <span>AI</span></div>
        <div class="app-subtitle">Adversarial stress-testing for trading strategies</div>
    </div>
    """, unsafe_allow_html=True)


# step indicator
def render_steps(current: int):
    steps = ["Portfolio", "Strategy", "Risk & Scenario", "Run"]
    html = '<div class="step-bar">'
    for i, s in enumerate(steps, 1):
        cls = "active" if i == current else ("done" if i < current else "step")
        html += f'<div class="step {cls}">{i}. {s}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# setup screen
def render_setup():
    render_header()
    render_steps(st.session_state.setup_step)

    step = st.session_state.setup_step

    # step 1 — portfolio builder
    if step == 1:
        st.markdown("### Build Your Portfolio")
        st.markdown("<p style='color:#8ab4c9'>Search by company name or ticker symbol (e.g. Apple, AAPL, Bitcoin, BTC-USD).</p>", unsafe_allow_html=True)

        # init portfolio dict in session state
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = {}  # { ticker: shares }

        # add asset row
        col1, col2 = st.columns([3, 1])
        with col1:
            new_query = st.text_input(
                "Search",
                placeholder = "e.g. Apple, NVDA, Bitcoin, Gold...",
                label_visibility = "collapsed",
                key = "ticker_input",
            ).strip()
        with col2:
            add_clicked = st.button("+ Add Asset", use_container_width=True)

        if add_clicked and new_query:
            with st.spinner(f"Searching for '{new_query}'..."):
                ticker, name, price = search_ticker(new_query)

            if ticker is None:
                st.error(f"Could not find '{new_query}'. Try a different name or ticker symbol.")
            elif ticker in st.session_state.portfolio:
                st.warning(f"{ticker} ({name}) is already in your portfolio.")
            else:
                st.session_state.portfolio[ticker] = 0
                if "prices" not in st.session_state:
                    st.session_state.prices = {}
                st.session_state.prices[ticker]  = price
                st.session_state.names           = getattr(st.session_state, "names", {})
                st.session_state.names[ticker]   = name
                st.rerun()

        # portfolio list
        if not st.session_state.portfolio:
            st.markdown("<div style='text-align:center;color:#555;padding:2rem'>No assets added yet. Search for a ticker above to get started.</div>", unsafe_allow_html=True)
        else:
            total = 0.0
            to_remove = []

            for ticker in list(st.session_state.portfolio.keys()):
                price    = st.session_state.prices.get(ticker, 0)
                names    = st.session_state.get("names", {})
                display  = f"{ticker} — {names.get(ticker, '')}" if names.get(ticker) else ticker
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                with c1:
                    st.markdown(f"<div style='padding:0.6rem 0;font-weight:700;color:#fff'>{display}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div style='padding:0.6rem 0;color:#8ab4c9;font-size:0.85rem'>${price:,.2f}/share</div>", unsafe_allow_html=True)
                with c3:
                    shares_val = st.text_input(
                        "Shares",
                        value = str(st.session_state.portfolio[ticker]),
                        key   = f"shares_{ticker}",
                        label_visibility = "collapsed",
                        placeholder = "Number of shares",
                    )
                    try:
                        n = max(0.0, float(shares_val)) if shares_val.strip() else 0.0
                    except ValueError:
                        n = 0.0
                    st.session_state.portfolio[ticker] = n
                    val = n * price
                    total += val
                    if n > 0:
                        st.markdown(f"<div style='color:#00ff88;font-size:0.85rem'>${val:,.2f}</div>", unsafe_allow_html=True)
                with c4:
                    if st.button("✕", key=f"remove_{ticker}", use_container_width=True):
                        to_remove.append(ticker)

            for t in to_remove:
                del st.session_state.portfolio[t]
                if t in st.session_state.prices:
                    del st.session_state.prices[t]
                st.rerun()

            # portfolio value
            if total > 0:
                st.markdown(f"""
                <div class="portfolio-value">
                    <div class="pv-label">Total Portfolio Value</div>
                    <div class="pv-amount">${total:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next: Choose Strategy →"):
            active = {t: s for t, s in st.session_state.portfolio.items() if s > 0}
            if len(active) < 1:
                st.error("Please add at least one asset with shares greater than 0.")
            else:
                st.session_state.shares = active
                st.session_state.setup_step = 2
                st.rerun()

    # step 2 — strategy
    elif step == 2:
        st.markdown("### Choose Your Trading Strategy")
        st.markdown("<p style='color:#888'>This shapes how your agent makes decisions.</p>", unsafe_allow_html=True)

        strategy_options = {
            "momentum":       "📈 Ride the Trend (Momentum) — buy what's been going up",
            "mean_reversion": "📉 Buy the Dip (Mean Reversion) — buy drops, sell rises",
            "trend_following":"➡️ Stay the Course (Trend Following) — hold steady, low turnover",
            "pure_rl":        "🤖 Let AI Decide (Pure RL) — fully self-learned, no bias",
        }

        strategy = st.radio(
            "Strategy",
            options   = list(strategy_options.keys()),
            format_func = lambda x: strategy_options[x],
            index     = list(strategy_options.keys()).index(st.session_state.strategy),
            label_visibility = "collapsed",
        )
        st.session_state.strategy = strategy

        horizon = st.radio(
            "How often do you rebalance?",
            options      = ["daily", "weekly"],
            format_func  = lambda x: "Every Day" if x == "daily" else "Once a Week",
            horizontal   = True,
            index        = ["daily", "weekly"].index(st.session_state.horizon),
        )
        st.session_state.horizon = horizon

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.setup_step = 1
                st.rerun()
        with col2:
            if st.button("Next: Risk & Scenario →"):
                st.session_state.setup_step = 3
                st.rerun()

    # step 3 — risk and scenario
    elif step == 3:
        st.markdown("### Risk Profile & Stress Scenario")

        risk_options = {
            "conservative": "🛡️ Play it Safe — low position size, 10% max loss tolerance",
            "moderate":     "⚖️ Balanced — medium position size, 20% max loss tolerance",
            "aggressive":   "🔥 Go Big — large position size, 35% max loss tolerance",
        }
        risk = st.radio(
            "How much loss can you stomach?",
            options      = list(risk_options.keys()),
            format_func  = lambda x: risk_options[x],
            index        = list(risk_options.keys()).index(st.session_state.risk_profile),
            label_visibility = "hidden",
        )
        st.session_state.risk_profile = risk

        scenario_options = {
            "adversarial": "⚔️ Full Adversarial — AI finds the worst possible attack",
            "flash_crash": "💥 Sudden Market Crash — sharp drop, partial recovery",
            "volatility":  "🌊 Wild Price Swings — amplified volatility and noise",
            "regime_change":"🔄 Market Shifts Direction — gradual trend reversal",
        }
        scenario = st.radio(
            "What stress scenario to throw at your strategy?",
            options      = list(scenario_options.keys()),
            format_func  = lambda x: scenario_options[x],
            index        = list(scenario_options.keys()).index(st.session_state.scenario),
        )
        st.session_state.scenario = scenario

        intensity_options = {
            "light":    f"⚡ Quick ({INTENSITY_PRESETS['light']['n_rounds']} rounds) — fast results",
            "standard": f"🔬 Standard ({INTENSITY_PRESETS['standard']['n_rounds']} rounds) — balanced",
            "deep":     f"🧠 Deep ({INTENSITY_PRESETS['deep']['n_rounds']} rounds) — most thorough",
        }
        intensity = st.radio(
            "Training intensity",
            options      = list(intensity_options.keys()),
            format_func  = lambda x: intensity_options[x],
            index        = list(intensity_options.keys()).index(st.session_state.intensity),
            horizontal   = True,
        )
        st.session_state.intensity = intensity

        covid = st.toggle(
            "🦠 Test against the real 2020 COVID crash at the end",
            value = st.session_state.covid_test,
        )
        st.session_state.covid_test = covid

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.setup_step = 2
                st.rerun()
        with col2:
            if st.button("⚔️ Start Battle"):
                st.session_state.screen = "battle"
                st.rerun()


# battle screen
def render_battle():
    render_header()
    st.markdown('<div class="battle-header">⚔️ ADVERSARY vs YOUR STRATEGY</div>', unsafe_allow_html=True)

    # score placeholders
    score_ph   = st.empty()
    progress   = st.progress(0)
    status_ph  = st.empty()
    feed_ph    = st.empty()

    # build portfolio
    active_shares = {t: s for t, s in st.session_state.shares.items() if s > 0}
    tickers       = list(active_shares.keys())

    status_ph.markdown("<p style='color:#888;text-align:center'>📡 Fetching market data...</p>", unsafe_allow_html=True)
    feats, weights, total_value = build_portfolio_from_shares(active_shares, split="train")

    status_ph.markdown("<p style='color:#888;text-align:center'>🤖 Initializing agents...</p>", unsafe_allow_html=True)
    trainer = MinimaxTrainer(
        features_dict   = feats,
        weights         = weights,
        initial_capital = total_value,
        scenario        = st.session_state.scenario,
        strategy        = st.session_state.strategy,
        risk_profile    = st.session_state.risk_profile,
        trading_horizon = st.session_state.horizon,
        intensity       = st.session_state.intensity,
    )

    n_rounds        = trainer.n_rounds
    adv_score       = 0
    strat_score     = 0
    feed_lines      = []
    loading_messages = [
        "Adversary scanning for weaknesses...",
        "Injecting stress scenario...",
        "Agent adapting to attack...",
        "Adversary recalibrating...",
        "Running evaluation episode...",
    ]

    def progress_callback(round_idx, total, info):
        nonlocal adv_score, strat_score

        # update scores
        if info["agent_reward"] > 0:
            strat_score += 1
            outcome     = "win"
        else:
            adv_score  += 1
            outcome     = "loss"

        # score board
        score_ph.markdown(f"""
        <div class="score-board">
            <div class="score-adversary">
                <div class="score-label">Adversary</div>
                <div class="score-value">{adv_score}</div>
            </div>
            <div class="score-vs">VS</div>
            <div class="score-strategy">
                <div class="score-label">Your Strategy</div>
                <div class="score-value">{strat_score}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # progress bar
        progress.progress(round_idx / total)

        # loading message
        msg = loading_messages[round_idx % len(loading_messages)]
        status_ph.markdown(f"<p style='color:#888;text-align:center'>⚙️ Round {round_idx}/{total} — {msg}</p>", unsafe_allow_html=True)

        # round feed
        pnl    = (info["portfolio_value"] / total_value - 1) * 100
        label  = info["scenario"]
        dd     = info["drawdown"] * 100
        css    = "round-win" if outcome == "win" else "round-loss"
        icon   = "✅" if outcome == "win" else "⚠️"
        line   = f'<div class="{css}">{icon} Round {round_idx}/{total} [{label}] &nbsp;|&nbsp; Return: {pnl:+.1f}% &nbsp;|&nbsp; Max DD: {dd:.1f}%</div>'
        feed_lines.append(line)
        feed_ph.markdown("".join(feed_lines), unsafe_allow_html=True)

        time.sleep(0.3)

    # run training
    status_ph.markdown("<p style='color:#00ff88;text-align:center'>⚔️ Battle started — agents are fighting...</p>", unsafe_allow_html=True)
    history = trainer.train(progress_callback=progress_callback)

    progress.progress(1.0)
    status_ph.markdown("<p style='color:#00ff88;text-align:center'>✅ Training complete — generating results...</p>", unsafe_allow_html=True)
    time.sleep(0.5)

    # generate curves and analysis
    curves       = trainer.get_three_curves()
    failure_modes = identify_failure_modes(
        history["scenario_label"],
        history["drawdown"],
        history["agent_reward"],
    )

    # groq summaries
    status_ph.markdown("<p style='color:#888;text-align:center'>🧠 AI analyzing results...</p>", unsafe_allow_html=True)
    groq_summary = generate_training_summary(
        st.session_state.strategy, tickers,
        st.session_state.risk_profile, st.session_state.scenario,
        history, curves, initial_capital=total_value,
    )
    groq_failure = generate_failure_mode_explanation(st.session_state.strategy, failure_modes)
    groq_rec     = generate_strategy_recommendation(
        st.session_state.strategy, st.session_state.risk_profile, curves, failure_modes
    )

    # store everything
    st.session_state.trainer       = trainer
    st.session_state.history       = history
    st.session_state.curves        = curves
    st.session_state.failure_modes = failure_modes
    st.session_state.groq_summary  = groq_summary
    st.session_state.groq_failure  = groq_failure
    st.session_state.groq_rec      = groq_rec
    st.session_state.screen        = "results"
    st.rerun()


# results screen
def render_results():
    render_header()

    curves        = st.session_state.curves
    history       = st.session_state.history
    failure_modes = st.session_state.failure_modes
    trainer       = st.session_state.trainer
    tickers       = [t for t, s in st.session_state.shares.items() if s > 0]

    h = curves["hardened_metrics"]
    a = curves["attacked_metrics"]
    b = curves["baseline_metrics"]

    # verdict
    survived = h["final_return_pct"] > a["final_return_pct"] or h["max_drawdown"] < a["max_drawdown"]
    if survived:
        st.markdown(f"""
        <div class="verdict-survived">
            <div class="verdict-title">✅ Strategy Survived</div>
            <div class="verdict-subtitle">Adversarial training improved resilience — hardened agent outperformed the unprepared strategy under the same attack</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-failed">
            <div class="verdict-title">⚠️ Adversary Wins</div>
            <div class="verdict-subtitle">The strategy needs more training or a different risk profile to withstand these conditions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # three curves chart
    st.markdown("### Portfolio Performance")
    fig = go.Figure()

    x = list(range(len(curves["baseline"])))

    fig.add_trace(go.Scatter(
        x=x, y=curves["baseline"],
        name="Baseline",
        line=dict(color="#888888", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curves["attacked"],
        name="Attacked (unprepared)",
        line=dict(color="#ff4444", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curves["hardened"],
        name="Hardened (trained)",
        line=dict(color="#00ff88", width=2),
    ))

    # adversarial event marker
    inj = curves.get("inj_step", 60)
    fig.add_vline(
        x=inj,
        line=dict(color="#ffffff", width=1.5, dash="dash"),
        annotation_text="Adversarial Event Injected",
        annotation_position="top",
        annotation_font_color="#ffffff",
        annotation_font_size=11,
    )

    fig.update_layout(
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="#ffffff"),
        legend        = dict(bgcolor="#1a1d24", bordercolor="#2a2d34", borderwidth=1),
        xaxis         = dict(title="Trading Days", gridcolor="#2a2d34", color="#888888"),
        yaxis         = dict(title="Portfolio Value ($)", gridcolor="#2a2d34", color="#888888"),
        margin        = dict(l=20, r=20, t=20, b=20),
        height        = 400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # dollar summary below chart
    initial = trainer.initial_capital
    b_end = curves["baseline"][-1]
    a_end = curves["attacked"][-1]
    h_end = curves["hardened"][-1]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div class="metric-name">Baseline (no attack)</div>
            <div style="color:#888888;font-size:1.3rem;font-weight:700">${b_end:,.2f}</div>
            <div style="color:#888888;font-size:0.8rem">{((b_end/initial)-1)*100:+.1f}% from ${initial:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div class="metric-name">Attacked (unprepared)</div>
            <div style="color:#ff4444;font-size:1.3rem;font-weight:700">${a_end:,.2f}</div>
            <div style="color:#ff4444;font-size:0.8rem">{((a_end/initial)-1)*100:+.1f}% from ${initial:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div class="metric-name">Hardened (trained)</div>
            <div style="color:#00ff88;font-size:1.3rem;font-weight:700">${h_end:,.2f}</div>
            <div style="color:#00ff88;font-size:0.8rem">{((h_end/initial)-1)*100:+.1f}% from ${initial:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # metrics comparison
    st.markdown("### How Did Hardening Help?")
    metrics_map = [
        ("Total Return",         f"{a['final_return_pct']:+.1f}%", f"{h['final_return_pct']:+.1f}%",
         h["final_return_pct"] > a["final_return_pct"]),
        ("Sharpe Ratio",         f"{a['sharpe']:+.3f}",            f"{h['sharpe']:+.3f}",
         h["sharpe"] > a["sharpe"]),
        ("Max Drawdown",         f"{a['max_drawdown']*100:.1f}%",  f"{h['max_drawdown']*100:.1f}%",
         h["max_drawdown"] < a["max_drawdown"]),
        ("Annualised Volatility",f"{a.get('volatility_ann', 0):.1f}%", f"{h.get('volatility_ann', 0):.1f}%",
         h.get("volatility_ann", 0) < a.get("volatility_ann", 0)),
    ]

    for name, att_val, hard_val, improved in metrics_map:
        icon = "✅" if improved else "⚠️"
        tag_cls = "metric-improved" if improved else "metric-worse"
        tag_txt = "improved" if improved else "worse"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-name">{name}</div>
            <div class="metric-values">
                <span class="metric-attacked">{att_val}</span>
                <span class="metric-arrow">→</span>
                <span class="metric-hardened">{hard_val}</span>
                <span class="{tag_cls}">{icon} {tag_txt}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # failure modes
    with st.expander("⚠️ View Failure Mode Analysis", expanded=False):
        for m in failure_modes:
            color = "#ff4444" if m["threat_level"] == "HIGH" else "#ffaa00" if m["threat_level"] == "MEDIUM" else "#00ff88"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-name">{m['scenario'].replace('_', ' ').title()} &nbsp;
                    <span style='color:{color};font-weight:700'>{m['threat_level']}</span>
                </div>
                <div style='color:#aaa;font-size:0.9rem'>
                    Avg drawdown: <b style='color:#ff4444'>{m['avg_drawdown_pct']:.1f}%</b> &nbsp;|&nbsp;
                    Worst: <b style='color:#ff4444'>{m['worst_drawdown_pct']:.1f}%</b> &nbsp;|&nbsp;
                    Rounds tested: {m['n_rounds']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # AI analysis
    st.markdown("### 🧠 AI Analysis")

    st.markdown(f"""
    <div class="ai-section">
        <div class="ai-section-title">📋 Training Summary</div>
        <hr class="ai-divider">
        <div class="ai-content">{st.session_state.groq_summary}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-section">
        <div class="ai-section-title">⚠️ Why Your Strategy Is Vulnerable</div>
        <hr class="ai-divider">
        <div class="ai-content">{st.session_state.groq_failure}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="ai-section">
        <div class="ai-section-title">📈 Recommendation</div>
        <hr class="ai-divider">
        <div class="ai-content">{st.session_state.groq_rec}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Q&A
    st.markdown("### 💬 Ask About Your Results")
    question = st.text_input("Ask anything about your strategy or the stress test results...")
    if question:
        with st.spinner("Thinking..."):
            answer = generate_qa_response(
                question,
                st.session_state.strategy,
                tickers,
                history,
                curves,
                failure_modes,
                initial_capital=trainer.initial_capital,
            )
        st.markdown(f"""
        <div class="ai-section">
            <div class="ai-content">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # run again button
    if st.button("🔄 Run New Battle"):
        for key in ["screen", "setup_step", "trainer", "history", "curves",
                    "failure_modes", "groq_summary", "groq_failure", "groq_rec"]:
            st.session_state[key] = "setup" if key == "screen" else (1 if key == "setup_step" else None)
        st.rerun()


# router
screen = st.session_state.screen
if screen == "setup":
    render_setup()
elif screen == "battle":
    render_battle()
elif screen == "results":
    render_results()