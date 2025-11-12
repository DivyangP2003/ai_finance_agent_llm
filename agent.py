# app.py
import os
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

# PyPortfolioOpt imports
from pypfopt import expected_returns, risk_models, EfficientFrontier

# AI agent imports
from agno.agent import Agent
from agno.models.google import Gemini

# --------------------------- Setup --------------------------- #
load_dotenv()
API_KEY_ENV = os.getenv("GOOGLE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # for alt-data news feed

st.set_page_config(page_title="AI Market Intelligence – Multi-Agent (v2)", page_icon="📊", layout="wide")

# --------------------------- Audit Logging DB Setup --------------------------- #
DB_PATH = "audit_log.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent TEXT,
            prompt TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_agent(agent_name, prompt, response):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, agent, prompt, response) VALUES (?, ?, ?, ?)",
              (datetime.utcnow().isoformat(), agent_name, prompt, response))
    conn.commit()
    conn.close()

def fetch_logs(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, agent, prompt, response FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# --------------------------- Utility & Data Functions --------------------------- #
@st.cache_data(ttl=60 * 15)
def download_close_prices(symbols, period="1y"):
    if not symbols:
        return pd.DataFrame()
    df = yf.download(symbols, period=period, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(df.name)
    return df

@st.cache_data(ttl=60 * 15)
def fetch_ticker_info(symbol):
    t = yf.Ticker(symbol)
    try:
        info = t.get_info() if hasattr(t, "get_info") else {}
    except Exception:
        info = {}
    return info

@st.cache_data(ttl=60 * 10)
def fetch_news_api(symbol, limit=20):
    """
    Fetch news via external NewsAPI (or similar) for better coverage.
    """
    if not NEWS_API_KEY:
        return []
    url = ("https://newsapi.org/v2/everything?"
           f"q={symbol}&pageSize={limit}&apiKey={NEWS_API_KEY}")
    try:
        r = requests.get(url)
        j = r.json()
        articles = j.get("articles", [])
    except Exception:
        articles = []
    processed = []
    for a in articles:
        processed.append({
            "title": a.get("title"),
            "source": a.get("source", {}).get("name", ""),
            "url": a.get("url", ""),
            "publishedAt": a.get("publishedAt", "")
        })
    return processed

@st.cache_data(ttl=60 * 10)
def fetch_news_yf(symbol, limit=10):
    t = yf.Ticker(symbol)
    try:
        news = t.news[:limit]
    except Exception:
        news = []
    processed = []
    for n in news:
        processed.append({
            "title": n.get("title") if isinstance(n, dict) else str(n),
            "publisher": n.get("publisher", "") if isinstance(n, dict) else "",
            "link": n.get("link", "") if isinstance(n, dict) else ""
        })
    return processed

def compute_returns(close_df):
    return close_df.pct_change().dropna()

def rolling_volatility(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252)

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def historical_var(returns, alpha=0.05):
    return np.quantile(returns.dropna(), alpha)

def compute_beta(asset_returns, benchmark_returns):
    merged = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if merged.shape[0] < 2:
        return np.nan
    x = merged.iloc[:,1].values
    y = merged.iloc[:,0].values
    cov = np.cov(x, y, ddof=1)
    if cov.shape == (2,2):
        beta = cov[0,1] / cov[0,0] if cov[0,0] != 0 else np.nan
        return float(beta)
    return np.nan

def sharpe_ratio(returns, risk_free_rate=0.0):
    mean = returns.mean() * 252
    std = returns.std() * np.sqrt(252)
    if std == 0:
        return np.nan
    return (mean - risk_free_rate) / std

# --------------------------- Agent Setup --------------------------- #
def create_agents():
    agents = {}

    agents["MarketAnalystAgent"] = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="Quantitative market analyst: trends, volatility, factor-like metrics, regimes.",
        instructions=[
            "Return a succinct analysis in markdown.",
            "Include explicit sections labeled 'Rationale:' and 'Recommendation:'.",
            "When you provide numbers, label them and explain their meaning.",
            "If you identify a market regime (e.g., risk-on / risk-off), state the evidence."
        ],
        markdown=True
    )

    agents["CompanyResearchAgent"] = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="Fundamental researcher: financials, earnings, qualitative signals.",
        instructions=[
            "Summarize company fundamentals and recent earnings.",
            "If recommending, include 'Rationale:' and 'Recommendation:'.",
            "Highlight material risks and catalysts."
        ],
        markdown=True
    )

    agents["SentimentAgent"] = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="Behavioral sentiment analyst: news, social, analyst tone.",
        instructions=[
            "Analyze sentiment of the provided headlines or snippets.",
            "Return an aggregated sentiment score (scale -1 to +1), a short summary, 'Rationale:', and 'Recommendation:'.",
            "Be explicit about uncertainty and sample size."
        ],
        markdown=True
    )

    agents["RiskAnalystAgent"] = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="Risk analyst: VaR, drawdown, correlation, stress tests.",
        instructions=[
            "Compute and interpret risk metrics; include 'Rationale:' and 'Recommendation:' sections.",
            "If VaR is presented, give the confidence level and interpretation."
        ],
        markdown=True
    )

    agents["PortfolioStrategistAgent"] = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="Portfolio strategist: allocation and optimization advice.",
        instructions=[
            "Propose weightings and justify them with 'Rationale:' and 'Recommendation:'.",
            "If constraints exist (max weight, min weight), state them explicitly."
        ],
        markdown=True
    )

    agents["TeamLeadAgent"] = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        description="Integrator: compile final insights from other agents into a structured report.",
        instructions=[
            "Integrate market, company, sentiment and risk analyses into a coherent, time-stamped report.",
            "Report must include a summary, top recommendations, and concise rationales for auditability."
        ],
        markdown=True
    )

    return agents

AGENTS = create_agents()

# --------------------------- Orchestration Functions --------------------------- #
def run_agent(agent_key, prompt):
    agent = AGENTS[agent_key]
    response = agent.run(prompt)
    # log prompt & response
    log_agent(agent_key, prompt, response.content)
    return response.content

def run_market_agent(symbols, close_df, benchmark="^GSPC"):
    if close_df.empty:
        return "No price data available for market analysis."

    returns = compute_returns(close_df)
    perf_6m = (close_df.iloc[-1] / close_df.iloc[0] - 1).to_dict()
    vol_ann = returns.std() * np.sqrt(252)
    vol = vol_ann.to_dict()

    prompt = (f"You are MarketAnalystAgent. Analyze quantitative metrics for: {', '.join(close_df.columns)}.\n"
              f"6-month returns approx:\n")
    for s, v in perf_6m.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nAnnualised volatility approx:\n"
    for s, v in vol.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += ("\nInstructions: Provide a short Market Overview, 
identify if a clear market regime exists (risk-on / risk-off), list top 3 signals, include 'Rationale:' & 'Recommendation:'.\n")

    return run_agent("MarketAnalystAgent", prompt)

def run_company_agent(symbol):
    info = fetch_ticker_info(symbol)
    news = fetch_news_api(symbol, limit=5) or fetch_news_yf(symbol, limit=5)
    prompt = (f"You are CompanyResearchAgent. Provide a concise company analysis for {symbol}.\n"
              f"Name: {info.get('longName', symbol)}; Sector: {info.get('sector', 'N/A')}; Market Cap: {info.get('marketCap', 'N/A')}\n"
              "Recent headlines:\n")
    for n in news:
        prompt += f"- {n.get('title')}\n"
    prompt += ("\nInstructions: Summarize fundamentals, list top 3 catalysts and risks. Include 'Rationale:' and 'Recommendation:'.\n")

    return run_agent("CompanyResearchAgent", prompt)

def run_sentiment_agent(symbol):
    news = fetch_news_api(symbol, limit=20)
    if not news:
        return "No news to analyze for sentiment."
    prompt = (f"You are SentimentAgent. Given these headlines for {symbol}:\n")
    for n in news:
        prompt += f"- {n.get('title')}\n"
    prompt += ("\nProduce an aggregated sentiment score (-1 to +1), a short summary, 'Rationale:' and 'Recommendation:'.\n")

    return run_agent("SentimentAgent", prompt)

def run_risk_agent(symbols, close_df, benchmark="^GSPC"):
    if close_df.empty:
        return "No price data for risk analysis."

    returns = compute_returns(close_df)
    var5 = {s: historical_var(returns[s], alpha=0.05) for s in returns.columns}
    mdd = {s: max_drawdown(close_df[s]) for s in close_df.columns}
    vol_ann = (returns.std() * np.sqrt(252)).to_dict()
    corr = returns.corr().round(3).to_dict()

    prompt = (f"You are RiskAnalystAgent. Risk metrics for {', '.join(symbols)}:\n"
              "VaR (5%):\n")
    for s, v in var5.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nMax Drawdown:\n"
    for s, v in mdd.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nAnnualised vol:\n"
    for s, v in vol_ann.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nCorrelation snapshot:\n"
    for s, row in corr.items():
        row_items = ", ".join([f"{k}:{v}" for k,v in row.items()])
        prompt += f"- {s}: {row_items}\n"
    prompt += ("\nInstructions: Interpret these risk metrics, list top 3 risk concerns, include 'Rationale:' & 'Recommendation:'.\n")

    return run_agent("RiskAnalystAgent", prompt)

def run_portfolio_agent(symbols, close_df, constraints=None):
    if close_df.empty:
        return "No price data for portfolio optimisation."

    mu = expected_returns.mean_historical_return(close_df)
    S = risk_models.sample_cov(close_df)

    ef = EfficientFrontier(mu, S)
    if constraints:
        # Example: constraints dict e.g. {"max_weight":0.4}
        max_w = constraints.get("max_weight")
        if max_w:
            ef.add_constraint(lambda w: w <= max_w)
        # add other constraints as needed

    raw_weights = ef.max_sharpe()  # optimize for max Sharpe
    cleaned = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)
    weights_str = {k: f"{v:.4f}" for k,v in cleaned.items() if v > 0}

    prompt = (f"You are PortfolioStrategistAgent. Based on optimisation results for symbols: {', '.join(symbols)}\n"
              f"Optimal weights (cleaned): {weights_str}\n"
              f"Expected annual return: {performance[0]:.2%}, Volatility: {performance[1]:.2%}, Sharpe: {performance[2]:.2f}\n"
              f"Constraints (if any): {constraints or 'None'}\n"
              "Instructions: Justify selection with 'Rationale:' and 'Recommendation:'. Provide a 2‐line rebalancing guideline.\n")

    return run_agent("PortfolioStrategistAgent", prompt)

def run_teamlead_agent(date_str, market_analysis, company_analyses, sentiment_analyses, risk_analysis, portfolio_recommendation):
    prompt = (f"You are TeamLeadAgent. Date: {date_str}.\n"
              "Integrate these inputs into a coherent investment report.\n\n"
              f"Market Analysis:\n{market_analysis}\n\n"
              f"Company Analyses:\n{company_analyses}\n\n"
              f"Sentiment Analyses:\n{sentiment_analyses}\n\n"
              f"Risk Analysis:\n{risk_analysis}\n\n"
              f"Portfolio Recommendation:\n{portfolio_recommendation}\n\n"
              "Instructions: Provide Executive Summary (2-4 sentences), Top 3 actionable recommendations with 'Rationale:' each, and an Audit Trail listing which agent produced which section.\n")

    return run_agent("TeamLeadAgent", prompt)

# --------------------------- Natural Language Query Interface --------------------------- #
def interpret_user_query(query, symbols, close_df):
    q = query.lower().strip()
    # Map some keywords
    if "volatility" in q:
        rolling = rolling_volatility(compute_returns(close_df))
        text = run_agent("MarketAnalystAgent", f"You are MarketAnalystAgent. Interpret rolling volatility trends for {', '.join(symbols)} based on data.\nInclude 'Rationale:' and 'Recommendation:'.")
        return text, {"type": "rolling_vol", "data": rolling}
    elif "sharpe" in q:
        returns = compute_returns(close_df)
        port_returns = returns.mean(axis=1)
        sr = sharpe_ratio(port_returns)
        text = run_agent("PortfolioStrategistAgent", f"You are PortfolioStrategistAgent. Interpret portfolio Sharpe of {sr:.4f} for symbols {', '.join(symbols)}.\nInclude 'Rationale:' and 'Recommendation:'.")
        return text, {"type": "sharpe", "value": sr}
    elif "sentiment" in q:
        sentiments = {s: run_sentiment_agent(s) for s in symbols}
        return "Sentiment outputs generated.", {"type": "sentiments", "data": sentiments}
    else:
        text = run_agent("TeamLeadAgent", f"You are TeamLeadAgent. User query: \"{query}\". Symbols: {', '.join(symbols)}. Provide guidance. Include 'Rationale:' and 'Recommendation:'.")
        return text, {"type": "generic", "text": text}

# --------------------------- Streamlit UI --------------------------- #
st.markdown("<h1 style='text-align:center; color:#1E88E5;'>📊 AI Market Intelligence – Multi-Agent v2</h1>", unsafe_allow_html=True)
st.markdown("For researchers, traders, asset managers & risk managers.")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
input_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, TSLA, GOOG")
api_key = st.sidebar.text_input("Enter your Google API Key (optional)", type="password")
news_key = st.sidebar.text_input("Enter your NewsAPI Key (optional)", type="password")
benchmark = st.sidebar.text_input("Benchmark ticker (for beta/vol):", "^GSPC")

symbols = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
elif API_KEY_ENV:
    os.environ["GOOGLE_API_KEY"] = API_KEY_ENV
if news_key:
    os.environ["NEWS_API_KEY"] = news_key

# Tabs layout
tabs = st.tabs(["Overview", "Company Deep Dives", "Risk & Correlation", "Portfolio Optimiser", "Chat Assistant", "Audit & Logs"])

# --- Overview Tab ---
with tabs[0]:
    st.header("Market Overview & Quick Insights")
    period = st.selectbox("Select tracking period:", ["6mo", "1y", "2y"], index=1)
    close_df = download_close_prices(symbols, period=period)
    if close_df.empty:
        st.warning("No price data found.")
    else:
        # price plot
        st.subheader("Price Chart")
        fig = go.Figure()
        for s in close_df.columns:
            fig.add_trace(go.Scatter(x=close_df.index, y=close_df[s], mode="lines", name=s))
        fig.update_layout(title=f"Price History ({period})", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Run MarketAnalystAgent"):
            with st.spinner("Running market analysis..."):
                ma = run_market_agent(symbols, close_df, benchmark=benchmark)
                st.markdown("### Market Analysis (XAI)")
                st.markdown(ma)

# --- Company Deep Dives Tab ---
with tabs[1]:
    st.header("Company Deep Dives & Sentiment")
    for s in symbols:
        with st.expander(f"{s} Research"):
            info = fetch_ticker_info(s)
            st.write(f"**{info.get('longName', s)}**  |  Sector: {info.get('sector','N/A')}  |  MarketCap: {info.get('marketCap','N/A')}")
            if st.button(f"Run CompanyResearchAgent for {s}", key=f"comp_{s}"):
                with st.spinner(f"Running CompanyResearchAgent for {s}..."):
                    ctext = run_company_agent(s)
                    st.markdown(ctext)
            if st.button(f"Run SentimentAgent for {s}", key=f"sent_{s}"):
                with st.spinner(f"Running SentimentAgent for {s}..."):
                    stext = run_sentiment_agent(s)
                    st.markdown(stext)
            news = fetch_news_api(s, limit=5) or fetch_news_yf(s, limit=5)
            if news:
                st.markdown("**Latest Headlines**")
                for n in news:
                    st.markdown(f"- {n.get('title')}  — _{n.get('source') or n.get('publisher','')}_, [{n.get('url')}]({n.get('url')})")

# --- Risk & Correlation Tab ---
with tabs[2]:
    st.header("Risk Metrics & Correlations")
    close_df_risk = download_close_prices(symbols, period="1y")
    if close_df_risk.empty:
        st.warning("No data for risk metrics.")
    else:
        if st.button("Run RiskAnalystAgent"):
            with st.spinner("Running risk analysis..."):
                rtext = run_risk_agent(symbols, close_df_risk, benchmark=benchmark)
                st.markdown("### Risk Analysis (XAI)")
                st.markdown(rtext)

        returns = compute_returns(close_df_risk)
        if not returns.empty:
            corr = returns.corr()
            st.subheader("Return Correlation Heatmap (1y)")
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1, zmax=1,
                colorbar=dict(title="Corr")
            ))
            fig_corr.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Rolling Volatility (21‐day annualised)")
            vol = rolling_volatility(returns)
            fig_vol = go.Figure()
            for c in vol.columns:
                fig_vol.add_trace(go.Scatter(x=vol.index, y=vol[c], mode="lines", name=c))
            fig_vol.update_layout(template="plotly_white")
            st.plotly_chart(fig_vol, use_container_width=True)

# --- Portfolio Optimiser Tab ---
with tabs[3]:
    st.header("Portfolio Optimiser (PyPortfolioOpt)")
    close_df_port = download_close_prices(symbols, period="1y")
    constraints_raw = st.text_input("Constraints (Python dict) e.g. {'max_weight':0.4}", "")
    try:
        constraints = eval(constraints_raw) if constraints_raw else None
    except Exception:
        constraints = None
    if st.button("Run PortfolioStrategistAgent"):
        with st.spinner("Optimising portfolio..."):
            ptext = run_portfolio_agent(symbols, close_df_port, constraints=constraints)
            st.markdown("### Allocation Recommendation (XAI)")
            st.markdown(ptext)

# --- Chat Assistant Tab ---
with tabs[4]:
    st.header("Natural-Language Research Assistant")
    st.markdown("Ask your question below; results will include visuals when appropriate.")
    user_query = st.text_input("Enter your question", "")
    if st.button("Ask"):
        with st.spinner("Processing query..."):
            close_df_chat = download_close_prices(symbols, period="1y")
            text_out, meta = interpret_user_query(user_query, symbols, close_df_chat)
            st.subheader("Assistant Response")
            st.markdown(text_out)
            if isinstance(meta, dict):
                if meta.get("type") == "rolling_vol":
                    vol_df = meta["data"]
                    st.subheader("Rolling Volatility Chart")
                    fig_rv = go.Figure()
                    for c in vol_df.columns:
                        fig_rv.add_trace(go.Scatter(x=vol_df.index, y=vol_df[c], mode="lines", name=c))
                    fig_rv.update_layout(template="plotly_white")
                    st.plotly_chart(fig_rv, use_container_width=True)
                elif meta.get("type") == "sharpe":
                    st.metric("Estimated Portfolio Sharpe (annualised)", f"{meta['value']:.3f}")
                elif meta.get("type") == "sentiments":
                    st.subheader("Sentiment Agent Texts")
                    for s, t in meta["data"].items():
                        with st.expander(s):
                            st.markdown(t)

# --- Audit & Logs Tab ---
with tabs[5]:
    st.header("Audit Trail & Logs")
    st.markdown("This table shows the most recent agent prompts & responses (immutable log).")
    logs = fetch_logs(limit=100)
    df_logs = pd.DataFrame(logs, columns=["timestamp", "agent", "prompt", "response"])
    st.dataframe(df_logs, use_container_width=True)

    st.markdown("---")
    st.markdown("Generate full integrated report (TeamLead) combining all analyses.")
    if st.button("Run full multi-agent orchestration"):
        with st.spinner("Running full orchestration..."):
            date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            market_analysis = run_market_agent(symbols, download_close_prices(symbols, period="1y"), benchmark=benchmark)
            company_analyses = {s: run_company_agent(s) for s in symbols}
            sentiment_analyses = {s: run_sentiment_agent(s) for s in symbols}
            risk_analysis = run_risk_agent(symbols, download_close_prices(symbols, period="1y"), benchmark=benchmark)
            portfolio_recommendation = run_portfolio_agent(symbols, download_close_prices(symbols, period="1y"))
            final_report = run_teamlead_agent(date_str, market_analysis, company_analyses, sentiment_analyses, risk_analysis, portfolio_recommendation)
            st.subheader("### Final Consolidated Report")
            st.markdown(final_report)

    st.markdown("---")
    st.markdown("**Notes**\n"
                "- Audit logs store each agent prompt & response with timestamp (immutable once written).\n"
                "- Portfolio optimisation uses PyPortfolioOpt to compute optimal weights based on historical returns & covariance. :contentReference[oaicite:4]{index=4}\n"
                "- Sentiment feed uses external NewsAPI for broader news coverage and alt-data. :contentReference[oaicite:5]{index=5}\n"
                "- For production: lock the DB in a proper data store (PostgreSQL, etc), implement robust error-handling, rate-limit news API, and store versioned agent models/prompts for full auditability.\n")

# End of app.py
