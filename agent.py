# app.py
"""
AI Market Intelligence (Multi-Agent) — Extended with:
1) MacroEconomicsAgent
2) Economic Calendar Integration (TradingEconomics optional)
3) Interactive Portfolio Optimizer (PyPortfolioOpt + Black-Litterman + user constraints)

Requirements (example):
pip install streamlit yfinance plotly pandas numpy pypfopt pandas_datareader requests feedparser python-dotenv

Notes:
- Provide TRADING_ECONOMICS_KEY in environment variables for economic calendar (optional).
- Provide GOOGLE_API_KEY as before for your Gemini agent usage (optional).
- Make sure PyPortfolioOpt is installed for portfolio optimization. If not installed, the optimizer UI will show an error prompt.
"""

import os
from datetime import datetime, timedelta
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import feedparser
import requests

# --- Optional imports that may not exist in all environments ---
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel, plotting as ppf_plotting
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

try:
    # pandas_datareader for FRED macro series
    import pandas_datareader.data as webreader
    PANDAS_DATAREADER_AVAILABLE = True
except Exception:
    PANDAS_DATAREADER_AVAILABLE = False

# --- AI agent imports (assuming your agno Agent wrapper) ---
try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    AGENTS_AVAILABLE = True
except Exception:
    AGENTS_AVAILABLE = False

# --------------------------- Setup --------------------------- #
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY", "")
TRADING_ECONOMICS_KEY = os.getenv("TRADING_ECONOMICS_KEY", "")  # optional

st.set_page_config(page_title="AI Market Intelligence (Multi-Agent) — Extended", page_icon="📊", layout="wide")

# --------------------------- Helper: caching --------------------------- #
@st.cache_data(ttl=60 * 15)
def download_close_prices(symbols, period="1y", interval="1d"):
    """Download historical Close prices for given symbols."""
    if not symbols:
        return pd.DataFrame()
    df = yf.download(symbols, period=period, interval=interval, progress=False)["Close"]
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
def fetch_news(symbol, limit=10):
    """
    Fetch recent Yahoo Finance news headlines via RSS feed fallback.
    Works even if yfinance's .news endpoint is broken.
    """
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        processed = []
        for entry in feed.entries[:limit]:
            processed.append({
                "title": entry.title,
                "publisher": entry.get("source", "Yahoo Finance"),
                "link": entry.link
            })
        return processed
    except Exception as e:
        st.warning(f"Failed to fetch RSS for {symbol}: {e}")
        return []

# --------------------------- Quantitative Analytics --------------------------- #
def compute_returns(close_df):
    return close_df.pct_change().dropna()

def rolling_volatility(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252)

def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def historical_var(returns, alpha=0.05, window=None):
    if window:
        tails = returns.rolling(window).apply(lambda x: np.quantile(x, alpha))
        return tails
    else:
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

# --------------------------- Multi-Agent Definitions --------------------------- #
def create_agents():
    agents = {}

    # Keep your original agents
    if AGENTS_AVAILABLE:
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
                "If VaR is presented, give the confidence level and the interpretation."
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

        # NEW: MacroEconomicsAgent
        agents["MacroEconomicsAgent"] = Agent(
            model=Gemini(id="gemini-2.0-flash"),
            description="Macro strategist: interprets key economic indicators and trends.",
            instructions=[
                "Summarize latest macro data (CPI, GDP, Unemployment, Fed funds rate) and identify the current policy regime.",
                "Assess potential market impacts and list short actionable notes.",
                "Include 'Rationale:' and 'Recommendation:' sections and be explicit about data sources and dates."
            ],
            markdown=True
        )
    else:
        # placeholder so the rest of the code runs but agent.run() will be unavailable
        agents = {}

    return agents

AGENTS = create_agents()

# --------------------------- Macro & Economic Calendar Integration --------------------------- #

@st.cache_data(ttl=60 * 30)
def fetch_macro_indicators(start_date=None, end_date=None):
    """
    Attempt to fetch key macro indicators from FRED (CPIAUCSL, UNRATE, GDPC1, FEDFUNDS).
    If pandas_datareader is not available or fetching fails, return an informative dict.
    """
    indicators = {
        "CPIAUCSL": None,  # CPI-U
        "UNRATE": None,    # Unemployment Rate
        "GDPC1": None,     # Real GDP
        "FEDFUNDS": None   # Effective Federal Funds Rate
    }
    dates = {"start": start_date, "end": end_date}
    try:
        if not PANDAS_DATAREADER_AVAILABLE:
            return {"error": "pandas_datareader not installed", "indicators": indicators}
        if end_date is None:
            end_date = datetime.today()
        if start_date is None:
            start_date = end_date - timedelta(days=365*2)  # last 2 years
        # fetch each series
        for k in indicators.keys():
            try:
                ser = webreader.DataReader(k, "fred", start_date, end_date)
                indicators[k] = ser
            except Exception as e:
                indicators[k] = None
        return {"error": None, "indicators": indicators, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"error": str(e), "indicators": indicators}

@st.cache_data(ttl=60 * 10)
def fetch_economic_calendar_tradingeconomics(country="united states", days=7):
    """
    Optional: fetch upcoming economic events from TradingEconomics calendar.
    Requires TRADING_ECONOMICS_KEY env var to be set.
    Docs: https://docs.tradingeconomics.com/
    """
    key = TRADING_ECONOMICS_KEY or ""
    if not key:
        return {"error": "no_key", "events": []}
    try:
        # Trading Economics example endpoint (calendar): adjust as per API docs if needed
        # We'll use a simple events endpoint filtered by country and date range
        # Note: please set TRADING_ECONOMICS_KEY env var with your API key
        to_date = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
        from_date = datetime.utcnow().strftime("%Y-%m-%d")
        url = f"https://api.tradingeconomics.com/calendar?country={country}&start_date={from_date}&end_date={to_date}&c={key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        events = r.json()
        # normalize important fields
        processed = []
        for e in events:
            processed.append({
                "date": e.get("date"),
                "country": e.get("country"),
                "category": e.get("category"),
                "impact": e.get("importance") or e.get("impact"),
                "event": e.get("event"),
                "actual": e.get("actual"),
                "consensus": e.get("consensus"),
                "previous": e.get("previous"),
                "source": e.get("source")
            })
        return {"error": None, "events": processed}
    except Exception as exc:
        return {"error": str(exc), "events": []}

def run_macro_agent():
    """
    Compose macro prompt with fetched indicators and feed to MacroEconomicsAgent if available.
    Otherwise return a summary built from data (non-LLM fallback).
    """
    # fetch latest macro indicators (2 years)
    macro_data = fetch_macro_indicators()
    fetched_at = macro_data.get("fetched_at", None)
    indicators = macro_data.get("indicators", {})
    # build a short report string
    report_text = f"Macro snapshot (data fetched at: {fetched_at} UTC)\n\n"
    for k, ser in indicators.items():
        if ser is None:
            report_text += f"- {k}: data unavailable\n"
        else:
            # pick latest value
            last_idx = ser.dropna().index.max()
            last_val = ser.dropna().iloc[-1].values[0]
            report_text += f"- {k}: latest {last_val} (date: {last_idx.date()})\n"
    # fetch economic calendar if possible
    calendar = fetch_economic_calendar_tradingeconomics()
    if calendar.get("error") is None and calendar.get("events"):
        report_text += "\nUpcoming economic events (TradingEconomics):\n"
        for e in calendar["events"][:8]:
            report_text += f"- {e.get('date')}: {e.get('event')} ({e.get('impact')})\n"
    else:
        if calendar.get("error") == "no_key":
            report_text += "\nEconomic calendar: TradingEconomics key not provided. Set TRADING_ECONOMICS_KEY environment variable to enable.\n"
        elif calendar.get("error"):
            report_text += f"\nEconomic calendar: failed to fetch ({calendar.get('error')}).\n"

    if AGENTS_AVAILABLE and "MacroEconomicsAgent" in AGENTS:
        # craft prompt and run the LLM agent
        prompt = "You are MacroEconomicsAgent. Interpret the following macro indicator snapshot and list potential market impacts and short recommendations (Rationale:, Recommendation:):\n\n"
        prompt += report_text
        response = AGENTS["MacroEconomicsAgent"].run(prompt)
        return response.content, report_text, calendar
    else:
        # Non-LLM fallback concise interpretation logic (basic heuristics)
        fallback = []
        try:
            # CPI trend
            cpi = indicators.get("CPIAUCSL")
            if cpi is not None and not cpi.empty:
                cpi_pct = (cpi.iloc[-1].values[0] / cpi.iloc[-13].values[0] - 1) * 100 if cpi.shape[0] > 13 else None
                fallback.append(f"CPI: ~{cpi_pct:.2f}% year-over-year (approx)" if cpi_pct is not None else "CPI: insufficient history")
            # Unemployment
            un = indicators.get("UNRATE")
            if un is not None and not un.empty:
                fallback.append(f"Unemployment: {un.iloc[-1].values[0]:.2f}%")
            # Fed funds
            fed = indicators.get("FEDFUNDS")
            if fed is not None and not fed.empty:
                fallback.append(f"Fed funds rate: {fed.iloc[-1].values[0]:.2f}%")
            # GDP
            gdpc = indicators.get("GDPC1")
            if gdpc is not None and not gdpc.empty:
                fallback.append(f"Real GDP (latest): {gdpc.iloc[-1].values[0]:.2f} (level)")
        except Exception:
            pass

        simple_summary = "Macro simple summary:\n" + "\n".join(fallback) + "\n\n" + "Note: For richer interpretation install agent model packages and provide API keys."
        return simple_summary, report_text, calendar

# --------------------------- Portfolio Optimizer --------------------------- #

def optimize_mean_variance(close_df, max_weight=None, min_weight=0.0, leverage=1.0, target_return=None):
    """
    Mean-variance optimization wrapper using PyPortfolioOpt.
    - close_df: DataFrame of closes (columns = assets)
    - max_weight: scalar cap applied uniformly to all assets (e.g., 0.4). If None, no per-asset cap.
    - min_weight: minimal weight per asset
    - leverage: max sum(abs(weights)) allowed (not directly used in EfficientFrontier but can be applied via volatility target or weight scaling)
    - target_return: if provided, use EfficientFrontier with target return
    Returns dictionary of optimized weights and metadata.
    """
    if not PYPFOPT_AVAILABLE:
        raise ImportError("PyPortfolioOpt not installed. pip install pypfopt")

    # compute expected returns and sample covariance
    mu = expected_returns.mean_historical_return(close_df)
    S = risk_models.sample_cov(close_df)
    ef = EfficientFrontier(mu, S)

    # apply basic bounds
    if max_weight is not None:
        ef.add_constraint(lambda w: w <= max_weight + 1e-8)  # elementwise via lambda not ideal; will use bounds below
        bounds = (min_weight, max_weight)
    else:
        bounds = (min_weight, 1.0)

    # enforce bounds using EfficientFrontier API
    ef = EfficientFrontier(mu, S, weight_bounds=bounds)

    if target_return is not None:
        try:
            weights = ef.efficient_return(target_return)
        except Exception:
            weights = ef.max_sharpe()  # fallback
    else:
        # maximize Sharpe by default
        weights = ef.max_sharpe()

    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    return {"weights": cleaned, "performance": perf}

def optimize_black_litterman(close_df, market_caps=None, absolute_views=None, tau=0.025, max_weight=None, min_weight=0.0):
    """
    Black-Litterman optimization using PyPortfolioOpt.
    - market_caps: dict of market caps for prior (if None, uses equal)
    - absolute_views: dict {asset: expected_return} representing user's views in absolute terms
    Returns dict with weights and performance metrics.
    """
    if not PYPFOPT_AVAILABLE:
        raise ImportError("PyPortfolioOpt not installed. pip install pypfopt")
    # get expected returns and cov
    mu = expected_returns.mean_historical_return(close_df)
    S = risk_models.sample_cov(close_df)

    tickers = list(close_df.columns)
    if market_caps is None:
        # equal market caps if not provided
        market_caps = {t: 1.0 for t in tickers}
    # build market cap series in same order
    mkt_caps = np.array([market_caps.get(t, 1.0) for t in tickers], dtype=float)
    # build Black-Litterman model
    bl = BlackLittermanModel(S, pi="market", market_caps=mkt_caps, tau=tau)
    if absolute_views:
        # views as dict: {ticker: expected_return_diff} or absolute expected returns
        # PyPortfolioOpt expects Q and P; for a simple single-asset absolute view we can use view dict via BL helper
        P = []
        Q = []
        for t, val in absolute_views.items():
            if t in tickers:
                row = [1.0 if x == t else 0.0 for x in tickers]
                P.append(row)
                Q.append(val)
        if P:
            bl = BlackLittermanModel(S, pi="market", market_caps=mkt_caps, tau=tau, absolute_views=(P, Q))
    bl_mu = bl.bl_expected_returns()
    bl_cov = bl.bl_cov()

    ef = EfficientFrontier(bl_mu, bl_cov, weight_bounds=(min_weight, max_weight if max_weight is not None else 1.0))
    weights = ef.max_sharpe()
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    return {"weights": cleaned, "performance": perf, "bl_mu": bl_mu}

# --------------------------- Orchestration --------------------------- #
def run_market_agent(symbols, close_df, benchmark="^GSPC"):
    if not AGENTS_AVAILABLE or "MarketAnalystAgent" not in AGENTS:
        return "MarketAnalystAgent unavailable - AGENTS package not installed or configured."

    if close_df.empty:
        return "No price data available for market analysis."

    returns = compute_returns(close_df)
    six_month = returns.loc[returns.index >= (close_df.index.max() - pd.DateOffset(months=6))]
    perf = (close_df.loc[close_df.index.max()] / close_df.loc[close_df.index.max() - pd.Timedelta(days=180)] - 1).to_dict() if (close_df.index.max() - pd.Timedelta(days=180)) in close_df.index else (close_df.pct_change(126).iloc[-1].to_dict() if close_df.shape[0] > 126 else close_df.pct_change().sum().to_dict())
    avg_vol = returns.std() * np.sqrt(252)
    volatility = avg_vol.to_dict()

    benchmark_prices = download_close_prices([benchmark], period="1y") if benchmark else pd.DataFrame()
    benchmark_returns = compute_returns(benchmark_prices) if not benchmark_prices.empty else pd.DataFrame()

    prompt = "You are MarketAnalystAgent. Analyze the following quantitative metrics for symbols: {}\n\n".format(", ".join(close_df.columns))
    prompt += "Latest date: {}\n\n".format(close_df.index.max().strftime("%Y-%m-%d"))
    prompt += "6-month approximate returns (latest available):\n"
    for s, v in perf.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nAnnualized volatility (approx):\n"
    for s, v in volatility.items():
        prompt += f"- {s}: {v:.2%}\n"
    if not benchmark_returns.empty:
        prompt += "\nBenchmark (S&P 500) recent vol and returns included for context.\n"
    prompt += "\nInstructions: Provide a short Market Overview, identify if there is a clear market regime (risk-on / risk-off), and list top 3 signals an analyst should watch. Include explicit sections labeled 'Rationale:' and 'Recommendation:'.\n"

    response = AGENTS["MarketAnalystAgent"].run(prompt)
    return response.content

def run_company_agent(symbol):
    if not AGENTS_AVAILABLE or "CompanyResearchAgent" not in AGENTS:
        return "CompanyResearchAgent unavailable."
    info = fetch_ticker_info(symbol)
    news = fetch_news(symbol, limit=6)
    prompt = f"You are CompanyResearchAgent. Provide a concise company analysis for {symbol}.\n\n"
    prompt += f"Basic info (truncated): Name: {info.get('longName', symbol)}; Sector: {info.get('sector', 'N/A')}; Market Cap: {info.get('marketCap', 'N/A')}\n"
    prompt += "Recent news headlines:\n"
    for n in news:
        prompt += f"- {n.get('title')}\n"
    prompt += "\nInstructions: Summarize fundamentals, list top 3 catalysts and top 3 risks. Provide 'Rationale:' and 'Recommendation:' sections. Keep it concise.\n"
    response = AGENTS["CompanyResearchAgent"].run(prompt)
    return response.content

def run_sentiment_agent(symbol):
    if not AGENTS_AVAILABLE or "SentimentAgent" not in AGENTS:
        return "SentimentAgent unavailable."
    news = fetch_news(symbol, limit=12)
    if not news:
        return "No news to analyze."
    prompt = f"You are SentimentAgent. Given the following headlines for {symbol}, produce an aggregated sentiment (-1 to +1), a short summary, and include 'Rationale:' and 'Recommendation:'.\n\nHeadlines:\n"
    for n in news:
        prompt += f"- {n.get('title')}\n"
    prompt += "\nInstructions: consider tone and frequency; be explicit about uncertainty and sample size.\n"
    response = AGENTS["SentimentAgent"].run(prompt)
    return response.content

def run_risk_agent(symbols, close_df, benchmark="^GSPC"):
    if not AGENTS_AVAILABLE or "RiskAnalystAgent" not in AGENTS:
        return "RiskAnalystAgent unavailable."
    if close_df.empty:
        return "No price data for risk analysis."

    returns = compute_returns(close_df)
    var_results = {}
    mdd_results = {}
    vol_results = {}
    for s in returns.columns:
        series = returns[s]
        var_95 = historical_var(series, alpha=0.05)
        mdd = max_drawdown(close_df[s])
        vol = series.std() * np.sqrt(252)
        var_results[s] = float(var_95)
        mdd_results[s] = float(mdd)
        vol_results[s] = float(vol)

    corr = returns.corr().round(3).to_dict()

    prompt = "You are RiskAnalystAgent. Compute risk metrics and interpret them.\n\n"
    prompt += "VaR (5%) per asset (approx):\n"
    for s, v in var_results.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nMax Drawdown (most recent period):\n"
    for s, v in mdd_results.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nAnnualized vol:\n"
    for s, v in vol_results.items():
        prompt += f"- {s}: {v:.2%}\n"
    prompt += "\nCorrelation snapshot (rounded):\n"
    for s, row in corr.items():
        row_items = ", ".join([f"{k}:{v}" for k, v in row.items()])
        prompt += f"- {s}: {row_items}\n"
    prompt += "\nInstructions: Provide an interpretation, list top 3 risk concerns, provide 'Rationale:' and 'Recommendation:' sections.\n"
    response = AGENTS["RiskAnalystAgent"].run(prompt)
    return response.content

def run_portfolio_agent(symbols, close_df, constraints=None):
    # Fallback wrapper: will call the interactive optimizer logic if PYPFOPT_AVAILABLE
    if not PYPFOPT_AVAILABLE:
        return "Portfolio optimization requires PyPortfolioOpt. Please install pypfopt to enable advanced optimization."
    # Use default constraints if none
    return "Use the Portfolio Strategist tab to run optimizer interactively."

# --------------------------- Natural Language Query Interface --------------------------- #
def interpret_user_query(query, symbols, close_df):
    q = query.lower()
    if "macro" in q or "cpi" in q or "unemployment" in q:
        macro_text, raw_snapshot, calendar = run_macro_agent()
        return macro_text, {"type": "macro", "snapshot": raw_snapshot, "calendar": calendar}
    # reuse earlier routing
    if "volatility" in q or "volatilit" in q:
        rolling = rolling_volatility(compute_returns(close_df))
        if AGENTS_AVAILABLE and "MarketAnalystAgent" in AGENTS:
            agent_response = AGENTS["MarketAnalystAgent"].run(
                f"You are MarketAnalystAgent. The user asked: '{query}'. Provide an interpretation of recent volatility patterns for {', '.join(symbols)}. "
                "Include 'Rationale:' and 'Recommendation:'."
            )
            return agent_response.content, {"type": "rolling_vol", "data": rolling}
        else:
            return "MarketAnalystAgent unavailable.", {"type": "rolling_vol", "data": rolling}
    elif "sharpe" in q or "sharpe ratio" in q:
        returns = compute_returns(close_df)
        port_returns = returns.mean(axis=1)
        sr = sharpe_ratio(port_returns)
        if AGENTS_AVAILABLE and "PortfolioStrategistAgent" in AGENTS:
            agent_response = AGENTS["PortfolioStrategistAgent"].run(
                f"You are PortfolioStrategistAgent. The user asked: '{query}'. Provide interpretation of portfolio Sharpe ratio: {sr:.4f} (annualized). "
                "Include 'Rationale:' and 'Recommendation:'."
            )
            return agent_response.content, {"type": "sharpe", "value": sr}
        else:
            return f"Estimated portfolio Sharpe (annualized): {sr:.4f}", {"type": "sharpe", "value": sr}
    elif "sentiment" in q:
        sentiments = {s: run_sentiment_agent(s) for s in symbols}
        return "Sentiment summary generated for requested symbols.", {"type": "sentiments", "data": sentiments}
    else:
        if AGENTS_AVAILABLE and "TeamLeadAgent" in AGENTS:
            team_resp = AGENTS["TeamLeadAgent"].run(f"You are TeamLeadAgent. The user query: {query}. The symbols in scope are: {', '.join(symbols)}. Provide an action plan and which analyses to run. Include 'Rationale:' and 'Recommendation:'.")
            return team_resp.content, {"type": "generic", "text": team_resp.content}
        return "No agents available to interpret query. Please ensure AGENTS are configured.", {"type": "error"}

# --------------------------- Streamlit UI --------------------------- #
st.title("📊 AI Market Intelligence — Multi-Agent Decision Framework (Extended)")
st.markdown("Target audience: researchers, traders, asset managers, and risk managers.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
input_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, TSLA, GOOG")
api_key = st.sidebar.text_input("Enter your Google API Key (optional)", type="password")
benchmark = st.sidebar.text_input("Benchmark ticker (for beta/vol):", "^GSPC")
period_default = st.sidebar.selectbox("Default historical period for charts:", ["6mo", "1y", "2y"], index=1)

symbols = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
elif api_key_env:
    os.environ["GOOGLE_API_KEY"] = api_key_env

tabs = st.tabs(["Overview", "Company Deep Dives", "Risk & Correlation", "Portfolio Strategist", "Macro & Calendar", "Chat Assistant", "Audit & Exports"])

# --- Overview Tab ---
with tabs[0]:
    st.header("Market Overview & Quick Analysis")
    period = st.selectbox("Historical period for charts:", ["6mo", "1y", "2y"], index=["6mo", "1y", "2y"].index(period_default))
    close_df = download_close_prices(symbols, period=period)
    if close_df.empty:
        st.warning("No price data found for the symbols provided.")
    else:
        st.subheader("Price Chart")
        fig = go.Figure()
        for s in close_df.columns:
            fig.add_trace(go.Scatter(x=close_df.index, y=close_df[s], mode="lines", name=s))
        fig.update_layout(title=f"Price history ({period})", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Run MarketAnalystAgent"):
            with st.spinner("Running MarketAnalystAgent..."):
                market_analysis_text = run_market_agent(symbols, close_df, benchmark=benchmark)
                st.markdown("### Market Analysis (XAI)")
                st.markdown(market_analysis_text)

# --- Company Deep Dives ---
with tabs[1]:
    st.header("Company Deep Dives")
    for s in symbols:
        with st.expander(f"{s} — Summary & Research"):
            info = fetch_ticker_info(s)
            st.subheader(info.get("longName", s))
            st.write(f"Sector: {info.get('sector', 'N/A')} • MarketCap: {info.get('marketCap', 'N/A')}")
            if st.button(f"Run CompanyResearchAgent for {s}", key=f"company_{s}"):
                with st.spinner(f"Running CompanyResearchAgent for {s}..."):
                    company_text = run_company_agent(s)
                    st.markdown(company_text)
            if st.button(f"Run SentimentAgent for {s}", key=f"sent_{s}"):
                with st.spinner(f"Running SentimentAgent for {s}..."):
                    sentiment_text = run_sentiment_agent(s)
                    st.markdown(sentiment_text)
            news = fetch_news(s, limit=6)
            if news:
                st.markdown("### 📰 Latest News Headlines")
                for n in news:
                    title = n.get("title", "Untitled")
                    publisher = n.get("publisher", "Unknown Source")
                    link = n.get("link", "")
                    st.markdown(
                        f"""
                        <div style='padding:10px 15px; margin-bottom:8px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.1);'>
                            <a href='{link}' target='_blank' style='text-decoration:none; color:#0066cc; font-weight:600;'>{title}</a><br>
                            <span style='font-size:13px; color:#555;'>🗞️ {publisher}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No recent news headlines found for this company.")

# --- Risk & Correlation Tab ---
with tabs[2]:
    st.header("Risk Metrics & Correlation")
    close_df_risk = download_close_prices(symbols, period="1y")
    if close_df_risk.empty:
        st.warning("No price data for risk metrics.")
    else:
        if st.button("Run RiskAnalystAgent"):
            with st.spinner("Running RiskAnalystAgent..."):
                risk_text = run_risk_agent(symbols, close_df_risk, benchmark=benchmark)
                st.markdown("### Risk Analysis (XAI)")
                st.markdown(risk_text)

        returns = compute_returns(close_df_risk)
        if not returns.empty:
            corr = returns.corr()
            st.subheader("Return Correlation Heatmap (1y)")
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorbar=dict(title="Corr")
            ))
            fig_corr.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

        vol = rolling_volatility(returns)
        st.subheader("Rolling Volatility (21-day annualized)")
        fig_vol = go.Figure()
        for c in vol.columns:
            fig_vol.add_trace(go.Scatter(x=vol.index, y=vol[c], mode="lines", name=c))
        fig_vol.update_layout(template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)

# --- Portfolio Strategist Tab ---
with tabs[3]:
    st.header("Portfolio Strategist — Allocation Proposals (Interactive)")
    st.markdown("This tab uses PyPortfolioOpt for mean-variance and Black-Litterman optimizations. If PyPortfolioOpt is missing, please install it.")
    if not PYPFOPT_AVAILABLE:
        st.error("PyPortfolioOpt not installed. Install with: pip install pypfopt")
    close_df_port = download_close_prices(symbols, period="1y")
    if close_df_port.empty:
        st.warning("No price data available for optimization.")
    else:
        st.subheader("Optimization Inputs")
        col1, col2 = st.columns(2)
        with col1:
            max_weight = st.number_input("Max weight per asset (0-1, leave blank for no cap)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
            min_weight = st.number_input("Min weight per asset (0-1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            leverage = st.number_input("Max leverage (sum abs weights). 1.0 = no leverage", min_value=0.0, value=1.0, step=0.1)
        with col2:
            optimizer_choice = st.selectbox("Optimizer", ["Mean-Variance (Max Sharpe)", "Mean-Variance (Target Return)", "Black-Litterman (Max Sharpe)"])
            target_return = None
            if optimizer_choice == "Mean-Variance (Target Return)":
                target_return = st.number_input("Target annual return (e.g., 0.10 for 10%)", min_value=0.0, value=0.10, step=0.01)
            bl_views_raw = st.text_area("Black-Litterman views (JSON mapping ticker -> expected annual return, optional)", value="{}")
        # sector constraints (optional)
        st.markdown("**Optional:** Sector caps. Provide a JSON mapping of sector -> max_weight (e.g. {\"Technology\":0.5}). Also provide a ticker->sector mapping if desired.")
        sector_caps_raw = st.text_area("Sector caps JSON", value="{}")
        ticker_sector_raw = st.text_area("Ticker -> sector mapping JSON (e.g. {\"AAPL\":\"Technology\", \"TSLA\":\"Consumer Discretionary\"})", value="{}")

        # Run optimizer
        if st.button("Run Optimizer"):
            st.info("Running optimizer...")
            # parse JSONs safely
            try:
                bl_views = json.loads(bl_views_raw) if bl_views_raw.strip() else {}
            except Exception:
                bl_views = {}
                st.warning("Invalid Black-Litterman views JSON; ignoring.")
            try:
                sector_caps = json.loads(sector_caps_raw) if sector_caps_raw.strip() else {}
            except Exception:
                sector_caps = {}
                st.warning("Invalid sector caps JSON; ignoring.")
            try:
                ticker_sector = json.loads(ticker_sector_raw) if ticker_sector_raw.strip() else {}
            except Exception:
                ticker_sector = {}
                st.warning("Invalid ticker->sector mapping JSON; ignoring.")

            # run requested optimizer
            try:
                if optimizer_choice.startswith("Mean-Variance"):
                    res = optimize_mean_variance(close_df_port, max_weight=max_weight, min_weight=min_weight, leverage=leverage, target_return=target_return)
                elif optimizer_choice.startswith("Black-Litterman"):
                    # try to build market caps from yfinance info if possible
                    market_caps = {}
                    for t in close_df_port.columns:
                        info = fetch_ticker_info(t)
                        market_caps[t] = info.get("marketCap", 1.0) or 1.0
                    # parse BL views: expecting mapping ticker->expected_return (e.g., 0.1)
                    bl_views_parsed = {k: float(v) for k, v in bl_views.items() if k in close_df_port.columns}
                    res = optimize_black_litterman(close_df_port, market_caps=market_caps, absolute_views=bl_views_parsed, max_weight=max_weight, min_weight=min_weight)
                else:
                    res = {"error": "unknown optimizer"}
            except Exception as e:
                st.error(f"Optimization error: {e}")
                res = None

            if res:
                weights = res.get("weights", {})
                perf = res.get("performance", None)
                st.subheader("Optimized Weights")
                if not weights:
                    st.warning("No weights returned.")
                else:
                    w_df = pd.DataFrame.from_dict(weights, orient="index", columns=["weight"])
                    w_df = w_df.sort_values("weight", ascending=False)
                    st.dataframe(w_df.style.format({"weight":"{:.4f}"}))

                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(labels=w_df.index, values=w_df["weight"], hole=0.3)])
                    fig_pie.update_layout(title="Optimized Portfolio Weights")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # performance metrics (if available)
                    if perf:
                        ann_ret, ann_vol, sharpe = perf
                        st.metric("Estimated annual return", f"{ann_ret:.2%}")
                        st.metric("Estimated annual volatility", f"{ann_vol:.2%}")
                        st.metric("Estimated Sharpe (rf=0)", f"{sharpe:.3f}")

                    # apply sector caps check (simple enforcement note - advanced enforcement requires custom constraints)
                    if sector_caps and ticker_sector:
                        # calculate sector weights
                        sector_weights = {}
                        for t, w in weights.items():
                            sector = ticker_sector.get(t, "Unknown")
                            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(w)
                        st.subheader("Sector allocations (computed)")
                        s_df = pd.DataFrame.from_dict(sector_weights, orient="index", columns=["weight"]).sort_values("weight", ascending=False)
                        st.dataframe(s_df.style.format({"weight":"{:.4f}"}))
                        # check caps
                        violated = []
                        for sec, cap in sector_caps.items():
                            w = sector_weights.get(sec, 0.0)
                            if w > cap:
                                violated.append((sec, w, cap))
                        if violated:
                            st.warning("Sector caps violated by the optimized portfolio. Consider stricter constraints or run optimization with custom sector constraints.")
                            for v in violated:
                                st.write(f"- {v[0]}: weight {v[1]:.2f} > cap {v[2]:.2f}")
                        else:
                            st.success("Sector caps satisfied.")

                    # allow CSV download
                    csv = w_df.to_csv()
                    st.download_button("Download weights CSV", csv, file_name="optimized_weights.csv", mime="text/csv")

# --- Macro & Calendar Tab ---
with tabs[4]:
    st.header("Macro Snapshot & Economic Calendar")
    st.markdown("Fetch key macro indicators and optional economic calendar (TradingEconomics).")
    if st.button("Fetch Macro Snapshot & Calendar"):
        with st.spinner("Fetching macro indicators..."):
            macro_text, raw_snapshot, calendar = run_macro_agent()
            st.subheader("Macro Agent Output")
            st.markdown(macro_text)
            st.subheader("Raw Snapshot (data sources)")
            st.code(raw_snapshot)
            if calendar and calendar.get("events"):
                st.subheader("Upcoming Economic Events (TradingEconomics)")
                for e in calendar["events"][:20]:
                    st.markdown(f"- **{e.get('date')}** — {e.get('event')} ({e.get('impact')}) — {e.get('country')}")
            else:
                if calendar.get("error") == "no_key":
                    st.info("TradingEconomics API key not provided. To enable economic calendar, set TRADING_ECONOMICS_KEY environment variable and redeploy.")
                elif calendar.get("error"):
                    st.warning(f"Economic calendar fetch error: {calendar.get('error')}")

# --- Chat Assistant Tab ---
with tabs[5]:
    st.header("Natural Language Research Assistant")
    st.markdown("Ask the system — it will route to an appropriate agent and return text + visuals when applicable.")
    user_query = st.text_input("Enter your question (e.g., 'Show me volatility trends for the S&P 500 in the last year')", "")
    if st.button("Ask") and user_query.strip():
        with st.spinner("Interpreting query and running agents..."):
            close_df_chat = download_close_prices(symbols, period="1y")
            text_out, meta = interpret_user_query(user_query, symbols, close_df_chat)
            st.subheader("Assistant Response")
            st.markdown(text_out)
            if isinstance(meta, dict):
                if meta.get("type") == "rolling_vol" and "data" in meta:
                    vol_df = meta["data"]
                    st.subheader("Rolling Volatility Visual")
                    fig_rv = go.Figure()
                    for c in vol_df.columns:
                        fig_rv.add_trace(go.Scatter(x=vol_df.index, y=vol_df[c], mode="lines", name=c))
                    fig_rv.update_layout(template="plotly_white")
                    st.plotly_chart(fig_rv, use_container_width=True)
                elif meta.get("type") == "sharpe":
                    st.metric("Estimated portfolio Sharpe (annualized)", f"{meta['value']:.3f}")
                elif meta.get("type") == "sentiments":
                    st.subheader("Sentiment Agent Outputs")
                    for s, text in meta["data"].items():
                        with st.expander(s):
                            st.markdown(text)
                elif meta.get("type") == "macro":
                    st.subheader("Macro snapshot (raw)")
                    st.code(meta.get("snapshot", ""))
                    calendar = meta.get("calendar", {})
                    if calendar and calendar.get("events"):
                        st.subheader("Upcoming events")
                        for e in calendar["events"][:10]:
                            st.markdown(f"- {e.get('date')}: {e.get('event')} ({e.get('impact')})")

# --- Audit & Exports Tab ---
with tabs[6]:
    st.header("Audit Trail & Report Generation")
    st.markdown("Run individual agents and then request an integrated report from TeamLeadAgent.")
    if st.button("Run full multi-agent orchestration and generate TeamLead report"):
        with st.spinner("Running agents and compiling report..."):
            close_for_run = download_close_prices(symbols, period="1y")
            date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            market_analysis = run_market_agent(symbols, close_for_run, benchmark=benchmark) if AGENTS_AVAILABLE else "Market agent unavailable"
            company_analyses = {s: run_company_agent(s) for s in symbols} if AGENTS_AVAILABLE else {}
            sentiment_analyses = {s: run_sentiment_agent(s) for s in symbols} if AGENTS_AVAILABLE else {}
            risk_analysis = run_risk_agent(symbols, close_for_run, benchmark=benchmark) if AGENTS_AVAILABLE else "Risk agent unavailable"
            portfolio_recommendation = run_portfolio_agent(symbols, close_for_run)
            if AGENTS_AVAILABLE and "TeamLeadAgent" in AGENTS:
                final_report = run_teamlead_agent(date_str, market_analysis, company_analyses, sentiment_analyses, risk_analysis, portfolio_recommendation)
                st.markdown("### TeamLead Consolidated Report")
                st.markdown(final_report)
            else:
                st.markdown("Agents not configured; here are the pieces gathered:")
                st.subheader("Market Analysis")
                st.markdown(market_analysis)
                st.subheader("Risk Analysis")
                st.markdown(risk_analysis)
                st.subheader("Portfolio Recommendation")
                st.markdown(portfolio_recommendation)

    st.markdown("---")
    st.markdown("**Notes & Limitations**")
    st.markdown("""
    - Agents include explicit `Rationale:` and `Recommendation:` for XAI/auditability if configured.\n
    - For production, add immutable audit logging (store prompts, responses, timestamps) to a secure DB.\n
    - Economic calendar is optional and requires TradingEconomics API key. FRED series are used for macro time series where possible.\n
    - Portfolio optimization uses PyPortfolioOpt; ensure realistic constraints and verify model outputs before trading.\n
    """)

# --------------------------- Helper: TeamLead aggregator wrapper --------------------------- #
def run_teamlead_agent(date_str, market_analysis, company_analyses, sentiment_analyses, risk_analysis, portfolio_recommendation):
    if not AGENTS_AVAILABLE or "TeamLeadAgent" not in AGENTS:
        return "TeamLeadAgent unavailable."
    prompt = (
        f"You are TeamLeadAgent. Today's date is {date_str}.\n"
        f"Integrate these sections into a clear, time-stamped investment report. Use the following inputs:\n\n"
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses:\n{company_analyses}\n\n"
        f"Sentiment Analyses:\n{sentiment_analyses}\n\n"
        f"Risk Analysis:\n{risk_analysis}\n\n"
        f"Portfolio Recommendation:\n{portfolio_recommendation}\n\n"
        "Instructions: Provide a short Executive Summary (2-4 sentences), Top 3 actionable recommendations with 'Rationale:' for each, and an Audit Trail (which agents produced the inputs). Keep the report structured with explicit headings.\n"
    )
    response = AGENTS["TeamLeadAgent"].run(prompt)
    return response.content

# EOF
