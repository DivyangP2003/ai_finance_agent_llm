# app.py
import os
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import feedparser
# --- AI agent imports ---
from agno.agent import Agent
from agno.models.google import Gemini
from fpdf import FPDF


# --------------------------- Setup --------------------------- #
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="AI Market Intelligence (Multi-Agent)", page_icon="📊", layout="wide")

# --------------------------- Helper: caching --------------------------- #
@st.cache_data(ttl=60 * 15)
def download_close_prices(symbols, period="1y"):
    """Download historical Close prices for given symbols."""
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

    return agents

AGENTS = create_agents()

# --------------------------- Orchestration --------------------------- #
def run_market_agent(symbols, close_df, benchmark="^GSPC"):
    """
    Enhanced version:
    Includes benchmark returns, volatility, and relative comparisons.
    """
    if close_df.empty:
        return "No price data available for market analysis."

    returns = compute_returns(close_df)
    six_month = returns.loc[returns.index >= (close_df.index.max() - pd.DateOffset(months=6))]

    # Compute recent performance and volatility
    perf = (close_df.iloc[-1] / close_df.iloc[max(0, len(close_df) - 126)] - 1).to_dict()
    avg_vol = returns.std() * np.sqrt(252)
    volatility = avg_vol.to_dict()

    # --- Benchmark Integration ---
    benchmark_prices = download_close_prices([benchmark], period="1y") if benchmark else pd.DataFrame()
    benchmark_returns = compute_returns(benchmark_prices) if not benchmark_prices.empty else pd.DataFrame()
    bench_ret = bench_vol = np.nan
    betas = {}

    if not benchmark_returns.empty:
        bench_ret = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0] - 1).iloc[0]
        bench_vol = benchmark_returns.std().iloc[0] * np.sqrt(252)
        for s in returns.columns:
            betas[s] = compute_beta(returns[s], benchmark_returns.iloc[:, 0])
    else:
        for s in returns.columns:
            betas[s] = np.nan

    # --- Build the prompt ---
    prompt = f"You are MarketAnalystAgent. Analyze the following quantitative metrics for symbols: {', '.join(close_df.columns)}.\n\n"
    prompt += f"Latest date: {close_df.index.max().strftime('%Y-%m-%d')}\n\n"

    prompt += "6-month returns (approx):\n"
    for s, v in perf.items():
        prompt += f"- {s}: {v:.2%}\n"

    prompt += "\nAnnualized volatility (approx):\n"
    for s, v in volatility.items():
        prompt += f"- {s}: {v:.2%}\n"

    if not np.isnan(bench_ret):
        prompt += f"\nBenchmark ({benchmark}) 1Y return: {bench_ret:.2%}, volatility: {bench_vol:.2%}\n"
        prompt += "Betas (vs benchmark):\n"
        for s, b in betas.items():
            if not np.isnan(b):
                prompt += f"- {s}: {b:.2f}\n"

    prompt += (
        "\nInstructions: Provide a short Market Overview, identify if there is a clear market regime "
        "(risk-on / risk-off), discuss any benchmark-relative patterns, and list top 3 signals to monitor. "
        "Include explicit sections labeled 'Rationale:' and 'Recommendation:'.\n"
    )

    response = AGENTS["MarketAnalystAgent"].run(prompt)
    return response.content


def run_company_agent(symbol):
    info = fetch_ticker_info(symbol)
    news = fetch_news(symbol, limit=6)
    # limit length of summary so the agent doesn't get huge payloads
    prompt = f"You are CompanyResearchAgent. Provide a concise company analysis for {symbol}.\n\n"
    prompt += f"Basic info (truncated): Name: {info.get('longName', symbol)}; Sector: {info.get('sector', 'N/A')}; Market Cap: {info.get('marketCap', 'N/A')}\n"
    prompt += "Recent news headlines:\n"
    for n in news:
        prompt += f"- {n.get('title')}\n"
    prompt += "\nInstructions: Summarize fundamentals, list top 3 catalysts and top 3 risks. Provide 'Rationale:' and 'Recommendation:' sections. Keep it concise.\n"
    response = AGENTS["CompanyResearchAgent"].run(prompt)
    return response.content

def run_sentiment_agent(symbol):
    news = fetch_news(symbol, limit=12)
    # Provide headlines to the SentimentAgent and ask for aggregated sentiment
    if not news:
        return "No news to analyze."
    prompt = f"You are SentimentAgent. Given the following headlines for {symbol}, produce an aggregated sentiment (-1 to +1), a short summary, and include 'Rationale:' and 'Recommendation:'.\n\nHeadlines:\n"
    for n in news:
        prompt += f"- {n.get('title')}\n"
    prompt += "\nInstructions: consider tone and frequency; be explicit about uncertainty and sample size.\n"
    response = AGENTS["SentimentAgent"].run(prompt)
    return response.content

def run_risk_agent(symbols, close_df, benchmark="^GSPC"):
    """
    Enhanced version:
    Adds benchmark-based VaR, correlation, and betas.
    """
    if close_df.empty:
        return "No price data for risk analysis."

    returns = compute_returns(close_df)
    var_results, mdd_results, vol_results = {}, {}, {}

    for s in returns.columns:
        series = returns[s]
        var_95 = historical_var(series, alpha=0.05)
        mdd = max_drawdown(close_df[s])
        vol = series.std() * np.sqrt(252)
        var_results[s] = float(var_95)
        mdd_results[s] = float(mdd)
        vol_results[s] = float(vol)

    # --- Benchmark Integration ---
    bench_stats = ""
    if benchmark:
        bench_prices = download_close_prices([benchmark], period="1y")
        if not bench_prices.empty:
            bench_returns = compute_returns(bench_prices)
            bench_var = historical_var(bench_returns.iloc[:, 0], alpha=0.05)
            bench_vol = bench_returns.std().iloc[0] * np.sqrt(252)
            bench_dd = max_drawdown(bench_prices.iloc[:, 0])
            bench_stats = (
                f"\nBenchmark ({benchmark}) VaR(5%): {bench_var:.2%}, "
                f"Vol: {bench_vol:.2%}, Max DD: {bench_dd:.2%}\n"
            )

    # --- Beta & correlation with benchmark ---
    beta_results = {}
    if benchmark and not bench_prices.empty:
        bench_returns = compute_returns(bench_prices)
        for s in returns.columns:
            beta_results[s] = compute_beta(returns[s], bench_returns.iloc[:, 0])
    else:
        for s in returns.columns:
            beta_results[s] = np.nan

    corr = returns.corr().round(3).to_dict()

    # --- Prompt build ---
    prompt = "You are RiskAnalystAgent. Compute and interpret risk metrics.\n\n"
    prompt += "VaR (5%) per asset:\n"
    for s, v in var_results.items():
        prompt += f"- {s}: {v:.2%}\n"

    prompt += "\nMax Drawdown:\n"
    for s, v in mdd_results.items():
        prompt += f"- {s}: {v:.2%}\n"

    prompt += "\nAnnualized volatility:\n"
    for s, v in vol_results.items():
        prompt += f"- {s}: {v:.2%}\n"

    if bench_stats:
        prompt += bench_stats
        prompt += "Betas (vs benchmark):\n"
        for s, b in beta_results.items():
            if not np.isnan(b):
                prompt += f"- {s}: {b:.2f}\n"

    prompt += "\nCorrelation snapshot (rounded):\n"
    for s, row in corr.items():
        row_items = ", ".join([f"{k}:{v}" for k, v in row.items()])
        prompt += f"- {s}: {row_items}\n"

    prompt += (
        "\nInstructions: Interpret these risk metrics, highlight benchmark-relative exposures (e.g., "
        "beta > 1 means higher market sensitivity), list top 3 risk concerns, and include 'Rationale:' "
        "and 'Recommendation:' sections.\n"
    )

    response = AGENTS["RiskAnalystAgent"].run(prompt)
    return response.content

def run_portfolio_agent(symbols, close_df, benchmark="^GSPC", constraints=None):
    """
    Enhanced benchmark-aware portfolio strategist.
    Provides equal-weight, risk-parity, and momentum-tilt allocations,
    and includes benchmark-relative context (return, volatility, beta).
    """

    if close_df.empty:
        return "No data available for portfolio construction."

    returns = compute_returns(close_df)

    # --- Simple candidate allocations ---
    n = len(symbols)
    equal = {s: round(1 / n, 4) for s in symbols}

    vols = returns.std() * np.sqrt(252)
    invvol = (1 / vols)
    invvol = invvol / invvol.sum()
    invvol_alloc = invvol.round(4).to_dict()

    # momentum tilt (6-month)
    six_month = close_df.pct_change(126).iloc[-1] if close_df.shape[0] > 126 else close_df.pct_change().iloc[-1]
    momentum = six_month.clip(lower=-1).fillna(0)
    if momentum.sum() <= 0:
        momentum_alloc = equal.copy()
    else:
        mom = (momentum + 0.0001) / (momentum.sum() + 0.0001)
        momentum_alloc = mom.round(4).to_dict()

    # --- Benchmark Integration ---
    bench_stats = ""
    if benchmark:
        bench_prices = download_close_prices([benchmark], period="1y")
        if not bench_prices.empty:
            bench_returns = compute_returns(bench_prices)
            bench_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0] - 1).iloc[0]
            bench_vol = bench_returns.std().iloc[0] * np.sqrt(252)
            bench_stats = f"Benchmark ({benchmark}) return: {bench_ret:.2%}, volatility: {bench_vol:.2%}"
        else:
            bench_stats = f"Benchmark ({benchmark}) data unavailable."

    # --- Build the prompt ---
    prompt = (
        f"You are PortfolioStrategistAgent.\n\n"
        f"Symbols in scope: {', '.join(symbols)}\n"
        f"{bench_stats}\n\n"
        "Candidate allocations:\n"
        f"- Equal-weight: {equal}\n"
        f"- Inverse-vol (risk-parity proxy): {invvol_alloc}\n"
        f"- Momentum-tilt: {momentum_alloc}\n\n"
        f"Constraints (if any): {constraints or 'None'}\n\n"
        "Instructions:\n"
        "- Choose the most suitable allocation for a moderately risk-tolerant institutional investor.\n"
        "- Consider benchmark-relative performance and potential tracking error.\n"
        "- If active risk is acceptable, propose small benchmark tilts (e.g., overweight momentum sectors).\n"
        "- Explain your reasoning with 'Rationale:' and 'Recommendation:' sections.\n"
        "- End with a 2-line guideline on rebalancing frequency.\n"
    )

    response = AGENTS["PortfolioStrategistAgent"].run(prompt)
    return response.content

def run_teamlead_agent(
    date_str,
    market_analysis,
    company_analyses,
    sentiment_analyses,
    risk_analysis,
    portfolio_recommendation,
    benchmark="^GSPC"
):
    """
    Enhanced TeamLeadAgent that produces a detailed, sectioned institutional report.
    """

    # --- Fetch simple benchmark stats ---
    bench_prices = download_close_prices([benchmark], period="1y")
    if not bench_prices.empty:
        bench_returns = compute_returns(bench_prices)
        bench_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0] - 1).iloc[0]
        bench_vol = bench_returns.std().iloc[0] * np.sqrt(252)
        bench_summary = f"Benchmark ({benchmark}) return: {bench_ret:.2%}, volatility: {bench_vol:.2%}"
    else:
        bench_summary = f"Benchmark ({benchmark}) data unavailable."

    # --- Flatten dicts for readability ---
    def flatten_dict(d):
        if isinstance(d, dict):
            return "\n".join([f"**{k}:** {v}" for k, v in d.items()])
        return str(d)

    # --- Rich Prompt ---
    prompt = f"""
You are **TeamLeadAgent**, the senior portfolio strategist and integrator.
Today's date: {date_str}.

**Benchmark Context:** {bench_summary}

Integrate the following agent outputs into a detailed institutional-style investment report.
Include clear markdown headings for each section.

---

### Agent Inputs

**Market Analysis:**
{market_analysis}

**Company Analyses:**
{flatten_dict(company_analyses)}

**Sentiment Analyses:**
{flatten_dict(sentiment_analyses)}

**Risk Analysis:**
{risk_analysis}

**Portfolio Recommendation:**
{portfolio_recommendation}

---

### Report Instructions

Create a **comprehensive and audit-ready investment report** with these sections:

1. **Executive Summary**  
   - 3–4 sentences summarizing overall market tone, benchmark-relative performance, and key takeaways.

2. **Market & Benchmark Overview**  
   - Summarize the market environment (risk-on/off), benchmark performance, volatility, and notable macro signals.

3. **Company Deep Dives**  
   - Highlight each company's fundamentals, catalysts, and risks (based on sub-agent inputs).

4. **Sentiment Insights**  
   - Summarize sentiment tone, key drivers, and overall behavioral bias (bullish/bearish/neutral).

5. **Risk Assessment**  
   - Interpret VaR, max drawdown, and correlation results.  
   - Explicitly state benchmark-relative exposures (e.g., beta > 1, sector concentration).

6. **Portfolio Strategy**  
   - Describe recommended allocation, rationale for each tilt, and benchmark-relative positioning (active vs passive stance).  
   - Include a short note on rebalancing or tactical hedges.

7. **Top 3 Actionable Recommendations**  
   - List each with a bold heading and **Rationale:** below.

8. **Audit Trail**  
   - Table of which agents contributed each section (MarketAnalystAgent, RiskAnalystAgent, etc.).  
   - Keep this concise but clear for governance/audit purposes.

Use a professional tone, markdown formatting, and numbered sections.
Avoid repeating raw data — interpret and summarize concisely.
    """

    response = AGENTS["TeamLeadAgent"].run(prompt)
    return response.content

# --------------------------- Natural Language Query Interface --------------------------- #
def interpret_user_query(query, symbols, close_df):
    """
    Simple router: sample common queries -> run relevant agent(s).
    For complex open-ended queries, send to TeamLeadAgent or MarketAnalystAgent to interpret and return steps.
    """
    q = query.lower()

    # Hard-coded routing for common queries
    if "volatility" in q or "volatilit" in q:
        # show rolling volatility & ask market agent to interpret
        rolling = rolling_volatility(compute_returns(close_df))
        # prepare visual separately in Streamlit
        agent_response = AGENTS["MarketAnalystAgent"].run(
            f"You are MarketAnalystAgent. The user asked: '{query}'. Provide an interpretation of recent volatility patterns for {', '.join(symbols)}. "
            "Include 'Rationale:' and 'Recommendation:'."
        )
        return agent_response.content, {"type": "rolling_vol", "data": rolling}
    elif "sharpe" in q or "sharpe ratio" in q:
        # compute portfolio-level sharpe assuming equal-weight unless user provides holdings
        returns = compute_returns(close_df)
        # portfolio returns equal-weight
        port_returns = returns.mean(axis=1)
        sr = sharpe_ratio(port_returns)
        agent_response = AGENTS["PortfolioStrategistAgent"].run(
            f"You are PortfolioStrategistAgent. The user asked: '{query}'. Provide interpretation of portfolio Sharpe ratio: {sr:.4f} (annualized). "
            "Include 'Rationale:' and 'Recommendation:'."
        )
        return agent_response.content, {"type": "sharpe", "value": sr}
    elif "sentiment" in q:
        # run sentiment agent for each symbol and summarize
        sentiments = {s: run_sentiment_agent(s) for s in symbols}
        return "Sentiment summary generated for requested symbols.", {"type": "sentiments", "data": sentiments}
    else:
        # fallback: let teamlead try to interpret and produce guidance
        team_resp = AGENTS["TeamLeadAgent"].run(f"You are TeamLeadAgent. The user query: {query}. The symbols in scope are: {', '.join(symbols)}. Provide an action plan and which analyses to run. Include 'Rationale:' and 'Recommendation:'.")
        return team_resp.content, {"type": "generic", "text": team_resp.content}

# --------------------------- Streamlit UI --------------------------- #
st.title("📊 AI Market Intelligence — Multi-Agent Decision Framework")
# Sidebar
st.sidebar.header("⚙️ Configuration")
input_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, TSLA, GOOG")
api_key = st.sidebar.text_input("Enter your Google API Key (optional)", type="password")
benchmark = st.sidebar.text_input("Benchmark ticker (for beta/vol):", "^GSPC")

symbols = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
elif api_key_env:
    os.environ["GOOGLE_API_KEY"] = api_key_env

tabs = st.tabs(["User Guide", "Overview", "Company Deep Dives", "Risk & Correlation", "Portfolio Strategist", "Chat Assistant", "Audit & Exports"])

# --- User Guide Tab ---
with tabs[0]:
    st.title("📘 AI Market Intelligence — Comprehensive User Guide")
    st.markdown("""
    Welcome to **AI Market Intelligence**, a multi-agent financial analysis system powered by Gemini AI.  
    This platform helps you understand markets, companies, risks, and optimal portfolio allocations — even if you have **no prior finance background**.

    ---
    """)

    st.header("🎯 What This App Does")
    st.markdown("""
    The app uses a set of specialized AI agents to analyze stocks, markets, and portfolios.  
    Each agent focuses on one domain — quantitative metrics, company fundamentals, sentiment, or risk — and the **TeamLeadAgent** combines them into a single benchmark-aware investment report.

    **Here’s what happens when you run the full orchestration:**
    - Data is downloaded from Yahoo Finance (prices, returns, volatility)
    - News headlines are analyzed for tone and bias
    - AI agents run independent analyses (Market, Risk, Sentiment, etc.)
    - The TeamLeadAgent synthesizes all insights into a professional report
    """)

    st.subheader("👥 Agents and Their Roles")
    st.markdown("""
    | Agent | Description | Output Example |
    |--------|--------------|----------------|
    | **MarketAnalystAgent** | Quantitative analyst — studies prices, trends, volatility, market regime. | “Market regime appears risk-on with tech outperforming.” |
    | **CompanyResearchAgent** | Company fundamentals & financial health. | “AAPL has strong margins and stable earnings growth.” |
    | **SentimentAgent** | Evaluates news tone and sentiment from headlines. | “Media sentiment mildly positive (+0.22).” |
    | **RiskAnalystAgent** | Measures risk (VaR, drawdown, beta, correlation). | “TSLA shows high beta (1.4) and largest drawdown.” |
    | **PortfolioStrategistAgent** | Suggests allocations (equal-weight, risk-parity, momentum-tilt). | “Allocate 40% AAPL, 35% GOOG, 25% TSLA.” |
    | **TeamLeadAgent** | Integrates all insights into a detailed investment report. | “Portfolio outperformed S&P 500 with lower volatility.” |
    """)

    st.markdown("---")

    st.header("⚙️ How to Use the App")
    st.markdown("""
    1. **Enter stock symbols** in the sidebar — e.g.:
       - `AAPL` (Apple)
       - `TSLA` (Tesla)
       - `GOOG` (Alphabet/Google)
       - You can enter multiple tickers separated by commas.
    2. **Select a benchmark** — a market index to compare performance against:
       - Default: `^GSPC` → S&P 500 (broad U.S. market)
       - Others: `^DJI` (Dow Jones), `^NDX` (Nasdaq 100), `^IXIC` (Nasdaq Composite)
    3. **Explore Tabs:**
       - **Overview:** Historical prices & MarketAnalystAgent.
       - **Company Deep Dives:** Company research & sentiment.
       - **Risk & Correlation:** Risk metrics, drawdown, volatility heatmaps.
       - **Portfolio Strategist:** Allocation recommendations.
       - **Chat Assistant:** Ask natural language queries.
       - **Audit & Exports:** Generate benchmark-aware reports.
    """)

    st.markdown("---")

    st.header("💬 Key Financial Terminologies (Plain English Guide)")
    st.markdown("""
    ### 🔢 Stock Market Basics
    - **Stock / Equity:** Ownership share in a company.
    - **Ticker Symbol:** Short code to identify a stock (e.g., `AAPL` = Apple).
    - **Index / Benchmark:** A collection of stocks used to represent the market (e.g., S&P 500).
    - **ETF (Exchange-Traded Fund):** A fund that tracks an index (like `SPY` for S&P 500).
    - **Price:** The latest traded value of a stock.
    - **Return:** The percentage change in price over a period.

    ### 📊 Market & Performance Metrics
    - **Volatility:** How much prices fluctuate. High = risky, Low = stable.
    - **Standard Deviation:** The math measure behind volatility.
    - **Drawdown:** The percentage fall from a recent peak — measures loss severity.
    - **Beta (β):** Sensitivity to market moves. β > 1 = more volatile than market, β < 1 = less volatile.
    - **Alpha (α):** Return in excess of the benchmark.
    - **Sharpe Ratio:** Risk-adjusted performance = (Return - Risk-free rate) / Volatility.
    - **VaR (Value at Risk):** Worst expected loss (e.g., “5% VaR = can lose 3% or more 5% of the time”).
    - **Max Drawdown:** Largest observed drop in value — a stress test of performance.

    ### 💰 Company & Fundamental Terms
    - **Market Cap:** Total company value = price × shares outstanding.
    - **Earnings Per Share (EPS):** Profit per share — measures profitability.
    - **P/E Ratio (Price-to-Earnings):** Valuation measure. High P/E → expensive stock.
    - **Revenue Growth:** Increase in company sales over time.
    - **Margins:** How much profit is kept from each dollar of sales.
    - **Cash Flow:** Real cash generated — shows business strength.
    - **Debt-to-Equity Ratio:** Measures leverage (how much debt the company has).
    - **Dividend Yield:** Annual dividend / price — investor income measure.

    ### 💬 Behavioral & Sentiment Terms
    - **Market Sentiment:** Overall tone (bullish = optimistic, bearish = pessimistic).
    - **Headline Sentiment:** News tone score from -1 (negative) to +1 (positive).
    - **Catalyst:** Event that might move prices (e.g., earnings release, product launch).
    - **Momentum:** Recent trend strength — stocks going up tend to keep rising (short term).

    ### ⚖️ Portfolio & Risk Terms
    - **Diversification:** Holding different assets to reduce risk.
    - **Correlation:** How assets move relative to each other (-1 = opposite, +1 = together).
    - **Risk Parity:** Balancing portfolio risk by inverse volatility.
    - **Equal Weight:** Every stock gets the same percentage.
    - **Momentum Tilt:** Overweight stocks that are trending upward.
    - **Hedging:** Using offsetting assets to protect against loss.
    - **Tracking Error:** How much a portfolio deviates from its benchmark.
    - **Benchmark-relative Return:** Outperformance or underperformance vs. the benchmark.

    ### 🏦 Economic & Market Context
    - **Risk-on Environment:** Investors prefer stocks and higher risk assets.
    - **Risk-off Environment:** Investors seek safety (bonds, gold, cash).
    - **Interest Rates:** Cost of borrowing money — affects valuations.
    - **Inflation:** Rate at which prices increase — reduces purchasing power.
    - **Yield Curve:** Shows interest rates for bonds of different maturities.
    - **Recession:** Period of declining economic activity.
    - **Market Regime:** Overall condition of market (bullish, bearish, volatile, stable).

    ### 🧠 AI & Analytical Concepts
    - **Agent:** An autonomous AI process that performs a specific analytical role.
    - **Multi-Agent System:** Several AI agents collaborating — like departments in a research team.
    - **Prompt:** The instructions given to each AI model (e.g., “Analyze volatility trends”).
    - **Explainability (XAI):** Making AI reasoning transparent and auditable.
    """)

    st.markdown("---")

    st.header("📘 How to Interpret AI Reports")
    st.markdown("""
    Every report includes structured sections for clarity:

    | Section | What It Means |
    |----------|---------------|
    | **Executive Summary** | High-level overview — performance, tone, and trends. |
    | **Market & Benchmark Overview** | Context on how markets and your stocks performed vs benchmark. |
    | **Company Deep Dives** | Company fundamentals and financial analysis. |
    | **Sentiment Insights** | News tone and behavioral signals. |
    | **Risk Assessment** | VaR, drawdown, volatility, and correlation results. |
    | **Portfolio Strategy** | AI’s allocation recommendation with rationale. |
    | **Recommendations** | Actionable buy/hold/sell or weighting guidance. |
    | **Audit Trail** | Lists which AI agent produced which section. |

    Each **Rationale** section explains *why* the recommendation was made.  
    Each **Recommendation** tells you *what* to do (e.g., overweight, hold, reduce exposure).
    """)

    st.markdown("---")

    st.header("🔍 Examples of Benchmarks You Can Use")
    st.markdown("""
    - `^GSPC` — S&P 500 (broad U.S. market)  
    - `^DJI` — Dow Jones Industrial Average  
    - `^NDX` — Nasdaq 100 (tech-heavy)  
    - `^IXIC` — Nasdaq Composite  
    - `^RUT` — Russell 2000 (small caps)  
    - `^FTSE` — UK FTSE 100  
    - `^N225` — Nikkei 225 (Japan)  
    - `^HSI` — Hang Seng Index (Hong Kong)  
    - `^STOXX50E` — Euro Stoxx 50 (Europe)
    """)

    st.markdown("---")

    st.header("🧠 AI Model Notes")
    st.markdown("""
    - The app uses **Google Gemini 2.0** models for reasoning.  
    - Each agent uses tailored prompts to focus on its domain.
    - The **TeamLeadAgent** merges analyses into a human-readable markdown report.
    - All outputs are auditable — sections are labeled by the agent that created them.
    """)

    st.markdown("---")

    st.header("💡 Tips for Beginners")
    st.markdown("""
    - Hover over terms or open this guide if you’re unsure about terminology.  
    - Start with **one stock** and learn how each agent’s report changes.  
    - Compare your stock to benchmarks like `^GSPC` or `^NDX` to understand context.  
    - Use the **Chat Assistant** tab to ask things like:
      - “What is volatility?”
      - “How has AAPL’s performance compared to the S&P 500?”
      - “Which stock has the lowest drawdown?”
    - You can generate **short** or **detailed** reports — both are benchmark-aware.
    """)

    st.success("✅ You’re all set! Start with the 'Overview' tab to explore your first analysis.")


# --- Overview Tab ---
with tabs[1]:
    st.header("Market Overview & Quick Analysis")
    period = st.selectbox("Historical period for charts:", ["6mo", "1y", "2y"], index=1)
    close_df = download_close_prices(symbols, period=period)
    if close_df.empty:
        st.warning("No price data found for the symbols provided.")
    else:
        st.subheader("Price Chart")
        fig = go.Figure()
        for s in close_df.columns:
            fig.add_trace(go.Scatter(x=close_df.index, y=close_df[s], mode="lines", name=s))
        fig.update_layout(title=f"Price history ({period})", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{period}_{'_'.join(symbols)}")  # ✅ FIXED

        if st.button("Run MarketAnalystAgent"):
            with st.spinner("Running MarketAnalystAgent..."):
                market_analysis_text = run_market_agent(symbols, close_df, benchmark=benchmark)
                st.markdown("### Market Analysis (XAI)")
                st.markdown(market_analysis_text)

# --- Company Deep Dives ---
with tabs[2]:
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
                    # Create a clickable card-like display
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
with tabs[3]:
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
            st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{'_'.join(symbols)}")  # ✅ FIXED

        vol = rolling_volatility(returns)
        st.subheader("Rolling Volatility (21-day annualized)")
        fig_vol = go.Figure()
        for c in vol.columns:
            fig_vol.add_trace(go.Scatter(x=vol.index, y=vol[c], mode="lines", name=c))
        fig_vol.update_layout(template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True, key=f"rolling_vol_{'_'.join(symbols)}")  # ✅ FIXED

# --- Portfolio Strategist Tab ---
with tabs[4]:
    st.header("Portfolio Strategist — Allocation Proposals")
    close_df_port = download_close_prices(symbols, period="1y")
    constraints_raw = st.text_input("Constraints (json) e.g. {'max_weight':0.4}", "")
    try:
        constraints = eval(constraints_raw) if constraints_raw else None
    except Exception:
        constraints = None
    if st.button("Run PortfolioStrategistAgent"):
        with st.spinner("Running PortfolioStrategistAgent..."):
            strategy_text = run_portfolio_agent(symbols, close_df_port, constraints=constraints)
            st.markdown("### Allocation Recommendation (XAI)")
            st.markdown(strategy_text)

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
                    st.plotly_chart(fig_rv, use_container_width=True, key=f"chat_vol_{'_'.join(symbols)}")  # ✅ FIXED
                elif meta.get("type") == "sharpe":
                    st.metric("Estimated portfolio Sharpe (annualized)", f"{meta['value']:.3f}")
                elif meta.get("type") == "sentiments":
                    st.subheader("Sentiment Agent Outputs")
                    for s, text in meta["data"].items():
                        with st.expander(s):
                            st.markdown(text)

# --- Audit & Exports Tab ---
with tabs[6]:
    st.header("Audit Trail & Report Generation")
    st.markdown("Run all agents together, integrate results, and generate a benchmark-aware TeamLead report.")

    if st.button("Run full multi-agent orchestration and generate TeamLead report"):
        with st.spinner("Running all agents and compiling final report..."):
            # Download full-year close data
            close_for_run = download_close_prices(symbols, period="1y")
            date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")

            # --- Run sub-agents with benchmark awareness ---
            market_analysis = run_market_agent(symbols, close_for_run, benchmark=benchmark)
            company_analyses = {s: run_company_agent(s) for s in symbols}
            sentiment_analyses = {s: run_sentiment_agent(s) for s in symbols}
            risk_analysis = run_risk_agent(symbols, close_for_run, benchmark=benchmark)
            portfolio_recommendation = run_portfolio_agent(symbols, close_for_run, benchmark=benchmark)

            # --- Generate benchmark-aware final report ---
            final_report = run_teamlead_agent(
                date_str=date_str,
                market_analysis=market_analysis,
                company_analyses=company_analyses,
                sentiment_analyses=sentiment_analyses,
                risk_analysis=risk_analysis,
                portfolio_recommendation=portfolio_recommendation,
                benchmark=benchmark
            )

            # --- Display results ---
            st.markdown("### 🧠 TeamLead Consolidated Report (Benchmark-Aware)")
            st.markdown(final_report)

    st.markdown("---")

