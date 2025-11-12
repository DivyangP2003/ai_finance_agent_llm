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
    Runs MarketAnalystAgent on given symbols. Returns agent content text.
    We'll provide the agent with key metrics precomputed and ask it to interpret them.
    """
    if close_df.empty:
        return "No price data available for market analysis."

    returns = compute_returns(close_df)
    six_month = returns.loc[returns.index >= (close_df.index.max() - pd.DateOffset(months=6))]
    perf = (close_df.loc[close_df.index.max()] / close_df.loc[close_df.index.max() - pd.Timedelta(days=180)] - 1).to_dict() if (close_df.index.max() - pd.Timedelta(days=180)) in close_df.index else (close_df.pct_change(126).iloc[-1].to_dict() if close_df.shape[0] > 126 else close_df.pct_change().sum().to_dict())
    avg_vol = returns.std() * np.sqrt(252)
    volatility = avg_vol.to_dict()

    benchmark_prices = download_close_prices([benchmark], period="1y") if benchmark else pd.DataFrame()
    benchmark_returns = compute_returns(benchmark_prices) if not benchmark_prices.empty else pd.DataFrame()

    # Prepare a concise prompt with key metrics and let the agent produce markdown with rationale & recommendation
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

    # run the agent
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
    # For each symbol compute VaR (historical 5%), max drawdown, rolling vol, and correlations
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

    # Prepare prompt for the risk analyst agent
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
    # only include top-level pairs to keep prompt short
    for s, row in corr.items():
        row_items = ", ".join([f"{k}:{v}" for k, v in row.items()])
        prompt += f"- {s}: {row_items}\n"
    prompt += "\nInstructions: Provide an interpretation, list top 3 risk concerns, provide 'Rationale:' and 'Recommendation:' sections.\n"
    response = AGENTS["RiskAnalystAgent"].run(prompt)
    return response.content

def run_portfolio_agent(symbols, close_df, constraints=None):
    """
    Portfolio strategist: propose naive allocations based on equal-weight, risk-parity proxy,
    or momentum tilt. For production, integrate PyPortfolioOpt or Black-Litterman.
    We'll provide a few candidate allocations and ask the agent to pick and justify one.
    """
    returns = compute_returns(close_df)
    # simple candidate allocations:
    n = len(symbols)
    equal = {s: round(1/n, 4) for s in symbols}
    # risk-parity proxy: weight inversely proportional to vol
    vols = returns.std() * np.sqrt(252)
    invvol = (1 / vols)
    invvol = invvol / invvol.sum()
    invvol_alloc = invvol.to_dict()
    # momentum tilt: 6-month returns normalized
    six_month = close_df.pct_change(126).iloc[-1] if close_df.shape[0] > 130 else close_df.pct_change().iloc[-1]
    momentum = six_month.clip(lower=-1).fillna(0)
    if momentum.sum() <= 0:
        momentum_alloc = equal.copy()
    else:
        mom = (momentum + 0.0001) / (momentum.sum() + 0.0001)
        momentum_alloc = mom.to_dict()

    prompt = f"You are PortfolioStrategistAgent. Consider the following allocation candidates for symbols: {', '.join(symbols)}.\n\n"
    prompt += "Candidate allocations:\n"
    prompt += f"- Equal-weight: {equal}\n"
    prompt += f"- Inverse-vol (risk-parity proxy): { {k: round(v,4) for k,v in invvol_alloc.items()} }\n"
    prompt += f"- Momentum-tilt: { {k: round(v,4) for k,v in momentum_alloc.items()} }\n"
    prompt += "\nConstraints (if any): {}\n".format(constraints or "None")
    prompt += "Instructions: Select the most suitable allocation for a moderately risk-tolerant institutional investor, justify selection in 'Rationale:' and 'Recommendation:' sections, and give a 2-line rebalancing guideline.\n"
    response = AGENTS["PortfolioStrategistAgent"].run(prompt)
    return response.content

def run_teamlead_agent(date_str, market_analysis, company_analyses, sentiment_analyses, risk_analysis, portfolio_recommendation):
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
st.markdown("Target audience: researchers, traders, asset managers, and risk managers.")

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

tabs = st.tabs(["Overview", "Company Deep Dives", "Risk & Correlation", "Portfolio Strategist", "Chat Assistant", "Audit & Exports"])

# --- Overview Tab ---
with tabs[0]:
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
            st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{'_'.join(symbols)}")  # ✅ FIXED

        vol = rolling_volatility(returns)
        st.subheader("Rolling Volatility (21-day annualized)")
        fig_vol = go.Figure()
        for c in vol.columns:
            fig_vol.add_trace(go.Scatter(x=vol.index, y=vol[c], mode="lines", name=c))
        fig_vol.update_layout(template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True, key=f"rolling_vol_{'_'.join(symbols)}")  # ✅ FIXED

# --- Portfolio Strategist Tab ---
with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
    st.header("Audit Trail & Report Generation")
    st.markdown("Run individual agents and then request an integrated report from TeamLeadAgent.")
    if st.button("Run full multi-agent orchestration and generate TeamLead report"):
        with st.spinner("Running agents and compiling report..."):
            close_for_run = download_close_prices(symbols, period="1y")
            date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            market_analysis = run_market_agent(symbols, close_for_run, benchmark=benchmark)
            company_analyses = {s: run_company_agent(s) for s in symbols}
            sentiment_analyses = {s: run_sentiment_agent(s) for s in symbols}
            risk_analysis = run_risk_agent(symbols, close_for_run, benchmark=benchmark)
            portfolio_recommendation = run_portfolio_agent(symbols, close_for_run)
            final_report = run_teamlead_agent(date_str, market_analysis, company_analyses, sentiment_analyses, risk_analysis, portfolio_recommendation)
            st.markdown("### TeamLead Consolidated Report")
            st.markdown(final_report)

    st.markdown("---")
    st.markdown("**Notes & Limitations**")
    st.markdown("""
    - Agents include explicit `Rationale:` and `Recommendation:` for XAI/auditability.\n
    - For production, add immutable audit logging (store prompts, responses, timestamps).\n
    - Sentiment currently uses Yahoo headlines; integrate institutional feeds for accuracy.\n
    - Portfolio optimization is illustrative; integrate PyPortfolioOpt / Black-Litterman for real optimization.\n
    """)
