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
import statsmodels.api as sm
import markdown as md
from typing import Dict, Tuple


# --------------------------- Setup --------------------------- #
load_dotenv()
api_key_env = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(page_title="AI Market Intelligence (Multi-Agent)", page_icon="📊", layout="wide")

# --------------------------- Country and Benchmark Mapping --------------------------- #
COUNTRY_BENCHMARKS = {
    # --- North America ---
    "United States": {
        "default": "^GSPC",
        "region": "US",
        "lang": "en-US",
        "benchmarks": {
            "S&P 500": {"ticker": "^GSPC", "exchange_suffix": ""},
            "Dow Jones Industrial Average": {"ticker": "^DJI", "exchange_suffix": ""},
            "Nasdaq 100": {"ticker": "^NDX", "exchange_suffix": ""},
            "Russell 2000": {"ticker": "^RUT", "exchange_suffix": ""},
            "Nasdaq Composite": {"ticker": "^IXIC", "exchange_suffix": ""}
        }
    },
    "Canada": {
        "default": "^GSPTSE",
        "region": "CA",
        "lang": "en-CA",
        "benchmarks": {
            "S&P/TSX Composite": {"ticker": "^GSPTSE", "exchange_suffix": ".TO"},
            "S&P/TSX 60": {"ticker": "^TX60", "exchange_suffix": ".TO"}
        }
    },

    # --- Latin America ---
    "Brazil": {
        "default": "^BVSP",
        "region": "BR",
        "lang": "pt-BR",
        "benchmarks": {
            "Bovespa Index (Ibovespa)": {"ticker": "^BVSP", "exchange_suffix": ".SA"},
            "IBrX 50": {"ticker": "^IBX50", "exchange_suffix": ".SA"}
        }
    },
    "Mexico": {
        "default": "^MXX",
        "region": "MX",
        "lang": "es-MX",
        "benchmarks": {
            "IPC (Bolsa Index)": {"ticker": "^MXX", "exchange_suffix": ".MX"}
        }
    },
    "Argentina": {
        "default": "^MERV",
        "region": "AR",
        "lang": "es-AR",
        "benchmarks": {
            "MERVAL": {"ticker": "^MERV", "exchange_suffix": ".BA"}
        }
    },

    # --- Europe ---
    "United Kingdom": {
        "default": "^FTSE",
        "region": "UK",
        "lang": "en-GB",
        "benchmarks": {
            "FTSE 100": {"ticker": "^FTSE", "exchange_suffix": ".L"},
            "FTSE 250": {"ticker": "^FTMC", "exchange_suffix": ".L"},
            "FTSE All-Share": {"ticker": "^FTAS", "exchange_suffix": ".L"}
        }
    },
    "Germany": {
        "default": "^GDAXI",
        "region": "DE",
        "lang": "de-DE",
        "benchmarks": {
            "DAX": {"ticker": "^GDAXI", "exchange_suffix": ".DE"},
            "MDAX": {"ticker": "^MDAXI", "exchange_suffix": ".DE"},
            "TecDAX": {"ticker": "^TECDAX", "exchange_suffix": ".DE"}
        }
    },
    "France": {
        "default": "^FCHI",
        "region": "FR",
        "lang": "fr-FR",
        "benchmarks": {
            "CAC 40": {"ticker": "^FCHI", "exchange_suffix": ".PA"},
            "SBF 120": {"ticker": "^SBF120", "exchange_suffix": ".PA"}
        }
    },
    "Italy": {
        "default": "^FTSEMIB",
        "region": "IT",
        "lang": "it-IT",
        "benchmarks": {
            "FTSE MIB": {"ticker": "^FTSEMIB", "exchange_suffix": ".MI"}
        }
    },
    "Spain": {
        "default": "^IBEX",
        "region": "ES",
        "lang": "es-ES",
        "benchmarks": {
            "IBEX 35": {"ticker": "^IBEX", "exchange_suffix": ".MC"}
        }
    },
    "Switzerland": {
        "default": "^SSMI",
        "region": "CH",
        "lang": "de-CH",
        "benchmarks": {
            "SMI (Swiss Market Index)": {"ticker": "^SSMI", "exchange_suffix": ".SW"}
        }
    },
    "Netherlands": {
        "default": "^AEX",
        "region": "NL",
        "lang": "nl-NL",
        "benchmarks": {
            "AEX": {"ticker": "^AEX", "exchange_suffix": ".AS"}
        }
    },
    "Sweden": {
        "default": "^OMXS30",
        "region": "SE",
        "lang": "sv-SE",
        "benchmarks": {
            "OMX Stockholm 30": {"ticker": "^OMXS30", "exchange_suffix": ".ST"}
        }
    },
    "Europe (Overall)": {
        "default": "^STOXX50E",
        "region": "EU",
        "lang": "en-EU",
        "benchmarks": {
            "Euro Stoxx 50": {"ticker": "^STOXX50E", "exchange_suffix": ""},
            "STOXX Europe 600": {"ticker": "^STOXX", "exchange_suffix": ""}
        }
    },

    # --- Asia ---
    "India": {
        "default": "^BSESN",
        "region": "IN",
        "lang": "en-IN",
        "benchmarks": {
            "BSE Sensex": {"ticker": "^BSESN", "exchange_suffix": ".BO"},
            "Nifty 50": {"ticker": "^NSEI", "exchange_suffix": ".NS"},
            "Nifty Bank": {"ticker": "^NSEBANK", "exchange_suffix": ".NS"}
        }
    },
    "Japan": {
        "default": "^N225",
        "region": "JP",
        "lang": "ja-JP",
        "benchmarks": {
            "Nikkei 225": {"ticker": "^N225", "exchange_suffix": ".T"},
            "TOPIX": {"ticker": "^TOPX", "exchange_suffix": ".T"}
        }
    },
    "China": {
        "default": "000001.SS",
        "region": "CN",
        "lang": "zh-CN",
        "benchmarks": {
            "SSE Composite Index": {"ticker": "000001.SS", "exchange_suffix": ".SS"},
            "CSI 300": {"ticker": "000300.SS", "exchange_suffix": ".SS"},
            "SZSE Component Index": {"ticker": "399001.SZ", "exchange_suffix": ".SZ"}
        }
    },
    "Hong Kong": {
        "default": "^HSI",
        "region": "HK",
        "lang": "zh-HK",
        "benchmarks": {
            "Hang Seng Index": {"ticker": "^HSI", "exchange_suffix": ".HK"},
            "Hang Seng Tech Index": {"ticker": "^HSTECH", "exchange_suffix": ".HK"}
        }
    },
    "South Korea": {
        "default": "^KS11",
        "region": "KR",
        "lang": "ko-KR",
        "benchmarks": {
            "KOSPI Composite": {"ticker": "^KS11", "exchange_suffix": ".KS"},
            "KOSDAQ": {"ticker": "^KQ11", "exchange_suffix": ".KQ"}
        }
    },
    "Taiwan": {
        "default": "^TWII",
        "region": "TW",
        "lang": "zh-TW",
        "benchmarks": {
            "TAIEX": {"ticker": "^TWII", "exchange_suffix": ".TW"}
        }
    },
    "Singapore": {
        "default": "^STI",
        "region": "SG",
        "lang": "en-SG",
        "benchmarks": {
            "Straits Times Index": {"ticker": "^STI", "exchange_suffix": ".SI"}
        }
    },
    "Australia": {
        "default": "^AXJO",
        "region": "AU",
        "lang": "en-AU",
        "benchmarks": {
            "ASX 200": {"ticker": "^AXJO", "exchange_suffix": ".AX"},
            "All Ordinaries": {"ticker": "^AORD", "exchange_suffix": ".AX"}
        }
    },

    # --- Middle East ---
    "Saudi Arabia": {
        "default": "^TASI",
        "region": "SA",
        "lang": "ar-SA",
        "benchmarks": {
            "Tadawul All Share Index": {"ticker": "^TASI", "exchange_suffix": ".SA"}
        }
    },
    "United Arab Emirates": {
        "default": "^DFMGI",
        "region": "AE",
        "lang": "ar-AE",
        "benchmarks": {
            "DFM General Index (Dubai)": {"ticker": "^DFMGI", "exchange_suffix": ".DU"},
            "ADX General (Abu Dhabi)": {"ticker": "^ADI", "exchange_suffix": ".AD"}
        }
    },
    "Qatar": {
        "default": "^QSI",
        "region": "QA",
        "lang": "ar-QA",
        "benchmarks": {
            "QE General": {"ticker": "^QSI", "exchange_suffix": ".QA"}
        }
    },

    # --- Africa ---
    "South Africa": {
        "default": "^J203",
        "region": "ZA",
        "lang": "en-ZA",
        "benchmarks": {
            "FTSE/JSE Top 40": {"ticker": "^J200", "exchange_suffix": ".JO"},
            "FTSE/JSE All Share": {"ticker": "^J203", "exchange_suffix": ".JO"}
        }
    },
}


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
def fetch_news(symbol, country="United States", limit=10):
    """
    Fetch recent Yahoo Finance news headlines via RSS feed.
    Region and language settings are dynamically fetched from COUNTRY_BENCHMARKS.
    """
    # Fallback to US if country not in mapping
    country_info = COUNTRY_BENCHMARKS.get(country, COUNTRY_BENCHMARKS["United States"])
    region = country_info.get("region", "US")
    lang = country_info.get("lang", "en-US")
    if "." not in symbol:
        # Append exchange suffix if known from selected benchmark
        try:
            # Use the *first* benchmark entry as representative for this country
            first_bench = next(iter(country_info["benchmarks"].values()))
            suffix = first_bench.get("exchange_suffix", "")
            if suffix:
                symbol += suffix
        except Exception:
            pass  # fallback quietly if structure changes

    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region={region}&lang={lang}"
        feed = feedparser.parse(url)

        processed = []
        for entry in feed.entries[:limit]:
            processed.append({
                "title": entry.title,
                "publisher": entry.get("source", "Yahoo Finance"),
                "link": entry.link
            })

        # Fallback to US feed if empty (for exotic tickers)
        if not processed and country != "United States":
            fallback_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            feed = feedparser.parse(fallback_url)
            processed = [{
                "title": e.title,
                "publisher": e.get("source", "Yahoo Finance"),
                "link": e.link
            } for e in feed.entries[:limit]]

        return processed

    except Exception as e:
        st.warning(f"⚠️ Failed to fetch RSS for {symbol} ({country}): {e}")
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

# --------------------------- Advanced Quantitative Analytics --------------------------- #
def compute_alpha_tracking_error(port_returns: pd.Series, bench_returns: pd.Series, risk_free_rate=0.0) -> Dict[str, float]:
    merged = pd.concat([port_returns, bench_returns], axis=1).dropna()
    if merged.shape[0] < 2:
        return {"beta": np.nan, "alpha_annual": np.nan, "tracking_error_annual": np.nan}
    y = merged.iloc[:, 0].values  # portfolio
    x = merged.iloc[:, 1].values  # benchmark
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha_daily = model.params[0]
    beta = model.params[1]
    mean_p = port_returns.mean() * 252
    mean_b = bench_returns.mean() * 252
    alpha_annual = (mean_p - risk_free_rate) - beta * (mean_b - risk_free_rate)
    tracking_err_annual = ((port_returns - bench_returns).std() * np.sqrt(252))
    return {"beta": float(beta), "alpha_annual": float(alpha_annual), "tracking_error_annual": float(tracking_err_annual)}

def drawdown_series(price: pd.Series) -> pd.DataFrame:
    price = price.dropna()
    peak = price.cummax()
    drawdown = (price - peak) / peak
    return pd.DataFrame({"price": price, "peak": peak, "drawdown": drawdown})

def plot_drawdown(price: pd.Series, title="Drawdown Chart"):
    dd = drawdown_series(price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd["price"], name="Price", yaxis="y1", mode="lines"))
    fig.add_trace(go.Scatter(x=dd.index, y=dd["drawdown"], name="Drawdown", yaxis="y2", fill="tozeroy", mode="lines"))
    fig.update_layout(
        title=title,
        xaxis=dict(domain=[0,1]),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right", tickformat=".0%", range=[dd["drawdown"].min()*1.1, 0]),
        legend=dict(orientation="h")
    )
    return fig

def compute_sector_exposure(symbols: list, weights: Dict[str, float] = None) -> Dict[str, float]:
    sector_weights = {}
    if not symbols:
        return sector_weights
    if weights is None:
        weights = {s: 1.0/len(symbols) for s in symbols}
    for s in symbols:
        try:
            info = fetch_ticker_info(s)
            sector = info.get("sector") or info.get("industry") or "Unknown"
        except Exception:
            sector = "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weights.get(s, 0.0)
    return sector_weights

def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window=63, symbol: str = "") -> pd.Series:
    """
    Compute rolling beta (OLS slope) between asset and benchmark returns.
    window: number of days (default 63 ≈ 3 months)
    symbol: optional label for easier plotting
    """
    merged = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if merged.shape[0] < window:
        return pd.Series(dtype=float)
    a = merged.iloc[:, 0]
    b = merged.iloc[:, 1]

    # Use rolling covariance/variance
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    beta = cov / var
    beta.name = f"{symbol}_rolling_beta_{window}d" if symbol else f"rolling_beta_{window}d"
    return beta


def expected_shortfall(returns: pd.Series, alpha=0.05):
    r = returns.dropna()
    if r.empty:
        return np.nan
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())

def stress_test_shock(symbols: list, returns_df: pd.DataFrame, benchmark_returns: pd.Series, shock_pct: float, betas: dict = None):
    results = {}
    if betas is None:
        betas = {}
    for s in returns_df.columns:
        b = betas.get(s, np.nan)
        if not np.isnan(b):
            est = b * shock_pct
        else:
            corr = returns_df[s].corr(benchmark_returns) if not benchmark_returns.empty else np.nan
            est = corr * shock_pct if not pd.isna(corr) else np.nan
        results[s] = float(est) if not pd.isna(est) else np.nan
    return results

def stress_test_historical_analogs(symbols: list, close_df: pd.DataFrame, bench_prices: pd.Series, window=30, top_n=3):
    if bench_prices is None or bench_prices.empty or close_df.empty:
        return []
    bench = bench_prices.dropna()
    # compute rolling window cumulative change (percent) ending at each index
    pct_change = bench.pct_change(periods=window).dropna()
    candidates = pct_change.nsmallest(top_n)
    scenarios = []
    bench_idx = bench.index
    for end_idx, drop in candidates.items():
        try:
            end_loc = bench_idx.get_loc(end_idx)
            start_loc = max(0, end_loc - window)
            start = bench_idx[start_loc]
            end = end_idx
            asset_drop = {}
            for s in close_df.columns:
                try:
                    series = close_df[s].loc[start:end]
                    cum = (series.iloc[-1] / series.iloc[0] - 1.0) if len(series) > 1 else np.nan
                    asset_drop[s] = float(cum) if not pd.isna(cum) else np.nan
                except Exception:
                    asset_drop[s] = np.nan
            scenarios.append({"start": start, "end": end, "bench_drop": float(drop), "asset_drops": asset_drop})
        except Exception:
            continue
    return scenarios

def sortino_ratio(returns: pd.Series, risk_free_rate=0.0, target=0.0):
    r = returns.dropna()
    if r.empty:
        return np.nan
    excess = r - target
    mean_ann = excess.mean() * 252
    downside = r[r < target]
    if downside.empty:
        return np.nan
    downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(252)
    if downside_dev == 0:
        return np.nan
    return float(mean_ann / downside_dev)
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
def run_market_agent(symbols, close_df, benchmark="^GSPC", country="United States"):
    """
    MarketAnalystAgent — now region-aware.
    Includes benchmark, volatility, and market regime comparisons with explicit country context.
    """
    if close_df.empty:
        return f"No price data available for market analysis in {country}."

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
    prompt = f"""
You are **MarketAnalystAgent**, analyzing equities in the **{country}** market.

Benchmark selected: **{benchmark}**  
Latest available data: {close_df.index.max().strftime('%Y-%m-%d')}

### Quantitative Overview
6-month returns (approx):
"""
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
        "\nInstructions:\n"
        "- Provide a short **Market Overview** focused on the {country} context.\n"
        "- Identify if the market regime appears **risk-on** or **risk-off**.\n"
        "- Discuss benchmark-relative trends and volatility.\n"
        "- Mention any regional factors (e.g., monetary policy, sector dominance).\n"
        "- Include clear sections labeled **'Rationale:'** and **'Recommendation:'**."
    )

    response = AGENTS["MarketAnalystAgent"].run(prompt)
    return response.content


def run_company_agent(symbol, country="United States", benchmark="^GSPC"):
    """
    CompanyResearchAgent — region-aware version.
    Adds country and benchmark context to company fundamentals analysis.
    """

    info = fetch_ticker_info(symbol)
    news = fetch_news(symbol, limit=6)

    # --- Prompt Building ---
    company_name = info.get("longName", symbol)
    sector = info.get("sector", "N/A")
    market_cap = info.get("marketCap", "N/A")
    currency = info.get("currency", "USD")

    prompt = f"""
You are **CompanyResearchAgent**, analyzing **{company_name} ({symbol})** in the **{country}** market.

Benchmark selected for comparison: **{benchmark}**  
Primary currency: **{currency}**  

### Company Overview
- Name: {company_name}  
- Sector: {sector}  
- Market Cap: {market_cap}

### Recent Headlines
"""
    for n in news:
        prompt += f"- {n.get('title', 'Untitled')}\n"

    prompt += f"""
### Instructions:
Provide a concise, professional company analysis contextualized for the **{country}** market.
If relevant, compare company performance, valuation, or growth trajectory versus the **{benchmark}** benchmark or sector peers.

Your output must include:
- **Fundamental summary:** Financial health, revenue trends, profitability, margins.
- **Regional context:** Economic or policy factors affecting the firm (e.g., local interest rates, regulation, or trade exposure).
- **Catalysts:** Top 3 upcoming events or drivers.
- **Risks:** Top 3 key risk factors (macroeconomic, operational, or valuation).
- **Rationale:** Explain your reasoning succinctly.
- **Recommendation:** Clear investment view (e.g., Overweight, Hold, Underweight).
- **Note:** Highlight if this company is a major component or outlier relative to the benchmark ({benchmark}).
"""

    response = AGENTS["CompanyResearchAgent"].run(prompt)
    return response.content


def run_sentiment_agent(symbol, country="United States", benchmark="^GSPC"):
    """
    SentimentAgent — region- and benchmark-aware version.
    Analyzes recent news headlines in the context of the selected country and benchmark.
    """

    news = fetch_news(symbol, limit=12)
    if not news:
        return f"No news found for {symbol} in the {country} market."

    # --- Build Prompt ---
    prompt = f"""
You are **SentimentAgent**, analyzing media and market sentiment for **{symbol}** in the **{country}** market.

Benchmark for contextual tone: **{benchmark}**

### News Headlines to Analyze
"""
    for n in news:
        prompt += f"- {n.get('title', 'Untitled')}\n"

    prompt += f"""
### Instructions:
Analyze the **tone**, **bias**, and **frequency** of sentiment in these headlines — considering the **{country}** media and investor environment.

Provide:
1. **Aggregated Sentiment Score:**  
   A numeric score between **-1 (very negative)** and **+1 (very positive)**.  
   Be explicit about uncertainty and sample size.

2. **Qualitative Summary:**  
   - Overall tone (bullish, bearish, or neutral).  
   - Whether sentiment aligns or diverges from the **{benchmark}** market mood.  
   - Note any **regional context** (e.g., regulation, macro policy, sector bias).

3. **Rationale:**  
   - Explain key factors driving the sentiment (e.g., product launches, earnings results, policy changes).  
   - Identify if tone is **short-term noise** or **structural**.

4. **Recommendation:**  
   - Summarize if sentiment supports **Buy**, **Hold**, or **Reduce** positioning.  
   - If relevant, relate it to benchmark sentiment (“positive relative to S&P 500 tone”).

Return the answer in clear markdown with sections:
**Sentiment Score**, **Summary**, **Rationale**, and **Recommendation**.
"""

    response = AGENTS["SentimentAgent"].run(prompt)
    return response.content

def run_risk_agent(symbols, close_df, benchmark="^GSPC", country="United States"):
    """
    RiskAnalystAgent — region- and benchmark-aware version.
    Computes and interprets risk metrics (VaR, volatility, drawdown, correlation, beta)
    in the context of the selected country's market and benchmark.
    """
    if close_df.empty:
        return f"No price data available for risk analysis in the {country} market."

    returns = compute_returns(close_df)
    var_results, mdd_results, vol_results = {}, {}, {}

    # --- Compute individual asset risk metrics ---
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
    beta_results = {}
    if benchmark:
        bench_prices = download_close_prices([benchmark], period="1y")
        if not bench_prices.empty:
            bench_returns = compute_returns(bench_prices)
            bench_var = historical_var(bench_returns.iloc[:, 0], alpha=0.05)
            bench_vol = bench_returns.std().iloc[0] * np.sqrt(252)
            bench_dd = max_drawdown(bench_prices.iloc[:, 0])
            bench_name = benchmark
            bench_stats = (
                f"\nBenchmark ({bench_name}) — VaR(5%): {bench_var:.2%}, "
                f"Volatility: {bench_vol:.2%}, Max Drawdown: {bench_dd:.2%}\n"
            )

            # Compute betas vs benchmark
            for s in returns.columns:
                beta_results[s] = compute_beta(returns[s], bench_returns.iloc[:, 0])
        else:
            for s in returns.columns:
                beta_results[s] = np.nan
    else:
        for s in returns.columns:
            beta_results[s] = np.nan

    # --- Correlation Matrix ---
    corr = returns.corr().round(3).to_dict()

    # --- Prompt Building ---
    prompt = f"""
You are **RiskAnalystAgent**, analyzing portfolio and security-level risk in the **{country}** market.

Selected benchmark: **{benchmark}**

### Asset Risk Metrics
VaR (5%) per asset:
"""
    for s, v in var_results.items():
        prompt += f"- {s}: {v:.2%}\n"

    prompt += "\nMax Drawdown (peak-to-trough):\n"
    for s, v in mdd_results.items():
        prompt += f"- {s}: {v:.2%}\n"

    prompt += "\nAnnualized Volatility (σ):\n"
    for s, v in vol_results.items():
        prompt += f"- {s}: {v:.2%}\n"

    # Include benchmark metrics if available
    if bench_stats:
        prompt += f"\n### Benchmark Risk Summary\n{bench_stats}"
        prompt += "\n### Beta (vs Benchmark):\n"
        for s, b in beta_results.items():
            if not np.isnan(b):
                prompt += f"- {s}: {b:.2f}\n"

    prompt += "\n### Correlation Matrix (rounded):\n"
    for s, row in corr.items():
        row_items = ", ".join([f"{k}:{v}" for k, v in row.items()])
        prompt += f"- {s}: {row_items}\n"

    # --- Region-specific Instructions ---
    prompt += f"""
### Instructions:
Interpret these risk metrics for the **{country}** market context:
- Discuss whether market volatility, drawdown, and VaR are high or low relative to the **{benchmark}**.
- Explain if risk levels suggest a **risk-on** or **risk-off** environment regionally.
- Highlight any **benchmark-relative exposures** (e.g., beta > 1 → more sensitive than the benchmark).
- Mention **correlation clusters** or **sector concentration** effects if visible.
- Note regional influences (e.g., policy shifts, currency volatility, geopolitical risks).

Structure your answer as:
**Summary**, **Rationale**, and **Recommendation**.
"""

    response = AGENTS["RiskAnalystAgent"].run(prompt)
    return response.content


def run_portfolio_agent(symbols, close_df, benchmark="^GSPC", constraints=None, country="United States"):
    """
    PortfolioStrategistAgent — region- and benchmark-aware version.
    Suggests allocations (equal-weight, risk-parity, momentum-tilt) contextualized for the selected country.
    """

    if close_df.empty:
        return f"No data available for portfolio construction in the {country} market."

    returns = compute_returns(close_df)

    # --- Simple candidate allocations ---
    n = len(symbols)
    equal = {s: round(1 / n, 4) for s in symbols}

    vols = returns.std() * np.sqrt(252)
    invvol = (1 / vols)
    invvol = invvol / invvol.sum()
    invvol_alloc = invvol.round(4).to_dict()

    # --- Momentum-tilt (6-month) ---
    six_month = close_df.pct_change(126).iloc[-1] if close_df.shape[0] > 126 else close_df.pct_change().iloc[-1]
    momentum = six_month.clip(lower=-1).fillna(0)
    if momentum.sum() <= 0:
        momentum_alloc = equal.copy()
    else:
        mom = (momentum + 0.0001) / (momentum.sum() + 0.0001)
        momentum_alloc = mom.round(4).to_dict()

    # --- Benchmark stats ---
    bench_stats = ""
    if benchmark:
        bench_prices = download_close_prices([benchmark], period="1y")
        if not bench_prices.empty:
            bench_returns = compute_returns(bench_prices)
            bench_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0] - 1).iloc[0]
            bench_vol = bench_returns.std().iloc[0] * np.sqrt(252)
            bench_stats = f"Benchmark ({benchmark}) — Return: {bench_ret:.2%}, Volatility: {bench_vol:.2%}"
        else:
            bench_stats = f"Benchmark ({benchmark}) data unavailable for {country}."

    # --- Build AI Prompt ---
    prompt = f"""
You are **PortfolioStrategistAgent**, designing an optimal multi-asset allocation strategy for the **{country}** market.

Selected Benchmark: **{benchmark}**  
{bench_stats}

### Portfolio Universe
Symbols in scope: {', '.join(symbols)}

### Candidate Allocations
- Equal-weight: {equal}
- Inverse-vol (risk-parity proxy): {invvol_alloc}
- Momentum-tilt (6M trend): {momentum_alloc}

### Constraints
{constraints or "None"}

### Instructions:
Formulate an allocation recommendation for a **moderately risk-tolerant institutional investor** in the **{country}** market.
Include benchmark-relative and regional insights:
- Discuss how the proposed portfolio might **track or deviate** from the **{benchmark}** (tracking error).
- If **active risk** is acceptable, identify small benchmark **tilts** (e.g., overweight tech, underweight cyclicals).
- Consider **local factors** (e.g., market liquidity, currency risk, macro policy, or regional concentration).
- Reference any **diversification** benefits visible in volatility or correlation patterns.
- If relevant, mention **cross-market hedging** or **ETF substitutes**.
- Explain your logic in two sections:
  - **Rationale:** Why this allocation suits current regional conditions.
  - **Recommendation:** Specific weights and guidance (e.g., rebalance quarterly, tilt toward growth sectors).

Output format:
**Summary**, **Rationale**, **Recommendation**, **Rebalancing Note** (2 lines).
"""

    response = AGENTS["PortfolioStrategistAgent"].run(prompt)
    return response.content


def run_teamlead_agent(
    date_str,
    market_analysis,
    company_analyses,
    sentiment_analyses,
    risk_analysis,
    portfolio_recommendation,
    benchmark="^GSPC",
    country="United States"
):
    """
    TeamLeadAgent — region- and benchmark-aware integrator.
    Produces a cohesive, country-contextual institutional report combining all sub-agent outputs.
    """

    # --- Fetch benchmark stats ---
    bench_prices = download_close_prices([benchmark], period="1y")
    if not bench_prices.empty:
        bench_returns = compute_returns(bench_prices)
        bench_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0] - 1).iloc[0]
        bench_vol = bench_returns.std().iloc[0] * np.sqrt(252)
        bench_summary = f"Benchmark ({benchmark}) — 1Y Return: {bench_ret:.2%}, Volatility: {bench_vol:.2%}"
    else:
        bench_summary = f"Benchmark ({benchmark}) data unavailable for {country}."

    # --- Helper to flatten dicts (Company & Sentiment outputs) ---
    def flatten_dict(d):
        if isinstance(d, dict):
            return "\n".join([f"**{k}:**\n{v}" for k, v in d.items()])
        return str(d)

    # --- Build AI Prompt ---
    prompt = f"""
You are **TeamLeadAgent**, the senior portfolio strategist and integrator.

Today's Date: {date_str}  
**Country/Region:** {country}  
**Selected Benchmark:** {benchmark}  
**Benchmark Summary:** {bench_summary}

Integrate the following agent outputs into a cohesive, benchmark-aware investment report for the **{country}** market.

---

### Agent Inputs

**Market Analysis (MarketAnalystAgent):**
{market_analysis}

**Company Analyses (CompanyResearchAgent):**
{flatten_dict(company_analyses)}

**Sentiment Analyses (SentimentAgent):**
{flatten_dict(sentiment_analyses)}

**Risk Analysis (RiskAnalystAgent):**
{risk_analysis}

**Portfolio Recommendation (PortfolioStrategistAgent):**
{portfolio_recommendation}

---

### Report Instructions

Produce a **comprehensive, professional, and benchmark-aware investment report** tailored for an institutional audience in the **{country}** market.

Your report must include the following sections in Markdown:

1. **Executive Summary**  
   - Summarize market tone, benchmark-relative performance, and key macro trends.  
   - Highlight whether the {country} market is in a **risk-on** or **risk-off** phase.

2. **Market & Benchmark Overview**  
   - Interpret the {benchmark} benchmark’s performance (return, volatility, trends).  
   - Discuss how analyzed stocks compare with the benchmark.  
   - Mention any relevant regional macro or policy context (interest rates, inflation, regulation).

3. **Company Deep Dives**  
   - Summarize key company fundamentals and catalysts.  
   - Contrast them with regional sector peers where relevant.

4. **Sentiment Insights**  
   - Aggregate tone from news and investor sentiment in the {country} market.  
   - Note if sentiment aligns or diverges from the benchmark’s mood.

5. **Risk Assessment**  
   - Interpret VaR, volatility, drawdowns, and betas relative to the {benchmark}.  
   - Identify key systemic and idiosyncratic risks specific to the {country} market.  
   - Include regional influences (e.g., currency swings, policy shocks, or commodity exposure).

6. **Portfolio Strategy & Allocation**  
   - Discuss recommended weights (equal, risk-parity, momentum).  
   - Explain regional tilts or deviations from the {benchmark}.  
   - Provide a concise **Rebalancing Note** (e.g., “Quarterly or on 20% volatility shift”).

7. **Top 3 Actionable Recommendations**  
   - Each with a **bold heading** and a **Rationale** section.  
   - Recommendations should align with the country’s market tone and benchmark dynamics.

8. **Audit Trail**  
   - Include a brief table showing which agents contributed each section.  
   - Use bullet points (e.g., “Market Analysis — MarketAnalystAgent”).

---

### Output Requirements:
- Write in **professional Markdown** suitable for institutional or client presentation.
- Avoid repetition; summarize key insights clearly.
- All quantitative references must be benchmark-relative (vs {benchmark}) when applicable.
- Keep tone objective, concise, and audit-friendly.
- Begin with a line stating: “**AI-Generated Institutional Market Report for {country} — Benchmark: {benchmark}**”.
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

def normalize_symbols(symbols, country, exchange_suffix=""):
    """
    Adds the correct exchange suffix to stock symbols based on the selected
    country and benchmark. Avoids duplication if suffix already present.
    """
    normalized = []
    for s in symbols:
        s = s.strip().upper()
        if not s:
            continue

        # Skip if already has a dot or caret (^ = index)
        if "." not in s and "^" not in s and exchange_suffix:
            s += exchange_suffix

        normalized.append(s)
    return normalized

# --------------------------- Streamlit UI --------------------------- #
st.title("📊 AI Market Intelligence — Multi-Agent Decision Framework")
st.sidebar.header("⚙️ Configuration")

# --- Country Selection ---
selected_country = st.sidebar.selectbox(
    "🌍 Select Country/Region",
    options=list(COUNTRY_BENCHMARKS.keys()),
    index=0
)

# --- Dynamic Benchmark Selection ---
country_info = COUNTRY_BENCHMARKS[selected_country]

benchmark_display = st.sidebar.selectbox(
    f"📈 Select Benchmark for {selected_country}",
    options=list(country_info["benchmarks"].keys()),
    index=0
)

benchmark_info = country_info["benchmarks"][benchmark_display]
benchmark = benchmark_info["ticker"]
exchange_suffix = benchmark_info.get("exchange_suffix", "")

# --- Stock input and API key ---
input_symbols = st.sidebar.text_input(
    f"Enter Stock Symbols ({selected_country} Market Codes)",
    "AAPL, TSLA, GOOG" if selected_country == "United States" else ""
)
api_key = st.sidebar.text_input("Enter your Google API Key (optional)", type="password")

symbols = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]

# Apply exchange suffix normalization AFTER user input
symbols = normalize_symbols(symbols, country=selected_country, exchange_suffix=exchange_suffix)

# --- Apply API key preference ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
elif api_key_env:
    os.environ["GOOGLE_API_KEY"] = api_key_env

tabs = st.tabs(["User Guide", "Overview", "Company Deep Dives", "Risk & Correlation", "AI Dashboard", "Portfolio Strategist", "Chat Assistant", "Audit & Exports"])

# --- User Guide Tab ---
with tabs[0]:
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
                market_analysis_text = run_market_agent(symbols, close_df, benchmark=benchmark, country=selected_country)
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
                    company_text = run_company_agent(s, country=selected_country, benchmark=benchmark)
                    st.markdown(company_text)
            if st.button(f"Run SentimentAgent for {s}", key=f"sent_{s}"):
                with st.spinner(f"Running SentimentAgent for {s}..."):
                    sentiment_text = run_sentiment_agent(s, country=selected_country, benchmark=benchmark)
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
                risk_text = run_risk_agent(symbols, close_df_risk, benchmark=benchmark, country=selected_country)
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
        # --- Advanced Quant Analytics Controls ---
        st.markdown("---")
        st.subheader("🧮 Advanced Quantitative Analytics")

        do_alpha = st.checkbox("Compute Alpha & Tracking Error (vs benchmark)", value=True)
        do_drawdown = st.checkbox("Show Drawdown Visualization", value=True)
        do_rolling_beta = st.checkbox("Show Rolling Beta (63d)", value=False)
        do_sector = st.checkbox("Compute Sector Exposure (yfinance info)", value=False)
        do_stress = st.checkbox("Run Stress Test (shock or historical analogs)", value=False)
        do_tail = st.checkbox("Expected Shortfall (CVaR) & Sortino", value=True)

        # Prepare benchmark returns if available
        bench_prices = download_close_prices([benchmark], period="1y") if benchmark else pd.DataFrame()
        bench_returns = compute_returns(bench_prices).iloc[:, 0] if not bench_prices.empty else pd.Series(dtype=float)

        # Portfolio-level metrics (equal-weight)
        if not returns.empty:
            # portfolio equal-weight returns
            port_returns = returns.mean(axis=1)

            if do_alpha and not bench_returns.empty:
                alpha_res = compute_alpha_tracking_error(port_returns, bench_returns)
                st.markdown("**Alpha & Tracking Error (portfolio vs benchmark)**")
                st.write({
                    "Beta (OLS)": alpha_res["beta"],
                    "Alpha (annualized)": f"{alpha_res['alpha_annual']:.2%}" if not np.isnan(alpha_res["alpha_annual"]) else np.nan,
                    "Tracking Error (annualized)": f"{alpha_res['tracking_error_annual']:.2%}" if not np.isnan(alpha_res["tracking_error_annual"]) else np.nan
                })

            if do_drawdown:
                st.markdown("**Drawdown Visualization (each asset)**")
                # show drawdown plot for each asset in an expander to prevent UI overload
                for c in close_df_risk.columns:
                    with st.expander(f"{c} drawdown"):
                        fig_dd = plot_drawdown(close_df_risk[c], title=f"{c} - Price & Drawdown")
                        st.plotly_chart(fig_dd, use_container_width=True)

            if do_rolling_beta and not bench_returns.empty:
                st.markdown("**Rolling Beta (63-day)**")
                fig_rb_all = go.Figure()
                for c in returns.columns:
                    rb = rolling_beta(returns[c], bench_returns, window=63, symbol=c)
                    if not rb.empty:
                        fig_rb_all.add_trace(go.Scatter(
                            x=rb.index,
                            y=rb.values,
                            mode="lines",
                            name=c,
                            line=dict(width=2)
                        ))
                fig_rb_all.update_layout(
                    title=f"Rolling Beta (63-day) vs Benchmark ({benchmark})",
                    xaxis_title="Date",
                    yaxis_title="Beta",
                    template="plotly_white",
                    legend=dict(orientation="h", y=-0.2)
                )
                st.plotly_chart(fig_rb_all, use_container_width=True)

            if do_sector:
                st.markdown("**Sector Exposure (approx via yfinance metadata)**")
                # equal weight by default
                weights = {s: 1/len(symbols) for s in symbols} if symbols else {}
                sector_exp = compute_sector_exposure(symbols, weights=weights)
                st.table(pd.DataFrame.from_dict(sector_exp, orient="index", columns=["Weight"]).sort_values("Weight", ascending=False))

            if do_stress:
                st.markdown("**Stress Test**")
                shock_pct = st.number_input("Shock percent on benchmark (e.g., -10 for -10%)", value=-10.0, format="%.2f")
                shock_pct /= 100.0
                # compute betas for assets
                betas = {}
                if not bench_returns.empty:
                    for s in returns.columns:
                        betas[s] = compute_beta(returns[s], bench_returns)
                shock_res = stress_test_shock(returns.columns.tolist(), returns, bench_returns, shock_pct, betas)
                st.write("Estimated immediate P&L % (approx):")
                st.table(pd.DataFrame.from_dict(shock_res, orient="index", columns=["Est P&L"]).applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"))

                st.markdown("**Historical analog scenarios**")
                scenarios = stress_test_historical_analogs(symbols, close_df_risk, bench_prices.iloc[:,0] if not bench_prices.empty else pd.Series(), window=30, top_n=3)
                if scenarios:
                    for scen in scenarios:
                        st.markdown(f"- **Window**: {scen['start'].date()} → {scen['end'].date()}, Benchmark drop: {scen['bench_drop']:.2%}")
                        st.table(pd.DataFrame.from_dict(scen["asset_drops"], orient="index", columns=["Cumulative Return"]).applymap(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"))
                else:
                    st.info("No historical analog scenarios available (benchmark data not found).")

            if do_tail:
                st.markdown("**Tail Risk: CVaR & Sortino**")
                alpha_level = st.slider("CVaR alpha", 0.01, 0.10, 0.05, step=0.01)
                # compute for each asset and portfolio
                es_table = {}
                for c in returns.columns:
                    es = expected_shortfall(returns[c], alpha=alpha_level)
                    sd = sortino_ratio(returns[c])
                    es_table[c] = {"CVaR": es, "Sortino": sd}
                df_es = pd.DataFrame(es_table).T
                df_es["CVaR_pct"] = df_es["CVaR"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                df_es["Sortino"] = df_es["Sortino"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                st.table(df_es[["CVaR_pct","Sortino"]])


# --- AI Dashboard Tab (REPLACE YOUR EXISTING AI DASHBOARD BLOCK WITH THIS) ---
with tabs[4]:
    st.header("📈 AI Market Intelligence Dashboard — Interactive (Ask Gemini)")

    close_df_dash = download_close_prices(symbols, period="1y")
    bench_prices_dash = download_close_prices([benchmark], period="1y") if benchmark else pd.DataFrame()
    returns_dash = compute_returns(close_df_dash) if not close_df_dash.empty else pd.DataFrame()

    # Utility: cache Gemini explanations to avoid repeated model calls for same prompt
    @st.cache_data(ttl=60 * 60)  # cache for 1 hour
    def get_gemini_explanation(prompt: str, agent_name: str = "MarketAnalystAgent") -> str:
        try:
            resp = AGENTS[agent_name].run(prompt)
            return resp.content
        except Exception as e:
            return f"⚠️ Gemini call failed: {e}"

    # Small helper to create the Ask-Gemini prompt
    def build_chart_prompt(title: str, chart_notes: str, numeric_summary: dict, symbols_list: list):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        summary_lines = "\n".join([f"- {k}: {v}" for k, v in numeric_summary.items()])
        prompt = f"""
You are MarketAnalystAgent (Gemini). The user asked for an explanation of the chart titled: "{title}".

Context:
- Symbols: {', '.join(symbols_list)}
- Chart generated: {ts}
- Chart notes: {chart_notes}

Key numeric summary:
{summary_lines}

Tasks (concise, markdown):
1. Provide a short plain-language Summary of the chart (3-4 sentences).
2. Highlight 3 Key Observations / Patterns (bullet list).
3. Provide 2 Risk Signals or red flags (if any).
4. Give 2 short Recommendations or actions (benchmark-relative if possible).
5. Provide a short confidence line (e.g., "Confidence: moderate — based on 1-year daily data").

Return in markdown with sections: Summary, Observations, Risks, Recommendations, Confidence.
"""
        return prompt

    # Sidebar area where Gemini answers will be displayed
    st.sidebar.header("Gemini Chart Explanation")
    if "gemini_chart_explanation" not in st.session_state:
        st.session_state["gemini_chart_explanation"] = None
        st.session_state["gemini_chart_meta"] = None

    if not close_df_dash.empty:
        # Layout: two columns for top charts
        col1, col2 = st.columns([2, 1])

        # ---------------------- Cumulative Returns vs Benchmark ----------------------
        with col1:
            st.subheader("Cumulative Returns vs Benchmark (1Y)")

            cum = (1 + returns_dash).cumprod()
            fig_cum = go.Figure()
            for c in cum.columns:
                fig_cum.add_trace(go.Scatter(x=cum.index, y=cum[c], mode="lines", name=c))
            if not bench_prices_dash.empty:
                bench_ret = compute_returns(bench_prices_dash).iloc[:, 0]
                bench_cum = (1 + bench_ret).cumprod()
                fig_cum.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode="lines",
                                             name=f"{benchmark}", line=dict(width=3, dash="dash")))

            fig_cum.update_layout(template="plotly_white", yaxis_title="Growth (1 = 0%)")
            st.plotly_chart(fig_cum, use_container_width=True)

            # Gemini button
            if st.button("Ask Gemini about this chart — Cumulative Returns", key="ask_cum"):
                # prepare short numeric summary
                numeric = {}
                for c in cum.columns:
                    numeric[f"{c} (1Y)"] = f"{(cum[c].iloc[-1] - 1):.2%}"
                if not bench_prices_dash.empty:
                    numeric[f"{benchmark} (1Y)"] = f"{(bench_cum.iloc[-1] - 1):.2%}"
                prompt = build_chart_prompt(
                    "Cumulative Returns vs Benchmark (1Y)",
                    "Shows indexed cumulative returns for selected symbols vs benchmark.",
                    numeric,
                    symbols
                )
                st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="MarketAnalystAgent")
                st.session_state["gemini_chart_meta"] = {"chart": "Cumulative Returns", "time": datetime.now().isoformat()}

        # ---------------------- Alpha vs Beta Scatter ----------------------
        with col2:
            st.subheader("Alpha vs Beta Scatter")
            betas = {}
            alphas = {}
            if not bench_prices_dash.empty and not returns_dash.empty:
                bench_ret_full = compute_returns(bench_prices_dash).iloc[:, 0]
                for c in returns_dash.columns:
                    res = compute_alpha_tracking_error(returns_dash[c], bench_ret_full)
                    betas[c] = res["beta"]
                    alphas[c] = res["alpha_annual"]

                fig_ab = go.Figure()
                fig_ab.add_trace(go.Scatter(
                    x=list(betas.values()),
                    y=list(alphas.values()),
                    mode="markers+text",
                    text=list(betas.keys()),
                    textposition="top center"
                ))
                fig_ab.update_layout(
                    xaxis_title="Beta (Market Sensitivity)",
                    yaxis_title="Alpha (Annualized)",
                    template="plotly_white",
                    height=420
                )
                st.plotly_chart(fig_ab, use_container_width=True)
            else:
                st.info("Benchmark data required for Alpha vs Beta chart.")

            if st.button("Ask Gemini about this chart — Alpha vs Beta", key="ask_ab"):
                numeric = {c: f"β={betas.get(c, np.nan):.2f}, α={alphas.get(c, np.nan):.2%}" for c in betas.keys()}
                prompt = build_chart_prompt(
                    "Alpha vs Beta Scatter",
                    "Scatter of asset beta vs alpha relative to selected benchmark.",
                    numeric,
                    symbols
                )
                st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="MarketAnalystAgent")
                st.session_state["gemini_chart_meta"] = {"chart": "Alpha vs Beta", "time": datetime.now().isoformat()}

        # ---------------------- Rolling Correlation vs Benchmark ----------------------
        st.subheader("Rolling 63-Day Correlation vs Benchmark")
        if not bench_prices_dash.empty and not returns_dash.empty:
            bench_ret_full = compute_returns(bench_prices_dash).iloc[:, 0]
            fig_corr_roll = go.Figure()
            for c in returns_dash.columns:
                roll_corr = returns_dash[c].rolling(63).corr(bench_ret_full)
                fig_corr_roll.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr.values, mode="lines", name=c))
            fig_corr_roll.update_layout(template="plotly_white", yaxis_title="Correlation", title="63-Day Rolling Correlation")
            st.plotly_chart(fig_corr_roll, use_container_width=True)
        else:
            st.info("Benchmark data required for rolling correlation.")

        if st.button("Ask Gemini about this chart — Rolling Correlation", key="ask_corr"):
            # build numeric summary: last corr + mean corr for each symbol
            numeric = {}
            if not bench_prices_dash.empty:
                for c in returns_dash.columns:
                    roll = returns_dash[c].rolling(63).corr(compute_returns(bench_prices_dash).iloc[:,0])
                    numeric[f"{c} last"] = f"{roll.iloc[-1]:.2f}" if not roll.empty else "N/A"
                    numeric[f"{c} mean"] = f"{roll.mean():.2f}" if not roll.empty else "N/A"
            prompt = build_chart_prompt(
                "Rolling 63-Day Correlation vs Benchmark",
                "Rolling correlation between each asset and the benchmark over 63-day windows.",
                numeric,
                symbols
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="MarketAnalystAgent")
            st.session_state["gemini_chart_meta"] = {"chart": "Rolling Correlation", "time": datetime.now().isoformat()}

        # ---------------------- Allocation Pie Charts ----------------------
        st.subheader("Portfolio Allocation Comparison (Equal • Risk-Parity • Momentum)")
        if not returns_dash.empty:
            equal = {s: 1/len(symbols) for s in symbols}
            vols = returns_dash.std() * np.sqrt(252)
            invvol = (1 / vols)
            invvol = (invvol / invvol.sum()).round(4).to_dict()
            six_month = close_df_dash.pct_change(126).iloc[-1] if close_df_dash.shape[0] > 126 else close_df_dash.pct_change().iloc[-1]
            mom = six_month / six_month.sum() if six_month.sum() != 0 else pd.Series(equal)
            mom = mom.round(4).to_dict()

            fig_alloc = go.Figure()
            fig_alloc.add_trace(go.Pie(labels=list(equal.keys()), values=list(equal.values()), domain=dict(x=[0, .33]), name="Equal Weight"))
            fig_alloc.add_trace(go.Pie(labels=list(invvol.keys()), values=list(invvol.values()), domain=dict(x=[.33, .66]), name="Risk Parity"))
            fig_alloc.add_trace(go.Pie(labels=list(mom.keys()), values=list(mom.values()), domain=dict(x=[.66, 1]), name="Momentum"))
            fig_alloc.update_layout(title="Equal • Risk-Parity • Momentum Allocations")
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("Need at least 1 symbol with data to show allocations.")

        if st.button("Ask Gemini about this chart — Allocations", key="ask_alloc"):
            numeric = {}
            if returns_dash is not None and not returns_dash.empty:
                numeric["Equal"] = ", ".join([f"{k}:{v:.2%}" for k,v in equal.items()])
                numeric["RiskParity"] = ", ".join([f"{k}:{v:.2%}" for k,v in invvol.items()])
                numeric["Momentum"] = ", ".join([f"{k}:{v:.2%}" for k,v in mom.items()])
            prompt = build_chart_prompt(
                "Portfolio Allocation Comparison",
                "Three pie charts showing Equal-weight, Risk-parity (inverse vol), and Momentum-tilt allocations.",
                numeric,
                symbols
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="PortfolioStrategistAgent")
            st.session_state["gemini_chart_meta"] = {"chart": "Allocations", "time": datetime.now().isoformat()}

        # ---------------------- Sector Exposure Treemap ----------------------
        st.subheader("Sector Exposure Treemap")
        sector_weights = compute_sector_exposure(symbols, weights={s:1/len(symbols) for s in symbols}) if symbols else {}
        if sector_weights:
            df_sector = pd.DataFrame({"Sector": list(sector_weights.keys()), "Weight": list(sector_weights.values())})
            fig_tree = go.Figure(go.Treemap(labels=df_sector["Sector"], parents=["Portfolio"] * len(df_sector), values=df_sector["Weight"]))
            fig_tree.update_layout(template="plotly_white")
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("Sector metadata not available for selected symbols.")

        if st.button("Ask Gemini about this chart — Sector Exposure", key="ask_sector"):
            numeric = {k: f"{v:.2%}" for k,v in sector_weights.items()}
            prompt = build_chart_prompt(
                "Sector Exposure Treemap",
                "Treemap showing portfolio weight by sector based on yfinance metadata.",
                numeric,
                symbols
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="MarketAnalystAgent")
            st.session_state["gemini_chart_meta"] = {"chart": "Sector Exposure", "time": datetime.now().isoformat()}

        # ---------------------- Stress Test Bar Chart ----------------------
        st.subheader("Stress Test: -10% Benchmark Shock Impact")
        if not bench_prices_dash.empty:
            bench_ret_full = compute_returns(bench_prices_dash).iloc[:, 0]
            betas_calc = {s: compute_beta(returns_dash[s], bench_ret_full) for s in symbols}
            shock = stress_test_shock(symbols, returns_dash, bench_ret_full, shock_pct=-0.10, betas=betas_calc)
            df_shock = pd.DataFrame({"Symbol": list(shock.keys()), "Impact (%)": [v * 100 for v in shock.values()]})
            fig_shock = go.Figure(go.Bar(x=df_shock["Symbol"], y=df_shock["Impact (%)"] ))
            fig_shock.update_layout(yaxis_title="Estimated Loss (%)", template="plotly_white")
            st.plotly_chart(fig_shock, use_container_width=True)
        else:
            st.info("Benchmark required for stress test.")

        if st.button("Ask Gemini about this chart — Stress Test", key="ask_stress"):
            numeric = {s: f"{shock.get(s, np.nan):.2%}" for s in symbols} if 'shock' in locals() else {}
            prompt = build_chart_prompt(
                "Stress Test: -10% Benchmark Shock Impact",
                "Bar chart estimating immediate P&L impact given a -10% benchmark shock using betas/correlation.",
                numeric,
                symbols
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="RiskAnalystAgent")
            st.session_state["gemini_chart_meta"] = {"chart": "Stress Test", "time": datetime.now().isoformat()}

        # ---------------------- Efficient Frontier (if small universe) ----------------------
        st.subheader("Efficient Frontier (Mean-Variance Preview)")
        if len(symbols) <= 5 and not returns_dash.empty:
            # Use your existing simple frontier builder (keeps tractable)
            def portfolio_performance(weights, mean_returns, cov_matrix):
                ret = np.dot(weights, mean_returns) * 252
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                return vol, ret

            mean_returns = returns_dash.mean()
            cov_matrix = returns_dash.cov()
            # Sample random portfolios instead of combinatorial enumeration (fast and interactive)
            n_points = 150
            rets = []
            vols = []
            weights_list = []
            rng = np.random.default_rng(seed=42)
            for _ in range(n_points):
                w = rng.random(len(mean_returns))
                w = w / w.sum()
                vol, ret = portfolio_performance(w, mean_returns, cov_matrix)
                rets.append(ret)
                vols.append(vol)
                weights_list.append(w)

            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=vols, y=rets, mode="markers", name="Portfolios", marker=dict(opacity=0.7)))
            eq_w = np.array([1/len(symbols)] * len(symbols))
            eq_vol, eq_ret = portfolio_performance(eq_w, mean_returns, cov_matrix)
            fig_ef.add_trace(go.Scatter(x=[eq_vol], y=[eq_ret], mode="markers", name="Equal Weight", marker=dict(size=12, symbol="star")))
            fig_ef.update_layout(xaxis_title="Volatility (Annualized)", yaxis_title="Expected Return (Annualized)", template="plotly_white")
            st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.info("Efficient frontier preview limited to ≤ 5 assets and requires data.")

        if st.button("Ask Gemini about this chart — Efficient Frontier", key="ask_ef"):
            numeric = {"Sample portfolios": n_points if 'n_points' in locals() else "N/A", "EqualWeight": f"Vol:{eq_vol:.3f}, Ret:{eq_ret:.3f}"}
            prompt = build_chart_prompt(
                "Efficient Frontier (Mean-Variance Preview)",
                "Cloud of randomly sampled portfolios (mean-variance) with an equal-weight marker.",
                numeric,
                symbols
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="PortfolioStrategistAgent")
            st.session_state["gemini_chart_meta"] = {"chart": "Efficient Frontier", "time": datetime.now().isoformat()}

        # ---------------------- Rolling Sharpe Ratio ----------------------
        st.subheader("Rolling Sharpe Ratio (63-Day)")
        if not returns_dash.empty:
            roll_sharpe_fig = go.Figure()
            window = 63
            for c in returns_dash.columns:
                roll_ret = returns_dash[c].rolling(window).mean() * 252
                roll_vol = returns_dash[c].rolling(window).std() * np.sqrt(252)
                rs = roll_ret / roll_vol
                roll_sharpe_fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name=c))
            roll_sharpe_fig.update_layout(template="plotly_white", yaxis_title="Sharpe Ratio")
            st.plotly_chart(roll_sharpe_fig, use_container_width=True)
        else:
            st.info("Data required to compute rolling Sharpe.")

        if st.button("Ask Gemini about this chart — Rolling Sharpe", key="ask_rs"):
            numeric = {}
            if not returns_dash.empty:
                for c in returns_dash.columns:
                    rs = (returns_dash[c].rolling(63).mean()*252) / (returns_dash[c].rolling(63).std()*np.sqrt(252))
                    numeric[f"{c} last_sharpe"] = f"{rs.iloc[-1]:.3f}" if not rs.empty else "N/A"
            prompt = build_chart_prompt(
                "Rolling Sharpe Ratio (63-Day)",
                "Rolling Sharpe ratio per asset using 63-day windows (annualized).",
                numeric,
                symbols
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="MarketAnalystAgent")
            st.session_state["gemini_chart_meta"] = {"chart": "Rolling Sharpe", "time": datetime.now().isoformat()}

        # ---------------------- Monte Carlo Simulation ----------------------
        st.subheader("Monte Carlo Simulation (1-Year)")
        symbol_mc = st.selectbox("Select symbol for simulation", symbols, key="mc_symbol_select")
        last_price = close_df_dash[symbol_mc].iloc[-1]
        vol_mc = returns_dash[symbol_mc].std() * np.sqrt(252)
        mu_mc = returns_dash[symbol_mc].mean() * 252
        days = 252
        paths = 200
        sim_paths = np.zeros((days, paths))
        rng = np.random.default_rng(seed=1)
        for i in range(paths):
            daily = rng.normal(mu_mc/days, vol_mc/np.sqrt(days), days)
            sim_paths[:, i] = last_price * np.cumprod(1 + daily)

        fig_mc = go.Figure()
        for i in range(min(paths, 100)):  # show at most 100 lines for performance
            fig_mc.add_trace(go.Scatter(x=np.arange(days), y=sim_paths[:, i], mode="lines", showlegend=False, line=dict(width=1)))
        fig_mc.update_layout(title=f"Monte Carlo Price Simulation: {symbol_mc}", template="plotly_white")
        st.plotly_chart(fig_mc, use_container_width=True)

        if st.button("Ask Gemini about this chart — Monte Carlo", key="ask_mc"):
            numeric = {
                "last_price": f"{last_price:.2f}",
                "mu_ann": f"{mu_mc:.2%}",
                "vol_ann": f"{vol_mc:.2%}",
                "paths": paths
            }
            prompt = build_chart_prompt(
                f"Monte Carlo Simulation: {symbol_mc}",
                "Simulated price paths for the next 1 year using geometric random-walk assumptions (normal returns).",
                numeric,
                [symbol_mc]
            )
            st.session_state["gemini_chart_explanation"] = get_gemini_explanation(prompt, agent_name="MarketAnalystAgent")
            st.session_state["gemini_chart_meta"] = {"chart": f"Monte Carlo ({symbol_mc})", "time": datetime.now().isoformat()}

    else:
        st.info("No data available for visualization in the dashboard. Provide symbols and a benchmark to enable charts.")

    # ---------------------- Sidebar: show the Gemini response when available ----------------------
    if st.session_state.get("gemini_chart_explanation"):
        meta = st.session_state.get("gemini_chart_meta", {})
        st.sidebar.markdown(f"**Chart:** {meta.get('chart', 'N/A')}")
        st.sidebar.markdown(f"**Generated at:** {meta.get('time', '')}")
        st.sidebar.markdown("---")
        st.sidebar.markdown(st.session_state["gemini_chart_explanation"])
        if st.sidebar.button("Clear explanation"):
            st.session_state["gemini_chart_explanation"] = None
            st.session_state["gemini_chart_meta"] = None


# --- Portfolio Strategist Tab ---
with tabs[5]:
    st.header("Portfolio Strategist — Allocation Proposals")
    close_df_port = download_close_prices(symbols, period="1y")
    constraints_raw = st.text_input("Constraints (json) e.g. {'max_weight':0.4}", "")
    try:
        constraints = eval(constraints_raw) if constraints_raw else None
    except Exception:
        constraints = None
    if st.button("Run PortfolioStrategistAgent"):
        with st.spinner("Running PortfolioStrategistAgent..."):
            strategy_text = run_portfolio_agent(symbols, close_df_port, constraints=constraints, benchmark=benchmark, country=selected_country)
            st.markdown("### Allocation Recommendation (XAI)")
            st.markdown(strategy_text)


with tabs[6]:
    st.header("💬 Conversational Chat Assistant")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # --- Bubble colors ---
    USER_COLOR = "#242017"      # Dark brown/grey
    AI_COLOR = "#0f0f0f"        # Deep black
    TEXT_COLOR = "white"

    def render_message(role, markdown_text):
        bg = USER_COLOR if role == "user" else AI_COLOR
        label = "You" if role == "user" else "AI"

        st.markdown(
            f"""
            <div style="
                background:{bg};
                padding:14px 18px;
                margin:12px 0;
                border-radius:14px;
                color:{TEXT_COLOR};
                font-family: 'Inter', sans-serif;
            ">
                <div style="font-weight:600; margin-bottom:6px; opacity:0.9;">
                    {label}
                </div>

                <div style="font-size:15px; line-height:1.6;">
                    {markdown_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Render all chat messages
    for role, msg in st.session_state["chat_history"]:
        render_message(role, msg)

    # --- User Input ---
    user_input = st.text_input("Ask anything about markets, stocks or portfolio:")

    def build_contextual_prompt(user_query):
        history = "\n".join([f"{r}: {m}" for r, m in st.session_state["chat_history"][-5:]])
        return f"""
You are a conversational financial assistant connected to a multi-agent research system.

User tickers: {symbols}
Country: {selected_country}
Benchmark: {benchmark}

Recent conversation:
{history}

User query: {user_query}

If needed, call internal market, risk, sentiment or portfolio agents and summarize naturally.
"""

    # --- Send button ---
    if st.button("Send") and user_input.strip():
        st.session_state["chat_history"].append(("user", user_input))

        prompt = build_contextual_prompt(user_input)
        ai_reply = AGENTS["TeamLeadAgent"].run(prompt).content

        st.session_state["chat_history"].append(("assistant", ai_reply))

        st.rerun()

# --- Audit & Exports Tab ---
with tabs[7]:
    st.header("Audit Trail & Report Generation")
    st.markdown("Run all agents together, integrate results, and generate a benchmark-aware TeamLead report.")

    if st.button("Run full multi-agent orchestration and generate TeamLead report"):
        with st.spinner("Running all agents and compiling final report..."):
            # Download full-year close data
            close_for_run = download_close_prices(symbols, period="1y")
            date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")

            # --- Run sub-agents with benchmark & country awareness ---
            
            # 1️⃣ Market Analysis (covers all symbols)
            market_analysis = run_market_agent(
                symbols=symbols,
                close_df=close_for_run,
                benchmark=benchmark,
                country=selected_country
            )
            
            # 2️⃣ Company Analyses (per symbol)
            company_analyses = {
                s: run_company_agent(s, country=selected_country, benchmark=benchmark)
                for s in symbols
            }
            
            # 3️⃣ Sentiment Analyses (per symbol)
            sentiment_analyses = {
                s: run_sentiment_agent(s, country=selected_country, benchmark=benchmark)
                for s in symbols
            }
            
            # 4️⃣ Risk Analysis (covers all symbols)
            risk_analysis = run_risk_agent(
                symbols=symbols,
                close_df=close_for_run,
                benchmark=benchmark,
                country=selected_country
            )
            
            # 5️⃣ Portfolio Recommendation (covers all symbols)
            portfolio_recommendation = run_portfolio_agent(
                symbols=symbols,
                close_df=close_for_run,
                benchmark=benchmark,
                country=selected_country
            )
            
            # 6️⃣ Final Integration Report
            final_report = run_teamlead_agent(
                date_str=date_str,
                market_analysis=market_analysis,
                company_analyses=company_analyses,
                sentiment_analyses=sentiment_analyses,
                risk_analysis=risk_analysis,
                portfolio_recommendation=portfolio_recommendation,
                benchmark=benchmark,
                country=selected_country
            )
            

            # --- Display results ---
            st.markdown("### 🧠 TeamLead Consolidated Report (Benchmark-Aware)")
            st.markdown(final_report)

    st.markdown("---")

