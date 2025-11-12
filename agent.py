import os
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

# --------------------------- Setup --------------------------- #
load_dotenv()  # Load .env if running locally
api_key_env = os.getenv("GOOGLE_API_KEY", "")

# Streamlit page config
st.set_page_config(page_title="AI Investment Strategist", page_icon="📈", layout="wide")

# --------------------------- Utility Functions --------------------------- #
def compare_stocks(symbols):
    """Fetch and compare 6-month performance for each stock."""
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")

            if hist.empty:
                st.warning(f"No data found for {symbol}, skipping it.")
                continue

            data[symbol] = hist['Close'].pct_change().sum()

        except Exception as e:
            st.error(f"Error fetching {symbol}: {e}")
    return data


def get_company_info(symbol):
    """Get basic company info from Yahoo Finance."""
    stock = yf.Ticker(symbol)
    try:
        info = stock.get_info() if hasattr(stock, "get_info") else {}
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A"),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A", "market_cap": "N/A", "summary": "N/A"}


def get_company_news(symbol):
    """Retrieve latest news for a stock."""
    stock = yf.Ticker(symbol)
    try:
        return stock.news[:5]
    except Exception:
        return [{"title": "No news available", "link": ""}]


# --------------------------- AI Agents --------------------------- #
market_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Analyzes and compares stock performance over time.",
    instructions=[
        "Retrieve and compare stock performance from Yahoo Finance.",
        "Calculate percentage change over a 6-month period.",
        "Rank stocks based on their relative performance."
    ],
    markdown=True
)

company_researcher = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Fetches company profiles, financials, and latest news.",
    instructions=[
        "Summarize company fundamentals and sector relevance.",
        "Summarize the most recent news affecting investor sentiment."
    ],
    markdown=True
)

stock_strategist = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Provides investment insights and recommends top stocks.",
    instructions=[
        "Analyze performance trends and fundamentals.",
        "Evaluate risk vs. reward and identify top investment opportunities."
    ],
    markdown=True
)

team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Compiles final investment insights into a structured report.",
    instructions=[
        "Integrate market analysis, company research, and strategy insights.",
        "Provide a ranked recommendation list for investors."
    ],
    markdown=True
)

# --------------------------- AI Logic --------------------------- #
def get_market_analysis(symbols):
    performance_data = compare_stocks(symbols)
    if not performance_data:
        return "No valid stock data found."
    analysis = market_analyst.run(f"Compare these stock performances: {performance_data}")
    return analysis.content


def get_company_analysis(symbol):
    info = get_company_info(symbol)
    news = get_company_news(symbol)
    response = company_researcher.run(
        f"Provide an analysis for {info['name']} in the {info['sector']} sector.\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n"
        f"Latest News: {news}"
    )
    return response.content


def get_stock_recommendations(symbols):
    market_analysis = get_market_analysis(symbols)
    company_data = {s: get_company_analysis(s) for s in symbols}
    recommendations = stock_strategist.run(
        f"Based on market analysis {market_analysis} and company data {company_data}, "
        f"which stocks should investors consider buying?"
    )
    return recommendations.content


def get_final_investment_report(symbols):
    market_analysis = get_market_analysis(symbols)
    company_analyses = [get_company_analysis(s) for s in symbols]
    stock_recommendations = get_stock_recommendations(symbols)

    final_report = team_lead.run(
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses:\n{company_analyses}\n\n"
        f"Stock Recommendations:\n{stock_recommendations}\n\n"
        f"Provide a comprehensive ranked list of stocks suitable for investment."
    )
    return final_report.content


# --------------------------- Streamlit UI --------------------------- #
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">📈 AI Investment Strategist</h1>
    <h3 style="text-align: center; color: #6c757d;">Generate personalized investment reports using AI and live market data.</h3>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Configuration")
input_symbols = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "AAPL, TSLA, GOOG")
api_key = st.sidebar.text_input("Enter your Google API Key (optional)", type="password")

stocks_symbols = [s.strip().upper() for s in input_symbols.split(",") if s.strip()]

if st.sidebar.button("Generate Investment Report"):
    if not stocks_symbols:
        st.sidebar.warning("Please enter at least one stock symbol.")
    elif not (api_key or api_key_env):
        st.sidebar.warning("Please enter your Google API Key.")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key or api_key_env

        with st.spinner("🔍 Generating investment report..."):
            try:
                report = get_final_investment_report(stocks_symbols)
                st.subheader("📊 Investment Report")
                st.markdown(report)

                # Plot stock performance
                st.markdown("### 📈 6-Month Stock Performance")
                stock_data = yf.download(stocks_symbols, period="6mo")['Close']
                fig = go.Figure()
                for s in stocks_symbols:
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[s], mode='lines', name=s))
                fig.update_layout(
                    title="Stock Performance Over Last 6 Months",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"⚠️ Error generating report: {e}")
