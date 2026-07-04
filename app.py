import streamlit as st
import pandas as pd
import os
import re
import json
import base64
import requests
from datetime import datetime
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Market X Growth Intelligence Advisor",
    page_icon="assets/market_x_logo.webp" if os.path.exists("assets/market_x_logo.webp") else "💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

MARKET_X_WEBSITE = "https://market-x.co.in/"
LOGO_PATH = "assets/market_x_logo.webp"


@st.cache_data
def get_logo_data_uri():
    try:
        with open(LOGO_PATH, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/webp;base64,{encoded}"
    except Exception:
        return None


LOGO_URI = get_logo_data_uri()
LOGO_IMG_TAG = (
    f'<img src="{LOGO_URI}" class="mx-logo" />'
    if LOGO_URI else '<div class="mx-logo-fallback">X</div>'
)


# =========================================================
# SAFE SECRET READER
# =========================================================
def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.environ.get(key, default)


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================
for folder in ["data", "leads", "reports"]:
    if not os.path.exists(folder):
        os.makedirs(folder)


# =========================================================
# DESIGN SYSTEM — TOKENS + GLOBAL STYLE
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --navy-950: #060B16;
        --navy-900: #0B1220;
        --navy-800: #101C33;
        --teal-600: #0D7B70;
        --teal-500: #12897D;
        --teal-400: #2FB3A3;
        --amber-500: #E8A33D;
        --amber-400: #F2B65B;
        --canvas: #F4F5FA;
        --card: #FFFFFF;
        --border: #E4E7EF;
        --ink-900: #0E1526;
        --ink-600: #4C5566;
        --ink-400: #8A93A6;
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
        --shadow-card: 0px 10px 30px rgba(14, 21, 38, 0.06);
        --shadow-hero: 0px 24px 60px rgba(6, 11, 22, 0.35);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background: var(--canvas);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Fraunces', serif;
        color: var(--ink-900);
        font-weight: 600;
    }

    p, span, label, div {
        color: var(--ink-900);
    }

    /* ---------- Logo ---------- */
    .mx-logo {
        height: 42px;
        width: 42px;
        border-radius: 10px;
        object-fit: cover;
        display: block;
    }

    .mx-logo-fallback {
        height: 42px;
        width: 42px;
        border-radius: 10px;
        background: var(--navy-900);
        color: var(--teal-400);
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ---------- Top brand bar ---------- */
    .mx-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 4px 22px 4px;
        margin-bottom: 4px;
    }

    .mx-topbar-left {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .mx-wordmark {
        font-family: 'Fraunces', serif;
        font-size: 19px;
        font-weight: 600;
        color: var(--ink-900);
        letter-spacing: 0.2px;
        line-height: 1.1;
    }

    .mx-wordmark-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        color: var(--ink-400);
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .mx-topbar-link a {
        text-decoration: none;
        color: var(--ink-900);
        font-weight: 600;
        font-size: 13.5px;
        padding: 9px 18px;
        border-radius: 999px;
        border: 1.5px solid var(--navy-900);
        transition: all 0.15s ease;
    }

    .mx-topbar-link a:hover {
        background: var(--navy-900);
        color: white;
    }

    /* ---------- Hero ---------- */
    .mx-hero {
        background: linear-gradient(148deg, var(--navy-950) 0%, var(--navy-800) 55%, #0C3A38 130%);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-hero);
        padding: 0;
        margin-bottom: 26px;
        overflow: hidden;
        position: relative;
    }

    .mx-hero-accent {
        height: 4px;
        width: 100%;
        background: linear-gradient(90deg, var(--teal-400), var(--amber-400), var(--teal-400));
    }

    .mx-hero-grid {
        display: grid;
        grid-template-columns: 1.55fr 1fr;
        gap: 0;
    }

    .mx-hero-left {
        padding: 44px 42px 44px 46px;
    }

    .mx-hero-kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--amber-400);
        font-weight: 600;
        margin-bottom: 16px;
    }

    .mx-hero-title {
        font-family: 'Fraunces', serif;
        font-size: 38px;
        line-height: 1.12;
        font-weight: 600;
        color: white;
        margin-bottom: 16px;
        max-width: 560px;
    }

    .mx-hero-subtitle {
        font-size: 15.5px;
        color: #C7D2E3;
        line-height: 1.65;
        max-width: 520px;
        font-weight: 400;
    }

    .mx-hero-cta {
        margin-top: 26px;
        display: flex;
        gap: 12px;
    }

    .mx-hero-cta a {
        text-decoration: none;
        font-size: 13.5px;
        font-weight: 600;
        padding: 11px 20px;
        border-radius: 999px;
    }

    .mx-cta-primary {
        background: var(--teal-400);
        color: var(--navy-950) !important;
    }

    .mx-cta-secondary {
        border: 1.5px solid rgba(255,255,255,0.35);
        color: white !important;
    }

    .mx-hero-right {
        background: rgba(255,255,255,0.05);
        border-left: 1px solid rgba(255,255,255,0.09);
        padding: 40px 36px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0;
    }

    .mx-diff-item {
        padding: 14px 0;
        border-bottom: 1px solid rgba(255,255,255,0.09);
    }

    .mx-diff-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }

    .mx-diff-item:first-child {
        padding-top: 0;
    }

    .mx-diff-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--teal-400);
        margin-bottom: 5px;
    }

    .mx-diff-value {
        font-family: 'Fraunces', serif;
        font-size: 17px;
        font-weight: 600;
        color: white;
        margin-bottom: 4px;
    }

    .mx-diff-copy {
        font-size: 12.5px;
        color: #A9B6CC;
        line-height: 1.5;
    }

    /* ---------- Section kicker ---------- */
    .section-kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: var(--teal-600);
        font-weight: 600;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* ---------- Cards ---------- */
    .glass-card {
        background: var(--card);
        padding: 30px 32px;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-card);
        margin-bottom: 20px;
    }

    .insight-box {
        background: #EEFAF8;
        border-left: 4px solid var(--teal-500);
        padding: 16px 20px;
        border-radius: var(--radius-sm);
        color: #0C3A38;
        margin: 18px 0 0 0;
        font-size: 14px;
        line-height: 1.6;
    }

    .warning-box {
        background: #FEF6E9;
        border-left: 4px solid var(--amber-500);
        padding: 16px 20px;
        border-radius: var(--radius-sm);
        color: #6B4A10;
        margin: 18px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    .success-box {
        background: #EEFAF8;
        border-left: 4px solid var(--teal-500);
        padding: 16px 20px;
        border-radius: var(--radius-sm);
        color: #0C3A38;
        margin: 18px 0;
        font-size: 14px;
        line-height: 1.6;
    }

    .small-text {
        font-size: 12.5px;
        color: var(--ink-400);
        line-height: 1.6;
    }

    .small-text a {
        color: var(--teal-500);
        font-weight: 600;
    }

    /* ---------- Streamlit widget overrides ---------- */
    div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1.5px solid var(--border);
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 14px;
        color: var(--ink-400);
        padding: 10px 6px;
        margin-right: 22px;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: var(--navy-900) !important;
        border-bottom: 2.5px solid var(--teal-500) !important;
    }

    div.stButton > button {
        border-radius: 999px;
        font-weight: 600;
        font-size: 14px;
        padding: 0.55rem 1.3rem;
        border: none;
        transition: all 0.15s ease;
    }

    div.stButton > button[kind="primary"] {
        background: var(--navy-900);
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--teal-600);
        color: white;
    }

    div.stButton > button[kind="secondary"] {
        background: white;
        color: var(--ink-900);
        border: 1.5px solid var(--border);
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--navy-900);
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 999px;
        font-weight: 600;
        background: var(--teal-500);
        color: white;
        border: none;
        padding: 0.55rem 1.3rem;
    }

    div[data-testid="stDownloadButton"] > button:hover {
        background: var(--teal-600);
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: var(--radius-sm) !important;
        border: 1.5px solid var(--border) !important;
        font-family: 'Inter', sans-serif;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--teal-500) !important;
        box-shadow: 0 0 0 1px var(--teal-500) !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--teal-500), var(--teal-400));
    }

    div[data-testid="stForm"] {
        border: 1px solid var(--border);
        padding: 28px 30px;
        border-radius: var(--radius-lg);
        background: white;
        box-shadow: var(--shadow-card);
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif;
        font-size: 26px;
        font-weight: 600;
        color: var(--navy-900);
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: var(--ink-400);
    }

    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 22px 0;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--navy-950);
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: #DCE3F0;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white;
        font-family: 'Fraunces', serif;
    }

    .mx-sidebar-logo {
        display: flex;
        justify-content: center;
        padding: 6px 0 18px 0;
    }

    .mx-sidebar-kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: var(--teal-400);
        font-weight: 600;
        margin: 6px 0 10px 0;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-size: 13.5px;
    }

    section[data-testid="stSidebar"] hr {
        border-top: 1px solid rgba(255,255,255,0.12);
        margin: 18px 0;
    }

    .mx-sidebar-list {
        font-size: 13px;
        line-height: 2;
        color: #B9C3D6;
    }

    .mx-sidebar-cta {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: var(--radius-sm);
        padding: 14px 16px;
        font-size: 12.5px;
        line-height: 1.6;
        color: #C7D2E3;
    }

    .mx-sidebar-cta a {
        color: var(--teal-400);
        font-weight: 700;
        text-decoration: none;
    }

    /* ---------- Footer ---------- */
    .mx-footer {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 22px 4px 6px 4px;
    }

    .mx-footer-text {
        font-size: 12.5px;
        color: var(--ink-400);
        line-height: 1.6;
    }

    .mx-footer-text a {
        color: var(--teal-600);
        font-weight: 700;
        text-decoration: none;
    }

    .mx-footer-brand {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: var(--ink-900);
        font-size: 13.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DEFAULT KNOWLEDGE BASE
# =========================================================
default_knowledge = {
    "company_profile.txt": """
Market X is a fast-growing consulting firm in India focused on FMCG, dairy, agriculture, government projects, brand development, business development, and market expansion.

Market X works with corporates, government bodies, and growth-stage businesses looking to enter new markets, improve sales, build distribution networks, conduct market research, and execute business development projects.

Market X has experience across Indian states including Bihar, Jharkhand, Uttar Pradesh, Kerala, Andhra Pradesh, Assam, and Arunachal Pradesh.

The firm supports clients through strategic advisory, market intelligence, execution planning, sales transformation, and business growth consulting.
""",

    "services.txt": """
Market X provides consulting services across FMCG, dairy, agriculture, plantation, government advisory, brand development, and business expansion.

Core services include:
1. FMCG distribution expansion
2. Route-to-market strategy
3. Distributor and channel partner identification
4. Dairy market development
5. Agriculture value chain consulting
6. Palm and cocoa plantation advisory
7. Government project consulting
8. Brand positioning and development
9. Market research
10. Competitor benchmarking
11. Sales transformation
12. Retail activation strategy
13. Business development support
14. Distributor due diligence
15. Beat planning and salesforce productivity diagnostics
16. GTM pilot design
17. Trade scheme benchmarking
18. Category and channel white-space analysis

Market X helps businesses identify market opportunities, design execution roadmaps, and improve sales performance.
""",

    "fmcg_strategy.txt": """
FMCG growth requires a structured understanding of market demand, retail universe, distributor capability, pricing, trade schemes, sales force productivity, and competitor presence.

A practical FMCG expansion approach includes:
1. Market mapping
2. Retail universe identification
3. Distributor profiling
4. Channel partner appointment
5. Sales beat planning
6. Trade scheme benchmarking
7. Retailer activation
8. Sales dashboard tracking
9. Competitor analysis
10. Pilot launch before full expansion
11. Salesman productivity review
12. SKU-wise velocity tracking
13. Outlet classification
14. Margin waterfall analysis
15. Secondary sales tracking

Many FMCG companies fail in new markets because they appoint distributors without understanding retailer density, credit cycles, local competition, category demand, and sales team capability.

Market X can support FMCG clients through route-to-market strategy, distribution expansion, market research, retailer activation planning, and performance tracking.
""",

    "dairy_strategy.txt": """
Dairy market development depends on cold chain, local consumption patterns, distributor reach, product freshness, retailer trust, pricing, and daily delivery discipline.

A dairy expansion strategy should evaluate:
1. Milk and value-added dairy product demand
2. Cold chain availability
3. Distributor storage capacity
4. Retailer network
5. Daily delivery route planning
6. Product shelf life
7. Consumer preference
8. Local and regional dairy competition
9. Institutional sales opportunity
10. Pricing and margin structure

Market X can support dairy companies through market assessment, distributor identification, sales channel design, institutional sales mapping, and rural or urban expansion planning.
""",

    "agriculture_strategy.txt": """
Agriculture and plantation projects require field-level understanding of farmers, land availability, crop suitability, procurement models, government schemes, processing opportunities, and value chain economics.

A structured agriculture consulting approach includes:
1. Farmer mapping
2. Land and crop suitability analysis
3. Value chain assessment
4. Procurement model design
5. Farmer engagement plan
6. Government linkage
7. Processing and market access strategy
8. Commercial feasibility assessment
9. Risk analysis
10. Implementation roadmap

Market X can support companies in palm plantation, cocoa plantation, agriculture value chain development, farmer engagement, and state-level project execution.
""",

    "government_projects.txt": """
Government project consulting requires understanding policy objectives, stakeholder management, implementation planning, field execution, monitoring frameworks, and impact reporting.

Important elements include:
1. Department objective mapping
2. Scheme and policy alignment
3. Beneficiary identification
4. District-level execution plan
5. Partner and vendor coordination
6. Monitoring and evaluation framework
7. Reporting dashboard
8. Field team management
9. Impact assessment
10. Documentation and compliance

Market X can support government and institutional clients through project planning, implementation support, market linkage, field research, and monitoring frameworks.
""",

    "brand_development.txt": """
Brand development requires clear positioning, customer understanding, competitor benchmarking, product differentiation, communication strategy, pricing, packaging, and channel alignment.

A brand development roadmap includes:
1. Consumer research
2. Brand positioning
3. Competitor benchmarking
4. Product portfolio assessment
5. Pricing strategy
6. Packaging and communication review
7. Market launch planning
8. Retail visibility strategy
9. Sales enablement
10. Brand performance tracking

Market X can support brands through market research, positioning strategy, launch planning, consumer insights, and business development support.
""",

    "big4_delivery_style.txt": """
A consulting-grade advisory response should include executive summary, current situation diagnosis, market opportunity, operating model implications, risks, recommended roadmap, KPIs, and decision questions.

For FMCG CXO clients, advisory should be practical, data-seeking, commercially grounded, and execution-oriented. It should not sound generic.

A strong consulting answer should include:
1. Key business hypothesis
2. Market entry logic
3. Channel economics
4. Distribution model
5. Retail activation
6. Pilot design
7. Governance cadence
8. KPI dashboard
9. Risk and mitigation
10. 30-60-90 day roadmap
11. Information required for deeper recommendation

Important FMCG KPIs include numeric distribution, weighted distribution, secondary sales, outlet productivity, salesman productivity, fill rate, beat adherence, order frequency, average order value, stock rotation, return rate, claim settlement cycle, gross margin, distributor ROI, and market share.
"""
}


def create_default_knowledge_files():
    for filename, content in default_knowledge.items():
        file_path = os.path.join("data", filename)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content.strip())


create_default_knowledge_files()


# =========================================================
# LOAD KNOWLEDGE BASE
# =========================================================
def load_knowledge_base():
    documents = []
    file_names = []

    for file in os.listdir("data"):
        if file.endswith(".txt"):
            path = os.path.join("data", file)
            with open(path, "r", encoding="utf-8") as f:
                documents.append(f.read())
                file_names.append(file)

    return documents, file_names


documents, file_names = load_knowledge_base()


# =========================================================
# BUSINESS GUARDRAILS
# =========================================================
allowed_keywords = [
    "business", "fmcg", "dairy", "agriculture", "sales", "distribution",
    "market", "brand", "strategy", "government", "plantation", "retail",
    "channel", "growth", "revenue", "consumer", "product", "pricing",
    "competitor", "competition", "go to market", "gtm", "route to market",
    "rtm", "startup", "consulting", "profit", "expansion", "dealer",
    "distributor", "rural", "urban", "customer", "trade", "scheme",
    "marketing", "launch", "market research", "market entry",
    "business development", "supply chain", "sales team", "retailer",
    "wholesaler", "modern trade", "general trade", "bihar", "jharkhand",
    "uttar pradesh", "kerala", "india", "palm", "cocoa", "food",
    "beverage", "milk", "edible oil", "company", "firm", "enterprise",
    "manufacturer", "export", "import", "roi", "margin", "profitability",
    "turnover", "lead generation", "customer acquisition", "market share",
    "sku", "secondary sales", "primary sales", "beat", "outlet", "kirana",
    "super stockist", "stockist", "cfa", "warehouse", "cold chain",
    "modern trade", "quick commerce", "ecommerce", "institutional sales",
    "horeca", "general trade", "distribution model", "pilot", "category",
    "trade promotion", "trade marketing", "activation", "launch plan"
]

blocked_keywords = [
    "movie", "song", "game", "sports", "cricket", "relationship",
    "dating", "medical", "health", "politics", "election", "religion",
    "hack", "illegal", "celebrity", "astrology", "betting", "casino"
]


def is_business_question(query):
    query_lower = query.lower()

    if any(word in query_lower for word in blocked_keywords):
        return False

    if any(word in query_lower for word in allowed_keywords):
        return True

    business_patterns = [
        r"\bhow do i grow\b",
        r"\bhow to expand\b",
        r"\bincrease sales\b",
        r"\benter .* market\b",
        r"\bfind .* distributor\b",
        r"\blaunch .* product\b"
    ]

    if any(re.search(pattern, query_lower) for pattern in business_patterns):
        return True

    return False


# =========================================================
# RETRIEVE BEST CONTEXT FROM KNOWLEDGE BASE
# =========================================================
def retrieve_context(query, docs, names, top_k=3):
    if len(docs) == 0:
        return "", ""

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(docs + [query])

        query_vector = vectors[-1]
        document_vectors = vectors[:-1]

        similarity_scores = cosine_similarity(query_vector, document_vectors)[0]
        ranked_indices = similarity_scores.argsort()[::-1][:top_k]

        selected_contexts = []
        selected_files = []

        for idx in ranked_indices:
            if similarity_scores[idx] >= 0.025:
                selected_contexts.append(docs[idx])
                selected_files.append(names[idx])

        return "\n\n---\n\n".join(selected_contexts), ", ".join(selected_files)

    except Exception:
        return "", ""


# =========================================================
# IDENTIFY SERVICE AREA
# =========================================================
def identify_service(query):
    q = query.lower()

    if any(word in q for word in ["fmcg", "retail", "distributor", "distribution", "route to market", "rtm", "dealer", "kirana", "stockist", "super stockist"]):
        return "FMCG distribution and route-to-market strategy"

    if any(word in q for word in ["dairy", "milk", "paneer", "curd", "cheese", "cold chain"]):
        return "Dairy market development and sales expansion"

    if any(word in q for word in ["agriculture", "farmer", "plantation", "palm", "cocoa", "value chain"]):
        return "Agriculture and plantation consulting"

    if any(word in q for word in ["government", "state", "department", "scheme", "project", "tender"]):
        return "Government project advisory and implementation support"

    if any(word in q for word in ["brand", "branding", "consumer", "positioning", "launch", "packaging"]):
        return "Brand development and market positioning"

    if any(word in q for word in ["market research", "survey", "competition", "competitor", "consumer insight", "benchmark"]):
        return "Market research and competitor benchmarking"

    return "Business growth consulting"


# =========================================================
# READINESS SCORE ENGINE
# =========================================================
def calculate_readiness_score(industry, target_market, turnover, current_distribution, monthly_sales, sales_team_size, challenge):
    score = 0
    observations = []

    if industry in ["FMCG", "Dairy", "Food Processing"]:
        score += 15
    else:
        score += 8

    if target_market and len(target_market.strip()) > 2:
        score += 15
    else:
        observations.append("Target geography is not clearly defined.")

    if turnover not in ["Not Disclosed", ""]:
        score += 12
    else:
        observations.append("Turnover is not disclosed, limiting commercial feasibility assessment.")

    if current_distribution in ["Strong distributor network", "Moderate distributor network"]:
        score += 18
    elif current_distribution == "Early-stage distribution":
        score += 10
        observations.append("Distribution appears early-stage and may need structured channel building.")
    else:
        observations.append("Current distribution strength is unclear.")

    if monthly_sales in ["₹50 Lakh - ₹2 Cr", "₹2 Cr - ₹10 Cr", "Above ₹10 Cr"]:
        score += 15
    elif monthly_sales == "Below ₹50 Lakh":
        score += 8
    else:
        observations.append("Monthly sales range is not disclosed.")

    if sales_team_size in ["11 - 50", "51 - 200", "Above 200"]:
        score += 15
    elif sales_team_size in ["1 - 5", "6 - 10"]:
        score += 8
        observations.append("Sales team may need expansion or productivity improvement for scale.")
    else:
        observations.append("Sales team size is unclear.")

    if challenge and len(challenge.strip()) > 20:
        score += 10
    else:
        observations.append("Business challenge needs more detail.")

    score = min(score, 100)

    if score >= 75:
        status = "Expansion-ready"
        interpretation = "The business appears ready for a structured market expansion or scale-up program."
    elif score >= 50:
        status = "Needs controlled pilot"
        interpretation = "The business has potential but should validate assumptions through a pilot before scaling."
    else:
        status = "Needs foundation work"
        interpretation = "The business should first strengthen market clarity, channel design, and execution basics."

    return score, status, interpretation, observations


# =========================================================
# REPORT GENERATOR
# =========================================================
def create_report_text(company, name, industry, target_market, turnover, service_area, score, status, interpretation, challenge, advisor_answer):
    report = f"""
MARKET X GROWTH INTELLIGENCE REPORT
Generated on: {datetime.now().strftime("%d %b %Y, %I:%M %p")}

CLIENT SNAPSHOT
Company: {company or "Not provided"}
Contact: {name or "Not provided"}
Industry: {industry}
Target Market: {target_market or "Not provided"}
Approximate Annual Turnover: {turnover}
Relevant Advisory Area: {service_area}

EXPANSION READINESS
Score: {score}/100
Status: {status}
Interpretation: {interpretation}

BUSINESS CHALLENGE
{challenge or "Not provided"}

ADVISOR RECOMMENDATION
{advisor_answer}

IMPORTANT NOTE
This report is a high-level advisory output generated using the Market X Growth Intelligence Advisor.
A detailed consulting engagement should include primary market research, distributor due diligence,
retailer mapping, competitor benchmarking, channel economics, and execution governance.

Prepared by Market X | {MARKET_X_WEBSITE}
"""
    return report


def get_download_buffer(text):
    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)
    return buffer


# =========================================================
# GROQ LLM CALL
# =========================================================
def call_groq_llm(user_query, context, service_area, diagnostic_data=None):
    try:
        api_key = get_secret("GROQ_API_KEY", "")
        model = get_secret("GROQ_MODEL", "llama-3.1-8b-instant")

        if api_key == "":
            return None, "AI advisory engine is not configured."

        diagnostic_json = json.dumps(diagnostic_data or {}, indent=2)

        system_prompt = """
You are Market X Growth Intelligence Advisor.

You are a senior business consulting assistant for Market X, an India-focused consulting firm.
Your style should be similar to a top-tier management consulting advisory note:
clear, structured, commercially practical, concise, and executive-friendly.

You only answer questions related to:
- FMCG
- Dairy
- Agriculture
- Plantation projects
- Palm plantation
- Cocoa plantation
- Distribution expansion
- Sales strategy
- Route-to-market strategy
- Go-to-market strategy
- Market research
- Competitor benchmarking
- Brand development
- Government projects
- Retail activation
- Business growth
- Business development
- Consulting

If the user asks anything unrelated to business consulting, politely refuse and redirect to business topics.

Rules:
- Do not make fake claims.
- Do not guarantee results.
- Do not invent market statistics.
- If data is missing, state what information is needed.
- Use Indian market context when relevant.
- Be practical and execution-oriented.
- Avoid generic motivational language.
- Avoid sounding desperate or overly salesy.
- Naturally explain where Market X can support.
- Prefer 30-60-90 day roadmaps, KPIs, risks, and decision points.
- Output in clean Markdown.
"""

        user_prompt = f"""
User question:
{user_query}

Relevant service area:
{service_area}

Diagnostic data:
{diagnostic_json}

Relevant Market X knowledge base:
{context}

Prepare a consulting-grade response using this structure:

## Executive View
Give a sharp 3-5 sentence diagnosis.

## Strategic Recommendation
Explain the recommended direction.

## 30-60-90 Day Roadmap
Use a table with columns: Phase, Focus, Key Actions, Output.

## KPI Dashboard
Use a table with columns: KPI, Why It Matters, Target Direction.

## Key Risks and Mitigation
Use a table with columns: Risk, Business Impact, Mitigation.

## How Market X Can Support
Explain 4-6 specific consulting workstreams.

## Information Needed for a Board-Level Plan
List the missing data required for deeper advisory.
"""

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.28,
            "max_tokens": 1400
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45
        )

        if response.status_code != 200:
            return None, "AI advisory engine is temporarily unavailable."

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return answer, None

    except Exception:
        return None, "AI advisory engine is temporarily unavailable."


# =========================================================
# FALLBACK RESPONSE
# =========================================================
def fallback_response(context, service_area):
    return f"""
## Executive View

Your question is related to **{service_area}**. A strong answer requires understanding the target geography, current distribution strength, sales velocity, retailer universe, competitor intensity, channel margins, and execution capability.

## Strategic Recommendation

The best approach is to avoid immediate full-scale expansion and first build a market-backed route-to-market plan. This should include market mapping, distributor due diligence, channel economics, retailer activation, and a controlled pilot.

## 30-60-90 Day Roadmap

| Phase | Focus | Key Actions | Output |
|---|---|---|---|
| 0-30 Days | Market diagnosis | Map priority districts, competitors, retailer universe, distributor options, and price architecture. | Market opportunity map |
| 31-60 Days | Channel design | Select distributor model, define margins, design beats, finalize pilot SKUs, and build sales governance. | Pilot GTM blueprint |
| 61-90 Days | Pilot execution | Launch in selected territory, track secondary sales, outlet productivity, fill rate, and scheme response. | Scale-up recommendation |

## KPI Dashboard

| KPI | Why It Matters | Target Direction |
|---|---|---|
| Numeric distribution | Measures outlet reach. | Increase |
| Secondary sales | Shows real market pull. | Increase |
| Distributor ROI | Ensures channel sustainability. | Improve |
| Beat adherence | Tracks sales execution discipline. | Improve |
| Fill rate | Reduces lost sales. | Improve |
| Outlet productivity | Measures quality of retail coverage. | Increase |

## Key Risks and Mitigation

| Risk | Business Impact | Mitigation |
|---|---|---|
| Wrong distributor selection | Weak market penetration and poor collections. | Conduct distributor due diligence. |
| Poor retailer mapping | Low sales productivity. | Build outlet universe before launch. |
| Weak trade economics | Distributor and retailer disengagement. | Benchmark margins and schemes. |
| No pilot governance | Expansion mistakes get multiplied. | Review pilot KPIs weekly. |

## How Market X Can Support

Market X can support through market research, route-to-market strategy, distributor identification, competitor benchmarking, sales beat planning, retail activation, pilot launch governance, and performance dashboard design.

## Information Needed for a Board-Level Plan

- Product category and SKU portfolio
- Target geography
- Current monthly sales
- Existing distributor network
- Sales team size
- Trade margin structure
- Competitor names
- Investment budget and timeline
"""


# =========================================================
# MAIN RESPONSE GENERATOR
# =========================================================
def generate_response(query, diagnostic_data=None):
    if not is_business_question(query):
        return """
## I can help with business consulting questions only

This advisor is designed for business, consulting, FMCG, dairy, agriculture, distribution, branding, market research, government projects, and growth strategy questions.

Please share a business challenge related to market expansion, sales growth, channel development, brand building, route-to-market, or business transformation.
"""

    context, source_file = retrieve_context(query, documents, file_names)
    service_area = identify_service(query)

    llm_answer, error = call_groq_llm(query, context, service_area, diagnostic_data)

    if llm_answer:
        return llm_answer

    return fallback_response(context, service_area)


# =========================================================
# SAVE LEAD
# =========================================================
def save_lead(data):
    lead_file = "leads/leads.csv"

    new_lead = pd.DataFrame([data])

    if os.path.exists(lead_file):
        old_data = pd.read_csv(lead_file)
        final_data = pd.concat([old_data, new_lead], ignore_index=True)
    else:
        final_data = new_lead

    final_data.to_csv(lead_file, index=False)


# =========================================================
# TOP BRAND BAR
# =========================================================
st.markdown(
    f"""
    <div class="mx-topbar">
        <div class="mx-topbar-left">
            {LOGO_IMG_TAG}
            <div>
                <div class="mx-wordmark">Market X</div>
                <div class="mx-wordmark-sub">Growth Intelligence</div>
            </div>
        </div>
        <div class="mx-topbar-link">
            <a href="{MARKET_X_WEBSITE}" target="_blank">market-x.co.in ↗</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO — headline + condensed differentiators, single unit
# =========================================================
st.markdown(
    f"""
    <div class="mx-hero">
        <div class="mx-hero-accent"></div>
        <div class="mx-hero-grid">
            <div class="mx-hero-left">
                <div class="mx-hero-kicker">Consulting-Grade AI Advisory</div>
                <div class="mx-hero-title">Boardroom-grade advisory for market expansion.</div>
                <div class="mx-hero-subtitle">
                    An execution-focused advisor for FMCG, dairy, agriculture, distribution, route-to-market,
                    brand development, and government projects across Indian markets — built by Market X.
                </div>
                <div class="mx-hero-cta">
                    <a href="{MARKET_X_WEBSITE}" target="_blank" class="mx-cta-primary">Visit Market X →</a>
                    <a href="#" class="mx-cta-secondary" onclick="return false;">Ask below ↓</a>
                </div>
            </div>
            <div class="mx-hero-right">
                <div class="mx-diff-item">
                    <div class="mx-diff-label">Diagnostic Depth</div>
                    <div class="mx-diff-value">CXO Lens</div>
                    <div class="mx-diff-copy">Frames every question around growth, channel economics, and execution risk.</div>
                </div>
                <div class="mx-diff-item">
                    <div class="mx-diff-label">Core Strength</div>
                    <div class="mx-diff-value">Route-to-Market</div>
                    <div class="mx-diff-copy">Built for distribution expansion, retail activation, and sales transformation.</div>
                </div>
                <div class="mx-diff-item">
                    <div class="mx-diff-label">Output Style</div>
                    <div class="mx-diff-value">Board Note</div>
                    <div class="mx-diff-copy">Roadmap, KPIs, risks, and mitigation — ready to bring into a leadership review.</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(f'<div class="mx-sidebar-logo">{LOGO_IMG_TAG}</div>', unsafe_allow_html=True)

    st.markdown('<div class="mx-sidebar-kicker">Advisor Console</div>', unsafe_allow_html=True)

    advisory_mode = st.radio(
        "Choose advisory focus",
        [
            "FMCG Market Expansion",
            "Distributor Evaluation",
            "Route-to-Market Strategy",
            "Brand Launch",
            "Dairy Expansion",
            "Agriculture / Plantation",
            "Government Project Advisory",
            "General Business Growth"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="mx-sidebar-kicker">This Advisor Delivers</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mx-sidebar-list">
        Expansion readiness scoring<br>
        Consulting-style diagnosis<br>
        30-60-90 day roadmap<br>
        KPI dashboard<br>
        Risk and mitigation matrix<br>
        Downloadable advisory note
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="mx-sidebar-cta">
            Want a deeper, human-led engagement?<br>
            Visit <a href="{MARKET_X_WEBSITE}" target="_blank">market-x.co.in</a>
            or use the Consultation Request tab.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Growth Advisor",
        "Expansion Diagnostic",
        "Market X Capabilities",
        "Consultation Request"
    ]
)


# =========================================================
# TAB 1 - GROWTH ADVISOR
# =========================================================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Executive Advisory</div>', unsafe_allow_html=True)
    st.subheader("Ask a Strategic Business Question")

    example_questions = {
        "FMCG Market Expansion": "We are an FMCG brand planning to expand in Bihar and Jharkhand. How should we design our route-to-market strategy?",
        "Distributor Evaluation": "How should we evaluate distributors before appointing them for a new FMCG territory?",
        "Route-to-Market Strategy": "What is the best route-to-market model for a packaged food brand entering Tier 2 and Tier 3 towns?",
        "Brand Launch": "We want to launch a new food brand in North India. What market research and launch plan should we follow?",
        "Dairy Expansion": "We are a dairy company planning to expand value-added dairy products in eastern India. What should we evaluate?",
        "Agriculture / Plantation": "We want to develop a palm plantation project with farmer engagement. What should be the roadmap?",
        "Government Project Advisory": "How should a government-linked agriculture project be planned and monitored at district level?",
        "General Business Growth": "Our sales have stagnated. How should we identify growth opportunities and improve market execution?"
    }

    user_query = st.text_area(
        "Describe your business challenge",
        value=example_questions.get(advisory_mode, ""),
        height=170
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        ask_button = st.button("Generate Advisory Note", type="primary")

    with col_b:
        clear_button = st.button("Clear Question", type="secondary")

    if clear_button:
        st.rerun()

    if ask_button:
        if user_query.strip() == "":
            st.warning("Please enter your business question.")
        else:
            diagnostic_data = {
                "advisory_mode": advisory_mode
            }

            with st.spinner("Preparing your consulting-grade advisory note..."):
                answer = generate_response(user_query, diagnostic_data)

            st.session_state["latest_answer"] = answer
            st.session_state["latest_query"] = user_query

            st.markdown("### Advisor Response")
            st.markdown(answer)

    if "latest_answer" in st.session_state:
        st.markdown(
            """
            <div class="success-box">
                <b>Boardroom-ready next step:</b> Use the Expansion Diagnostic tab to convert this advisory into a qualified market-entry report,
                or submit a Consultation Request to work directly with the Market X team.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2 - EXPANSION DIAGNOSTIC
# =========================================================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Growth Readiness Assessment</div>', unsafe_allow_html=True)
    st.subheader("Expansion Diagnostic")

    st.write(
        "This diagnostic converts a business requirement into a readiness score, risk view, and downloadable advisory report."
    )

    d1, d2 = st.columns(2)

    with d1:
        diag_company = st.text_input("Company Name", key="diag_company")
        diag_name = st.text_input("Your Name", key="diag_name")

        diag_industry = st.selectbox(
            "Industry",
            [
                "FMCG",
                "Dairy",
                "Agriculture",
                "Government",
                "Food Processing",
                "Retail",
                "Manufacturing",
                "Startup",
                "Other"
            ],
            key="diag_industry"
        )

        diag_turnover = st.selectbox(
            "Approximate Annual Turnover",
            [
                "Below ₹10 Cr",
                "₹10 Cr - ₹50 Cr",
                "₹50 Cr - ₹100 Cr",
                "₹100 Cr - ₹500 Cr",
                "Above ₹500 Cr",
                "Not Disclosed"
            ],
            key="diag_turnover"
        )

    with d2:
        diag_target_market = st.text_input(
            "Target Market / Geography",
            placeholder="Example: Bihar, Jharkhand, UP, Kerala",
            key="diag_target_market"
        )

        diag_current_distribution = st.selectbox(
            "Current Distribution Strength",
            [
                "Not Disclosed",
                "No existing distribution",
                "Early-stage distribution",
                "Moderate distributor network",
                "Strong distributor network"
            ],
            key="diag_current_distribution"
        )

        diag_monthly_sales = st.selectbox(
            "Current Monthly Sales",
            [
                "Not Disclosed",
                "Below ₹50 Lakh",
                "₹50 Lakh - ₹2 Cr",
                "₹2 Cr - ₹10 Cr",
                "Above ₹10 Cr"
            ],
            key="diag_monthly_sales"
        )

        diag_sales_team_size = st.selectbox(
            "Sales Team Size",
            [
                "Not Disclosed",
                "1 - 5",
                "6 - 10",
                "11 - 50",
                "51 - 200",
                "Above 200"
            ],
            key="diag_sales_team_size"
        )

    diag_challenge = st.text_area(
        "Business Challenge",
        placeholder="Example: We want to appoint distributors in Bihar but do not know which districts to prioritize.",
        height=110,
        key="diag_challenge"
    )

    run_diag = st.button("Run Expansion Diagnostic", type="primary")

    if run_diag:
        score, status, interpretation, observations = calculate_readiness_score(
            industry=diag_industry,
            target_market=diag_target_market,
            turnover=diag_turnover,
            current_distribution=diag_current_distribution,
            monthly_sales=diag_monthly_sales,
            sales_team_size=diag_sales_team_size,
            challenge=diag_challenge
        )

        diagnostic_data = {
            "company": diag_company,
            "industry": diag_industry,
            "target_market": diag_target_market,
            "turnover": diag_turnover,
            "current_distribution": diag_current_distribution,
            "monthly_sales": diag_monthly_sales,
            "sales_team_size": diag_sales_team_size,
            "challenge": diag_challenge,
            "readiness_score": score,
            "readiness_status": status
        }

        service_area = identify_service(f"{diag_industry} {diag_challenge} {diag_target_market}")

        diagnostic_query = f"""
        Company is in {diag_industry}. Target market is {diag_target_market}.
        Annual turnover is {diag_turnover}. Current distribution: {diag_current_distribution}.
        Monthly sales: {diag_monthly_sales}. Sales team size: {diag_sales_team_size}.
        Business challenge: {diag_challenge}.
        Prepare a market expansion diagnostic and route-to-market recommendation.
        """

        with st.spinner("Running Market X diagnostic..."):
            advisor_answer = generate_response(diagnostic_query, diagnostic_data)

        st.session_state["diagnostic_answer"] = advisor_answer
        st.session_state["diagnostic_score"] = score
        st.session_state["diagnostic_status"] = status
        st.session_state["diagnostic_interpretation"] = interpretation
        st.session_state["diagnostic_observations"] = observations

        st.markdown("### Expansion Readiness Score")

        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric("Readiness Score", f"{score}/100")

        with k2:
            st.metric("Status", status)

        with k3:
            st.metric("Advisory Area", service_area)

        st.progress(score / 100)

        box_class = "success-box" if score >= 75 else "warning-box"
        st.markdown(
            f"""
            <div class="{box_class}">
                <b>{status}:</b> {interpretation}
            </div>
            """,
            unsafe_allow_html=True
        )

        if observations:
            st.markdown("### Diagnostic Observations")
            for obs in observations:
                st.write(f"- {obs}")

        st.markdown("### Consulting Recommendation")
        st.markdown(advisor_answer)

        report_text = create_report_text(
            company=diag_company,
            name=diag_name,
            industry=diag_industry,
            target_market=diag_target_market,
            turnover=diag_turnover,
            service_area=service_area,
            score=score,
            status=status,
            interpretation=interpretation,
            challenge=diag_challenge,
            advisor_answer=advisor_answer
        )

        dl1, dl2 = st.columns([1, 1.4])

        with dl1:
            st.download_button(
                label="Download Advisory Report",
                data=get_download_buffer(report_text),
                file_name=f"market_x_growth_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

        with dl2:
            st.markdown(
                f"""
                <div style="padding-top: 10px;" class="small-text">
                    <a href="{MARKET_X_WEBSITE}" target="_blank">Discuss this report with a Market X consultant →</a>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 3 - CAPABILITIES
# =========================================================
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Consulting Capabilities</div>', unsafe_allow_html=True)
    st.subheader("What Market X Can Deliver")

    capability_rows = [
        {
            "Capability": "FMCG Route-to-Market",
            "Business Question": "Where should we expand and what channel model should we use?",
            "Typical Output": "District prioritization, distributor model, beat plan, launch roadmap"
        },
        {
            "Capability": "Distributor Due Diligence",
            "Business Question": "Which distributor is commercially and operationally fit?",
            "Typical Output": "Distributor scorecard, risk profile, appointment recommendation"
        },
        {
            "Capability": "Market Research",
            "Business Question": "What is the real demand, competition, and pricing structure?",
            "Typical Output": "Retail survey, competitor benchmarking, channel insights"
        },
        {
            "Capability": "Sales Transformation",
            "Business Question": "Why is sales productivity low?",
            "Typical Output": "KPI dashboard, beat redesign, sales governance rhythm"
        },
        {
            "Capability": "Brand Development",
            "Business Question": "How should we position, price, and launch our brand?",
            "Typical Output": "Positioning strategy, launch plan, retail visibility roadmap"
        },
        {
            "Capability": "Government Project Advisory",
            "Business Question": "How should a field project be designed and monitored?",
            "Typical Output": "Implementation plan, monitoring framework, impact dashboard"
        }
    ]

    st.dataframe(pd.DataFrame(capability_rows), use_container_width=True, hide_index=True)

    st.markdown("### Board-Level FMCG Metrics Market X Can Track")

    metric_rows = [
        {
            "Metric": "Numeric Distribution",
            "Meaning": "Number of outlets carrying the product",
            "Why It Matters": "Measures reach"
        },
        {
            "Metric": "Weighted Distribution",
            "Meaning": "Presence in high-sales-value outlets",
            "Why It Matters": "Measures quality of reach"
        },
        {
            "Metric": "Secondary Sales",
            "Meaning": "Sales from distributor to retailer",
            "Why It Matters": "Shows actual market movement"
        },
        {
            "Metric": "Outlet Productivity",
            "Meaning": "Average sales per active outlet",
            "Why It Matters": "Shows sales quality"
        },
        {
            "Metric": "Beat Adherence",
            "Meaning": "Salesperson visits as per plan",
            "Why It Matters": "Tracks field discipline"
        },
        {
            "Metric": "Distributor ROI",
            "Meaning": "Distributor return after costs",
            "Why It Matters": "Ensures channel sustainability"
        }
    ]

    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        <div class="insight-box">
            <b>Why clients choose Market X:</b> Large FMCG and food businesses don't just need advice —
            they need decision support, governance metrics, and execution visibility. This is what
            positions Market X as an execution-focused consulting partner, not just an advisory shop.
            <br><br>
            <a href="{MARKET_X_WEBSITE}" target="_blank">Learn more about our frameworks →</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 4 - CONSULTATION REQUEST
# =========================================================
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Qualified Lead Capture</div>', unsafe_allow_html=True)
    st.subheader("Request a Market Expansion Consultation")

    with st.form("lead_form"):
        f1, f2 = st.columns(2)

        with f1:
            name = st.text_input("Your Name *")
            designation = st.text_input("Designation")
            company = st.text_input("Company Name")
            email = st.text_input("Email")

        with f2:
            phone = st.text_input("Phone Number *")
            industry = st.selectbox(
                "Industry",
                [
                    "FMCG",
                    "Dairy",
                    "Agriculture",
                    "Government",
                    "Food Processing",
                    "Retail",
                    "Manufacturing",
                    "Startup",
                    "Other"
                ]
            )

            turnover = st.selectbox(
                "Approximate Annual Turnover",
                [
                    "Below ₹10 Cr",
                    "₹10 Cr - ₹50 Cr",
                    "₹50 Cr - ₹100 Cr",
                    "₹100 Cr - ₹500 Cr",
                    "Above ₹500 Cr",
                    "Not Disclosed"
                ]
            )

            consultation_type = st.selectbox(
                "Consultation Need",
                [
                    "Market Expansion",
                    "Distributor Appointment",
                    "Route-to-Market Strategy",
                    "Market Research",
                    "Sales Transformation",
                    "Brand Launch",
                    "Government Project",
                    "Agriculture / Plantation",
                    "Other"
                ]
            )

        location = st.text_input(
            "Target Market / Location",
            placeholder="Example: Bihar, Jharkhand, UP, Kerala"
        )

        challenge = st.text_area(
            "Business Challenge *",
            placeholder="Briefly describe your business challenge",
            height=110
        )

        urgency = st.selectbox(
            "Timeline",
            [
                "Immediate",
                "Within 30 days",
                "1 - 3 months",
                "3 - 6 months",
                "Exploratory"
            ]
        )

        submitted = st.form_submit_button("Submit Consultation Request", type="primary")

        if submitted:
            if name.strip() == "" or phone.strip() == "" or challenge.strip() == "":
                st.error("Please fill Name, Phone Number, and Business Challenge.")
            else:
                score, status, interpretation, observations = calculate_readiness_score(
                    industry=industry,
                    target_market=location,
                    turnover=turnover,
                    current_distribution="Not Disclosed",
                    monthly_sales="Not Disclosed",
                    sales_team_size="Not Disclosed",
                    challenge=challenge
                )

                lead_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name,
                    "designation": designation,
                    "company": company,
                    "email": email,
                    "phone": phone,
                    "industry": industry,
                    "turnover": turnover,
                    "consultation_type": consultation_type,
                    "target_location": location,
                    "challenge": challenge,
                    "urgency": urgency,
                    "readiness_score": score,
                    "readiness_status": status
                }

                save_lead(lead_data)

                st.success("Your consultation request has been submitted.")
                st.markdown(
                    f"""
                    <div class="success-box">
                        <b>Initial readiness view:</b> {status} with a score of {score}/100.
                        A Market X consultant will use this information to prepare a focused discussion.
                        You can also reach us directly at
                        <a href="{MARKET_X_WEBSITE}" target="_blank">market-x.co.in</a>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <hr>
    <div class="mx-footer">
        {LOGO_IMG_TAG}
        <div>
            <div class="mx-footer-brand">Market X — Strategy, Brand &amp; Transformation Consulting</div>
            <div class="mx-footer-text">
                <a href="{MARKET_X_WEBSITE}" target="_blank">market-x.co.in</a> ·
                This advisor provides high-level guidance. Recommendations should be validated through
                primary market research, distributor due diligence, and a client-specific engagement.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
