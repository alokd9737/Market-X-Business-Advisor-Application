# app.py
import streamlit as st
import pandas as pd
import os
import re
import json
import requests
from datetime import datetime
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Market X Growth Intelligence Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure assets folder exists
Path("assets").mkdir(exist_ok=True)

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
for folder in ["data", "leads", "reports", "assets"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# =========================================================
# CUSTOM CSS - BIG 4 STYLE (updated)
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .mx-header {
        display:flex;
        align-items:center;
        justify-content:space-between;
        background: linear-gradient(90deg,#07111f 0%, #0f3b6b 60%);
        padding:18px 22px;
        border-radius:12px;
        color: #fff;
        box-shadow: 0 10px 30px rgba(2,6,23,0.35);
        margin-bottom:18px;
    }
    .mx-logo {
        display:flex;
        align-items:center;
        gap:14px;
    }
    .mx-title {
        font-weight:800;
        font-size:22px;
        margin:0;
        line-height:1;
    }
    .mx-sub {
        font-size:13px;
        color: #dbeafe;
        margin-top:4px;
    }

    .mx-nav {
        display:flex;
        gap:10px;
        align-items:center;
    }
    .mx-pill {
        background: rgba(255,255,255,0.06);
        padding:8px 12px;
        border-radius:8px;
        color:#e6f0ff;
        font-weight:600;
        font-size:13px;
    }

    .mx-card {
        background: #fff;
        border-radius:12px;
        padding:18px;
        box-shadow: 0 8px 20px rgba(15,23,42,0.06);
        border: 1px solid #e6eef8;
    }

    .mx-metric-title { color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase; }
    .mx-metric-value { font-size:26px; font-weight:800; color:#0f172a; }

    .glass-card {
        background: linear-gradient(135deg, rgba(7,17,31,0.95) 0%, rgba(16,42,86,0.95) 48%, rgba(15,118,110,0.95) 100%);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0px 18px 45px rgba(15, 23, 42, 0.25);
    }

    .small-text {
        font-size:13px;
        color:#64748b;
    }

    .mx-footer { color:#64748b; font-size:13px; padding-top:12px; }

    /* Responsive tweaks */
    @media (max-width: 800px) {
        .mx-header { flex-direction:column; gap:12px; align-items:flex-start; }
        .mx-nav { width:100%; justify-content:flex-start; flex-wrap:wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DEFAULT KNOWLEDGE BASE (unchanged)
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
# BUSINESS GUARDRAILS (unchanged)
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
"""
    return report

def get_download_buffer(text):
    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)
    return buffer

# =========================================================
# GROQ LLM CALL (unchanged)
# =========================================================
def call_groq_llm(user_query, context, service_area, diagnostic_data=None):
    try:
        api_key = get_secret("GROQ_API_KEY", "")
        model = get_secret("GROQ_MODEL", "llama-3.1-8b-instant")

        if api_key == "":
            return None, "Groq API key missing."

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

        url = "[api.groq.com](https://api.groq.com/openai/v1/chat/completions)"

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
            return None, f"Groq API error: {response.status_code} - {response.text}"

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return answer, None

    except Exception as e:
        return None, str(e)

# =========================================================
# FALLBACK RESPONSE (unchanged)
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
""", "Blocked: Non-business question"

    context, source_file = retrieve_context(query, documents, file_names)
    service_area = identify_service(query)

    llm_answer, error = call_groq_llm(query, context, service_area, diagnostic_data)

    if llm_answer:
        return llm_answer, f"Groq LLM | Knowledge files: {source_file or 'No specific file matched'}"

    fallback = fallback_response(context, service_area)
    return fallback, f"Fallback used. Reason: {error}"

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
# UI: Header with logo uploader and nav
# =========================================================
def render_header(default_logo_path="assets/market_x_logo.png"):
    # Left column: logo uploader and preview
    col1, col2 = st.columns([1, 3], gap="small")
    with col1:
        uploaded = st.file_uploader("Upload logo (optional)", type=["png","jpg","jpeg"], key="logo_uploader")
        logo_path = default_logo_path
        if uploaded:
            logo_bytes = uploaded.read()
            try:
                with open(default_logo_path, "wb") as f:
                    f.write(logo_bytes)
                logo_path = default_logo_path
            except Exception:
                logo_path = None

        if logo_path and os.path.exists(logo_path):
            st.image(logo_path, width=120)
        else:
            st.markdown("<div style='font-weight:800;color:#07111f;font-size:20px'>MARKET X</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="mx-header">
                <div style="display:flex;flex-direction:column;">
                    <div class="mx-title">Market X Growth Intelligence</div>
                    <div class="mx-sub">Boardroom-grade business advisory for market expansion.</div>
                </div>
                <div class="mx-nav">
                    <div class="mx-pill">Overview</div>
                    <div class="mx-pill">Readiness</div>
                    <div class="mx-pill">Reports</div>
                    <div class="mx-pill">Leads</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Render header
render_header()

# =========================================================
# VALUE CARDS (improved layout)
# =========================================================
st.markdown("<div style='display:flex;gap:18px;margin-top:8px'>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4, gap="large")
with m1:
    st.markdown('<div class="mx-card"><div class="mx-metric-title">Readiness Score</div><div class="mx-metric-value">--</div><div class="small-text">Run assessment to compute</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="mx-card"><div class="mx-metric-title">Active Pilots</div><div class="mx-metric-value">--</div><div class="small-text">No active pilots</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="mx-card"><div class="mx-metric-title">Leads (30d)</div><div class="mx-metric-value">--</div><div class="small-text">Qualified</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="mx-card"><div class="mx-metric-title">Reports</div><div class="mx-metric-value">--</div><div class="small-text">Downloadable</div></div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Glass hero / quick advisory
st.markdown(
    """
    <div class="glass-card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-weight:800;font-size:20px">Quick Advisory</div>
                <div style="margin-top:6px;color:#dbeafe;max-width:900px">Start with a 90-day pilot: market mapping, distributor due diligence, SKU pilot, and weekly KPI governance. Use numeric distribution and secondary sales as primary success metrics.</div>
            </div>
            <div>
                <!-- placeholder for CTA -->
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR: Input form for assessment and report generation
# =========================================================
st.sidebar.header("Run Readiness Assessment")
company = st.sidebar.text_input("Company name")
contact_name = st.sidebar.text_input("Contact person")
industry = st.sidebar.selectbox("Industry", ["FMCG", "Dairy", "Agriculture", "Food Processing", "Other"])
target_market = st.sidebar.text_input("Target geography (state/district)")
turnover = st.sidebar.selectbox("Approx annual turnover", ["Not Disclosed", "Below ₹1 Cr", "₹1 Cr - ₹10 Cr", "₹10 Cr - ₹50 Cr", "Above ₹50 Cr"])
current_distribution = st.sidebar.selectbox("Current distribution strength", ["Not Disclosed", "Early-stage distribution", "Moderate distributor network", "Strong distributor network"])
monthly_sales = st.sidebar.selectbox("Monthly sales range", ["Not Disclosed", "Below ₹50 Lakh", "₹50 Lakh - ₹2 Cr", "₹2 Cr - ₹10 Cr", "Above ₹10 Cr"])
sales_team_size = st.sidebar.selectbox("Sales team size", ["Not Disclosed", "1 - 5", "6 - 10", "11 - 50", "51 - 200", "Above 200"])
challenge = st.sidebar.text_area("Describe the business challenge (brief)")

run_assess = st.sidebar.button("Run assessment and generate report")

# =========================================================
# MAIN: handle assessment, LLM call, report download
# =========================================================
if run_assess:
    # calculate readiness
    score, status, interpretation, observations = calculate_readiness_score(
        industry, target_market, turnover, current_distribution, monthly_sales, sales_team_size, challenge
    )

    # prepare diagnostic data for LLM
    diagnostic_data = {
        "company": company,
        "contact": contact_name,
        "industry": industry,
        "target_market": target_market,
        "turnover": turnover,
        "current_distribution": current_distribution,
        "monthly_sales": monthly_sales,
        "sales_team_size": sales_team_size,
        "challenge": challenge
    }

    # generate advisory via LLM
    query = f"Provide a board-level advisory for {company or 'a client'} in {industry} targeting {target_market or 'unspecified geography'}."
    advisor_answer, meta = generate_response(query, diagnostic_data)

    # show results
    st.markdown(f"### Expansion Readiness — **{score}/100**  \n**Status:** {status}")
    st.markdown(f"**Interpretation:** {interpretation}")
    if observations:
        st.markdown("**Observations:**")
        for obs in observations:
            st.markdown(f"- {obs}")

    st.markdown("---")
    st.markdown("## Advisor Recommendation")
    st.markdown(advisor_answer)

    # create report and download
    report_text = create_report_text(company, contact_name, industry, target_market, turnover, identify_service(query), score, status, interpretation, challenge, advisor_answer)
    buffer = get_download_buffer(report_text)
    st.download_button("Download report (TXT)", buffer, file_name=f"marketx_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")

    # save lead
    lead_data = {
        "timestamp": datetime.now().isoformat(),
        "company": company,
        "contact": contact_name,
        "industry": industry,
        "target_market": target_market,
        "turnover": turnover,
        "score": score
    }
    try:
        save_lead(lead_data)
        st.success("Lead saved.")
    except Exception:
        st.warning("Could not save lead to CSV. Check file permissions.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("<div class='mx-footer'>Market X Growth Intelligence — Boardroom-grade advisory for market expansion.</div>", unsafe_allow_html=True)
