import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Market X Business Advisor",
    page_icon="💼",
    layout="wide"
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
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 35px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.18);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #dbeafe;
        line-height: 1.5;
    }

    .lead-box {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
    }

    .cta-box {
        background-color: #eff6ff;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #bfdbfe;
        margin-top: 15px;
    }

    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
        text-align: center;
        min-height: 120px;
    }

    .small-text {
        font-size: 14px;
        color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CREATE REQUIRED FOLDERS
# =========================================================
if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("leads"):
    os.makedirs("leads")


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
    "turnover", "lead generation", "customer acquisition", "market share"
]

blocked_keywords = [
    "movie", "song", "game", "sports", "cricket", "relationship",
    "dating", "medical", "health", "politics", "election", "religion",
    "hack", "illegal", "celebrity"
]


def is_business_question(query):
    query_lower = query.lower()

    if any(word in query_lower for word in blocked_keywords):
        return False

    if any(word in query_lower for word in allowed_keywords):
        return True

    return False


# =========================================================
# RETRIEVE BEST CONTEXT FROM KNOWLEDGE BASE
# =========================================================
def retrieve_context(query, docs, names):
    if len(docs) == 0:
        return "", ""

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(docs + [query])

        query_vector = vectors[-1]
        document_vectors = vectors[:-1]

        similarity_scores = cosine_similarity(query_vector, document_vectors)
        best_index = similarity_scores.argmax()
        best_score = similarity_scores[0][best_index]

        if best_score < 0.03:
            return "", ""

        return docs[best_index], names[best_index]

    except Exception:
        return "", ""


# =========================================================
# IDENTIFY SERVICE AREA
# =========================================================
def identify_service(query):
    q = query.lower()

    if any(word in q for word in ["fmcg", "retail", "distributor", "distribution", "route to market", "rtm", "dealer"]):
        return "FMCG distribution and route-to-market strategy"

    if any(word in q for word in ["dairy", "milk", "paneer", "curd", "cheese"]):
        return "Dairy market development and sales expansion"

    if any(word in q for word in ["agriculture", "farmer", "plantation", "palm", "cocoa", "value chain"]):
        return "Agriculture and plantation consulting"

    if any(word in q for word in ["government", "state", "department", "scheme", "project"]):
        return "Government project advisory and implementation support"

    if any(word in q for word in ["brand", "branding", "consumer", "positioning", "launch"]):
        return "Brand development and market positioning"

    if any(word in q for word in ["market research", "survey", "competition", "competitor", "consumer insight"]):
        return "Market research and competitor benchmarking"

    return "Business growth consulting"


# =========================================================
# GROQ LLM CALL
# =========================================================
def call_groq_llm(user_query, context, service_area):
    try:
        api_key = get_secret("GROQ_API_KEY", "")
        model = get_secret("GROQ_MODEL", "llama-3.1-8b-instant")

        if api_key == "":
            return None, "Groq API key missing."

        system_prompt = """
You are Market X Business Advisor.

You are a professional consulting assistant for a business consulting firm.

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

If a user asks anything unrelated to business consulting, politely refuse and redirect them to business topics.

Your role:
1. Understand the user's business problem.
2. Give clear, practical, structured business advice.
3. Highlight common mistakes or risks.
4. Naturally explain how Market X can help.
5. Encourage the user to share industry, target market, turnover, and contact details for a customized roadmap.

Important rules:
- Do not sound desperate or overly salesy.
- Do not make fake claims.
- Do not guarantee results.
- Do not answer medical, legal, political, entertainment, personal, or unrelated questions.
- Keep the response professional and consulting-oriented.
- Use Indian business context when relevant.
- Format with headings and bullet points.
"""

        user_prompt = f"""
User question:
{user_query}

Relevant service area:
{service_area}

Relevant Market X knowledge base:
{context}

Now provide a useful consulting-style answer.

Response format:
1. Business Perspective
2. Key Steps / Recommendations
3. Common Risks
4. How Market X Can Support
5. Suggested Next Action
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
            "temperature": 0.35,
            "max_tokens": 900
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=40
        )

        if response.status_code != 200:
            return None, f"Groq API error: {response.status_code} - {response.text}"

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return answer, None

    except Exception as e:
        return None, str(e)


# =========================================================
# FALLBACK RESPONSE IF GROQ FAILS
# =========================================================
def fallback_response(context, service_area):
    if context:
        return f"""
## Business Perspective

Your question is related to **{service_area}**.

Based on Market X's consulting approach, the following points are important:

{context[:1400]}

## Recommended Next Steps

For this type of business challenge, you should evaluate:

1. Current market presence
2. Target geography and customer segment
3. Existing distributor or channel strength
4. Competitor activity
5. Pricing and trade scheme structure
6. Sales team productivity
7. Retailer or customer activation plan
8. Execution timeline and investment requirement

## Common Risks

Many businesses struggle because they expand without proper market mapping, wrong distributor selection, weak channel margins, unclear sales accountability, and poor retailer activation.

## How Market X Can Support

Market X can support through:

- Market research
- Route-to-market planning
- Distributor identification
- Sales expansion strategy
- Brand development
- Competitor benchmarking
- Government or institutional project advisory
- Execution roadmap preparation

## Suggested Next Action

Please share your industry, target location, approximate turnover, and current challenge so the Market X team can suggest a more specific roadmap.
"""

    return f"""
## Business Perspective

Your question appears related to **{service_area}**.

A practical consulting approach would be to first understand:

1. Which market you want to enter
2. Which customer segment you want to target
3. How strong your existing sales or distribution network is
4. Which competitors are already active
5. What pricing and trade schemes are currently used
6. What investment and timeline you are planning

## Common Risks

Many companies expand without market research, distributor due diligence, retailer mapping, channel margin analysis, or execution tracking. This can lead to slow sales, poor distributor performance, and weak market penetration.

## How Market X Can Support

Market X can help with market research, distribution expansion, go-to-market strategy, channel partner identification, brand development, competitor benchmarking, and execution planning.

## Suggested Next Action

Share your business category and target geography so that our consulting team can suggest a practical next-step roadmap.
"""


# =========================================================
# MAIN RESPONSE GENERATOR
# =========================================================
def generate_response(query):
    if not is_business_question(query):
        return """
## I can help with business questions only

I am designed to assist only with business, consulting, FMCG, dairy, agriculture, distribution, branding, market research, government projects, and growth strategy questions.

If you are facing a business challenge related to market expansion, sales growth, channel development, brand building, or business transformation, please share your question and I will be happy to help.
""", "Blocked: Non-business question"

    context, source_file = retrieve_context(query, documents, file_names)
    service_area = identify_service(query)

    llm_answer, error = call_groq_llm(query, context, service_area)

    if llm_answer:
        return llm_answer, "Groq LLM"

    fallback = fallback_response(context, service_area)
    return fallback, f"Fallback used. Reason: {error}"


# =========================================================
# SAVE LEAD
# =========================================================
def save_lead(name, company, email, phone, industry, turnover, location, challenge):
    lead_file = "leads/leads.csv"

    new_lead = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "company": company,
        "email": email,
        "phone": phone,
        "industry": industry,
        "turnover": turnover,
        "target_location": location,
        "challenge": challenge
    }])

    if os.path.exists(lead_file):
        old_data = pd.read_csv(lead_file)
        final_data = pd.concat([old_data, new_lead], ignore_index=True)
    else:
        final_data = new_lead

    final_data.to_csv(lead_file, index=False)


# =========================================================
# UI HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Market X Business Advisor</div>
        <div class="hero-subtitle">
            AI-powered business advisory assistant for FMCG, dairy, agriculture, distribution, brand development, government projects and market expansion.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP CARDS
# =========================================================
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        """
        <div class="metric-card">
            <h4>Business-Only</h4>
            <p>Focused on consulting and market growth questions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <h4>AI Powered</h4>
            <p>Uses Groq LLM with business guardrails.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
            <h4>Lead Capture</h4>
            <p>Captures consultation requests from visitors.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("About This Advisor")

    st.write("This assistant is designed for business and consulting-related questions only.")

    st.markdown("### Key Areas")
    st.write("- FMCG growth")
    st.write("- Dairy expansion")
    st.write("- Agriculture consulting")
    st.write("- Distribution strategy")
    st.write("- Brand development")
    st.write("- Market research")
    st.write("- Government projects")
    st.write("- Plantation projects")

    st.markdown("---")

    st.markdown("### AI Status")

    groq_key = get_secret("GROQ_API_KEY", "")

    if groq_key:
        st.success("Groq API key available")
    else:
        st.warning("Groq API key missing. Fallback mode will be used.")

    st.markdown("### Model")
    st.code(get_secret("GROQ_MODEL", "llama-3.1-8b-instant"))


# =========================================================
# MAIN LAYOUT
# =========================================================
left_col, right_col = st.columns([1.35, 1])


with left_col:
    st.subheader("Ask a Business Question")

    user_query = st.text_area(
        "Describe your business challenge",
        placeholder="Example: We want to expand our FMCG distribution in Bihar. How should we start?",
        height=160
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        ask_button = st.button("Get Business Advice", type="primary")

    with col_b:
        clear_button = st.button("Clear")

    if clear_button:
        st.rerun()

    if ask_button:
        if user_query.strip() == "":
            st.warning("Please enter your business question.")
        else:
            with st.spinner("Market X Business Advisor is preparing your response..."):
                answer, source = generate_response(user_query)

            st.markdown("### Advisor Response")
            st.markdown(answer)

            with st.expander("Technical response source"):
                st.write(source)

            st.markdown(
                """
                <div class="cta-box">
                    <b>Want a customized business roadmap?</b><br>
                    Submit your details on the right side and the Market X team can review your requirement.
                </div>
                """,
                unsafe_allow_html=True
            )


with right_col:
    st.markdown('<div class="lead-box">', unsafe_allow_html=True)

    st.subheader("Request Consultation")

    with st.form("lead_form"):
        name = st.text_input("Your Name *")
        company = st.text_input("Company Name")
        email = st.text_input("Email")
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

        location = st.text_input(
            "Target Market / Location",
            placeholder="Example: Bihar, Jharkhand, UP, Kerala"
        )

        challenge = st.text_area(
            "Business Challenge *",
            placeholder="Briefly describe your business challenge"
        )

        submitted = st.form_submit_button("Submit Request")

        if submitted:
            if name.strip() == "" or phone.strip() == "" or challenge.strip() == "":
                st.error("Please fill Name, Phone Number, and Business Challenge.")
            else:
                save_lead(
                    name=name,
                    company=company,
                    email=email,
                    phone=phone,
                    industry=industry,
                    turnover=turnover,
                    location=location,
                    challenge=challenge
                )
                st.success("Thank you. Your consultation request has been submitted.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <p class="small-text">
    Disclaimer: This advisor provides high-level business guidance. For customized consulting recommendations, please connect with the Market X team.
    </p>
    """,
    unsafe_allow_html=True
)
