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
