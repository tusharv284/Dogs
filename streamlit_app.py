import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dog App Analytics", layout="wide")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "dog_app_data.csv")
    df = pd.read_csv(csv_path)
    df["ownership_experience_encoded"] = df["ownership_years"].map(
        {"<1": 1, "1-3": 2, "4-7": 3, "8+": 4}
    )
    return df

df = load_data()

st.title("🐾 India Dog Care App — Survey Analytics Dashboard")

# ── FILTERS ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

age = col1.multiselect(
    "Age Group",
    sorted(df["age_group"].unique()),
    default=sorted(df["age_group"].unique())
)

region = col2.multiselect(
    "Region",
    sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

df_f = df[(df["age_group"].isin(age)) & (df["region"].isin(region))].copy()
df_no_na = df_f.dropna(subset=["monthly_spend_inr"]).copy()

# ── KPI METRICS ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric("Respondents", len(df_f))
c2.metric("Avg Spend", f"₹{df_no_na['monthly_spend_inr'].mean():.0f}")
c3.metric("App Interest %", f"{(df_f['app_use_likelihood'] != 'No').mean() * 100:.0f}%")
c4.metric("Avg Dogs", f"{df_f['num_dogs'].mean():.1f}")

tabs = st.tabs([
    "Overview",
    "Spending",
    "Challenges",
    "Features",
    "Advanced Analytics"
])

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2)

    fig1 = px.histogram(
        df_f,
        x="app_use_likelihood",
        title="App Adoption Intent",
        color="app_use_likelihood",
        color_discrete_map={"Yes": "#2ecc71", "Maybe": "#f39c12", "No": "#e
