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
    df = pd.read_csv("dog_app_data.csv")
    df["ownership_experience_encoded"] = df["ownership_years"].map(
        {"<1":1,"1-3":2,"4-7":3,"8+":4}
    )
    return df

df = load_data()

st.title("?? India Dog Care App — Survey Analytics Dashboard")

# FILTERS
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
df_no_na = df_f.dropna(subset=["monthly_spend_inr"])

# KPI METRICS
c1,c2,c3,c4 = st.columns(4)

c1.metric("Respondents", len(df_f))
c2.metric("Avg Spend", f"?{df_no_na['monthly_spend_inr'].mean():.0f}")
c3.metric("App Interest %", f"{(df_f['app_use_likelihood']!='No').mean()*100:.0f}%")
c4.metric("Avg Dogs", f"{df_f['num_dogs'].mean():.1f}")

tabs = st.tabs([
"Overview",
"Spending",
"Challenges",
"Features",
"Advanced Analytics"
])

# OVERVIEW
with tabs[0]:

    col1,col2 = st.columns(2)

    fig1 = px.histogram(
        df_f,
        x="app_use_likelihood",
        title="App Adoption Intent"
    )

    col1.plotly_chart(fig1,use_container_width=True)

    fig2 = px.histogram(
        df_no_na,
        x="monthly_spend_inr",
        nbins=30,
        title="Monthly Dog Spending Distribution"
    )

    col2.plotly_chart(fig2,use_container_width=True)

# SPENDING
with tabs[1]:

    col1,col2 = st.columns(2)

    fig3 = px.box(
        df_no_na,
        x="age_group",
        y="monthly_spend_inr",
        title="Age vs Spending"
    )

    col1.plotly_chart(fig3,use_container_width=True)

    fig4 = px.bar(
        df_f.groupby("residence_type")["num_dogs"].mean().reset_index(),
        x="residence_type",
        y="num_dogs",
        title="Residence Type vs Average Dogs"
    )

    col2.plotly_chart(fig4,use_container_width=True)

# CHALLENGES
with tabs[2]:

    challenge_counts = df_f["biggest_challenge"].value_counts()

    fig5 = px.bar(
        challenge_counts.reset_index(),
        x="index",
        y="biggest_challenge",
        title="Biggest Challenges for Dog Owners"
    )

    st.plotly_chart(fig5,use_container_width=True)

# FEATURES
with tabs[3]:

    feature_data = {
        "Feature":[
        "Vet booking",
        "Dog parks",
        "Grooming",
        "Lost dog alert",
        "Marketplace",
        "Community",
        "Health tracking"
        ],
        "Interest":[
        78,72,66,60,55,48,52
        ]
    }

    feature_df = pd.DataFrame(feature_data)

    fig6 = px.bar(
        feature_df,
        x="Feature",
        y="Interest",
        title="Feature Interest (%)"
    )

    st.plotly_chart(fig6,use_container_width=True)

# ADVANCED ANALYTICS
with tabs[4]:

    st.subheader("Correlation Heatmap")

    num_cols = [
    "monthly_spend_inr",
    "num_dogs",
    "num_services_used",
    "num_features_valued",
    "app_interest_scale"
    ]

    corr = df_no_na[num_cols].corr()

    fig7 = ff.create_annotated_heatmap(
        corr.values,
        x=num_cols,
        y=num_cols
    )

    st.plotly_chart(fig7,use_container_width=True)

    st.subheader("Customer Personas (KMeans Clustering)")

    X = df_no_na[
    ["monthly_spend_inr",
    "num_dogs",
    "num_services_used",
    "ownership_experience_encoded"]
    ]

    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4)
    df_no_na["cluster"] = kmeans.fit_predict(X_scaled)

    fig8 = px.scatter(
        df_no_na,
        x="monthly_spend_inr",
        y="num_dogs",
        color="cluster",
        size="num_services_used",
        title="Dog Owner Personas"
    )

    st.plotly_chart(fig8,use_container_width=True)

    st.subheader("Ownership Experience vs App Adoption")

    fig9 = px.bar(
        df_f.groupby(["ownership_years","app_use_likelihood"])
        .size()
        .reset_index(name="count"),
        x="ownership_years",
        y="count",
        color="app_use_likelihood",
        title="Ownership Experience vs App Adoption"
    )

    st.plotly_chart(fig9,use_container_width=True)

st.caption("MBA Project — India Dog Care App Market Analysis")
