import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Dog App Analytics', layout='wide')

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'dog_app_data.csv')
    df = pd.read_csv(csv_path)
    df['ownership_experience_encoded'] = df['ownership_years'].map(
        {'<1': 1, '1-3': 2, '4-7': 3, '8+': 4}
    )
    return df

df = load_data()

st.title('India Dog Care App - Survey Analytics Dashboard')

# FILTERS
col1, col2 = st.columns(2)
age = col1.multiselect('Age Group', sorted(df['age_group'].unique()), default=sorted(df['age_group'].unique()))
region = col2.multiselect('Region', sorted(df['region'].unique()), default=sorted(df['region'].unique()))

df_f = df[(df['age_group'].isin(age)) & (df['region'].isin(region))].copy()
df_no_na = df_f.dropna(subset=['monthly_spend_inr']).copy()

# KPI METRICS
c1, c2, c3, c4 = st.columns(4)
c1.metric('Respondents', len(df_f))
c2.metric('Avg Spend (INR)', f"{df_no_na['monthly_spend_inr'].mean():.0f}")
c3.metric('App Interest %', f"{(df_f['app_use_likelihood'] != 'No').mean() * 100:.0f}%")
c4.metric('Avg Dogs', f"{df_f['num_dogs'].mean():.1f}")

tabs = st.tabs(['Overview', 'Spending', 'Challenges', 'Features', 'Advanced Analytics'])

# OVERVIEW
with tabs[0]:
    col1, col2 = st.columns(2)
    color_map = {'Yes': '#2ecc71', 'Maybe': '#f39c12', 'No': '#e74c3c'}
    fig1 = px.histogram(df_f, x='app_use_likelihood', title='App Adoption Intent',
                        color='app_use_likelihood', color_discrete_map=color_map)
    col1.plotly_chart(fig1, use_container_width=True)
    fig2 = px.histogram(df_no_na, x='monthly_spend_inr', nbins=30,
                        title='Monthly Dog Spending Distribution (INR)')
    col2.plotly_chart(fig2, use_container_width=True)

# SPENDING
with tabs[1]:
    col1, col2 = st.columns(2)
    fig3 = px.box(df_no_na, x='age_group', y='monthly_spend_inr',
                  title='Age Group vs Monthly Spending', color='age_group')
    col1.plotly_chart(fig3, use_container_width=True)
    fig4 = px.bar(df_f.groupby('residence_type')['num_dogs'].mean().reset_index(),
                  x='residence_type', y='num_dogs',
                  title='Residence Type vs Avg Number of Dogs', color='residence_type')
    col2.plotly_chart(fig4, use_container_width=True)

# CHALLENGES
with tabs[2]:
    challenge_counts = df_f['biggest_challenge'].value_counts().reset_index()
    challenge_counts.columns = ['challenge', 'count']
    fig5 = px.bar(challenge_counts, x='challenge', y='count',
                  title='Biggest Challenges for Dog Owners', color='challenge')
    st.plotly_chart(fig5, use_container_width=True)

# FEATURES
with tabs[3]:
    feature_df = pd.DataFrame({
        'Feature': ['Vet Booking', 'Dog Parks', 'Grooming', 'Lost Dog Alert',
                    'Marketplace', 'Community', 'Health Tracking'],
        'Interest': [78, 72, 66, 60, 55, 48, 52]
    }).sort_values('Interest', ascending=False)
    fig6 = px.bar(feature_df, x='Feature', y='Interest',
                  title='Feature Interest (%)', color='Interest',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig6, use_container_width=True)

# ADVANCED ANALYTICS
with tabs[4]:
    st.subheader('Correlation Heatmap')
    num_cols = ['monthly_spend_inr', 'num_dogs', 'num_services_used',
                'num_features_valued', 'app_interest_scale']
    corr = df_no_na[num_cols].corr().round(2)
    fig7 = ff.create_annotated_heatmap(
        z=corr.values.tolist(), x=num_cols, y=num_cols,
        colorscale='RdBu', showscale=True
    )
    fig7.update_layout(title='Correlation Matrix')
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader('Customer Personas (KMeans Clustering)')
    X = df_no_na[['monthly_spend_inr', 'num_dogs', 'num_services_used', 'ownership_experience_encoded']]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_no_na = df_no_na.copy()
    df_no_na['cluster'] = kmeans.fit_predict(X_scaled).astype(str)
    persona_labels = {'0': 'Casual Owner', '1': 'Engaged Parent',
                      '2': 'Multi-Dog Household', '3': 'Premium Spender'}
    df_no_na['persona'] = df_no_na['cluster'].map(persona_labels)
    fig8 = px.scatter(df_no_na, x='monthly_spend_inr', y='num_dogs',
                      color='persona', size='num_services_used',
                      hover_data=['ownership_years', 'region'],
                      title='Dog Owner Personas (KMeans Clustering)')
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader('Ownership Experience vs App Adoption')
    adoption_df = (df_f.groupby(['ownership_years', 'app_use_likelihood'])
                   .size().reset_index(name='count'))
    color_map = {'Yes': '#2ecc71', 'Maybe': '#f39c12', 'No': '#e74c3c'}
    fig9 = px.bar(adoption_df, x='ownership_years', y='count',
                  color='app_use_likelihood', barmode='group',
                  category_orders={'ownership_years': ['<1', '1-3', '4-7', '8+'],
                                   'app_use_likelihood': ['Yes', 'Maybe', 'No']},
                  color_discrete_map=color_map,
                  title='Ownership Experience vs App Adoption')
    st.plotly_chart(fig9, use_container_width=True)

st.caption('MBA Project - India Dog Care App Market Analysis')
