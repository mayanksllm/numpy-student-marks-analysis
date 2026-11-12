import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit.components.v1 as components

st.set_page_config(
    page_title="EV Charging Station Utilisation Analyser",
    page_icon="⚡",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_ev_sessions.csv")
    stations = pd.read_csv("station_utilisation_summary.csv")
    return df, stations

df, station_stats = load_data()

st.title("⚡ EV Charging Station Utilisation Analyser")
st.markdown("**Geospatial Telemetry Analytics, K-Means Station Clustering & Hourly Demand Forecasting (Jodhpur Network)**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cleaned Charging Sessions", f"{len(df):,}", "30-Day Telemetry Log")
col2.metric("Total Energy Delivered", f"{df['energy_delivered_kwh'].sum():,.1f} kWh", "Across 5 Hubs")
col3.metric("Most Congested Hub", "EV-Station-C (Paota)", "90.34% Utilisation")
col4.metric("Demand Model Accuracy", "R² = 0.9925", "Linear Regression")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Geospatial Map & Expansion",
    "📊 Utilisation & Peak EDA",
    "🤖 Live Demand Predictor",
    "📋 Raw vs. Clean Data Audit"
])

with tab1:
    st.subheader("Interactive Station Cluster & Expansion Map")
    st.markdown("🔴 **Over-Utilised (>65%)** | 🟡 **Optimal (35%–65%)** | 🟢 **Under-Utilised (<35%)** | 🔵 **Recommended New Sites**")
    with open("ev_station_clusters_map.html", "r", encoding="utf-8") as f:
        map_html = f.read()
    components.html(map_html, height=520)

    st.subheader("Station KPI Summary & K-Means Tiers")
    st.dataframe(
        station_stats[["station", "location", "total_chargers", "total_sessions", "total_energy_kwh", "avg_duration_min", "utilisation_rate_%", "utilisation_tier"]],
        use_container_width=True
    )

with tab2:
    st.subheader("Exploratory Data Analysis & Peak Load Patterns")
    selected_station = st.selectbox("Filter by Charging Station:", ["All Stations"] + sorted(df["station"].unique().tolist()))
    filtered_df = df if selected_station == "All Stations" else df[df["station"] == selected_station]

    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(7, 4.2))
        hourly_trend = filtered_df.groupby(["hour", "day_type"])["energy_delivered_kwh"].sum().reset_index()
        sns.lineplot(data=hourly_trend, x="hour", y="energy_delivered_kwh", hue="day_type", marker="o", linewidth=2.2, ax=ax1)
        ax1.set_title(f"Hourly Energy Demand: Weekday vs. Weekend ({selected_station})", weight="bold")
        ax1.set_xlabel("Hour of Day (0-23)")
        ax1.set_ylabel("Energy Delivered (kWh)")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        sns.boxplot(data=filtered_df, x="charger_type", y="session_duration_min", palette="Set2", ax=ax2)
        ax2.set_title(f"Session Duration by Charger Tier ({selected_station})", weight="bold")
        ax2.set_xlabel("Charger Type")
        ax2.set_ylabel("Duration (Minutes)")
        st.pyplot(fig2)

with tab3:
    st.subheader("Interactive Hourly Energy Demand Simulator")
    hourly_df = df.groupby(["station", "hour", "day_type", "is_peak_hour"]).agg(
        session_count=("session_id", "count"),
        avg_duration=("session_duration_min", "mean"),
        hourly_energy_kwh=("energy_delivered_kwh", "sum")
    ).reset_index()

    X = pd.get_dummies(hourly_df[["station", "hour", "day_type", "is_peak_hour", "session_count", "avg_duration"]], drop_first=True)
    y = hourly_df["hourly_energy_kwh"]
    lr = LinearRegression().fit(X, y)

    sc1, sc2, sc3, sc4 = st.columns(4)
    in_station = sc1.selectbox("Station", sorted(df["station"].unique().tolist()))
    in_hour = sc2.slider("Hour of Day", 6, 23, 18)
    in_day = sc3.selectbox("Day Type", ["Weekday", "Weekend"])
    in_sessions = sc4.slider("Expected Sessions in Hour", 1, 40, 15)
    in_dur = st.slider("Average Session Duration (Minutes)", 20, 180, 65)

    input_row = pd.DataFrame([{
        "station": in_station,
        "hour": in_hour,
        "day_type": in_day,
        "is_peak_hour": 1 if in_hour in [8, 9, 10, 11, 17, 18, 19, 20] else 0,
        "session_count": in_sessions,
        "avg_duration": in_dur
    }])
    input_encoded = pd.get_dummies(input_row, drop_first=True).reindex(columns=X.columns, fill_value=0)
    predicted_kwh = max(0.0, float(lr.predict(input_encoded)[0]))

    st.success(f"### Predicted Hourly Grid Load: **{predicted_kwh:.2f} kWh**")
    if predicted_kwh > 650:
        st.warning("⚠️ High Grid Load Alert: Peak-hour load balancing or DC fast-bay expansion recommended.")

with tab4:
    st.subheader("Data Cleaning & Schema Audit (Stage 4)")
    st.markdown("""
    - **Raw Records (`ev_charging_raw_dataset.csv`):** `3,604` rows
    - **Anomalies Resolved:** Removed `2` duplicate `session_id` entries, dropped `1` zero-duration failed session (`EV99991`), imputed `1` missing `charger_type` (`EV99992`), and engineered `6` temporal/utilization features.
    - **Final Clean Records (`cleaned_ev_sessions.csv`):** `3,601` rows
    """)
    st.dataframe(df.head(25), use_container_width=True)
