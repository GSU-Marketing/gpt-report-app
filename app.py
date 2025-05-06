import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(layout="wide")
st.title("📊 GPT-Powered Graduate-Marketing Data Explorer - BETA")

# --- Cached functions ---
@st.cache_data
def load_data_from_github(url):
    return pd.read_parquet(url)

@st.cache_data
def preprocess_timestamps(df):
    for col in ["Ping Timestamp", "Applications Created Date", "Applications Submitted Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data
def get_filtered_data(df, programs, statuses, terms):
    if programs and "All" not in programs:
        df = df[df['Applications Applied Program'].isin(programs)]
    if statuses and "All" not in statuses:
        df = df[df['Person Status'].isin(statuses)]
    if terms and "All" not in terms:
        df = df[df['Applications Applied Term'].isin(terms)]
    return df

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/GSU-Marketing/gpt-report-app/main/streamlit-test.parquet"

# --- Load Data ---
try:
    st.sidebar.subheader("📂 Upload or Use Default")
    dev_key = st.sidebar.text_input("🔐 Dev Key (Optional)", type="password")
    uploaded_file = st.sidebar.file_uploader("Upload your data file", type=["xlsx", "parquet"])

    if uploaded_file and dev_key == st.secrets.get("DEV_KEY", ""):
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)
        st.sidebar.success("✅ Using uploaded file.")
    else:
        df = load_data_from_github(DEFAULT_DATA_URL)
        st.sidebar.info("Using default GitHub data.")

    df = preprocess_timestamps(df)

except Exception as load_error:
    st.error("🚨 Failed to load data. Please try refreshing the app.")
    st.stop()

# --- Sidebar View Switcher ---
view = st.sidebar.selectbox("Select Dashboard Page", [
    "Page 1: Funnel Overview",
    "Page 2: Geography & Program",
    "Page 3: Engagement & Traffic"
])

# --- Filters ---
st.sidebar.subheader("🔎 Filter Data")
programs = ["All"] + sorted(df['Applications Applied Program'].dropna().unique())
statuses = ["All"] + sorted(df['Person Status'].dropna().unique())
terms = ["All"] + sorted(df['Applications Applied Term'].dropna().unique())

selected_programs = st.sidebar.multiselect("Programs:", programs, default=["All"])
selected_statuses = st.sidebar.multiselect("Statuses:", statuses, default=["All"])
selected_terms = st.sidebar.multiselect("Terms:", terms, default=["All"])

filtered_df = get_filtered_data(df, selected_programs, selected_statuses, selected_terms)

# Dummy route for page flow (avoid syntax error)
if view == "Page 2: Geography & Program":
    st.subheader("🌍 Geography & Program Breakdown")

    top_programs = filtered_df['Applications Applied Program'].value_counts().head(10).reset_index()
    top_programs.columns = ['Program', 'Count']
    fig = px.bar(top_programs, x='Count', y='Program', orientation='h', title="Top Applied Programs")
    st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Created Date" in filtered_df.columns and "Application APOS Program Modality" in filtered_df.columns:
        modality_df = filtered_df.dropna(subset=["Applications Created Date", "Application APOS Program Modality"])
        modality_counts = modality_df["Application APOS Program Modality"].value_counts().reset_index()
        modality_counts.columns = ["Modality", "Count"]
        fig = px.pie(modality_counts, names="Modality", values="Count", title="Modality Distribution (by Application Created Date)")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if all(col in filtered_df.columns for col in ["Ping Timestamp", "Person Citizenship", "Person Status", "Ping IP Address"]):
        filtered_df["Citizenship Type"] = filtered_df["Person Citizenship"].str.lower().fillna("").apply(lambda x: "Domestic" if x == "united states" else "International")
        zip_vis_df = filtered_df.dropna(subset=["Ping Timestamp"])
        zip_vis_grouped = zip_vis_df.groupby([pd.to_datetime(zip_vis_df["Ping Timestamp"]).dt.to_period("M"), "Citizenship Type", "Person Status"]).agg({"Ping IP Address": "count"}).reset_index()
        zip_vis_grouped["Ping Timestamp"] = zip_vis_grouped["Ping Timestamp"].astype(str)
        fig = px.bar(zip_vis_grouped, x="Ping Timestamp", y="Ping IP Address", color="Person Status", facet_col="Citizenship Type",
                     title="Domestic vs International Leads by Status Over Time")
        st.plotly_chart(fig, config={'displayModeBar': False})
