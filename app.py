import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.express as px

# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(layout="wide")
st.title("📊 GPT-Powered Graduate Data Explorer")

# --- Cached functions ---
@st.cache_data
def load_data_from_github(url):
    return pd.read_parquet(url)

@st.cache_data
def preprocess_timestamps(df):
    df["Ping Timestamp"] = pd.to_datetime(df["Ping Timestamp"], errors="coerce")
    return df

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/GSU-Marketing/gpt-report-app/main/streamlit-test.parquet"

# --- Load Data ---
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

# --- Sidebar Filters ---
st.sidebar.subheader("🔎 Filter Data")
programs = ["All"] + sorted(df['Applications Applied Program'].dropna().unique())
statuses = ["All"] + sorted(df['Person Status'].dropna().unique())
terms = ["All"] + sorted(df['Applications Applied Term'].dropna().unique())

selected_program = st.sidebar.selectbox("Program:", programs)
selected_status = st.sidebar.selectbox("Status:", statuses)
selected_term = st.sidebar.selectbox("Term:", terms)

filtered_df = df.copy()
if selected_program != "All":
    filtered_df = filtered_df[filtered_df['Applications Applied Program'] == selected_program]
if selected_status != "All":
    filtered_df = filtered_df[filtered_df['Person Status'] == selected_status]
if selected_term != "All":
    filtered_df = filtered_df[filtered_df['Applications Applied Term'] == selected_term]

# --- Visualizations ---
st.subheader("📈 Visualizations")

col1, col2 = st.columns(2)

with col1:
    if "Ping Timestamp" in filtered_df.columns:
        time_series = filtered_df.dropna(subset=["Ping Timestamp"])
        fig = px.histogram(time_series, x="Ping Timestamp", nbins=50, title="Inquiries Over Time")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if "Zip Code" in filtered_df.columns:
        fig = px.histogram(filtered_df, x="Zip Code", title="Geographic Distribution")
        st.plotly_chart(fig, use_container_width=True)

# --- GPT Analysis ---
st.subheader("💬 Ask GPT About the Data")
prompt = st.text_area("Type your analysis question or use one of the templates below:")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔍 Program Interest Trends"):
        prompt = "Analyze trends in program interest over time."
    if st.button("📊 Conversion Funnel Analysis"):
        prompt = "Analyze the funnel from inquiry to application."
with col2:
    if st.button("🌍 Geographic Breakdown"):
        prompt = "Provide a breakdown of where applicants are from."
    if st.button("📅 Seasonal Patterns"):
        prompt = "Analyze any seasonal spikes or declines."

if st.button("🧠 Run GPT Analysis") and prompt.strip():
    st.info("⏳ Generating insight...")
    try:
        sample_csv = filtered_df.sample(min(50, len(filtered_df))).to_csv(index=False)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst. Provide clear, concise analysis."},
                {"role": "user", "content": f"Here is a sample of the dataset:\n{sample_csv}\n\n{prompt}"}
            ]
        )
        st.success("✅ GPT Response:")
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"❌ Error: {e}")
