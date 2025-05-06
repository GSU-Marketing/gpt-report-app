import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

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

# --- Filter Setup ---
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

# --- Page Navigation ---
page = st.sidebar.selectbox("Select Dashboard Page", [
    "Page 1: Funnel Overview",
    "Page 2: Geography & Programs",
    "Page 3: Engagement Trends"
])

# --- Page 1: Funnel Overview ---
if page == "Page 1: Funnel Overview":
    st.subheader("🎯 Lead Funnel Overview")

    inquiries = len(df[df['Person Status'] == 'Inquiry'])
    applicants = len(df[df['Person Status'] == 'Applicant'])
    enrolled = len(df[df['Person Status'] == 'Enrolled'])

    col1, col2, col3 = st.columns(3)
    col1.metric("🧠 Inquiries", inquiries)
    col2.metric("📄 Applicants", applicants)
    col3.metric("🎓 Enrolled", enrolled)

    funnel_fig = go.Figure(go.Funnel(
        y=["Inquiry", "Applicant", "Enrolled"],
        x=[inquiries, applicants, enrolled]
    ))
    st.plotly_chart(funnel_fig, use_container_width=True)

    st.subheader("📆 Leads Over Time")
    leads_df = df[df['Person Status'].isin(['Inquiry', 'Applicant', 'Enrolled'])].dropna(subset=['Ping Timestamp'])
    fig = px.histogram(leads_df, x="Ping Timestamp", color="Person Status", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🗓️ Leads by Term")
    df_term = df.copy()
    df_term["Term"] = df_term["Applications Applied Term"].combine_first(df_term["Person Inquiry Term"])
    df_term = df_term[df_term["Person Status"].isin(["Inquiry", "Applicant", "Enrolled"])]
    term_counts = df_term.groupby(["Term", "Person Status"]).size().reset_index(name="Count")
    fig = px.bar(term_counts, x="Term", y="Count", color="Person Status", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# --- Page 2: Geography & Programs ---
elif page == "Page 2: Geography & Programs":
    st.subheader("📍 Geography & Program Interest")

    if "Zip Code" in df.columns:
        df["Zip Code"] = df["Zip Code"].astype(str)
        fig = px.histogram(df, x="Zip Code", title="Leads by Zip Code")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏫 Top Programs")
    top_programs = df['Applications Applied Program'].value_counts().head(10).reset_index()
    top_programs.columns = ['Program', 'Count']
    fig = px.bar(top_programs, x='Count', y='Program', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# --- Page 3: Engagement Trends ---
elif page == "Page 3: Engagement Trends":
    st.subheader("📊 Engagement Trends")

    st.markdown("**Engagement Duration**")
    fig = px.histogram(df, x="Ping Duration (seconds)", nbins=30)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Activity by Hour of Day**")
    df['Hour'] = pd.to_datetime(df['Ping Timestamp'], errors='coerce').dt.hour
    fig = px.histogram(df, x="Hour", nbins=24)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Traffic Sources (UTM)**")
    if "Ping UTM Source" in df.columns:
        fig = px.pie(df, names="Ping UTM Source", title="Traffic Sources")
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
