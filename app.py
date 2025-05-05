import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(layout="wide")
st.title("📊 GPT-Powered Grad Report App")

# --- Tabs ---
tab1, tab2 = st.tabs(["🧠 AI Report Generator", "📈 Looker Dashboard"])

# --- Cached functions ---
@st.cache_data
def load_data_from_github(url):
    return pd.read_parquet(url)

@st.cache_data
def preprocess_timestamps(df):
    df["Ping Timestamp"] = pd.to_datetime(df["Ping Timestamp"], errors="coerce")
    return df

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/GSU-Marketing/gpt-report-app/main/streamlit-test.parquet"

# --- Tab 1 ---
with tab1:
    st.subheader("🧠 Upload Data or Use Default")

    # Dev override
    dev_key = st.sidebar.text_input("🔐 Dev Key (Optional)", type="password")
    uploaded_file = st.file_uploader("Upload your data file", type=["xlsx", "parquet"])

    if uploaded_file and dev_key == st.secrets.get("DEV_KEY", ""):
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()
        st.success("✅ Using uploaded file.")
    else:
        df = load_data_from_github(DEFAULT_DATA_URL)
        st.info("📂 Using default GitHub data.")

    df = preprocess_timestamps(df)

    # --- Filters ---
    st.sidebar.subheader("🔎 Filter Data")
    programs = ["All"] + sorted(df['Applications Applied Program'].dropna().unique())
    statuses = ["All"] + sorted(df['Person Status'].dropna().unique())
    terms = ["All"] + sorted(df['Applications Applied Term'].dropna().unique())

    selected_program = st.sidebar.selectbox("Program:", programs, key="program_filter")
    selected_status = st.sidebar.selectbox("Status:", statuses, key="status_filter")
    selected_term = st.sidebar.selectbox("Term:", terms, key="term_filter")

    filtered_df = df
    if selected_program != "All":
        filtered_df = filtered_df[filtered_df['Applications Applied Program'] == selected_program]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df['Person Status'] == selected_status]
    if selected_term != "All":
        filtered_df = filtered_df[filtered_df['Applications Applied Term'] == selected_term]

    # --- Date range filtering ---
    st.subheader("🗓️ Filter by Date Range")

    min_date = df["Ping Timestamp"].min()
    max_date = df["Ping Timestamp"].max()
    fiscal_years = list(range(min_date.year, max_date.year + 2))
    calendar_years = list(range(min_date.year, max_date.year + 1))

    view_type = st.selectbox("Select Time View:", ["All", "Fiscal Year", "Calendar Year"])
    selected_year = None

    if view_type == "Fiscal Year":
        selected_year = st.selectbox("Select Fiscal Year:", fiscal_years)
        fy_start = pd.Timestamp(f"{selected_year - 1}-07-01")
        fy_end = pd.Timestamp(f"{selected_year}-06-30")
        filtered_df = filtered_df[(filtered_df["Ping Timestamp"] >= fy_start) & (filtered_df["Ping Timestamp"] <= fy_end)]
        st.markdown(f"📆 Showing data from **{fy_start.date()} to {fy_end.date()}**")

    elif view_type == "Calendar Year":
        selected_year = st.selectbox("Select Calendar Year:", calendar_years)
        cy_start = pd.Timestamp(f"{selected_year}-01-01")
        cy_end = pd.Timestamp(f"{selected_year}-12-31")
        filtered_df = filtered_df[(filtered_df["Ping Timestamp"] >= cy_start) & (filtered_df["Ping Timestamp"] <= cy_end)]
        st.markdown(f"📆 Showing data from **{cy_start.date()} to {cy_end.date()}**")

    # --- Filtered preview ---
    st.subheader("📄 Filtered Data Preview")
    st.dataframe(filtered_df.head())
    st.markdown(f"**Total rows after filtering:** {len(filtered_df)}")

    # --- GPT Prompts ---
    st.subheader("📋 Choose a Report Template:")
    col1, col2 = st.columns(2)
    prompt = ""

    with col1:
        if st.button("🔍 Program Interest Trends"):
            prompt = "Analyze the trends in program interest over time. Which programs are growing or shrinking in popularity?"

        if st.button("📊 Conversion Funnel Analysis"):
            prompt = "Analyze the funnel from inquiry to applicant to enrolled. Where are most students dropping off?"

    with col2:
        if st.button("🌍 Geographical Distribution of Applicants"):
            prompt = "Provide a breakdown of the applicants' geographical locations, and identify key regions of growth or decline."

        if st.button("📆 Time-Based Application Trends"):
            prompt = "Analyze application trends over time. Identify seasonal spikes, fiscal year patterns, and program-specific changes."

    # --- GPT Input & Run ---
    st.subheader("💬 Ask GPT About the Filtered Data")
    user_input = st.text_area("Type your analysis request:", value=prompt, key="gpt_analysis_input")

    if st.button("🔎 Run GPT Analysis") and user_input.strip():
        st.info("⏳ Generating GPT-powered insight...")
        try:
            sample_csv = filtered_df.sample(min(50, len(filtered_df))).to_csv(index=False)

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Provide clear, concise analysis."},
                    {"role": "user", "content": f"Here is a sample of the dataset:\n{sample_csv}\n\n{user_input}"}
                ]
            )
            st.success("✅ GPT Response:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error during GPT call: {e}")

# --- Tab 2: Looker Dashboard ---
with tab2:
    st.subheader("📈 Embedded Looker Studio Dashboard")
    st.markdown(
        """
        <iframe width="600" height="450" src="https://lookerstudio.google.com/embed/reporting/c8b7472a-4864-40ae-b1d7-482c9cf581da/page/WIh1E" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
        """,
        unsafe_allow_html=True,
    )
