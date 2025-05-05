import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# Setup
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(layout="wide")
st.title("📊 GPT-Powered Grad Report App")

# Create tabs
tab1, tab2 = st.tabs(["🧠 AI Report Generator", "📈 Looker Dashboard"])

# --- TAB 1: AI Report ---
with tab1:
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success("Using uploaded file.")
    else:
        df = pd.read_excel("Streamlit_test.xlsx")
        st.info("Using default test data.")

    # --- FILTERS ---
   st.subheader("📋 Or choose a report template:")

col1, col2 = st.columns(2)
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


    # Apply filters
    filtered_df = df.copy()
    if selected_program != "All":
        filtered_df = filtered_df[filtered_df['Inquiry Program'] == selected_program]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df['Status'] == selected_status]
    if selected_term != "All":
        filtered_df = filtered_df[filtered_df['Term'] == selected_term]

    # GPT PROMPT SECTION
    st.subheader("💬 Ask GPT About the Filtered Data")
    prompt = st.text_area("Type your analysis request:")

    st.subheader("💬 Ask GPT About the Filtered Data")

st.subheader("🗓️ Filter by Date Range")

# Ensure Ping Timestamp is datetime
df["Ping Timestamp"] = pd.to_datetime(df["Ping Timestamp"], errors="coerce")
filtered_df["Ping Timestamp"] = pd.to_datetime(filtered_df["Ping Timestamp"], errors="coerce")

# Generate fiscal and calendar years available
min_date = df["Ping Timestamp"].min()
max_date = df["Ping Timestamp"].max()

# Get list of unique fiscal years
fiscal_years = list(range(min_date.year, max_date.year + 2))  # +2 to catch edge case June
calendar_years = list(range(min_date.year, max_date.year + 1))

# Select view type and year
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


# --- Predefined Templates ---
st.markdown("**📌 Quick Templates:**")
templates = {
    "Executive Summary": "Give me an executive summary of this dataset.",
    "Conversion Funnel": "Summarize the conversion rates from inquiry to application to enrollment.",
    "Geographic Insights": "Analyze which states or countries generate the most enrollments.",
    "Program Performance": "Which programs are generating the most inquiries and enrollments?",
    "Enrollment Trends": "Describe any trends in enrollment over time.",
}

selected_template = st.selectbox("Choose a predefined report template:", ["None"] + list(templates.keys()))

# Set prompt from template or let user type
if selected_template != "None":
    prompt = templates[selected_template]
else:
    prompt = st.text_area("Or type your own prompt here:")


# --- TAB 2: LOOKER ---
with tab2:
    st.subheader("📈 Embedded Looker Studio Dashboard")
    st.markdown(
        """
        <iframe width="100%" height="600" 
        src="https://lookerstudio.google.com/embed/reporting/c8b7472a-4864-40ae-b1d7-482c9cf581da/page/p_5vd9ebic9c"
        frameborder="0" style="border:0" allowfullscreen></iframe>
        """,
        unsafe_allow_html=True,
    )
