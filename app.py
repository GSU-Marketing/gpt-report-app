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
def get_filtered_data(df, program, status, term):
    if program != "All":
        df = df[df['Applications Applied Program'] == program]
    if status != "All":
        df = df[df['Person Status'] == status]
    if term != "All":
        df = df[df['Applications Applied Term'] == term]
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

# --- Filters with session state ---
st.sidebar.subheader("🔎 Filter Data")
programs = ["All"] + sorted(df['Applications Applied Program'].dropna().unique())
statuses = ["All"] + sorted(df['Person Status'].dropna().unique())
terms = ["All"] + sorted(df['Applications Applied Term'].dropna().unique())

if "Prospect" in statuses:
    statuses.remove("Prospect")

if "program_filter" not in st.session_state:
    st.session_state.program_filter = "All"
if "status_filter" not in st.session_state:
    st.session_state.status_filter = "All"
if "term_filter" not in st.session_state:
    st.session_state.term_filter = "All"

selected_program = st.sidebar.selectbox("Program:", programs, index=programs.index(st.session_state.program_filter))
selected_status = st.sidebar.selectbox("Status:", statuses, index=statuses.index(st.session_state.status_filter))
selected_term = st.sidebar.selectbox("Term:", terms, index=terms.index(st.session_state.term_filter))

st.session_state.program_filter = selected_program
st.session_state.status_filter = selected_status
st.session_state.term_filter = selected_term

if selected_program == "All" and selected_status == "All" and selected_term == "All":
    filtered_df = df.copy()
else:
    filtered_df = get_filtered_data(df, selected_program, selected_status, selected_term)

# --- PAGE 1: Funnel Overview ---
if view == "Page 1: Funnel Overview":
    st.subheader("🎯 Lead Funnel Overview")

    inquiries = len(filtered_df[filtered_df['Person Status'] == 'Inquiry'])
    applicants = len(filtered_df[filtered_df['Person Status'] == 'Applicant'])
    enrolled = len(filtered_df[filtered_df['Person Status'] == 'Enrolled'])

    col1, col2, col3 = st.columns(3)
    col1.metric("🧠 Inquiries", inquiries)
    col2.metric("📄 Applicants", applicants)
    col3.metric("🎓 Enrolled", enrolled)

    funnel_data = pd.DataFrame({
        "Stage": ["Inquiry", "Applicant", "Enrolled"],
        "Count": [inquiries, applicants, enrolled]
    })
    funnel_fig = px.bar(funnel_data, x="Count", y="Stage", orientation="h",
                        text="Count", color="Stage",
                        title="Lead Funnel", color_discrete_sequence=px.colors.qualitative.Set2)
    funnel_fig.update_traces(textposition='outside')
    st.plotly_chart(funnel_fig, config={'displayModeBar': False})

    leads_over_time = filtered_df[filtered_df['Person Status'].isin(['Inquiry', 'Applicant', 'Enrolled'])]
    leads_over_time = leads_over_time.dropna(subset=["Ping Timestamp"])
    fig = px.histogram(leads_over_time, x="Ping Timestamp", color="Person Status", barmode="group",
                       title="Leads Over Time")
    st.plotly_chart(fig, config={'displayModeBar': False})

    df_term = filtered_df.copy()
    df_term["Term"] = df_term["Applications Applied Term"].combine_first(df_term["Person Inquiry Term"])
    df_term = df_term[df_term["Person Status"].isin(["Inquiry", "Applicant", "Enrolled"])]
    term_counts = df_term.groupby(["Term", "Person Status"]).size().reset_index(name="Count")
    fig = px.bar(term_counts, x="Term", y="Count", color="Person Status", barmode="group",
                 title="Leads by Term")
    st.plotly_chart(fig, config={'displayModeBar': False})

# --- PAGE 3: Engagement & Traffic ---
if view == "Page 3: Engagement & Traffic":
    st.subheader("📈 Engagement & Traffic Sources")

    if "Ping UTM Source" in filtered_df.columns:
        traffic_df = filtered_df[filtered_df["Ping UTM Source"].notna()]
        if not traffic_df.empty:
            fig = px.pie(traffic_df, names="Ping UTM Source", title="Traffic Sources (Non-null)")
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Source data to display.")

    if "Ping UTM Medium" in filtered_df.columns:
        medium_df = filtered_df[filtered_df['Ping UTM Medium'].notna()]
        if not medium_df.empty:
            fig = px.bar(medium_df, x="Ping UTM Medium", title="Traffic by UTM Medium")
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Medium data to display.")

    if "Ping UTM Campaign" in filtered_df.columns:
        campaign_df = filtered_df[filtered_df['Ping UTM Campaign'].notna()]
        if not campaign_df.empty:
            campaign_counts = campaign_df['Ping UTM Campaign'].value_counts().reset_index()
            campaign_counts.columns = ["Ping UTM Campaign", "Count"]
            fig = px.bar(campaign_counts, x="Ping UTM Campaign", y="Count", title="Traffic by UTM Campaign")
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Campaign data to display.")

    if "Ping Timestamp" in filtered_df.columns:
        filtered_df["Hour"] = pd.to_datetime(filtered_df["Ping Timestamp"], errors='coerce').dt.hour
        fig = px.histogram(filtered_df.dropna(subset=["Hour"]), x="Hour", nbins=24, title="Activity by Hour of Day")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Created Date" in filtered_df.columns:
        created_counts = filtered_df.dropna(subset=["Applications Created Date"])
        fig = px.histogram(created_counts, x="Applications Created Date", title="Applications Created Over Time")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Submitted Date" in filtered_df.columns:
        submitted_counts = filtered_df.dropna(subset=["Applications Submitted Date"])
        fig = px.histogram(submitted_counts, x="Applications Submitted Date", title="Applications Submitted Over Time")
        st.plotly_chart(fig, config={'displayModeBar': False})

# --- PAGE 3: Engagement & Traffic ---
elif view == "Page 3: Engagement & Traffic":
    st.subheader("📈 Engagement & Traffic Sources")

    if "Ping UTM Source" in filtered_df.columns:
        traffic_df = filtered_df[filtered_df["Ping UTM Source"].notna()]
        if not traffic_df.empty:
            fig = px.pie(traffic_df, names="Ping UTM Source", title="Traffic Sources (Non-null)")
            st.plotly_chart(fig, config={'displayModeBar': False})

    if "Ping UTM Medium" in filtered_df.columns:
        medium_df = filtered_df[filtered_df['Ping UTM Medium'].notna()]
        if not medium_df.empty:
            medium_counts = medium_df['Ping UTM Medium'].value_counts().reset_index()
            medium_counts.columns = ["Medium", "Count"]
            fig = px.bar(medium_counts, x="Medium", y="Count", title="Traffic by UTM Medium")
            st.plotly_chart(fig, config={'displayModeBar': False})

    if "Ping UTM Campaign" in filtered_df.columns:
        campaign_df = filtered_df[filtered_df['Ping UTM Campaign'].notna()]
        if not campaign_df.empty:
            campaign_counts = campaign_df['Ping UTM Campaign'].value_counts().reset_index()
            campaign_counts.columns = ["Campaign", "Count"]
            fig = px.bar(campaign_counts, x="Campaign", y="Count", title="Traffic by UTM Campaign")
            st.plotly_chart(fig, config={'displayModeBar': False})

    if "Ping Timestamp" in filtered_df.columns:
        filtered_df["Hour"] = pd.to_datetime(filtered_df["Ping Timestamp"], errors='coerce').dt.hour
        fig = px.histogram(filtered_df, x="Hour", nbins=24, title="Activity by Hour of Day")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Created Date" in filtered_df.columns:
        created_counts = filtered_df.dropna(subset=["Applications Created Date"])
        fig = px.histogram(created_counts, x="Applications Created Date", title="Applications Created Over Time")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Submitted Date" in filtered_df.columns:
        submitted_counts = filtered_df.dropna(subset=["Applications Submitted Date"])
        fig = px.histogram(submitted_counts, x="Applications Submitted Date", title="Applications Submitted Over Time")
        st.plotly_chart(fig, config={'displayModeBar': False})

# Dummy route for page flow (avoid syntax error)
if view == "Page 2: Geography & Program":
    st.subheader("🌍 Geography & Program Breakdown")

    top_programs = filtered_df['Applications Applied Program'].value_counts().head(10).reset_index()
    top_programs.columns = ['Program', 'Count']
    fig = px.bar(top_programs, x='Count', y='Program', orientation='h', title="Top Applied Programs")
    st.plotly_chart(fig, config={'displayModeBar': False})

    if "Application APOS Program Modality" in filtered_df.columns and "Applications Created Date" in filtered_df.columns:
        modality_df = filtered_df.dropna(subset=["Application APOS Program Modality", "Applications Created Date"])
        modality_df = modality_df[modality_df["Person Status"] != "Prospect"]
        modality_grouped = modality_df.groupby("Application APOS Program Modality")["Ping IP Address"].count().reset_index()
        modality_grouped.columns = ["Modality", "Count"]
        if not modality_grouped.empty:
            fig = px.pie(modality_grouped, names="Modality", values="Count", title="Modality Distribution (by Application Created Date)")
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No modality data available for the current filters.")

    if all(col in filtered_df.columns for col in ["Ping Timestamp", "Person Citizenship", "Person Status", "Ping IP Address"]):
        filtered_df["Citizenship Type"] = filtered_df["Person Citizenship"].str.lower().fillna("").apply(lambda x: "Domestic" if x == "united states" else "International")
        filtered_df = filtered_df[filtered_df['Person Status'] != 'Prospect']
        zip_vis_df = filtered_df.dropna(subset=["Ping Timestamp"])
        zip_vis_grouped = zip_vis_df.groupby([pd.to_datetime(zip_vis_df["Ping Timestamp"]).dt.to_period("M"), "Citizenship Type", "Person Status"]).agg({"Ping IP Address": "count"}).reset_index()
        zip_vis_grouped["Ping Timestamp"] = zip_vis_grouped["Ping Timestamp"].astype(str)
        fig = px.bar(zip_vis_grouped, x="Ping Timestamp", y="Ping IP Address", color="Person Status", facet_col="Citizenship Type",
                     title="Domestic vs International Leads by Status Over Time")
        st.plotly_chart(fig, config={'displayModeBar': False})
