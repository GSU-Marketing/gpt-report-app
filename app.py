
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
import pickle
import io


# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(layout="wide")
gsu_colors = ['#0055CC', '#00A3AD', '#FDB913', '#C8102E']
st.image("GSU Logo Stacked.png", width=160)
st.markdown("## GPT-Powered Graduate-Marketing Data Explorer", unsafe_allow_html=True)

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

@st.cache_data
def load_data_from_gdrive():
    creds_data = base64.b64decode(st.secrets["gdrive"]["token_pickle"])
    creds = pickle.loads(creds_data)
    service = build('drive', 'v3', credentials=creds)

    file_id = st.secrets["gdrive"]["file_id"]
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()

    from googleapiclient.http import MediaIoBaseDownload
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return pd.read_parquet(fh)


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
    elif "gdrive" in st.secrets:
        df = load_data_from_gdrive()
        st.sidebar.success("✅ Using secure Google Drive data.")
    else:
        st.error("🚨 No data source available. Please upload a file or configure Google Drive access.")
        st.stop()

    df = preprocess_timestamps(df)

except Exception as load_error:
    st.error("🚨 Failed to load data. Please try refreshing the app.")
    st.exception(load_error)  # This will show the error traceback
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
    stacked = st.sidebar.checkbox("📱 Mobile View", value=False)

    if stacked:
        st.metric("🧠 Inquiries", inquiries)
        st.metric("📄 Applicants", applicants)
        st.metric("🎓 Enrolled", enrolled)
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("🧠 Inquiries", inquiries)
        col2.metric("📄 Applicants", applicants)
        col3.metric("🎓 Enrolled", enrolled)

    funnel_data = pd.DataFrame({
        "Stage": ["Inquiry", "Applicant", "Enrolled"],
        "Count": [inquiries, applicants, enrolled]
    })
    funnel_fig = px.bar(funnel_data, x="Count", y="Stage",
                        text="Count", color="Stage",
                        title="Lead Funnel", color_discrete_sequence=gsu_colors, orientation="h")
    funnel_fig.update_traces(textposition='outside')
    st.plotly_chart(funnel_fig, use_container_width=stacked, config={'displayModeBar': False})

    leads_over_time = filtered_df[filtered_df['Person Status'].isin(['Inquiry', 'Applicant', 'Enrolled'])]
    leads_over_time = leads_over_time.dropna(subset=["Ping Timestamp"])
    fig = px.histogram(leads_over_time, x="Ping Timestamp", color="Person Status", barmode="group",
                       title="Leads Over Time", color_discrete_sequence=gsu_colors)
    st.plotly_chart(fig, use_container_width=stacked, config={'displayModeBar': False})

    df_term = filtered_df.copy()
    df_term["Term"] = df_term["Applications Applied Term"].combine_first(df_term["Person Inquiry Term"])
    df_term = df_term[df_term["Person Status"].isin(["Inquiry", "Applicant", "Enrolled"])]
    term_counts = df_term.groupby(["Term", "Person Status"]).size().reset_index(name="Count")
    fig = px.bar(term_counts, x="Term", y="Count", color="Person Status", barmode="group",
                 title="Leads by Term", color_discrete_sequence=gsu_colors)
    st.plotly_chart(fig, use_container_width=stacked, config={'displayModeBar': False})


# --- PAGE 2: Geography & Program ---
elif view == "Page 2: Geography & Program":
    st.subheader("🌍 Geography & Program Breakdown")

    top_programs = filtered_df['Applications Applied Program'].value_counts().head(10).reset_index()
    top_programs.columns = ['Program', 'Count']
    fig = px.bar(top_programs, x='Count', y='Program', orientation='h', title="Top Applied Programs",
                 color_discrete_sequence=gsu_colors)
    st.plotly_chart(fig, config={'displayModeBar': False})

    # Registration Hours by Term (replacement for modality)
    reg_cols = [col for col in filtered_df.columns if "Registration Hours" in col]
    reg_df = filtered_df[reg_cols].copy()
    melted = reg_df.melt(var_name="Term", value_name="Hours").dropna()

    show_avg = st.sidebar.checkbox("Show Average Hours per Person", value=False)
    if show_avg:
        avg_df = melted.groupby("Term")["Hours"].mean().reset_index()
        fig = px.bar(avg_df, x="Term", y="Hours", title="Average Registration Hours by Term",
                     color_discrete_sequence=gsu_colors)
    else:
        sum_df = melted.groupby("Term")["Hours"].sum().reset_index()
        fig = px.bar(sum_df, x="Term", y="Hours", title="Total Registration Hours by Term",
                     color_discrete_sequence=gsu_colors)

    st.plotly_chart(fig, config={'displayModeBar': False})

# --- PAGE 3: Engagement & Traffic ---
elif view == "Page 3: Engagement & Traffic":
    st.subheader("📈 Engagement & Traffic Sources")

    if "Ping UTM Source" in filtered_df.columns:
        traffic_df = filtered_df[filtered_df["Ping UTM Source"].notna()]
        if not traffic_df.empty:
            fig = px.pie(traffic_df, names="Ping UTM Source", title="Traffic Sources (UTM Source)")
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Source data to display.")

    if "Ping UTM Medium" in filtered_df.columns:
        medium_df = filtered_df[filtered_df["Ping UTM Medium"].notna()]
        if not medium_df.empty:
            medium_counts = medium_df["Ping UTM Medium"].value_counts().reset_index()
            medium_counts.columns = ["UTM Medium", "Count"]
            fig = px.bar(medium_counts, x="UTM Medium", y="Count", title="Traffic by UTM Medium",
                         color_discrete_sequence=gsu_colors)
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Medium data to display.")

    if "Ping UTM Campaign" in filtered_df.columns:
        campaign_df = filtered_df[filtered_df["Ping UTM Campaign"].notna()]
        if not campaign_df.empty:
            campaign_counts = campaign_df["Ping UTM Campaign"].value_counts().reset_index()
            campaign_counts.columns = ["Campaign", "Count"]
            fig = px.bar(campaign_counts, x="Campaign", y="Count", title="Traffic by UTM Campaign",
                         color_discrete_sequence=gsu_colors)
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Campaign data to display.")

    if "Ping Timestamp" in filtered_df.columns:
        filtered_df["Hour"] = pd.to_datetime(filtered_df["Ping Timestamp"], errors='coerce').dt.hour
        fig = px.histogram(filtered_df.dropna(subset=["Hour"]), x="Hour", nbins=24,
                           title="Activity by Hour of Day", color_discrete_sequence=gsu_colors)
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Created Date" in filtered_df.columns:
        created_counts = filtered_df.dropna(subset=["Applications Created Date"])
        if not created_counts.empty:
            fig = px.histogram(created_counts, x="Applications Created Date",
                               title="Applications Created Over Time",
                               color_discrete_sequence=gsu_colors)
            st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Submitted Date" in filtered_df.columns:
        submitted_counts = filtered_df.dropna(subset=["Applications Submitted Date"])
        if not submitted_counts.empty:
            fig = px.histogram(submitted_counts, x="Applications Submitted Date",
                               title="Applications Submitted Over Time",
                               color_discrete_sequence=gsu_colors)
            st.plotly_chart(fig, config={'displayModeBar': False})
