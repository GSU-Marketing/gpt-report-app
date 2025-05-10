import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
import pickle
import io
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(layout="wide")

@st.cache_resource
def get_gsheet_client():
    creds_dict = dict(st.secrets["gcp"])
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

def log_visitor_to_sheet(ip, page, session_id, prompt=None):
    client = get_gsheet_client()
    sheet = client.open("Visitor Logs").sheet1
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([ip, now, page, session_id, prompt or ""])

from streamlit_javascript import st_javascript

ip = st_javascript("await fetch('https://api.ipify.org?format=json').then(r => r.json()).then(d => d.ip)")

import uuid

# Generate a session_id only once per session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())





# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
gsu_colors = ['#0055CC', '#00A3AD', '#FDB913', '#C8102E']
st.image("GSU Logo Stacked.png", width=160)
st.markdown("## GPT-Powered Graduate-Marketing Data Explorer", unsafe_allow_html=True)

# --- Cached functions ---
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

def summarize_funnel_metrics(df):
    stages = ["Inquiry", "Applicant", "Enrolled"]
    counts = {stage: len(df[df["Person Status"] == stage]) for stage in stages}
    return "\n".join([f"{stage}: {count}" for stage, count in counts.items()])


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

# --- GPT Helper Function ---
def ask_gpt(prompt: str, system_prompt: str, temperature: float = 0.4):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning("⚠️ GPT error: API limit reached or input is too large.")
        st.info("ℹ️ Details have been logged for review.")
    return None

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
        st.sidebar.caption("🔐 Data Source: Private Google Drive")
    else:
        st.error("🚨 No data source available. Please upload a file or configure Google Drive access.")
        st.stop()

    df = preprocess_timestamps(df)

except Exception as load_error:
    st.error("🚨 Failed to load data. Please try refreshing the app.")
    st.exception(load_error)
    st.stop()

# --- Sidebar View Switcher ---
view = st.sidebar.selectbox("Select Dashboard Page", [
    "Page 1: Funnel Overview",
    "Page 2: Programs & Registration Hours",
    "Page 3: Engagement & Channels",
    "Page 4: Admin Dashboard"
])


if ip and "visitor_logged" not in st.session_state:

    try:
        log_visitor_to_sheet(ip, page=view, session_id=st.session_state.session_id)
        st.session_state.visitor_logged = True
    except Exception as e:
        st.warning(f"⚠️ Initial visitor log failed: {e}")



# --- Filters with session state ---
st.sidebar.subheader("🔎 Filter Data")
programs = ["All"] + sorted(df['Applications Applied Program'].dropna().astype(str).str.strip().loc[lambda x: (x != "") & (x.str.lower() != "nan")].unique())
statuses = ["All"] + sorted(df['Person Status'].dropna().astype(str).str.strip().loc[lambda x: (x != "") & (x.str.lower() != "nan")].unique())
terms = ["All"] + sorted(df['Applications Applied Term'].dropna().astype(str).str.strip().loc[lambda x: (x != "") & (x.str.lower() != "nan")].unique())

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


# --- Apply Date Range Filter ---
from datetime import datetime

min_date = pd.to_datetime("2022-07-01")
max_date = pd.to_datetime("2025-12-31")

assert pd.api.types.is_datetime64_any_dtype(filtered_df["Ping Timestamp"])


ping_dates = pd.to_datetime(filtered_df["Ping Timestamp"], errors="coerce")
valid_dates = ping_dates.dropna()
data_min = valid_dates.min()
data_max = valid_dates.max()

start_date = max(min_date, data_min)
end_date = min(max_date, data_max)

selected_dates = st.sidebar.date_input(
    "📅 Date Range (Ping Timestamp)",
    (start_date.date(), end_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start, end = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    st.sidebar.caption(f"📆 Showing data from **{start.date()}** to **{end.date()}**")

    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df["Ping Timestamp"], errors="coerce") >= start) &
        (pd.to_datetime(filtered_df["Ping Timestamp"], errors="coerce") <= end)
    ]


# --- GPT Sidebar Chat + Summary ---
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Ask a question about your data")
user_question = st.sidebar.text_area("What would you like to know?", placeholder="e.g. What is the most common program?", height=100)

if user_question:
    with st.spinner("Asking AI..."):
        cols_to_include = ["Person Status", "Applications Applied Program", "Applications Applied Term", "Ping Timestamp"]
        data_sample = filtered_df[cols_to_include].head(300).to_csv(index=False)

        answer = ask_gpt(
            prompt=f"Here is the data:\n\n{data_sample}\n\nQuestion: {user_question}",
            system_prompt="You are a data analyst assistant that answers questions about CSV-style data."
        )

        if answer:
            st.sidebar.success("✅ Answer ready")
            st.sidebar.write(answer)
            try:
                log_visitor_to_sheet(ip, page=view, session_id=st.session_state.session_id, prompt=user_question)
            except Exception as e:
                st.warning("⚠️ Visitor GPT log failed.")

 

    
# Optional summary toggle
if st.sidebar.checkbox("🧠 Show automatic summary", value=False):
    with st.spinner("Generating summary..."):
        summary_sample = filtered_df.head(300).to_csv(index=False)
        summary = ask_gpt(
            prompt=summary_sample,
            system_prompt="You are a data analyst. Summarize the following dataset."
        )
        if summary:
            st.sidebar.markdown("### 📊 Data Summary")
            st.sidebar.write(summary)



# --- PAGE 1: Funnel Overview ---
if view == "Page 1: Funnel Overview":
    st.subheader("🪣 Funnel Overview")

    inquiries = len(filtered_df[filtered_df['Person Status'] == 'Inquiry'])
    applicants = len(filtered_df[filtered_df['Person Status'] == 'Applicant'])
    enrolled = len(filtered_df[filtered_df['Person Status'] == 'Enrolled'])
    stacked = st.sidebar.checkbox("📱 Mobile View", value=True)

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

    # Remove rows with missing, NaN, or blank Term values
    df_term = df_term[df_term["Term"].notna()]
    df_term = df_term[df_term["Term"].astype(str).str.strip().str.lower() != "nan"]
    df_term = df_term[df_term["Term"].astype(str).str.strip() != ""]
    term_counts = df_term.groupby(["Term", "Person Status"]).size().reset_index(name="Count")
    fig = px.bar(term_counts, x="Term", y="Count", color="Person Status", barmode="group",
                 title="Leads by Term", color_discrete_sequence=gsu_colors)
    st.plotly_chart(fig, use_container_width=stacked, config={'displayModeBar': False})

    if st.sidebar.checkbox("🧠 Show Page Summary", value=False):
        st.markdown("### 🧠 Summary")
        with st.spinner("Summarizing Page 1..."):
            max_rows = 1000 // max(len(filtered_df.columns), 1)
            
            summary_input = summarize_funnel_metrics(filtered_df)
            summary = ask_gpt(
            prompt=f"Provide a funnel drop-off summary using this data:\n\n{summary_input}",
            system_prompt="You are a data analyst providing a brief summary of the data."
            )

            if summary:
                st.info(summary)


# --- PAGE 2: Geography & Program ---
elif view == "Page 2: Programs & Registration Hours":
    st.subheader("📊 Programs & Registration Hours")


    top_programs = (
        filtered_df['Applications Applied Program']
        .dropna()
        .astype(str)
        .loc[lambda x: (x.str.strip() != "") & (x.str.lower() != "nan")]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_programs.columns = ['Program', 'Count']

    fig = px.bar(top_programs, x='Count', y='Program', orientation='h', title="Top Applied Programs",
                 color_discrete_sequence=gsu_colors)
    st.plotly_chart(fig, config={'displayModeBar': False})

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

    if st.sidebar.checkbox("🧠 Show Page Summary", value=True):
        st.markdown("### 🧠 Summary")
        with st.spinner("Summarizing Page 2..."):

            max_rows = 1000 // max(len(filtered_df.columns), 1)
            page_sample = filtered_df.head(max_rows).to_csv(index=False)
            summary = ask_gpt(
                prompt=f"Summarize this data with attention to geographic and program details:\n\n{page_sample}",
                system_prompt="You are a data analyst summarizing geographic and program-related patterns."
            )
            if summary:
                st.info(summary)


# --- PAGE 3: Engagement & Traffic ---
elif view == "Page 3: Engagement & Channels":
    st.subheader("📡 Engagement & Channels")

    if "Ping UTM Source" in filtered_df.columns:
        traffic_df = filtered_df[filtered_df["Ping UTM Source"].notna()]
        traffic_df = traffic_df[traffic_df["Ping UTM Source"].astype(str).str.strip().str.lower() != "nan"]
        traffic_df = traffic_df[traffic_df["Ping UTM Source"].astype(str).str.strip() != ""]
        if not traffic_df.empty:
            fig = px.pie(traffic_df, names="Ping UTM Source", title="Traffic Sources (UTM Source)")
            st.plotly_chart(fig, config={'displayModeBar': False})
        else:
            st.info("ℹ️ No UTM Source data to display.")

    if "Ping UTM Medium" in filtered_df.columns:
        medium_df = filtered_df[filtered_df["Ping UTM Medium"].notna()]
        medium_df = medium_df[medium_df["Ping UTM Medium"].astype(str).str.strip().str.lower() != "nan"]
        medium_df = medium_df[medium_df["Ping UTM Medium"].astype(str).str.strip() != ""]
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
        campaign_df = campaign_df[campaign_df["Ping UTM Campaign"].astype(str).str.strip().str.lower() != "nan"]
        campaign_df = campaign_df[campaign_df["Ping UTM Campaign"].astype(str).str.strip() != ""]
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
        created_counts = created_counts[created_counts["Applications Created Date"].astype(str).str.lower() != "nan"]
        if not created_counts.empty:
            fig = px.histogram(created_counts, x="Applications Created Date",
                               title="Applications Created Over Time",
                               color_discrete_sequence=gsu_colors)
            st.plotly_chart(fig, config={'displayModeBar': False})

    if "Applications Submitted Date" in filtered_df.columns:
        submitted_counts = filtered_df.dropna(subset=["Applications Submitted Date"])
        submitted_counts = submitted_counts[submitted_counts["Applications Submitted Date"].astype(str).str.lower() != "nan"]
        if not submitted_counts.empty:
            fig = px.histogram(submitted_counts, x="Applications Submitted Date",
                               title="Applications Submitted Over Time",
                               color_discrete_sequence=gsu_colors)
            st.plotly_chart(fig, config={'displayModeBar': False})


    if st.sidebar.checkbox("🧠 Show Page Summary", value=True):
        st.markdown("### 🧠 Summary")
        with st.spinner("Summarizing Page 3..."):

            max_rows = 1000 // max(len(filtered_df.columns), 1)
            page_sample = filtered_df.head(max_rows).to_csv(index=False)
            summary = ask_gpt(
                prompt=f"Summarize this marketing traffic data:\n\n{page_sample}",
                system_prompt="You are a data analyst summarizing UTM and engagement metrics."
            )
            if summary:
                st.info(summary)



# --- PAGE 4: Admin View ---
elif view == "Page 4: Admin Dashboard":

    def get_visitor_logs():
        client = get_gsheet_client()
        sheet = client.open("Visitor Logs").sheet1
        records = sheet.get_all_records()
        return pd.DataFrame(records)

    st.subheader("🔒 Admin Dashboard")
    admin_key = st.text_input("Enter Admin Access Key", type="password")
    if admin_key != st.secrets["ADMIN_KEY"]:
        st.warning("Access denied.")
        st.stop()

    st.success("Access granted. Welcome, Admin.")

    try:
        logs_df = get_visitor_logs()
        st.dataframe(logs_df)
        st.download_button("📥 Download Logs CSV", logs_df.to_csv(index=False), "visitor_logs.csv")
    except Exception as e:
        st.error("❌ Failed to load visitor logs.")
        st.exception(e)
