import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
import pickle
import io
import us  # pip install us
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(layout="wide")

import os
import requests


LOCAL_MMDB_PATH = "/tmp/GeoLite2-City.mmdb"  # Streamlit supports /tmp for temp files

@st.cache_resource
def get_geoip_reader():
    import geoip2.database

    def download_large_file_from_drive(file_id, destination):
        session = requests.Session()
        URL = "https://docs.google.com/uc?export=download"

        response = session.get(URL, params={"id": file_id}, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
        if token:
            response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if b"<html" in chunk[:100].lower():
                    raise ValueError("❌ ERROR: File is not a valid .mmdb binary. Likely an HTML error page was downloaded.")
                f.write(chunk)


    if not os.path.exists(LOCAL_MMDB_PATH):
        with st.spinner("⬇️ Downloading GeoLite2-City.mmdb from Google Drive..."):
            download_large_file_from_drive("1_wuanXceHz-XUaXSMrIT-lBM7qWrCPqI", LOCAL_MMDB_PATH)

    return geoip2.database.Reader(LOCAL_MMDB_PATH)



@st.cache_resource
def get_gsheet_client():
    creds_dict = dict(st.secrets["gcp"])
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client

@st.cache_data(ttl=3600)
def load_google_sheet(sheet_name="STAGE 5"):
    client = get_gsheet_client()
    worksheet = client.open("2025-2026 Grad Marketing Data (Refreshed Weekly)").worksheet(sheet_name)
    data = worksheet.get_all_records()
    return pd.DataFrame(data), worksheet

def ip_to_geo(ip, reader):
    try:
        response = reader.city(ip)
        return {
            "Zip Code": response.postal.code,
            "city": response.city.name,
            "region": response.subdivisions.most_specific.name,
            "country": response.country.name
        }


    except:
        return {"Zip Code": None, "city": None, "region": None, "country": None}

def enrich_geo_fields(df, reader):
    def enrich_row(row):
        geo = ip_to_geo(row.get("Ping IP Address", ""), reader)
        if pd.notna(row.get("Zip Code")) and str(row["Zip Code"]).strip():
            geo["Zip Code"] = row["Zip Code"]  # Preserve manual ZIPs
        return pd.Series(geo)


    enriched = df.apply(enrich_row, axis=1)
    for col in ["Zip Code", "city", "region", "country"]:
        df[col] = enriched[col]
    return df


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
st.image("logo.png", width=160)
st.markdown("## GPT-Powered Graduate-Marketing Data Explorer", unsafe_allow_html=True)
if "mobile_view" not in st.session_state:
    st.session_state.mobile_view = True

st.sidebar.subheader("🖼️ View Settings")
st.session_state.mobile_view = st.sidebar.checkbox("📱 Enable Mobile View", value=st.session_state.mobile_view)


# --- Cached functions ---
@st.cache_data
def preprocess_timestamps(df):
    for col in ["Ping Timestamp", "Applications Created Date", "Applications Submitted Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def get_enriched_geo_df(filtered_df, _reader):  # 👈 Note the leading underscore
    return enrich_geo_fields(filtered_df.copy(), _reader)


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
    client = get_gsheet_client()
    sheet = client.open_by_key(st.secrets["gdrive"]["file_id"]).sheet1
    records = sheet.get_all_records()
    return pd.DataFrame(records)
@st.cache_data
def load_us_states_geojson():
    file_id = "1hsUzy5HmhEa_s5vu3uUdpnps-hpbQ_WQ"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    return json.loads(r.content)


## not using the below, testing new grab from google sheets ###
#def load_data_from_gdrive():
    #creds_data = base64.b64decode(st.secrets["gdrive"]["token_pickle"])
    #creds = pickle.loads(creds_data)
    #service = build('drive', 'v3', credentials=creds)

    #file_id = st.secrets["gdrive"]["file_id"]
    #request = service.files().get_media(fileId=file_id)
    #fh = io.BytesIO()

    #from googleapiclient.http import MediaIoBaseDownload
    #downloader = MediaIoBaseDownload(fh, request)
    #done = False
    #while not done:
     #   status, done = downloader.next_chunk()

    #fh.seek(0)
    #return pd.read_excel(fh)  # ✅ Correct for Excel


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
    "Page 4: Admin Dashboard",
    "Page 5: Geographic Insights"
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


# Step 1: Initial filtering
base_filtered_df = get_filtered_data(df, selected_program, selected_status, selected_term)

# Step 2: Get actual date bounds from filtered data
ping_dates = pd.to_datetime(base_filtered_df["Ping Timestamp"], errors="coerce").dropna()
data_min, data_max = ping_dates.min(), ping_dates.max()

# Step 3: Let user override date range (defaults to range of filtered data)
selected_dates = st.sidebar.date_input(
    "📅 Date Range (Ping Timestamp)",
    value=(data_min.date(), data_max.date()),
    min_value=data_min.date(),
    max_value=data_max.date()
)

# Step 4: Final filter — apply date selection
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start, end = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
    filtered_df = base_filtered_df[(ping_dates >= start) & (ping_dates <= end)]
    st.sidebar.caption(f"📆 Showing data from **{start.date()}** to **{end.date()}**")
else:
    filtered_df = base_filtered_df



    


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

    # 🛡 Ensure numeric values only
    melted["Hours"] = pd.to_numeric(melted["Hours"], errors="coerce")
    melted = melted.dropna(subset=["Hours"])



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

    if st.sidebar.checkbox("🧠 Show Page Summary", value=False):
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


    if st.sidebar.checkbox("🧠 Show Page Summary", value=False):
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

# --- PAGE 5: Geographic Insights ---
elif view == "Page 5: Geographic Insights":
    st.subheader("🌍 Geographic Insights")

    reader = get_geoip_reader()

    # ✅ 1. Enrich geo data
    geo_df = get_enriched_geo_df(filtered_df, reader)

    # ✅ 2. Create U.S.-only DataFrame with normalized state names
    us_states_geojson = load_us_states_geojson()
    valid_state_names = [f["properties"]["NAME"] for f in us_states_geojson["features"]]

    # --- Full Geo Data: unfiltered, for global views ---
    geo_df_all = geo_df.copy()

# --- U.S. Only (for ZIPs, U.S. state maps) ---
    geo_df_us = geo_df.copy()
    geo_df_us["region"] = geo_df_us["region"].apply(
        lambda x: us.states.lookup(str(x)).name if us.states.lookup(str(x)) else str(x)
    ).astype(str).str.strip()
    geo_df_us = geo_df_us[geo_df_us["region"].isin(valid_state_names)]



# --- Optional debug of unmatched states ---
    if st.sidebar.checkbox("🔎 Show Unmatched Regions", value=False):
        unmatched = geo_df_all[~geo_df_all["region"].isin(valid_state_names)]
        if not unmatched.empty:
            st.warning("⚠️ The following regions don't match U.S. states in the GeoJSON:")
            st.dataframe(
                unmatched["region"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "Unmatched", "region": "Count"})
            )
    
    # --- Top cities and ZIPs outside Georgia (non-GA, USA only) ---
    non_ga_df = geo_df_us[geo_df_us["region"] != "Georgia"].copy()

    zip_counts_non_ga = (
        non_ga_df["Zip Code"]
        .astype(str)
        .str.zfill(5)
        .replace("nan", pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
    )
    zip_counts_non_ga.columns = ["Zip Code", "Count"]

    non_ga_df["City_Region"] = non_ga_df["city"].fillna("") + ", " + non_ga_df["region"].fillna("")
    city_counts_non_ga = non_ga_df["City_Region"].value_counts().reset_index()
    city_counts_non_ga.columns = ["City, Region", "Count"]

# --- US ZIP distribution ---
    zip_counts = (
        geo_df_us["Zip Code"]
        .astype(str)
        .str.zfill(5)
        .replace("nan", pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
    )
    zip_counts.columns = ["Zip Code", "Count"]

# --- State-level choropleth ---
    state_counts = geo_df_us["region"].value_counts().reset_index()
    state_counts.columns = ["region", "count"]
    
    from streamlit_folium import st_folium
    import folium

    # Prepare state-level data
    state_choro_data = dict(zip(state_counts["region"], state_counts["count"]))

    # Load the GeoJSON separately as a folium object
    m = folium.Map(location=[37.8, -96], zoom_start=4, tiles="cartodbpositron")

    # Add choropleth layer
    folium.Choropleth(
        geo_data=us_states_geojson,
        name="choropleth",
        data=state_counts,
        columns=["region", "count"],
        key_on="feature.properties.NAME",
        fill_color="Blues",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Lead Count by U.S. State"
    ).add_to(m)

    # Optional: add hover tooltip with state names
    folium.GeoJsonTooltip(fields=["NAME"]).add_to(
        folium.GeoJson(us_states_geojson)
    )

    from streamlit_folium import st_folium

    # ✅ Actually display the map
    st_data = st_folium(m, width=900, height=600)

 

# --- 🌍 Global map — switch to geo_df_all ---
    country_counts = geo_df_all["country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Count"]

    fig_country = px.choropleth(
        country_counts,
        locations="Country",
        locationmode="country names",
        color="Count",
        title="Global Lead Distribution"
    )

# --- 🌎 Domestic vs International using geo_df_all ---
    def normalize_country(country):
        country = str(country or "").strip().lower()
        return "USA" if country in {"united states", "us", "usa", "u.s.", "u.s.a.", "united states of america"} else "International"

    geo_df_all["is_domestic"] = geo_df_all["country"].apply(normalize_country)

    
  
    dom_counts = geo_df_all["is_domestic"].value_counts().reset_index()
    dom_counts.columns = ["Region", "Count"]

    fig_domestic = px.pie(dom_counts, names="Region", values="Count", title="USA vs International Leads")

# --- 🌆 Global cities ---
    city_region_df = geo_df_all.copy()
    city_region_df["City_Region"] = city_region_df["city"].fillna("") + ", " + city_region_df["region"].fillna("")
    city_counts = city_region_df["City_Region"].value_counts().reset_index()
    city_counts.columns = ["City, Region", "Count"]


    # --- Clean up noisy/invalid values for better UX ---
    zip_counts["Zip Code"] = zip_counts["Zip Code"].astype(str)
    zip_counts_non_ga["Zip Code"] = zip_counts_non_ga["Zip Code"].astype(str)
    city_counts["City, Region"] = city_counts["City, Region"].str.strip()
    city_counts_non_ga["City, Region"] = city_counts_non_ga["City, Region"].str.strip()

    zip_counts = zip_counts[zip_counts["Zip Code"].str.lower() != "none"]
    city_counts = city_counts[~city_counts["City, Region"].str.contains("None", case=False, na=False)]
    city_counts_non_ga = city_counts_non_ga[~city_counts_non_ga["City, Region"].str.contains("None", case=False, na=False)]
    zip_counts_non_ga = zip_counts_non_ga[zip_counts_non_ga["Zip Code"].str.lower() != "none"]





    # --- Tab Layout ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📬 Top ZIP Codes", 
        "🗺 US States", 
        "🌐 Countries", 
        "🏙 Cities", 
        "🇺🇸 vs 🌍",
        "🌆 Cities Outside GA", 
        "📮 ZIPs Outside GA"
    ])

    with tab1:
        st.markdown("### 📬 Top ZIP Codes")
        st.dataframe(zip_counts.head(10))

    with tab2:
        st.markdown("### 🗺 Lead Density by U.S. State")
        st.plotly_chart(fig_zip)

    with tab3:
        st.markdown("### 🌐 Global Heatmap by Country")
        st.plotly_chart(fig_country)

    with tab4:
        st.markdown("### 🏙 Top 10 Cities by Engagement")
        st.dataframe(city_counts.head(10))

    with tab5:
        st.markdown("### 🇺🇸 Domestic vs 🌍 International")
        st.plotly_chart(fig_domestic)

    with tab6:
        st.markdown("### 🌆 Top Cities Outside Georgia")
        st.dataframe(city_counts_non_ga.head(10))

    with tab7:
        st.markdown("### 📮 Top ZIPs Outside Georgia")
        st.dataframe(zip_counts_non_ga.head(10))

