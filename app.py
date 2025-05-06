import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

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
    for col in ["Ping Timestamp", "Applications Created Date", "Applications Submitted Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
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

    funnel_fig = go.Figure(go.Funnel(
        y = ["Inquiry", "Applicant", "Enrolled"],
        x = [inquiries, applicants, enrolled]
    ))
    st.plotly_chart(funnel_fig, use_container_width=True)

    leads_over_time = filtered_df[filtered_df['Person Status'].isin(['Inquiry', 'Applicant', 'Enrolled'])]
    leads_over_time = leads_over_time.dropna(subset=["Ping Timestamp"])
    fig = px.histogram(leads_over_time, x="Ping Timestamp", color="Person Status", barmode="group",
                       title="Leads Over Time")
    st.plotly_chart(fig, use_container_width=True)

    df_term = filtered_df.copy()
    df_term["Term"] = df_term["Applications Applied Term"].combine_first(df_term["Person Inquiry Term"])
    df_term = df_term[df_term["Person Status"].isin(["Inquiry", "Applicant", "Enrolled"])]
    term_counts = df_term.groupby(["Term", "Person Status"]).size().reset_index(name="Count")
    fig = px.bar(term_counts, x="Term", y="Count", color="Person Status", barmode="group",
                 title="Leads by Term")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: Geography & Program ---
elif view == "Page 2: Geography & Program":
    st.subheader("🌍 Geography & Program Breakdown")

    top_programs = filtered_df['Applications Applied Program'].value_counts().head(10).reset_index()
    top_programs.columns = ['Program', 'Count']
    fig = px.bar(top_programs, x='Count', y='Program', orientation='h', title="Top Applied Programs")
    st.plotly_chart(fig, use_container_width=True)

    if "Zip Code" in filtered_df.columns:
        zip_counts = filtered_df["Zip Code"].astype(str).value_counts().reset_index()
        zip_counts.columns = ["Zip Code", "Count"]
        fig = px.choropleth(zip_counts, locations="Zip Code", locationmode="USA-states",
                            color="Count", scope="usa", title="Lead Distribution by Zip Code")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 3: Engagement & Traffic ---
elif view == "Page 3: Engagement & Traffic":
    st.subheader("📈 Engagement & Traffic Sources")

    if "Ping Duration (seconds)" in filtered_df.columns:
        fig = px.histogram(filtered_df, x="Ping Duration (seconds)", nbins=30,
                           title="Engagement Duration Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if "Ping UTM Source" in filtered_df.columns:
        traffic_df = filtered_df[filtered_df["Ping UTM Source"].notna()]
        fig = px.pie(traffic_df, names="Ping UTM Source", title="Traffic Sources (Non-null)")
        st.plotly_chart(fig, use_container_width=True)

    if "Ping Timestamp" in filtered_df.columns:
        filtered_df["Hour"] = pd.to_datetime(filtered_df["Ping Timestamp"], errors='coerce').dt.hour
        fig = px.histogram(filtered_df, x="Hour", nbins=24, title="Activity by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

    # Applications Created and Submitted Time Series
    if "Applications Created Date" in filtered_df.columns:
        created_counts = filtered_df.dropna(subset=["Applications Created Date"])
        fig = px.histogram(created_counts, x="Applications Created Date", title="Applications Created Over Time")
        st.plotly_chart(fig, use_container_width=True)

    if "Applications Submitted Date" in filtered_df.columns:
        submitted_counts = filtered_df.dropna(subset=["Applications Submitted Date"])
        fig = px.histogram(submitted_counts, x="Applications Submitted Date", title="Applications Submitted Over Time")
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