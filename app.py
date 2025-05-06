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

# --- Filters ---
st.sidebar.subheader("🔎 Filter Data")
programs = ["All"] + sorted(df['Applications Applied Program'].dropna().unique())
statuses = ["All"] + sorted(df['Person Status'].dropna().unique())
terms = ["All"] + sorted(df['Applications Applied Term'].dropna().unique())

selected_program = st.sidebar.selectbox("Program:", programs)
selected_status = st.sidebar.selectbox("Status:", statuses)
selected_term = st.sidebar.selectbox("Term:", terms)

filtered_df = get_filtered_data(df, selected_program, selected_status, selected_term)

# --- PAGE 1: Funnel Overview ---
if view == "Page 1: Funnel Overview":
    st.subheader("🎯 Lead Funnel Overview")

    inquiries = len(filtered_df[filtered_df['Person Status'] == 'Inquiry'])
    applicants = len(filtered_df[filtered_df['Person Status'] == 'Applicant'])
    enrolled = len(filtered_df[filtered_df['Person Status'] == 'Enrolled'])

    stacked = st.sidebar.checkbox("📱 Mobile View", value=False)

    if stacked:
        for label, count in zip(["🧠 Inquiries", "📄 Applicants", "🎓 Enrolled"], [inquiries, applicants, enrolled]):
            st.metric(label, count)
    else:
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

    if "Person Citizenship" in filtered_df.columns:
        citizenship_clean = filtered_df["Person Citizenship"].str.lower().fillna("")
        filtered_df["Citizenship Type"] = citizenship_clean.apply(lambda x: "Domestic" if x == "united states" else "International")
        citizenship_df = filtered_df["Citizenship Type"].value_counts().reset_index()
        citizenship_df.columns = ["Type", "Count"]
        fig = px.pie(citizenship_df, names="Type", values="Count", title="Domestic vs International Leads")
        st.plotly_chart(fig, config={'displayModeBar': False})

# --- PAGE 2: Geography & Program ---
elif view == "Page 2: Geography & Program":
    st.subheader("🌍 Geography & Program Breakdown")

    top_programs = filtered_df['Applications Applied Program'].value_counts().head(10).reset_index()
    top_programs.columns = ['Program', 'Count']
    fig = px.bar(top_programs, x='Count', y='Program', orientation='h', title="Top Applied Programs")
    st.plotly_chart(fig, config={'displayModeBar': False})

    if "Zip Code" in filtered_df.columns:
        zip_counts = filtered_df["Zip Code"].astype(str).value_counts().reset_index()
        zip_counts.columns = ["Zip Code", "Count"]
        fig = px.bar(zip_counts.head(20), x="Count", y="Zip Code", orientation="h", title="Top ZIP Codes")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Application APOS Program Modality" in filtered_df.columns:
        modality_counts = filtered_df["Application APOS Program Modality"].value_counts().reset_index()
        modality_counts.columns = ["Modality", "Count"]
        fig = px.bar(modality_counts, x="Modality", y="Count", title="Application Modality Distribution")
        st.plotly_chart(fig, config={'displayModeBar': False})

# --- PAGE 3: Engagement & Traffic ---
elif view == "Page 3: Engagement & Traffic":
    st.subheader("📈 Engagement & Traffic Sources")

    if "Ping UTM Source" in filtered_df.columns:
        traffic_df = filtered_df[filtered_df["Ping UTM Source"].notna()]
        fig = px.pie(traffic_df, names="Ping UTM Source", title="Traffic Sources (Non-null)")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Ping UTM Medium" in filtered_df.columns:
        medium_df = filtered_df[filtered_df['Ping UTM Medium'].notna()]
        fig = px.bar(medium_df.value_counts('Ping UTM Medium').reset_index(name='Count'),
                     x="Ping UTM Medium", y="Count", title="Traffic by UTM Medium")
        st.plotly_chart(fig, config={'displayModeBar': False})

    if "Ping UTM Campaign" in filtered_df.columns:
        campaign_df = filtered_df[filtered_df['Ping UTM Campaign'].notna()]
        fig = px.bar(campaign_df.value_counts('Ping UTM Campaign').reset_index(name='Count'),
                     x="Ping UTM Campaign", y="Count", title="Traffic by UTM Campaign")
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
    with st.spinner("🔎 GPT is analyzing the data..."):
        try:
            sample_csv = filtered_df.sample(min(25, len(filtered_df))).to_csv(index=False)
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
            st.error("❌ Error: GPT analysis failed. Please retry later.")
