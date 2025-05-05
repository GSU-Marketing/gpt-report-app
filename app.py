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
    st.subheader("📌 Filter Your Data (Optional)")
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_program = st.selectbox("Filter by Program:", ["All"] + sorted(df['Inquiry Program'].dropna().unique().tolist()))
    with col2:
        selected_status = st.selectbox("Filter by Status:", ["All"] + sorted(df['Status'].dropna().unique().tolist()))
    with col3:
        selected_term = st.selectbox("Filter by Term:", ["All"] + sorted(df['Term'].dropna().unique().tolist()))

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

    if st.button("Generate AI Report") and prompt:
        csv_data = filtered_df.to_csv(index=False)
        full_prompt = f"You are a university data analyst. Here is a CSV dataset:\n\n{csv_data}\n\n{prompt}"
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": full_prompt}]
        )
        st.write(response.choices[0].message.content)

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
