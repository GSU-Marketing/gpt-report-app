import openai
from openai import OpenAI
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os

# Load env variables if needed
load_dotenv()

# Get the API key from secrets.toml or .env
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Create OpenAI client for v1+
client = OpenAI(api_key=openai.api_key)

# Streamlit UI
st.title("📊 GPT-4 Data Report Generator")

uploaded_file = st.file_uploader("Upload your cleaned dataset (Excel or CSV)", type=["csv", "xlsx"])

query = st.text_input("Enter a custom question or leave blank for summary:")

if st.button("Generate GPT-4 Report") and uploaded_file:
    # Read the uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Convert DataFrame to string summary
    data_summary = df.head(10).to_string()

    prompt = f"""
    You are a senior data analyst. Analyze this dataset and generate an executive summary.

    Data (first 10 rows):
    {data_summary}

    {'Custom question: ' + query if query else ''}
    """

    # Send to GPT-4
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a senior data analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content
    st.subheader("📄 GPT-4 Report")
    st.write(report)
