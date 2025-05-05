import streamlit as st

def show_looker_dashboard(looker_url: str, height: int = 800):
    """
    Embed a Looker Studio dashboard via iframe.

    Args:
        looker_url (str): The full Looker Studio embed URL.
        height (int): Height of the iframe (default is 800).
    """
    st.markdown(
        f"""
        <iframe src="{looker_url}" width="100%" height="{height}" frameborder="0" style="border:0" allowfullscreen 
        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-forms">
        </iframe>
        """,
        unsafe_allow_html=True,
    )
