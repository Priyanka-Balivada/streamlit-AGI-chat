import streamlit as st

st.sidebar.page_link("ui.py", label="Home", icon="🏠")
st.sidebar.page_link("pages/rag.py", label="Page 1", icon="1️⃣")
st.sidebar.page_link("pages/rag4.py", label="Page 2", icon="2️⃣", disabled=True)
st.sidebar.page_link("http://www.google.com", label="Google", icon="🌎")