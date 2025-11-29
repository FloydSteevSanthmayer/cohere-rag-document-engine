import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Cohere RAG PDF", layout="wide")
st.title("Cohere RAG PDF — Demo")
st.write("Starter launcher: replace with full pipeline implementation.")

if not os.getenv("COHERE_API_KEY"):
    st.warning("COHERE_API_KEY not set. See .env.example")
if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("OPENROUTER_API_KEY not set. See .env.example")
