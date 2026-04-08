import streamlit as st
from pipelines.data_organizers.file_pathways import APP_STYLES_FOLDER

def load_css(path=APP_STYLES_FOLDER / 'main.css'):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
