import streamlit as st
from pipelines.data_organizers.file_pathways import APP_STYLES_FOLDER


def load_css(path=APP_STYLES_FOLDER / 'main.css') -> None:
    """
    Inject a CSS file into the Streamlit page as a <style> block.

    Streamlit has no native stylesheet API, so the file's contents are
    passed through st.markdown with HTML escaping disabled. 

    Args:
        path: CSS file to load. Defaults to main.css in APP_STYLES_FOLDER.
    """
    with open(path) as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
