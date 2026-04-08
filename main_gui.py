import streamlit as st
from pipelines.data_organizers.file_pathways import MASTER_CSVS_FOLDER, RUNTIME_FOLDERS, MAIN_CSV
from pipelines.data_organizers.csv_loader import load_clean
# Statistics
from pipelines.statistics import master_descriptive_gen, all_single_var_desc_gen, multivar_desc_gen
from pipelines.statistics.png_generators import desc_gen, hm_gen, pg_gen, pp_gen
# ML
from pipelines.ml import gmm_analysis, olr, rm_anova_icc, ols
# Data Organizers
from pipelines.data_organizers import csv_merger, type_converter
# Styles
from app_styles import load_css

import os

# --- Setup ---
for folder in RUNTIME_FOLDERS:
    os.makedirs(folder, exist_ok=True)

if not MAIN_CSV.exists():
    st.error(f"\n CSV file not found at `{MASTER_CSVS_FOLDER}`.\n\n" 
             f"Place your dataset in the `{MASTER_CSVS_FOLDER}`" 
             f"and check the 'config.py' to make sure {MAIN_CSV} is the name of your file in 'MAIN_CSV_NAME")
    st.stop()

@st.cache_data
def get_df():
    return load_clean()

df = get_df()
var_options = list(df.columns)

# --- UI ---
st.set_page_config(
    page_title="Statistical & ML Toolkit", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="auto",
                   )

load_css.load_css()

st.title("Statistical & ML Analysis Toolkit")

signature = 'Developed by Liam Sweeney'
st.markdown(f"<p style='text-align: center; color: gray;'>{signature}</p>", unsafe_allow_html=True)


with st.sidebar:
    st.header("Variable Selection")

    st.markdown("""
    <p>General Variables</p>
    """, unsafe_allow_html=True)
    selected = st.multiselect("General", var_options, label_visibility='collapsed')
    st.divider()

    st.markdown("""
    <p>Dependent Variables</p>
    """, unsafe_allow_html=True)
    endo_selected = st.multiselect("Dependent / Endogenous", var_options, label_visibility='collapsed')
    st.divider()
    
    st.markdown("""
    <p>Independent Variables</p>
    """, unsafe_allow_html=True)
    exo_selected = st.multiselect("Independetnt / Exogenous", [var for var in var_options if var not in endo_selected], label_visibility='collapsed')

# --- N Var Verification ---
def warn_min(n=2):
    if len(selected) < n:
        st.warning(f"Please select at least {n} variable{'s' if n > 1 else ''}.")
        return False
    return True

def endo_max(n=1):
    if len(endo_selected) > n:
        st.warning(f"Please select at most {n} endogenous/dependent variable{'s' if n > 1 else ''}.")
        return False
    return True

def endo_min(n=1):
    if len(endo_selected) < n:
        st.warning(f"Please select at least {n} endogenous/dependent  variable{'s' if n > 1 else ''}.")
        return False
    return True

def exo_min(n=1):
    if len(exo_selected) < n:
        st.warning(f"Please select at least {n} exogenous/independent variable{'s' if n > 1 else ''}.")
        return False
    return True

# --- Multivariate Analyses ---
st.divider()
st.subheader("Statistical Analyses")
col1, col2 = st.columns(2)

with col1:
    if st.button("Uni/Multivariate Exploration", use_container_width=True):
        if warn_min(1):
            multivar_desc_gen.explore_multi_variables(*selected)
            st.success("Uni/Multivariate Exploration CSV Generated")

with col2:
    if st.button("RM-Anova & ICC", use_container_width=True):
        if warn_min(2):
            rm_anova_icc.rm_anova_icc(*selected)
            st.success("RM-Anova & ICC CSV Generated")
st.divider()

# --- ML Analyses ---
st.subheader("ML Analyses")
col3, col4, col5 = st.columns(3)

with col3:
    if st.button("GMM Analysis", use_container_width=True):
        if warn_min(2):
            gmm_analysis.gmm_analysis(*selected)
            st.success("GMM Analysis Generated")

with col4:
    if st.button("Ordinal Logistic Regression (OLR)", use_container_width=True):
        if endo_max(1) and endo_min(1) and exo_min(1):
            result = olr.run_olr(endo=endo_selected, exo=exo_selected)
            st.success("OLR Analysis Generated")

with col5:
    if st.button("Ordinary Least Squares (OLS) Regression", use_container_width=True):
        if endo_max(1) and endo_min(1) and exo_min(1):
            result = ols.run_ols(endo=endo_selected, exo=exo_selected)
            st.success("OLS Analysis Generated")
st.divider()

# --- Data Visualizations ---
st.subheader("Data Visualizations")
col6, col7, col8, col9 = st.columns(4)

with col6:
    if st.button("Descriptive Visualization", use_container_width=True):
        if warn_min(2):
            desc_gen.desc_visualization(*selected)
            st.success("Descriptive Visualization Generated")

with col7:
    if st.button("Heatmap Visualization", use_container_width=True):
        if warn_min(2):
            hm_gen.heatmap_visualizations(*selected)
            st.success("Heatmap Vsiualization Generated")

with col8:
    if st.button("PairGrid Visualization", use_container_width=True):
        if warn_min(2):
            pg_gen.pairgrid_visualizations(*selected)
            st.success("PairGrid Vsiualization Generated")

with col9:
    if st.button("PairPlot Visualization", use_container_width=True):
        if warn_min(2):
            pp_gen.pair_plot_visualizations(*selected)
            st.success("PairPlot Vsiualization Generated")
st.divider()

# --- Full-Data Generators ---
st.subheader("Full-Data Generators")
col10, col11 = st.columns(2)

with col10:
    if st.button("Master Descriptive CSV Generator", use_container_width=True):
        master_descriptive_gen.master_descriptive_csv_generator()
        st.success("Master Descriptive CSV Generated")

with col11:
    if st.button("All Variable Descriptives Generator", use_container_width=True):
        all_single_var_desc_gen.all_single_var_descriptive_generator()
        st.success("All Variable Descriptives Generated")
st.divider()

# --- Full-Data Generators ---
st.subheader("Full-Data Generators")
col12, col13 = st.columns(2)

with col12:
    if st.button("File Type Converter", use_container_width=True):
        type_converter.to_csv()
        st.success("Data files converted to CSVs")

with col13:
    if st.button("CSV Merger", use_container_width=True):
        csv_merger.merge_csv()
        st.success("All CSVs in the 'Unmerged CSV' folder have been merged")
