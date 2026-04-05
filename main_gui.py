import streamlit as st
from pipelines.data_organizers.file_pathways import MASTER_CSVS_FOLDER, RUNTIME_FOLDERS, ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER, MAIN_CSV
from pipelines.data_organizers.csv_loader import load_clean
# Statistics
from pipelines.statistics import master_descriptive_gen, all_single_var_desc_gen, multivar_desc_gen
from pipelines.statistics.png_generators import desc_gen, hm_gen, pg_gen, pp_gen
# ML
from pipelines.ml import gmm_analysis, olr_mlm, rm_anova_icc
# Data Organizers
from pipelines.data_organizers import csv_merger, type_converter

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
st.title("Statistical & ML Analysis Toolkit")

selected = st.multiselect("Select variables", var_options)

def warn_min(n=2):
    if len(selected) < n:
        st.warning(f"Please select at least {n} variable{'s' if n > 1 else ''}.")
        return False
    return True

# --- Multivariate Analyses ---
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

# --- ML Analyses ---
st.subheader("ML Analyses")
col3, col4 = st.columns(2)

with col3:
    if st.button("GMM Analysis", use_container_width=True):
        if warn_min(2):
            gmm_analysis.gmm_analysis(*selected)

with col4:
    if st.button("Ordinal Logistic Regression & Multilevel Modeling", use_container_width=True):
        if warn_min(2):
            olr_mlm(*selected)

# --- Data Visualizations ---
st.subheader("Data Visualizations")
col5, col6, col7, col8 = st.columns(4)

with col5:
    if st.button("Descriptive Visualization", use_container_width=True):
        if warn_min(2):
            desc_gen.desc_visualization(*selected)

with col6:
    if st.button("Heatmap Visualization", use_container_width=True):
        if warn_min(2):
            hm_gen.heatmap_visualizations(*selected)

with col7:
    if st.button("PairGrid Visualization", use_container_width=True):
        if warn_min(2):
            pg_gen.pairgrid_visualizations(*selected)

with col8:
    if st.button("PairPlot Visualization", use_container_width=True):
        if warn_min(2):
            pp_gen.pair_plot_visualizations(*selected)

# --- Full-Data Generators ---
st.subheader("Full-Data Generators")
col9, col10 = st.columns(2)

with col9:
    if st.button("Master Descriptive CSV Generator", use_container_width=True):
        master_descriptive_gen.master_descriptive_csv_generator()
        st.success("Master Descriptive CSV Generated")

with col10:
    if st.button("All Variable Descriptives Generator", use_container_width=True):
        all_single_var_desc_gen.all_single_var_descriptive_generator()
        st.success("All Variable Descriptives Generated")

# --- Full-Data Generators ---
st.subheader("Full-Data Generators")
col11, col12 = st.columns(2)

with col11:
    if st.button("File Type Converter", use_container_width=True):
        type_converter.to_csv()
        st.success("Data files converted to CSVs")

with col12:
    if st.button("CSV Merger", use_container_width=True):
        csv_merger.merge_csv()
        st.success("All CSVs in the 'Unmerged CSV' folder have been merged")