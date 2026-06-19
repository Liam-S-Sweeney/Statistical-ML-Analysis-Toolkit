import streamlit as st
import os
import time
import logging
from pipelines.data_organizers.file_pathways import MASTER_CSVS_FOLDER, RUNTIME_FOLDERS, MAIN_CSV
from pipelines.data_organizers.csv_loader import load_clean
# Statistics
from pipelines.statistics import master_descriptive_gen, all_single_var_desc_gen, multivar_desc_gen, chi_sqr
from pipelines.statistics.png_generators import desc_gen, hm_gen, pg_gen, pp_gen
# ML
from pipelines.ml import blr, gmm_analysis, lda, olr, ols_mra, rm_anova_icc, ols, nbr, blr_mra
# Data Organizers
from pipelines.data_organizers import csv_merger, type_converter
# Utility
from pipelines.utility import overlap_checker
# Styles
from app_styles import load_css


# --- Setup ---
for folder in RUNTIME_FOLDERS:
    os.makedirs(folder, exist_ok=True)

csv_available = MAIN_CSV.exists()

if csv_available:
    @st.cache_data
    def get_df():
        return load_clean()
    df = get_df()
    var_options = list(df.columns)
else:
    df = None
    var_options = []

# Log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("toolkit.log")
    ]
)

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
    if csv_available:
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
        exo_selected = st.multiselect("Independent / Exogenous", var_options, label_visibility='collapsed')
        st.divider()

        st.markdown("""
        <p>Moderator Variable</p>
        """, unsafe_allow_html=True)
        mod_selected = st.multiselect("Moderator", var_options, label_visibility='collapsed')
        overlap_checker.check_for_var_overlap(endo=endo_selected, exo=exo_selected, mod=mod_selected)
        st.divider()

    else:
        st.info("Load a CSV to enable variable selection.")

# --- N Var Verification ---
def gen_min(n=2):
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

def exo_max(n=1):
    if len(exo_selected) > n:
        st.warning(f"Please select at most {n} exogenous/independent variable{'s' if n > 1 else ''}.")
        return False
    return True

def exo_min(n=1):
    if len(exo_selected) < n:
        st.warning(f"Please select at least {n} exogenous/independent variable{'s' if n > 1 else ''}.")
        return False
    return True

def mod_max(n=1):
    if len(mod_selected) > n:
        st.warning(f"Please select at most {n} moderator variable{'s' if n > 1 else ''}.")
        return False
    return True

def mod_min(n=1):
    if len(mod_selected) < n:
        st.warning(f"Please select at least {n} moderator variable{'s' if n > 1 else ''}.")
        return False
    return True

# --- Multivariate Analyses ---
st.divider()
st.subheader("Statistical Analyses")

col_var_expl, col_rma_icc, col_chi = st.columns(3)
with col_var_expl:
    if st.button("Uni/Multivariate Exploration", use_container_width=True):
        if csv_available and gen_min(1):
            multivar_desc_gen.explore_multi_variables(*selected)
            st.success("Uni/Multivariate Exploration CSV Generated")

with col_rma_icc:
    if st.button("RM-Anova & ICC", use_container_width=True):
        if csv_available and gen_min(2):
            rm_anova_icc.rm_anova_icc(*selected)
            st.success("RM-Anova & ICC CSV Generated")

with col_chi:
    if st.button("Chi-Square Test", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_min(1):
            result = chi_sqr.run_chi_sqr(endo=endo_selected, exo=exo_selected)
            st.success("Chi-Square Test Analysis Generated")

st.divider()

# --- ML Analyses ---
st.subheader("ML Analyses")
col_gmm, col_lda, col_olr, col_nbr, col_blr, col_blr_mra, col_ols, col_ols_mra = st.columns(8)

with col_gmm:
    if st.button("GMM Analysis", use_container_width=True):
        if csv_available and gen_min(2):
            gmm_analysis.gmm_analysis(*selected)
            st.success("GMM Analysis Generated")

with col_lda:
    if st.button("LDA Analysis", use_container_width=True):
        if csv_available and gen_min(2):
            lda.lda_model(*selected)
            st.success("LDA Analysis Generated")

with col_olr:
    if st.button("OLR", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_min(1):
            result = olr.run_olr(endo=endo_selected, exo=exo_selected)
            st.success("OLR Analysis Generated")

with col_nbr:
    if st.button("NBR", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_min(1):
            result = nbr.run_nbr(endo=endo_selected, exo=exo_selected)
            st.success("NBR Analysis Generated")         

with col_blr:
    if st.button("BLR", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_min(1):
            result = blr.run_blr(endo=endo_selected, exo=exo_selected)
            st.success("BLR Analysis Generated")   

with col_blr_mra:
    if st.button("BLR MRA + SSA", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_max(1) and exo_min(1) and mod_max(1) and mod_min(1):
            result = blr_mra.run_mra(
                endo=endo_selected, 
                focal_var=exo_selected, 
                moderator_var=mod_selected,
            )
            if result is not None:
                st.error(str(result))
            else:
                st.success("BLR MRA Generated")  

with col_ols:
    if st.button("OLSR", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_min(1):
            result = ols.run_ols(endo=endo_selected, exo=exo_selected)
            st.success("OLS Analysis Generated")

with col_ols_mra:
    if st.button("OLS MRA + SSA", use_container_width=True):
        if csv_available and endo_max(1) and endo_min(1) and exo_max(1) and exo_min(1) and mod_max(1) and mod_min(1):
            result = ols_mra.run_mra(
                endo=endo_selected, 
                focal_var=exo_selected, 
                moderator_var=mod_selected,
            )
            st.success("OLS MRA Generated")


st.divider()

# --- Data Visualizations ---
st.subheader("Data Visualizations")
col_des_vis, col_hm, col_pg, col_pp = st.columns(4)

with col_des_vis:
    if st.button("Descriptive", use_container_width=True):
        if csv_available and gen_min(2):
            desc_gen.desc_visualization(*selected)
            st.success("Descriptive Visualization Generated")

with col_hm:
    if st.button("Heatmap", use_container_width=True):
        if csv_available and gen_min(2):
            hm_gen.heatmap_visualizations(*selected)
            st.success("Heatmap Visualization Generated")

with col_pg:
    if st.button("PairGrid", use_container_width=True):
        if csv_available and gen_min(2):
            pg_gen.pairgrid_visualizations(*selected)
            st.success("PairGrid Visualization Generated")

with col_pp:
    if st.button("PairPlot", use_container_width=True):
        if csv_available and gen_min(2):
            pp_gen.pair_plot_visualizations(*selected)
            st.success("PairPlot Visualization Generated")
st.divider()

# --- Full-Data Generators ---
st.subheader("Full-Data Generators")
col_md_gen, col_av_gen = st.columns(2)

with col_md_gen:
    if st.button("Master Descriptive CSV Generator", use_container_width=True):
        master_descriptive_gen.master_descriptive_csv_generator()
        st.success("Master Descriptive CSV Generated")

with col_av_gen:
    if st.button("All Variable Descriptives Generator", use_container_width=True):
        all_single_var_desc_gen.all_single_var_descriptive_generator()
        st.success("All Variable Descriptives Generated")
st.divider()

# --- Data Preparation ---
if not csv_available:
    st.warning(f"No CSV found at `{MASTER_CSVS_FOLDER}`. Analysis features disabled — use the File Converter or CSV Merger below to prepare your data.")
st.subheader("Data Preparation")
col_to_csv, col_csv_merger = st.columns(2)

with col_to_csv:
    if st.button("File Type Converter", use_container_width=True):
        type_converter.to_csv()
        st.success("Data files converted to CSVs")

with col_csv_merger:
    if st.button("CSV Merger", use_container_width=True):
        csv_merger.merge_csv()
        st.success("All CSVs in the 'Unmerged CSV' folder have been merged")
        time.sleep(1)
        st.rerun()