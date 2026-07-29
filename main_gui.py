"""
Statistical & ML Analysis Toolkit — Streamlit entry point.

Deployment note
---------------
This app is self-contained: a visitor uploads a CSV, picks variables and
settings in the sidebar, and every result is rendered in the browser with a
download button. Nothing depends on files being placed on the server's disk,
which is what makes it deployable to Streamlit Community Cloud.
"""

import logging
import os

import pandas as pd
import streamlit as st

from app_styles import load_css
from app_utils.results import run_and_render
from pipelines.data_organizers import session_data
from pipelines.data_organizers.file_pathways import (
    ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER,
    FIGURE_PNGS_OUTPUT_FOLDER,
    GMM_ANALYSIS_OUTPUT_FOLDER,
    LDA_OUTPUT_FOLDER,
    MASTER_VAR_DESC_OUTPUT_FOLDER,
    MULTI_VAR_ANALYSIS_OUTPUT_FOLDER,
    REGRESSION_ANALYSIS_OUTPUT_FOLDER,
    RUNTIME_FOLDERS,
)
# Statistics
from pipelines.statistics import (
    all_single_var_desc_gen,
    chi_sqr,
    master_descriptive_gen,
    multivar_desc_gen,
)
from pipelines.statistics.png_generators import desc_gen, hm_gen, pg_gen, pp_gen
# ML
from pipelines.ml import blr, gmm_analysis, lda, mblr, mols, nbr, olr, ols, rm_anova_icc
# Utility
from pipelines.utility import overlap_checker

# --- Setup ---
for folder in RUNTIME_FOLDERS:
    os.makedirs(folder, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)

ALL_OUTPUT_FOLDERS = [
    ALL_VAR_DESC_ANALYSIS_OUTPUT_FOLDER,
    FIGURE_PNGS_OUTPUT_FOLDER,
    GMM_ANALYSIS_OUTPUT_FOLDER,
    LDA_OUTPUT_FOLDER,
    MASTER_VAR_DESC_OUTPUT_FOLDER,
    MULTI_VAR_ANALYSIS_OUTPUT_FOLDER,
    REGRESSION_ANALYSIS_OUTPUT_FOLDER,
]

st.set_page_config(
    page_title="Statistical & ML Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

load_css.load_css()

st.title("Statistical & ML Analysis Toolkit")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by Liam Sweeney</p>",
    unsafe_allow_html=True,
)


# --- Data upload ---
@st.cache_data(show_spinner=False)
def _read_upload(file_bytes: bytes, name: str) -> pd.DataFrame:
    """Parse an uploaded file. Cached on content so re-runs don't re-parse."""
    import io

    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    if name.lower().endswith(".sav"):
        import pyreadstat

        with open("/tmp/_upload.sav", "wb") as handle:
            handle.write(file_bytes)
        frame, _ = pyreadstat.read_sav("/tmp/_upload.sav")
        return frame
    if name.lower().endswith(".json"):
        return pd.read_json(io.BytesIO(file_bytes))
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


with st.expander("Dataset", expanded=not session_data.has_active_df()):
    uploads = st.file_uploader(
        "Upload one or more data files (CSV, Excel, SPSS .sav, JSON)",
        type=["csv", "xlsx", "xls", "sav", "json"],
        accept_multiple_files=True,
    )

    merge_key = st.text_input(
        "ID column for merging (leave blank if uploading a single file)",
        value=session_data.get_setting("ID_VAR") or "",
        help="When several files are uploaded they are outer-joined on this column.",
    )

    if uploads:
        frames = {up.name: _read_upload(up.getvalue(), up.name) for up in uploads}

        if len(frames) == 1:
            name, merged = next(iter(frames.items()))
        elif merge_key:
            merged = None
            skipped = []
            for name, frame in frames.items():
                if merge_key not in frame.columns:
                    skipped.append(name)
                    continue
                prefix = name.rsplit(".", 1)[0]
                frame = frame.rename(
                    columns={
                        col: f"{prefix}_{col}" for col in frame.columns if col != merge_key
                    }
                )
                merged = frame if merged is None else merged.merge(
                    frame, on=merge_key, how="outer"
                )
            if skipped:
                st.warning(f"Skipped (no '{merge_key}' column): {', '.join(skipped)}")
            name = f"{len(frames) - len(skipped)} files merged on {merge_key}"
        else:
            st.error("Multiple files uploaded — enter an ID column to merge them on.")
            merged = None

        if merged is not None:
            session_data.set_active_df(merged, name)
            # Only seed ID_VAR from the merge key when one was actually given;
            # this block re-runs on every rerun and would otherwise wipe the
            # ID column chosen in the sidebar.
            if merge_key:
                session_data.set_setting("ID_VAR", merge_key)

    if session_data.has_active_df():
        active = session_data.get_active_df()
        st.success(
            f"Active dataset: **{session_data.get_active_name()}** — "
            f"{active.shape[0]:,} rows × {active.shape[1]:,} columns"
        )
        st.dataframe(active.head(20), use_container_width=True)
    else:
        st.info("Upload a dataset to enable the analyses below.")


data_available = session_data.has_active_df()
df = session_data.get_active_df()
var_options = list(df.columns) if data_available else []

def _seed_widget(key: str, stored, options: list) -> None:
    """
    Initialise a keyed selectbox once, and reset it if its value is no longer a
    valid option (e.g. the user swapped datasets). Streamlit owns the value
    after this; we never pass `index` again, which is what makes the selection
    survive reruns.
    """
    current = st.session_state.get(key)
    if current is None:
        st.session_state[key] = stored if stored in options else options[0]
    elif current not in options:
        st.session_state[key] = options[0]


# --- Sidebar: variables + settings ---
selected: list[str] = []
endo_selected: list[str] = []
exo_selected: list[str] = []
mod_selected: list[str] = []

with st.sidebar:
    if data_available:
        st.header("Variable Selection")
        selected = st.multiselect("General", var_options)
        endo_selected = st.multiselect("Dependent / Endogenous", var_options)
        exo_selected = st.multiselect("Independent / Exogenous", var_options)
        mod_selected = st.multiselect("Moderator", var_options)
        overlap_checker.check_for_var_overlap(
            endo=endo_selected, exo=exo_selected, mod=mod_selected
        )

        st.divider()
        st.header("Settings")

        with st.expander("Data cleaning"):
            id_options = ["(none)"] + var_options
            _seed_widget("cfg_id_var", session_data.get_setting("ID_VAR"), id_options)
            id_choice = st.selectbox(
                "Subject ID column",
                id_options,
                key="cfg_id_var",
                help="Required for chi-square, RM-ANOVA, and all regressions.",
            )
            session_data.set_setting("ID_VAR", "" if id_choice == "(none)" else id_choice)

            codes = st.text_input(
                "Missing-value codes (comma separated)",
                value=", ".join(str(c) for c in session_data.get_setting("MISSING_CODES")),
            )
            try:
                session_data.set_setting(
                    "MISSING_CODES",
                    [float(c.strip()) for c in codes.split(",") if c.strip()],
                )
            except ValueError:
                st.warning("Missing-value codes must be numeric.")

            session_data.set_setting(
                "IMPOSSIBLE_ZERO_VARS",
                st.multiselect(
                    "Variables where 0 is impossible",
                    var_options,
                    default=[
                        v
                        for v in session_data.get_setting("IMPOSSIBLE_ZERO_VARS")
                        if v in var_options
                    ],
                    help="Zeros in these columns are treated as missing, not as real values.",
                ),
            )

        with st.expander("GMM / LDA"):
            dx_options = ["(none)"] + var_options
            _seed_widget("cfg_dx", session_data.get_setting("DX"), dx_options)
            dx_choice = st.selectbox(
                "Diagnostic / outcome column", dx_options, key="cfg_dx"
            )
            session_data.set_setting("DX", "" if dx_choice == "(none)" else dx_choice)

            k_min, k_max = st.slider("K range", 1, 20, (
                int(session_data.get_setting("K_MIN")),
                int(session_data.get_setting("K_MAX")),
            ))
            session_data.set_setting("K_MIN", k_min)
            session_data.set_setting("K_MAX", k_max)
            session_data.set_setting(
                "N_INIT", st.number_input("n_init", 1, 100, int(session_data.get_setting("N_INIT")))
            )
            session_data.set_setting(
                "RAND_STATE",
                st.number_input("Random seed", 0, 9999, int(session_data.get_setting("RAND_STATE"))),
            )
            session_data.set_setting(
                "DPI", st.number_input("Figure DPI", 72, 600, int(session_data.get_setting("DPI")))
            )
    else:
        st.info("Upload a dataset to enable variable selection.")


# --- Selection guards ---
def _require(condition: bool, message: str) -> bool:
    if not condition:
        st.warning(message)
        return False
    return True


def gen_min(n=2):
    return _require(len(selected) >= n, f"Select at least {n} general variable(s).")


def endo_one():
    return _require(len(endo_selected) == 1, "Select exactly one dependent variable.")


def exo_min(n=1):
    return _require(len(exo_selected) >= n, f"Select at least {n} independent variable(s).")


def exo_one():
    return _require(len(exo_selected) == 1, "Select exactly one independent variable.")


def mod_one():
    return _require(len(mod_selected) == 1, "Select exactly one moderator variable.")


def _require_id() -> bool:
    return _require(
        bool(session_data.get_setting("ID_VAR")),
        "Set a Subject ID column in the sidebar (Settings → Data cleaning) first.",
    )


def _id() -> str:
    id_var = session_data.get_setting("ID_VAR")
    if not id_var:
        raise ValueError(
            "No Subject ID column is set. Choose one in the sidebar under "
            "Settings -> Data cleaning."
        )
    return id_var


def _blocked() -> bool:
    if not data_available:
        st.warning("Upload a dataset first.")
        return True
    return False


# --- Statistical analyses ---
st.divider()
st.subheader("Statistical Analyses")

col_var_expl, col_rma_icc, col_chi = st.columns(3)

with col_var_expl:
    if st.button("Uni/Multivariate Exploration", use_container_width=True):
        if not _blocked() and gen_min(1):
            run_and_render(
                "Uni/Multivariate Exploration",
                multivar_desc_gen.explore_multi_variables,
                ALL_OUTPUT_FOLDERS,
                *selected,
            )

with col_rma_icc:
    if st.button("RM-ANOVA & ICC", use_container_width=True):
        if not _blocked() and gen_min(2):
            if _require_id():
                run_and_render(
                    "RM-ANOVA & ICC",
                    rm_anova_icc.rm_anova_icc,
                    ALL_OUTPUT_FOLDERS,
                    *selected,
                    id_var=_id(),
                )

with col_chi:
    if st.button("Chi-Square Test", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_min(1):
            run_and_render(
                "Chi-Square Test",
                chi_sqr.run_chi_sqr,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                exo=exo_selected,
                id_var=_id(),
            )

# --- ML analyses ---
st.divider()
st.subheader("ML Analyses")

col_gmm, col_lda, col_olr, col_nbr = st.columns(4)
col_blr, col_blr_mra, col_ols, col_ols_mra = st.columns(4)

with col_gmm:
    if st.button("GMM Analysis", use_container_width=True):
        if not _blocked() and gen_min(2):
            if not session_data.get_setting("DX"):
                st.warning("Set a diagnostic/outcome column in the sidebar first.")
            else:
                run_and_render(
                    "GMM Analysis",
                    gmm_analysis.gmm_analysis,
                    ALL_OUTPUT_FOLDERS,
                    *selected,
                    dx_col_=session_data.get_setting("DX"),
                    dpi_=int(session_data.get_setting("DPI")),
                    k_min=int(session_data.get_setting("K_MIN")),
                    k_max=int(session_data.get_setting("K_MAX")),
                    n_init=int(session_data.get_setting("N_INIT")),
                    rand_state=int(session_data.get_setting("RAND_STATE")),
                )

with col_lda:
    if st.button("LDA Analysis", use_container_width=True):
        if not _blocked() and gen_min(2):
            if not session_data.get_setting("DX"):
                st.warning("Set a diagnostic/outcome column in the sidebar first.")
            else:
                run_and_render(
                    "LDA Analysis",
                    lda.lda_model,
                    ALL_OUTPUT_FOLDERS,
                    *selected,
                    dx_col_=session_data.get_setting("DX"),
                    dpi_=int(session_data.get_setting("DPI")),
                )

with col_olr:
    if st.button("OLR", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_min(1):
            run_and_render(
                "Ordinal Logistic Regression",
                olr.run_olr,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                exo=exo_selected,
                id_var=_id(),
            )

with col_nbr:
    if st.button("NBR", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_min(1):
            run_and_render(
                "Negative Binomial Regression",
                nbr.run_nbr,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                exo=exo_selected,
                id_var=_id(),
            )

with col_blr:
    if st.button("BLR", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_min(1):
            run_and_render(
                "Binary Logistic Regression",
                blr.run_blr,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                exo=exo_selected,
                id_var=_id(),
            )

with col_blr_mra:
    if st.button("BLR MRA + SSA", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_one() and mod_one():
            run_and_render(
                "BLR Moderated Regression",
                mblr.run_mblr,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                focal_var=exo_selected,
                moderator_var=mod_selected,
                id_var=_id(),
            )

with col_ols:
    if st.button("OLSR", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_min(1):
            run_and_render(
                "OLS Regression",
                ols.run_ols,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                exo=exo_selected,
            )

with col_ols_mra:
    if st.button("OLS MRA + SSA", use_container_width=True):
        if not _blocked() and _require_id() and endo_one() and exo_one() and mod_one():
            run_and_render(
                "OLS Moderated Regression",
                mols.run_mols,
                ALL_OUTPUT_FOLDERS,
                endo=endo_selected,
                focal_var=exo_selected,
                moderator_var=mod_selected,
                id_var=_id(),
            )

# --- Visualizations ---
st.divider()
st.subheader("Data Visualizations")

col_des_vis, col_hm, col_pg, col_pp = st.columns(4)

for column, label, fn in (
    (col_des_vis, "Descriptive", desc_gen.desc_visualization),
    (col_hm, "Heatmap", hm_gen.heatmap_visualizations),
    (col_pg, "PairGrid", pg_gen.pairgrid_visualizations),
    (col_pp, "PairPlot", pp_gen.pair_plot_visualizations),
):
    with column:
        if st.button(label, use_container_width=True):
            if not _blocked() and gen_min(2):
                run_and_render(label, fn, ALL_OUTPUT_FOLDERS, *selected)

# --- Full-data generators ---
st.divider()
st.subheader("Full-Data Generators")

col_md_gen, col_av_gen = st.columns(2)

with col_md_gen:
    if st.button("Master Descriptive CSV", use_container_width=True):
        if not _blocked():
            run_and_render(
                "Master Descriptives",
                master_descriptive_gen.master_descriptive_csv_generator,
                ALL_OUTPUT_FOLDERS,
            )

with col_av_gen:
    if st.button("All Variable Descriptives", use_container_width=True):
        if not _blocked():
            run_and_render(
                "All Variable Descriptives",
                all_single_var_desc_gen.all_single_var_descriptive_generator,
                ALL_OUTPUT_FOLDERS,
            )