"""
Session-scoped store for the active dataset and runtime settings.

osted deployment, where a visitor has no way to place a file in `files/`.

This module holds the active DataFrame (and the settings that used to live as
module-level constants in `config.py`) in Streamlit's session state, so
`load_clean()` can serve uploaded data without any pipeline module changing.

It degrades gracefully: outside a Streamlit runtime (e.g. under pytest) it falls
back to a plain dict, so the existing test suite keeps working unchanged.

"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import streamlit as st

    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover - streamlit in requirements.txt
    _HAS_STREAMLIT = False

_FALLBACK_STORE: dict[str, Any] = {}

_DF_KEY = "_active_df"
_NAME_KEY = "_active_df_name"
_CFG_PREFIX = "_cfg_"

DEFAULT_SETTINGS: dict[str, Any] = {
    # Data cleaning
    "ID_VAR": "",
    "MISSING_CODES": [-99, -999, -9999],
    "IMPOSSIBLE_ZERO_VARS": [],
    # Moderated Regression
    "DATA_THRESHOLD": 1,
    # GMM
    "DX": "",
    "DPI": 300,
    "K_MIN": 1,
    "K_MAX": 11,
    "N_INIT": 10,
    "RAND_STATE": 0,
    "REG_COVAR": 1e-3,
    # LDA
    "CV": 5,
}


def _store() -> Any:
    """
    Return session state if inside a Streamlit run,
    else return a plain dict.
    """
    if _HAS_STREAMLIT:
        try:
            # Touching session_state outside a script run raises; fall back
            st.session_state  # noqa: B018 - silence quality insureance B018
            return st.session_state
        except Exception:
            return _FALLBACK_STORE
    return _FALLBACK_STORE


# --- Active Dataset ---


def set_active_df(df: pd.DataFrame, name: str = "uploaded.csv") -> None:
    store = _store()
    store[_DF_KEY] = df
    store[_NAME_KEY] = name


def get_active_df() -> pd.DataFrame | None:
    return _store().get(_NAME_KEY)


def get_active_name() -> pd.DataFrame | None:
    return _store().get(_NAME_KEY)


def has_active_df() -> bool:
    return get_active_df() is not None


def clear_active_df() -> None:
    store = _store()
    for key in (_DF_KEY, _NAME_KEY):
        try:
            del store[key]
        except (KeyError, AttributeError):
            pass


# --- Settings ---


def get_setting(key: str) -> Any:
    """Runtime value for a setting, falling back to the packaged default"""
    return _store().get(f"{_CFG_PREFIX}{key}", DEFAULT_SETTINGS.get(key))


def set_setting(key: str, value: Any) -> None:
    _store()[f"{_CFG_PREFIX}{key}"] = value
