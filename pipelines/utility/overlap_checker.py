import streamlit as st

def check_for_var_overlap(endo,exo,mod):
    overlap = set(endo) & set(exo) & set(mod)
    if overlap:
        st.warning(f'Check endogenous, exogenous/focal, and moderator variables, there is an overlap: {overlap}')
        return False
    else:
        return True