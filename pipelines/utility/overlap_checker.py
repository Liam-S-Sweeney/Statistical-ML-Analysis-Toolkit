import streamlit as st

def check_for_var_overlap(endo,exo):
    overlap = set(endo) & set(exo)
    if overlap:
        st.warning(f'Check endogenous and exogenous variables, there is an overlap: {overlap}')
        return False
    else:
        return True