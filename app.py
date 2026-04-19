import os
import streamlit as st
import pandas as pd
import numpy as np
from features import FEATURE_NAMES, FeatureEngineer

st.set_page_config(page_title="Feature Processor", page_icon="⚙️")
st.title("⚙️ Feature Processor")

if "input_data" not in st.session_state:
    if os.path.exists("sample_data.csv"):
        st.session_state.input_data = pd.read_csv("sample_data.csv")
    else:
        st.session_state.input_data = pd.DataFrame()

tab1, tab2 = st.tabs(["Data", "Feature Names"])

with tab1:
    st.header("Feature Data")
    if not st.session_state.input_data.empty:
        st.dataframe(st.session_state.input_data, use_container_width=True)
        st.success(f"{len(st.session_state.input_data)} rows × {len(st.session_state.input_data.columns)} columns")
    else:
        st.info("No data loaded")

with tab2:
    st.header("Features")
    cols = st.columns(3)
    for i, name in enumerate(FEATURE_NAMES):
        cols[i % 3].write(f"• {name}")