import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from features import FEATURE_NAMES

st.set_page_config(page_title="Financial Distress AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 Financial Distress AI Agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "data" not in st.session_state:
    st.session_state.data = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Panel", "💬 Chatbot", "📈 Visual Analysis", "📋 Summary & Decisions"])

with tab1:
    st.header("Upload Financial Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.analysis_done = False
        st.success(f"Loaded {len(st.session_state.data)} rows")
    
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(st.session_state.data))
        with col2:
            st.metric("Columns", len(st.session_state.data.columns))
        with col3:
            st.metric("Missing Values", st.session_state.data.isna().sum().sum())

with tab2:
    st.header("AI Chatbot")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    user_input = st.chat_input("Ask about financial distress predictions...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        data_info = ""
        if st.session_state.data is not None:
            data_info = f"\n\nCurrent data has {len(st.session_state.data)} rows."
        
        response = f"I understand: '{user_input}'.{data_info}\n\nI can help with:\n- Analyzing financial data\n- Predicting financial distress\n- Generate summary & recommendations\n- Visualize trends\n\nPlease upload data in the Data Panel first."
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with tab3:
    st.header("Visual Analysis")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        viz_type = st.selectbox("Select Visualization", 
            ["Correlation Heatmap", "Distribution", "Time Series", "Pair Plot"])
        
        if viz_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        
        elif viz_type == "Distribution":
            col = st.selectbox("Select Column", df.columns)
            fig, ax = plt.subplots(figsize=(10, 4))
            if df[col].dtype in [np.number]:
                sns.histplot(df[col], kde=True, ax=ax)
            else:
                df[col].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)
        
        elif viz_type == "Time Series":
            if "year" in df.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                for col in df.select_dtypes(include=[np.number]).columns[:5]:
                    df.groupby("year")[col].mean().plot(ax=ax, label=col)
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("'year' column not found")
        
        elif viz_type == "Pair Plot":
            cols = st.multiselect("Select Columns", df.select_dtypes(include=[np.number]).columns, 
                default=list(df.select_dtypes(include=[np.number]).columns[:4]))
            if cols:
                fig = sns.pairplot(df[cols])
                st.pyplot(fig)
    else:
        st.info("Upload data in Data Panel to enable visualizations")

with tab4:
    st.header("Summary, Recommendations & Decisions")
    
    if st.button("Generate Analysis"):
        if st.session_state.data is not None:
            st.session_state.analysis_done = True
        else:
            st.warning("Please upload data first")
    
    if st.session_state.analysis_done and st.session_state.data is not None:
        df = st.session_state.data
        numeric_df = df.select_dtypes(include=[np.number])
        
        st.subheader("📊 Data Summary")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Features:** {len(df.columns)}")
        st.write(f"**Missing Values:** {df.isna().sum().sum()}")
        
        if len(numeric_df.columns) > 0:
            st.write("**Key Statistics:**")
            st.dataframe(numeric_df.describe(), use_container_width=True)
            
            corr_matrix = numeric_df.corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        st.subheader("💡 Recommendations")
        
        cols = st.columns(3)
        with cols[0]:
            st.info("**Immediate Actions**")
            st.write("• Review high-risk indicators")
            st.write("• Verify data completeness")
        with cols[1]:
            st.warning("**Short-term (1-3 months)**")
            st.write("• Monitor cash flow trends")
            st.write("• Reduce operational costs")
        with cols[2]:
            st.success("**Long-term (6-12 months)**")
            st.write("• Diversify revenue streams")
            st.write("• Build financial reserves")
        
        st.subheader("🎯 Key Decisions")
        
        decision_list = [
            ("Review", "Review all financial projections for accuracy", "High"),
            ("Monitor", "Monitor key distress indicators monthly", "High"),
            ("Reduce", "Reduce non-essential expenditures", "Medium"),
            ("Diversify", "Explore new revenue sources", "Medium"),
            ("Plan", "Develop contingency financing plan", "Low")
        ]
        
        for decision, desc, priority in decision_list:
            with st.expander(f"{decision} - {priority} Priority"):
                st.write(desc)
        
        if st.button("Export Report"):
            st.success("Report exported (functionality placeholder)")
    else:
        st.info("Upload data and click 'Generate Analysis'")