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
if "company_results" not in st.session_state:
    st.session_state.company_results = None

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Panel", "💬 Chatbot", "📈 Visual Analysis", "📋 Summary & Decisions"])

with tab1:
    st.header("Upload Financial Data")
    col_upload1, col_upload2 = st.columns([3, 1])
    with col_upload1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    with col_upload2:
        st.write("")
        st.write("")
        if st.button("Load Sample Data"):
            if os.path.exists("sample_financial_data.csv"):
                st.session_state.data = pd.read_csv("sample_financial_data.csv")
                st.session_state.company_results = None
                st.success(f"Loaded sample data: {len(st.session_state.data)} rows")
            else:
                st.error("Sample file not found")
        
        if st.button("Run Analysis", type="primary"):
            if st.session_state.data is not None:
                st.session_state.data = pd.read_csv("sample_financial_data.csv") if not uploaded_file else st.session_state.data
                df = st.session_state.data
                numeric_df = df.select_dtypes(include=[np.number])
                potential_company_cols = [c for c in df.columns if any(x in c.lower() for x in ["company", "firm", "name", "id"])]
                
                if potential_company_cols:
                    company_col = potential_company_cols[0]
                    companies = df[company_col].unique()
                    
                    results = []
                    for company in companies:
                        company_data = df[df[company_col] == company]
                        company_numeric = company_data.select_dtypes(include=[np.number])
                        
                        risk_score = 0
                        
                        if company_numeric.isna().sum().sum() / max(1, company_numeric.size) > 0.1:
                            risk_score += 30
                        
                        if len(company_numeric.columns) > 1:
                            corr = company_numeric.corr()
                            negative_corr = (corr < -0.5).sum().sum()
                            if negative_corr > 3:
                                risk_score += 25
                        
                        if len(company_numeric) > 0:
                            low_values = (company_numeric < company_numeric.quantile(0.1).values).sum().sum()
                            if low_values > max(1, len(company_numeric)) * 0.1:
                                risk_score += 20
                        
                        if "year" in df.columns:
                            recent = company_data[company_data["year"] == company_data["year"].max()].select_dtypes(include=[np.number]).mean()
                            past = company_data[company_data["year"] == company_data["year"].min()].select_dtypes(include=[np.number]).mean()
                            decline = ((recent - past) < 0).sum()
                            if decline > max(1, len(recent)) * 0.5:
                                risk_score += 25
                        
                        distress_prob = min(100, risk_score)
                        
                        if distress_prob >= 60:
                            status = "Distressed"
                        elif distress_prob >= 40:
                            status = "At Risk"
                        else:
                            status = "Healthy"
                        
                        results.append({
                            "Company": company,
                            "Distress_Probability": distress_prob,
                            "Status": status
                        })
                    
                    st.session_state.company_results = pd.DataFrame(results)
                    st.success("Analysis complete!")
                else:
                    st.warning("No company column detected")
            else:
                st.warning("Please load data first")
    
    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.company_results = None
        st.success(f"Loaded {len(st.session_state.data)} rows")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        potential_company_cols = [c for c in df.columns if any(x in c.lower() for x in ["company", "firm", "name", "id", "entity", "org"])]
        if potential_company_cols:
            st.subheader("Detected Companies")
            company_col = st.selectbox("Select Company Column", potential_company_cols)
            companies = df[company_col].unique()
            st.write(f"**{len(companies)} companies detected:** {', '.join(map(str, companies[:10]))}{'...' if len(companies) > 10 else ''}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())

with tab2:
    st.header("AI Chatbot")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    user_input = st.chat_input("Ask about companies, statistics, or predictions...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        response = ""
        
        if st.session_state.company_results is not None:
            results = st.session_state.company_results
            user_lower = user_lower = user_input.lower()
            
            if any(x in user_lower for x in ["distress", "healthy", "risk", "status"]):
                response = "## Company Financial Status\n\n"
                for _, row in results.iterrows():
                    emoji = "🚨" if row["Status"] == "Distressed" else "⚠️" if row["Status"] == "At Risk" else "✅"
                    response += f"- **{row['Company']}**: {emoji} {row['Status']} ({row['Distress_Probability']}%)\n"
            
            elif any(x in user_lower for x in ["statistics", "stats", "summary", "data", "average", "mean"]):
                response = "## Summary Statistics by Company\n\n"
                if st.session_state.data is not None:
                    df = st.session_state.data
                    potential_company_cols = [c for c in df.columns if any(x in c.lower() for x in ["company", "firm", "name", "id"])]
                    if potential_company_cols:
                        company_col = potential_company_cols[0]
                        for company in df[company_col].unique()[:5]:
                            company_data = df[df[company_col] == company]
                            numeric_data = company_data.select_dtypes(include=[np.number])
                            if len(numeric_data.columns) > 0:
                                stats = numeric_data.describe().loc[["mean", "std", "min", "max"]]
                                response += f"### {company}\n"
                                response += f"- Records: {len(company_data)}\n"
                                response += f"- Mean values: {numeric_data.mean().mean():.2f}\n"
                                response += f"- Std deviation: {numeric_data.std().mean():.2f}\n\n"
            
            else:
                response = f"I can answer about:\n- Company financial status (distressed/healthy)\n- Summary statistics per company\n- Risk analysis\n\nTry asking: 'What is the status of each company?' or 'Show statistics per company'"
        else:
            response = "Please upload data and generate analysis in the Summary tab first. Then I can answer questions about your companies."
        
        if not response:
            response = f"I understand: '{user_input}'.\n\nUpload data and generate analysis to enable company-specific questions."
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with tab3:
    st.header("Visual Analysis")
    
    if st.button("Run Analysis", type="primary"):
        if st.session_state.data is not None:
            df = st.session_state.data
            numeric_df = df.select_dtypes(include=[np.number])
            potential_company_cols = [c for c in df.columns if any(x in c.lower() for x in ["company", "firm", "name", "id"])]
            
            if potential_company_cols:
                company_col = potential_company_cols[0]
                companies = df[company_col].unique()
                
                results = []
                for company in companies:
                    company_data = df[df[company_col] == company]
                    company_numeric = company_data.select_dtypes(include=[np.number])
                    
                    risk_score = 0
                    
                    if company_numeric.isna().sum().sum() / max(1, company_numeric.size) > 0.1:
                        risk_score += 30
                    
                    if len(company_numeric.columns) > 1:
                        corr = company_numeric.corr()
                        negative_corr = (corr < -0.5).sum().sum()
                        if negative_corr > 3:
                            risk_score += 25
                    
                    if len(company_numeric) > 0:
                        low_values = (company_numeric < company_numeric.quantile(0.1).values).sum().sum()
                        if low_values > max(1, len(company_numeric)) * 0.1:
                            risk_score += 20
                    
                    if "year" in df.columns:
                        recent = company_data[company_data["year"] == company_data["year"].max()].select_dtypes(include=[np.number]).mean()
                        past = company_data[company_data["year"] == company_data["year"].min()].select_dtypes(include=[np.number]).mean()
                        decline = ((recent - past) < 0).sum()
                        if decline > max(1, len(recent)) * 0.5:
                            risk_score += 25
                    
                    distress_prob = min(100, risk_score)
                    
                    if distress_prob >= 60:
                        status = "Distressed"
                    elif distress_prob >= 40:
                        status = "At Risk"
                    else:
                        status = "Healthy"
                    
                    results.append({
                        "Company": company,
                        "Distress_Probability": distress_prob,
                        "Status": status
                    })
                
                st.session_state.company_results = pd.DataFrame(results)
                st.success("Analysis complete!")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        potential_company_cols = [c for c in df.columns if any(x in c.lower() for x in ["company", "firm", "name", "id"])]
        
        viz_type = st.selectbox("Select Visualization", 
            ["All Companies Status", "Company Comparison", "Correlation Heatmap", "Distribution", "Time Series"])
        
        plt.close('all')
        
        if viz_type == "All Companies Status":
            if st.session_state.company_results is not None:
                results = st.session_state.company_results
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ["#e74c3c" if s == "Distressed" else "#f39c12" if s == "At Risk" else "#2ecc71" for s in results["Status"]]
                ax.barh(results["Company"], results["Distress_Probability"], color=colors)
                ax.set_xlabel("Distress Probability (%)")
                ax.set_xlim(0, 100)
                ax.axvline(x=60, color="red", linestyle="--", alpha=0.7, label="Distressed")
                ax.axvline(x=40, color="orange", linestyle="--", alpha=0.7, label="At Risk")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Run analysis first to see company status")
        
        elif viz_type == "Company Comparison":
            if potential_company_cols:
                company_col = potential_company_cols[0]
                if len(numeric_cols) > 0:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    comp_means = df.groupby(company_col)[numeric_cols[:5]].mean()
                    comp_means.plot(kind="bar", ax=ax)
                    ax.set_title(f"Financial Metrics by {company_col}")
                    ax.legend(loc='best', fontsize=8)
                    plt.xticks(rotation=0)
                    st.pyplot(fig)
        
        elif viz_type == "Correlation Heatmap":
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
                ax.set_title("Feature Correlation Heatmap")
                st.pyplot(fig)
            else:
                st.warning("Need more numeric columns")
        
        elif viz_type == "Distribution":
            if len(numeric_cols) > 0:
                col = st.selectbox("Select Column", list(numeric_cols))
                fig, ax = plt.subplots(figsize=(10, 4))
                for company in potential_company_cols:
                    company_data = df
                sns.histplot(data=df, x=col, hue=potential_company_cols[0] if potential_company_cols else None, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)
        
        elif viz_type == "Time Series":
            if "year" in df.columns and len(numeric_cols) > 0:
                company_col = potential_company_cols[0] if potential_company_cols else None
                metric = st.selectbox("Select Metric", list(numeric_cols))
                fig, ax = plt.subplots(figsize=(10, 5))
                if company_col:
                    for company in df[company_col].unique():
                        company_df = df[df[company_col] == company]
                        ax.plot(company_df["year"], company_df[metric], marker="o", label=company)
                    ax.legend()
                else:
                    ax.plot(df["year"], df[metric], marker="o")
                ax.set_xlabel("Year")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} Over Time")
                st.pyplot(fig)
            else:
                st.warning("Data needs 'year' column for time series")
    else:
        st.info("Upload data in Data Panel to enable visualizations")

with tab4:
    st.header("Summary, Recommendations & Decisions")
    
    if st.button("Generate Analysis"):
        if st.session_state.data is not None:
            df = st.session_state.data
            numeric_df = df.select_dtypes(include=[np.number])
            potential_company_cols = [c for c in df.columns if any(x in c.lower() for x in ["company", "firm", "name", "id"])]
            
            if potential_company_cols:
                company_col = potential_company_cols[0]
                companies = df[company_col].unique()
                
                results = []
                for company in companies:
                    company_data = df[df[company_col] == company]
                    company_numeric = company_data.select_dtypes(include=[np.number])
                    
                    risk_score = 0
                    
                    if company_numeric.isna().sum().sum() / max(1, company_numeric.size) > 0.1:
                        risk_score += 30
                    
                    if len(company_numeric.columns) > 1:
                        corr = company_numeric.corr()
                        negative_corr = (corr < -0.5).sum().sum()
                        if negative_corr > 3:
                            risk_score += 25
                    
                    low_values = (company_numeric < company_numeric.quantile(0.1).values).sum().sum()
                    if low_values > max(1, len(company_numeric)) * 0.1:
                        risk_score += 20
                    
                    if "year" in df.columns:
                        recent = company_data[company_data["year"] == company_data["year"].max()].select_dtypes(include=[np.number]).mean()
                        past = company_data[company_data["year"] == company_data["year"].min()].select_dtypes(include=[np.number]).mean()
                        decline = ((recent - past) < 0).sum()
                        if decline > max(1, len(recent)) * 0.5:
                            risk_score += 25
                    
                    distress_prob = min(100, risk_score)
                    
                    if distress_prob >= 60:
                        status = "Distressed"
                    elif distress_prob >= 40:
                        status = "At Risk"
                    else:
                        status = "Healthy"
                    
                    results.append({
                        "Company": company,
                        "Distress_Probability": distress_prob,
                        "Status": status
                    })
                
                st.session_state.company_results = pd.DataFrame(results)
            else:
                st.warning("No company column detected. Please ensure your data has a company name/ID column.")
        else:
            st.warning("Please upload data first")
    
    if st.session_state.company_results is not None:
        results = st.session_state.company_results
        
        st.subheader("🎯 Financial Health Verdict by Company")
        
        for _, row in results.iterrows():
            with st.expander(f"{row['Company']} - {row['Status']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Distress Probability", f"{row['Distress_Probability']}%")
                with col2:
                    if row["Status"] == "Distressed":
                        st.error("🚨 FINANCIALLY DISTRESSED")
                    elif row["Status"] == "At Risk":
                        st.warning("⚠️ FINANCIALLY AT RISK")
                    else:
                        st.success("✅ FINANCIALLY HEALTHY")
        
        st.subheader("📊 Overall Summary")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Total Companies", len(results))
        with col_s2:
            healthy_count = len(results[results["Status"] == "Healthy"])
            st.metric("Healthy", healthy_count)
        with col_s3:
            at_risk_count = len(results[results["Status"] == "At Risk"])
            st.metric("At Risk", at_risk_count)
        with col_s4:
            distressed_count = len(results[results["Status"] == "Distressed"])
            st.metric("Distressed", distressed_count)
        
        fig_verdict, ax_verdict = plt.subplots(figsize=(10, 4))
        colors = ["#e74c3c" if s == "Distressed" else "#f39c12" if s == "At Risk" else "#2ecc71" for s in results["Status"]]
        ax_verdict.barh(results["Company"], results["Distress_Probability"], color=colors)
        ax_verdict.set_xlim(0, 100)
        ax_verdict.set_xlabel("Distress Probability (%)")
        ax_verdict.axvline(x=60, color="red", linestyle="--", label="Distressed Threshold")
        ax_verdict.axvline(x=40, color="orange", linestyle="--", label="At Risk Threshold")
        ax_verdict.legend()
        st.pyplot(fig_verdict)
        
        st.subheader("💡 Recommendations")
        
        cols = st.columns(3)
        with cols[0]:
            st.info("**For Distressed Companies**")
            st.write("• Immediate financial audit")
            st.write("• Negotiate creditor terms")
            st.write("• Reduce costs drastically")
            st.write("• Seek professional advisor")
        with cols[1]:
            st.warning("**For At-Risk Companies**")
            st.write("• Monthly monitoring")
            st.write("• Improve cash flow")
            st.write("• Diversify revenue")
            st.write("• Build reserves")
        with cols[2]:
            st.success("**For Healthy Companies**")
            st.write("• Maintain practices")
            st.write("• Consider growth")
            st.write("• Monitor trends")
            st.write("• Explore expansion")
        
        if st.button("Export Report"):
            st.success("Report exported (functionality placeholder)")
    else:
        st.info("Upload data and click 'Generate Analysis'. Ensure your data has a company name/ID column.")