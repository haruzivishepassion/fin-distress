import os
import pickle
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from features import FEATURE_NAMES

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Financial Distress AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 Financial Distress AI Agent")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
FEATURE_COLS = [f"X{i}" for i in range(1, 19)] + ["year"]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model_bundle = load_model()
MODEL_AVAILABLE = model_bundle is not None

for key, default in [("chat_history", []), ("data", None), ("company_results", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

def predict_with_model(feature_row: pd.DataFrame) -> float:
    mdl = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feats = model_bundle["features"]
    row = feature_row[feats].copy()
    row_scaled = scaler.transform(row)
    return float(mdl.predict_proba(row_scaled)[0, 1])

def map_columns_to_features(df: pd.DataFrame):
    col_lower = {c.lower(): c for c in df.columns}
    if all(f"x{i}" in col_lower for i in range(1, 19)):
        mapped = pd.DataFrame(index=df.index)
        for i in range(1, 19):
            mapped[f"X{i}"] = pd.to_numeric(df[col_lower[f"x{i}"]], errors="coerce").fillna(0)
        mapped["year"] = pd.to_numeric(df[col_lower["year"]], errors="coerce").fillna(1) if "year" in col_lower else 1
        return mapped
    MAPPING = {
        "X1":["net_profit_ratio","net_profit","npm","profit_ratio","net_margin"],
        "X2":["eps","earnings_per_share","persistent_eps"],
        "X3":["roa","return_on_assets","return on assets"],
        "X4":["operating_margin","operating_profit_rate","ebit_margin"],
        "X5":["roe","return_on_equity","return on equity"],
        "X6":["cash_flow","cash_flow_rate","operating_cash"],
        "X7":["debt_ratio","total_debt_ratio","leverage","debt_to_assets"],
        "X8":["net_income_equity","net_income_to_equity"],
        "X9":["liability_to_equity","debt_equity","debt_to_equity"],
        "X10":["working_capital","working_capital_ratio","wc_to_assets"],
        "X11":["current_ratio","liquidity_ratio"],
        "X12":["quick_ratio","acid_test","acid_ratio"],
        "X13":["total_debt","total_liabilities_ratio"],
        "X14":["cash_turnover","asset_turnover"],
        "X15":["ar_turnover","accounts_receivable_turnover","receivable_turnover"],
        "X16":["inventory_turnover","stock_turnover"],
        "X17":["op_profit_capital","operating_profit_capital"],
        "X18":["net_profit_assets","net_profit_to_assets","profitability"],
    }
    mapped = pd.DataFrame(index=df.index)
    matched_any = False
    for feat, synonyms in MAPPING.items():
        for syn in synonyms:
            if syn in col_lower:
                mapped[feat] = pd.to_numeric(df[col_lower[syn]], errors="coerce").fillna(0)
                matched_any = True
                break
        else:
            mapped[feat] = 0.0
    if not matched_any:
        return None
    mapped["year"] = pd.to_numeric(df[col_lower["year"]], errors="coerce").fillna(1) if "year" in col_lower else 1
    return mapped

def _fallback_heuristic(company_data):
    risk_score = 50.0
    if "year" in company_data.columns and len(company_data) > 1:
        recent = company_data[company_data["year"] == company_data["year"].max()].select_dtypes(include=[np.number]).mean()
        past = company_data[company_data["year"] == company_data["year"].min()].select_dtypes(include=[np.number]).mean()
        if len(recent) > 0:
            decline_frac = float(((recent - past) < 0).mean())
            risk_score += (decline_frac - 0.5) * 60
    missing_frac = company_data.select_dtypes(include=[np.number]).isna().mean().mean()
    if missing_frac > 0.2:
        risk_score += 10
    return float(np.clip(risk_score, 0, 100)), "Heuristic"

def analyze_companies(df):
    pc = [c for c in df.columns if any(x in c.lower() for x in ["company","firm","name","id","entity","org"])]
    if not pc:
        return None
    company_col = pc[0]
    results = []
    for company in df[company_col].unique():
        company_data = df[df[company_col] == company].copy()
        feature_df = map_columns_to_features(company_data)
        if MODEL_AVAILABLE and feature_df is not None:
            avg = feature_df[FEATURE_COLS].mean(axis=0).to_frame().T
            prob = predict_with_model(avg)
            distress_pct = round(prob * 100, 1)
            method = "ML Model"
        else:
            distress_pct, method = _fallback_heuristic(company_data)
        status = "Distressed" if distress_pct >= 60 else "At Risk" if distress_pct >= 35 else "Healthy"
        results.append({"Company": company, "Distress_Probability": distress_pct, "Status": status, "Method": method})
    return pd.DataFrame(results)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
if MODEL_AVAILABLE:
    st.sidebar.success("✅ ML Model loaded\n`GradientBoostingClassifier`\nAccuracy ≈ **85%** | AUC ≈ **0.68**")
else:
    st.sidebar.warning("⚠️ model.pkl not found. Using fallback heuristic.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Expected CSV format:**\n- Columns: `company`, `year`, `X1`–`X18`\n- Or common financial ratio names")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Panel","💬 Chatbot","📈 Visual Analysis","📋 Summary & Decisions"])

# ── Tab 1 ────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Upload Financial Data")
    col_upload1, col_upload2 = st.columns([3, 1])
    with col_upload1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    with col_upload2:
        st.write(""); st.write("")
        if st.button("Load Sample Data"):
            if os.path.exists("sample_financial_data.csv"):
                st.session_state.data = pd.read_csv("sample_financial_data.csv")
                st.session_state.company_results = None
                st.success(f"Loaded sample data: {len(st.session_state.data)} rows")
            else:
                st.error("Sample file not found")
        if st.button("Load & Analyze", type="primary"):
            if st.session_state.data is not None:
                with st.spinner("Running ML analysis…"):
                    results = analyze_companies(st.session_state.data)
                if results is not None:
                    st.session_state.company_results = results
                    st.success(f"Analysis complete! {len(results)} companies analysed.")
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
        pc = [c for c in df.columns if any(x in c.lower() for x in ["company","firm","name","id","entity","org"])]
        if pc:
            st.subheader("Detected Companies")
            company_col = st.selectbox("Select Company Column", pc)
            companies = df[company_col].unique()
            st.write(f"**{len(companies)} companies detected:** {', '.join(map(str, companies[:10]))}{'…' if len(companies)>10 else ''}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df)); col2.metric("Columns", len(df.columns)); col3.metric("Missing Values", int(df.isna().sum().sum()))
        fdf = map_columns_to_features(df)
        if fdf is not None and MODEL_AVAILABLE:
            st.success("✅ Columns mapped to X1–X18 feature space — ML model will be used")
        elif not MODEL_AVAILABLE:
            st.warning("⚠️ No ML model available — using fallback heuristic")
        else:
            st.info("ℹ️ Columns not matched to X1–X18. Ensure your CSV has X1–X18 column names or standard ratio names.")

# ── Tab 2 ────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("AI Chatbot")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    user_input = st.chat_input("Ask about companies, statistics, or predictions…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ul = user_input.lower()
        if st.session_state.company_results is not None:
            results = st.session_state.company_results
            if any(x in ul for x in ["distress","healthy","risk","status","which"]):
                response = "## Company Financial Status\n\n"
                for _, r in results.iterrows():
                    e = "🚨" if r["Status"]=="Distressed" else "⚠️" if r["Status"]=="At Risk" else "✅"
                    response += f"- **{r['Company']}**: {e} {r['Status']} — {r['Distress_Probability']}% risk *(via {r['Method']})*\n"
            elif any(x in ul for x in ["statistics","stats","summary","average","mean"]):
                response = "## Summary Statistics by Company\n\n"
                if st.session_state.data is not None:
                    df = st.session_state.data
                    pc = [c for c in df.columns if any(x in c.lower() for x in ["company","firm","name","id"])]
                    if pc:
                        for co in df[pc[0]].unique()[:5]:
                            cd = df[df[pc[0]]==co].select_dtypes(include=[np.number])
                            response += f"### {co}\n- Records: {len(df[df[pc[0]]==co])}\n- Mean: {cd.mean().mean():.2f}\n- Std: {cd.std().mean():.2f}\n\n"
            elif any(x in ul for x in ["model","accuracy","ml","algorithm"]):
                response = f"## ML Model Information\n- **Algorithm**: Gradient Boosting Classifier\n- **Accuracy**: ~85%\n- **AUC-ROC**: ~0.68\n- **Balancing**: SMOTE\n- **Features**: X1–X18 + year\n- **Available**: {'Yes ✅' if MODEL_AVAILABLE else 'No ⚠️'}"
            else:
                response = "I can answer about:\n- Company status (distressed/at risk/healthy)\n- Summary statistics\n- ML model accuracy\n\nTry: *'Which companies are at risk?'*"
        else:
            response = "Please upload data and click **Load & Analyze** first."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    if st.button("Clear Chat"):
        st.session_state.chat_history = []; st.rerun()

# ── Tab 3 ────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Visual Analysis")
    if st.button("Run Visual Analysis", type="primary"):
        if st.session_state.data is not None:
            with st.spinner("Analysing…"):
                res = analyze_companies(st.session_state.data)
            if res is not None:
                st.session_state.company_results = res; st.success("Analysis complete!")
            else:
                st.warning("No company column detected")
        else:
            st.info("Upload data first")
    if st.session_state.data is not None:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pc = [c for c in df.columns if any(x in c.lower() for x in ["company","firm","name","id"])]
        viz_opts = ["All Companies Status","Company Comparison","Correlation Heatmap","Distribution","Time Series"]
        if MODEL_AVAILABLE:
            viz_opts.append("Feature Importance")
        viz_type = st.selectbox("Select Visualisation", viz_opts)
        plt.close("all")
        if viz_type == "All Companies Status":
            if st.session_state.company_results is not None:
                r = st.session_state.company_results
                fig, ax = plt.subplots(figsize=(10, max(4, len(r)*0.6)))
                colors = ["#e74c3c" if s=="Distressed" else "#f39c12" if s=="At Risk" else "#2ecc71" for s in r["Status"]]
                ax.barh(r["Company"], r["Distress_Probability"], color=colors)
                ax.set_xlabel("Distress Probability (%)"); ax.set_xlim(0,100)
                ax.axvline(60, color="red", linestyle="--", alpha=0.7, label="Distressed (60%)")
                ax.axvline(35, color="orange", linestyle="--", alpha=0.7, label="At Risk (35%)")
                ax.legend(); st.pyplot(fig)
            else:
                st.warning("Run analysis first")
        elif viz_type == "Feature Importance" and MODEL_AVAILABLE:
            imp = pd.Series(model_bundle["model"].feature_importances_, index=model_bundle["features"]).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#2ecc71" if v<0.05 else "#f39c12" if v<0.15 else "#e74c3c" for v in imp]
            ax.barh(imp.index, imp.values, color=colors)
            ax.set_xlabel("Feature Importance"); ax.set_title("Feature Importance (Gradient Boosting)"); st.pyplot(fig)
        elif viz_type == "Company Comparison" and pc:
            fig, ax = plt.subplots(figsize=(12,5))
            df.groupby(pc[0])[list(numeric_cols[:5])].mean().plot(kind="bar", ax=ax)
            ax.set_title(f"Metrics by {pc[0]}"); plt.xticks(rotation=0); st.pyplot(fig)
        elif viz_type == "Correlation Heatmap" and len(numeric_cols)>1:
            fig, ax = plt.subplots(figsize=(12,10))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
            st.pyplot(fig)
        elif viz_type == "Distribution" and len(numeric_cols)>0:
            col = st.selectbox("Select Column", list(numeric_cols))
            fig, ax = plt.subplots(figsize=(10,4))
            sns.histplot(data=df, x=col, hue=pc[0] if pc else None, kde=True, ax=ax)
            st.pyplot(fig)
        elif viz_type == "Time Series" and "year" in df.columns:
            metric = st.selectbox("Select Metric", list(numeric_cols))
            fig, ax = plt.subplots(figsize=(10,5))
            if pc:
                for co in df[pc[0]].unique():
                    cdf = df[df[pc[0]]==co]; ax.plot(cdf["year"], cdf[metric], marker="o", label=co)
                ax.legend()
            else:
                ax.plot(df["year"], df[metric], marker="o")
            ax.set_xlabel("Year"); ax.set_ylabel(metric); st.pyplot(fig)
    else:
        st.info("Upload data to enable visualisations")

# ── Tab 4 ────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Summary, Recommendations & Decisions")
    if st.button("Generate Analysis"):
        if st.session_state.data is not None:
            with st.spinner("Running ML model…"):
                res = analyze_companies(st.session_state.data)
            if res is not None:
                st.session_state.company_results = res
            else:
                st.warning("No company column detected")
        else:
            st.warning("Please upload data first")
    if st.session_state.company_results is not None:
        results = st.session_state.company_results
        st.subheader("🎯 Financial Health Verdict by Company")
        for _, row in results.iterrows():
            with st.expander(f"{row['Company']} — {row['Status']}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Distress Probability", f"{row['Distress_Probability']}%")
                    st.caption(f"Prediction method: {row['Method']}")
                with c2:
                    if row["Status"]=="Distressed": st.error("🚨 FINANCIALLY DISTRESSED")
                    elif row["Status"]=="At Risk": st.warning("⚠️ FINANCIALLY AT RISK")
                    else: st.success("✅ FINANCIALLY HEALTHY")
        st.subheader("📊 Overall Summary")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Companies", len(results))
        c2.metric("Healthy", len(results[results["Status"]=="Healthy"]))
        c3.metric("At Risk", len(results[results["Status"]=="At Risk"]))
        c4.metric("Distressed", len(results[results["Status"]=="Distressed"]))
        fig, ax = plt.subplots(figsize=(10, max(3, len(results)*0.5)))
        colors = ["#e74c3c" if s=="Distressed" else "#f39c12" if s=="At Risk" else "#2ecc71" for s in results["Status"]]
        ax.barh(results["Company"], results["Distress_Probability"], color=colors)
        ax.set_xlim(0,100); ax.set_xlabel("Distress Probability (%)")
        ax.axvline(60, color="red", linestyle="--", label="Distressed"); ax.axvline(35, color="orange", linestyle="--", label="At Risk")
        ax.legend(); st.pyplot(fig)
        st.subheader("💡 Recommendations")
        c1,c2,c3 = st.columns(3)
        with c1: st.error("**For Distressed**\n\n• Immediate financial audit\n• Negotiate creditor terms\n• Reduce costs\n• Seek advisor")
        with c2: st.warning("**For At Risk**\n\n• Monthly monitoring\n• Improve cash flow\n• Diversify revenue\n• Build reserves")
        with c3: st.success("**For Healthy**\n\n• Maintain practices\n• Consider growth\n• Monitor trends\n• Explore expansion")
        csv = results.to_csv(index=False)
        st.download_button("📥 Export Report as CSV", csv, "financial_distress_report.csv", "text/csv")
    else:
        st.info("Upload data and click **Generate Analysis**.")
