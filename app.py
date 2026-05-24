import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0f1117; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #f0f0f0 !important;
}

.risk-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 12px 0;
}
.risk-high {
    background: linear-gradient(135deg, #ff4d4d22, #ff000011);
    border: 1.5px solid #ff4d4d88;
}
.risk-low {
    background: linear-gradient(135deg, #00c85322, #00ff6611);
    border: 1.5px solid #00c85388;
}
.metric-card {
    background: #1a1d2e;
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid #2a2d3e;
    text-align: center;
    margin: 6px 0;
}
.metric-label { font-size: 12px; color: #888; letter-spacing: 1px; text-transform: uppercase; }
.metric-value { font-size: 26px; font-weight: 600; color: #f0f0f0; margin-top: 4px; }
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px #4f46e555;
}
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7c3aed;
    margin: 20px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from src.pipeline.predict_pipeline import PredictPipeline
    return PredictPipeline()


@st.cache_data(show_spinner=False)
def load_model_report():
    from src.utils import load_object
    return load_object("artifacts/model_report.pkl")


def gauge_chart(probability: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": "#f0f0f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar": {"color": "#4f46e5"},
            "bgcolor": "#1a1d2e",
            "bordercolor": "#2a2d3e",
            "steps": [
                {"range": [0, 30],  "color": "#00c85322"},
                {"range": [30, 60], "color": "#f59e0b22"},
                {"range": [60, 100],"color": "#ef444422"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Default Probability", "font": {"color": "#888", "size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


def model_comparison_chart(report: dict):
    names   = list(report.keys())
    roc     = [report[m]["roc_auc"]   for m in names]
    acc     = [report[m]["accuracy"]  for m in names]
    f1      = [report[m]["f1_score"]  for m in names]

    fig = go.Figure()
    for metric, values, color in [
        ("ROC-AUC",  roc, "#4f46e5"),
        ("Accuracy", acc, "#06b6d4"),
        ("F1 Score", f1,  "#10b981"),
    ]:
        fig.add_trace(go.Bar(
            name=metric, x=names, y=values,
            marker_color=color, opacity=0.85,
        ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": "#2a2d3e"},
        yaxis={"gridcolor": "#2a2d3e", "range": [0, 1]},
        margin=dict(t=20, b=10, l=10, r=10),
        height=320,
    )
    return fig


def feature_importance_chart(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
    names, vals = zip(*pairs)
    short_names = [n.replace("NumberOfTime","#Time").replace("NumberOf","#") for n in names]

    fig = go.Figure(go.Bar(
        x=list(vals), y=list(short_names),
        orientation="h",
        marker=dict(
            color=list(vals),
            colorscale=[[0,"#4f46e5"],[1,"#7c3aed"]],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"},
        xaxis={"gridcolor": "#2a2d3e"},
        yaxis={"gridcolor": "#2a2d3e"},
        margin=dict(t=10, b=10, l=10, r=10),
        height=340,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Default Risk")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**About this App**")
    st.caption(
        "Predicts the probability of a borrower defaulting on a loan "
        "using the *Give Me Some Credit* dataset and ensemble ML models."
    )

    st.markdown("---")
    artifacts_ready = os.path.exists("artifacts/model.pkl")
    if artifacts_ready:
        st.success("✅ Model artifacts loaded")
    else:
        st.warning("⚠️ Run training first")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training… this may take a few minutes"):
                try:
                    from src.pipeline.train_pipeline import run_training_pipeline
                    best_name, best_info, _ = run_training_pipeline()
                    st.success(f"Done! Best: {best_name}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error: {ex}")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Loan Default Risk Predictor")
st.markdown(
    "Assess the likelihood a borrower will experience serious financial distress "
    "within the next two years."
)
st.markdown("---")

FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Single Prediction":

    if not os.path.exists("artifacts/model.pkl"):
        st.info("Please train the model first using the sidebar button.")
        st.stop()

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("### Borrower Information")

        st.markdown('<p class="section-header">Financial Behavior</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        revolving_util = c1.slider(
            "Revolving Utilization (%)",
            0.0, 1.0, 0.35, 0.01,
            help="Total balance on credit cards / credit limits"
        )
        debt_ratio = c2.slider(
            "Debt Ratio",
            0.0, 5.0, 0.35, 0.01,
            help="Monthly debt payments / gross monthly income"
        )

        st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        age          = c3.number_input("Age", 18, 100, 42)
        monthly_inc  = c4.number_input("Monthly Income ($)", 0, 100000, 5000, step=100)
        dependents   = st.slider("Number of Dependents", 0, 10, 1)

        st.markdown('<p class="section-header">Credit History</p>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        open_lines    = c5.number_input("Open Credit Lines", 0, 60, 8)
        real_estate   = c6.number_input("Real Estate Loans", 0, 20, 1)

        st.markdown('<p class="section-header">Past Due History</p>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        due_30_59  = c7.number_input("30–59 Days Late", 0, 20, 0)
        due_60_89  = c8.number_input("60–89 Days Late", 0, 20, 0)
        due_90     = c9.number_input("90+ Days Late",   0, 20, 0)

        predict_btn = st.button("⚡ Predict Default Risk")

    with col_result:
        st.markdown("### Risk Assessment")

        if predict_btn:
            try:
                from src.pipeline.predict_pipeline import PredictPipeline, CustomData

                custom_data = CustomData(
                    revolving_utilization=revolving_util,
                    age=age,
                    past_due_30_59=due_30_59,
                    debt_ratio=debt_ratio,
                    monthly_income=monthly_inc,
                    open_credit_lines=open_lines,
                    times_90_days_late=due_90,
                    real_estate_loans=real_estate,
                    past_due_60_89=due_60_89,
                    dependents=dependents,
                )
                df_input = custom_data.get_data_as_dataframe()
                pipeline = load_pipeline()
                pred, proba = pipeline.predict(df_input)
                prob = proba[0]

                st.plotly_chart(gauge_chart(prob), use_container_width=True)

                if pred[0] == 1:
                    st.markdown(
                        f'<div class="risk-card risk-high">'
                        f'<h2 style="color:#ff4d4d;margin:0">⚠️ HIGH RISK</h2>'
                        f'<p style="color:#ccc;margin:8px 0 0">Probability of default: <b>{prob*100:.1f}%</b></p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="risk-card risk-low">'
                        f'<h2 style="color:#00c853;margin:0">✅ LOW RISK</h2>'
                        f'<p style="color:#ccc;margin:8px 0 0">Probability of default: <b>{prob*100:.1f}%</b></p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Key metrics
                st.markdown("**Input Summary**")
                kc1, kc2, kc3 = st.columns(3)
                kc1.metric("Age", age)
                kc2.metric("Monthly Income", f"${monthly_inc:,}")
                kc3.metric("Debt Ratio", f"{debt_ratio:.2f}")
                kc4, kc5, kc6 = st.columns(3)
                kc4.metric("Revolving Util.", f"{revolving_util*100:.0f}%")
                kc5.metric("Times 90+ Late", due_90)
                kc6.metric("Dependents", dependents)

            except Exception as ex:
                st.error(f"Prediction error: {ex}")
        else:
            st.info("👈 Fill in the borrower details and click **Predict Default Risk**")

            # Show feature importance if model exists
            try:
                from src.utils import load_object
                model  = load_object("artifacts/model.pkl")
                fig_fi = feature_importance_chart(model, FEATURE_COLUMNS)
                if fig_fi:
                    st.markdown("**Feature Importances (Best Model)**")
                    st.plotly_chart(fig_fi, use_container_width=True)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":

    if not os.path.exists("artifacts/model.pkl"):
        st.info("Please train the model first using the sidebar button.")
        st.stop()

    st.markdown("### Batch Prediction via CSV Upload")
    st.caption(
        f"Upload a CSV with columns: `{'`, `'.join(FEATURE_COLUMNS)}`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
            st.markdown(f"**Rows uploaded:** {len(df_batch)}")
            st.dataframe(df_batch.head(10), use_container_width=True)

            if st.button("⚡ Run Batch Prediction"):
                pipeline = load_pipeline()
                preds, probas = pipeline.predict(df_batch[FEATURE_COLUMNS])
                df_batch["Prediction"]  = preds
                df_batch["Default_Prob"] = np.round(probas, 4)
                df_batch["Risk_Label"]   = df_batch["Prediction"].map(
                    {0: "✅ Low Risk", 1: "⚠️ High Risk"}
                )

                st.success("Batch prediction complete!")
                st.dataframe(df_batch, use_container_width=True)

                # Distribution chart
                fig_dist = px.histogram(
                    df_batch, x="Default_Prob", nbins=30,
                    color_discrete_sequence=["#4f46e5"],
                    title="Distribution of Default Probabilities",
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ccc"},
                    xaxis={"gridcolor": "#2a2d3e"},
                    yaxis={"gridcolor": "#2a2d3e"},
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                csv_out = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results",
                    data=csv_out,
                    file_name="loan_predictions.csv",
                    mime="text/csv",
                )
        except Exception as ex:
            st.error(f"Error: {ex}")
    else:
        # Show sample template
        sample = pd.DataFrame(
            [[0.35, 42, 0, 0.4, 5000, 8, 0, 1, 0, 1]],
            columns=FEATURE_COLUMNS,
        )
        st.markdown("**Sample CSV template:**")
        st.dataframe(sample, use_container_width=True)
        csv_template = sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV Template",
            data=csv_template,
            file_name="loan_template.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    if not os.path.exists("artifacts/model_report.pkl"):
        st.info("Please train the model first using the sidebar button.")
        st.stop()

    report = load_model_report()

    # Best model banner
    best_name = max(report, key=lambda k: report[k]["roc_auc"])
    best      = report[best_name]

    st.markdown(f"### 🏆 Best Model: **{best_name}**")
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val in [
        (m1, "ROC-AUC",   f"{best['roc_auc']:.4f}"),
        (m2, "Accuracy",  f"{best['accuracy']:.4f}"),
        (m3, "F1 Score",  f"{best['f1_score']:.4f}"),
        (m4, "Precision", f"{best['precision']:.4f}"),
        (m5, "Recall",    f"{best['recall']:.4f}"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### All Models Comparison")
        st.plotly_chart(model_comparison_chart(report), use_container_width=True)

    with col_right:
        st.markdown("#### Feature Importances")
        try:
            from src.utils import load_object
            model  = load_object("artifacts/model.pkl")
            fig_fi = feature_importance_chart(model, FEATURE_COLUMNS)
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.caption("Feature importances not available for this model type.")
        except Exception as ex:
            st.caption(f"Could not load feature importances: {ex}")

    st.markdown("---")
    st.markdown("#### Detailed Metrics Table")
    rows = []
    for name, info in report.items():
        rows.append({
            "Model":     name,
            "ROC-AUC":   info["roc_auc"],
            "Accuracy":  info["accuracy"],
            "F1 Score":  info["f1_score"],
            "Precision": info["precision"],
            "Recall":    info["recall"],
            "Best ⭐":   "⭐" if name == best_name else "",
        })
    df_table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        df_table.style.highlight_max(
            subset=["ROC-AUC", "Accuracy", "F1 Score"],
            color="#4f46e522",
        ),
        use_container_width=True,
    )
