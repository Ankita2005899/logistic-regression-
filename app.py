import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_preprocessing import (load_raw_data, create_target,
                                    get_feature_types, get_available_datasets)
from src.model_training import train_and_save, load_pipeline, get_model_path
from src.model_evaluation import get_metrics
from src.prediction import predict_student, explain_prediction
from src.visualization import (
    fig_pass_fail_bar, fig_pass_fail_pie, fig_age_dist, fig_studytime_dist,
    fig_absences_dist, fig_gender_dist, fig_internet_dist, fig_famsup_dist,
    fig_studytime_vs_result, fig_failures_vs_result, fig_absences_vs_result,
    fig_famrel_vs_result, fig_freetime_vs_result, fig_schoolsup_vs_result,
    fig_confusion_matrix, fig_roc_curve, fig_metrics_bar, fig_prob_bar,
    fig_gauge, fig_factors,
)

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
    .stApp { background-color: #F4F6FB; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1E1B4B 0%,#3730A3 100%); }
    [data-testid="stSidebar"] .stMarkdown *, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSidebar"] .stRadio label * { color: #FFFFFF !important; }
    .hero { background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#0D9488 120%);
            color:#fff; padding:22px 28px; border-radius:16px; margin-bottom:18px;
            box-shadow:0 6px 18px rgba(79,70,229,.25); }
    .hero h1 { color:#fff; margin:0; font-size:30px; }
    .hero p  { color:#E0E7FF; margin:6px 0 0; font-size:15px; }
    .metric-card { background:#fff; border-radius:14px; padding:16px 20px; margin-bottom:12px;
                   box-shadow:0 2px 10px rgba(79,70,229,.10); border-left:5px solid #0D9488; }
    .metric-card h3 { margin:0; font-size:13px; color:#64748B; font-weight:600; }
    .metric-card p  { margin:4px 0 0; font-size:25px; font-weight:700; color:#1E1B4B; }
    .section-header { background:linear-gradient(90deg,#4F46E5,#7C3AED); color:#fff !important;
                      padding:8px 18px; border-radius:10px; font-size:17px; font-weight:700; margin:16px 0 12px; }
    .result-card { border-radius:18px; padding:22px 26px; text-align:center; margin-bottom:12px;
                   box-shadow:0 6px 18px rgba(0,0,0,.08); }
    .result-card .tag { font-size:13px; letter-spacing:1.5px; font-weight:700; opacity:.85; }
    .result-card .big { font-size:34px; font-weight:800; margin:4px 0; }
    .result-card .sub { font-size:15px; }
    .r-low  { background:linear-gradient(135deg,#D1FAE5,#A7F3D0); color:#065F46; }
    .r-med  { background:linear-gradient(135deg,#FEF3C7,#FDE68A); color:#92400E; }
    .r-high { background:linear-gradient(135deg,#FEE2E2,#FECACA); color:#991B1B; }
    .action-box { background:#fff; border-radius:14px; padding:14px 18px; border-left:5px solid #4F46E5;
                  box-shadow:0 2px 10px rgba(79,70,229,.10); font-size:15px; }
    .note { color:#64748B; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- friendly labels
LABELS = {
    "school": ("School", "Which school the student attends"),
    "sex": ("Gender", ""),
    "age": ("Age (years)", "Student age"),
    "address": ("Home area", "Urban or rural home address"),
    "famsize": ("Family size", ""),
    "Pstatus": ("Parents' status", "Living together or apart"),
    "Medu": ("Mother's education", "0 none · 1 primary · 2 5th–9th grade · 3 secondary · 4 higher education"),
    "Fedu": ("Father's education", "0 none · 1 primary · 2 5th–9th grade · 3 secondary · 4 higher education"),
    "Mjob": ("Mother's job", ""),
    "Fjob": ("Father's job", ""),
    "reason": ("Reason for choosing school", ""),
    "guardian": ("Guardian", ""),
    "traveltime": ("Travel time to school", "1 = under 15 min · 2 = 15–30 min · 3 = 30–60 min · 4 = over 1 hour"),
    "studytime": ("Weekly study time", "1 = under 2 h · 2 = 2–5 h · 3 = 5–10 h · 4 = over 10 h"),
    "failures": ("Past class failures", "Number of earlier failed classes (3 means 3 or more)"),
    "schoolsup": ("Extra school support", "Extra educational support from the school"),
    "famsup": ("Family educational support", ""),
    "paid": ("Paid extra classes", "Paid tuition in the subject"),
    "activities": ("Extra-curricular activities", ""),
    "nursery": ("Attended nursery school", ""),
    "higher": ("Wants higher education", "Plans to continue to university"),
    "internet": ("Internet access at home", ""),
    "romantic": ("In a relationship", ""),
    "famrel": ("Family relationship quality", "1 = very bad … 5 = excellent"),
    "freetime": ("Free time after school", "1 = very little … 5 = a lot"),
    "goout": ("Going out with friends", "1 = very rarely … 5 = very often"),
    "health": ("Current health status", "1 = very bad … 5 = very good"),
    "absences": ("School absences", "Number of classes missed"),
    "subject": ("Subject", ""),
}
GROUPS = {
    "👤 Profile": ["school", "sex", "age", "address", "guardian", "romantic"],
    "👪 Family": ["famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "famrel", "famsup"],
    "📘 Study & school": ["studytime", "failures", "absences", "schoolsup", "paid", "higher",
                          "reason", "traveltime", "nursery", "internet", "activities"],
    "🎯 Lifestyle": ["freetime", "goout", "health"],
}
VALUE_NAMES = {"F": "Female", "M": "Male", "U": "Urban", "R": "Rural", "GT3": "More than 3 people",
               "LE3": "3 or fewer", "T": "Living together", "A": "Living apart",
               "GP": "Gabriel Pereira", "MS": "Mousinho da Silveira", "yes": "Yes", "no": "No"}
PRESETS = {
    "Typical student (dataset median)": {},
    "Low-risk example": {"studytime": 4, "failures": 0, "absences": 0, "higher": "yes", "goout": 2,
                         "schoolsup": "no", "age": 16, "famrel": 5},
    "High-risk example": {"studytime": 1, "failures": 2, "absences": 20, "higher": "no", "goout": 5,
                          "schoolsup": "yes", "age": 19, "famrel": 2},
}

available_subjects = get_available_datasets()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🎓 Student Performance")
    st.markdown("##### Early-warning dashboard")
    st.markdown("---")
    st.markdown("**📚 Dataset**")
    selected_subject_label = st.selectbox("Dataset", list(available_subjects.keys()), index=0,
                                          label_visibility="collapsed")
    dataset_key = available_subjects[selected_subject_label]
    st.markdown("---")
    page = st.radio("Navigation", [
        "🔮  Predict a Student", "🏠  Home Dashboard", "📊  Dataset Analysis",
        "📈  Data Visualization", "⚙️  Model Training", "📋  Model Evaluation", "ℹ️  About Project",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Algorithm:** Logistic Regression")


@st.cache_data
def get_data(d_key):
    raw = load_raw_data(d_key)
    return raw, create_target(raw)


def ensure_model(key):
    """Train automatically if no saved model exists (needed on a fresh Streamlit Cloud server)."""
    if not os.path.exists(get_model_path(key)):
        with st.spinner("Training the model for the first time (a few seconds)..."):
            train_and_save(key)


raw_df, df = get_data(dataset_key)


def card(title, value, suffix=""):
    st.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{value}{suffix}</p></div>',
                unsafe_allow_html=True)


# ============================================================================= PREDICT
if page == "🔮  Predict a Student":
    ensure_model(dataset_key)
    st.markdown("""<div class="hero"><h1>🔮 Predict a Student's Result</h1>
    <p>Enter the student's details on the left. The prediction, risk level and the reasons behind it
    update instantly on the right.</p></div>""", unsafe_allow_html=True)

    feature_df = df.drop(columns=["result"])
    num_cols, cat_cols = get_feature_types(feature_df)

    def default_for(col):
        if col in num_cols:
            return int(feature_df[col].median())
        return feature_df[col].mode()[0]

    def k(col):
        return f"in_{dataset_key}_{col}"

    for col in feature_df.columns:                   # initialise widget state once
        if k(col) not in st.session_state:
            st.session_state[k(col)] = default_for(col)

    def apply_preset():
        over = PRESETS[st.session_state["preset"]]
        for col in feature_df.columns:
            st.session_state[k(col)] = over.get(col, default_for(col))

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="section-header">1 · Student details (input)</div>', unsafe_allow_html=True)
        st.selectbox("Quick start with an example", list(PRESETS), key="preset", on_change=apply_preset,
                     help="Fills every field with a sample student. You can still edit any value.")
        groups = {g: [c for c in cols if c in feature_df.columns] for g, cols in GROUPS.items()}
        used = {c for cols in groups.values() for c in cols}
        groups["👤 Profile"] += [c for c in feature_df.columns if c not in used]
        tabs = st.tabs(list(groups))
        for tab, (gname, cols) in zip(tabs, groups.items()):
            with tab:
                c1, c2 = st.columns(2)
                for i, col in enumerate(cols):
                    label, help_ = LABELS.get(col, (col, ""))
                    with (c1 if i % 2 == 0 else c2):
                        if col in num_cols:
                            mn, mx = int(feature_df[col].min()), int(feature_df[col].max())
                            st.slider(label, mn, mx, key=k(col), help=help_ or None)
                        else:
                            opts = sorted(feature_df[col].dropna().unique().tolist())
                            st.selectbox(label, opts, key=k(col), help=help_ or None,
                                         format_func=lambda v: VALUE_NAMES.get(
                                             v, str(v).replace("_", " ").title()))

    input_data = {col: st.session_state[k(col)] for col in feature_df.columns}
    for col in num_cols:
        input_data[col] = float(input_data[col])
    res = predict_student(input_data, dataset_key)

    with right:
        st.markdown('<div class="section-header">2 · Prediction (output)</div>', unsafe_allow_html=True)
        threshold = st.slider("Alert threshold for P(FAIL)", 0.10, 0.90, 0.30, 0.05,
                              help="A student is flagged as 'at risk' when the probability of FAIL reaches "
                                   "this value. A lower threshold catches more at-risk students but raises "
                                   "more false alarms. 0.30 is the value recommended in the report.")
        p_fail = res["fail_prob"] / 100.0
        if p_fail >= max(0.5, threshold):
            cls, tag, big = "r-high", "HIGH RISK", "Likely to FAIL"
            action = ("**Priority support:** arrange a counselling meeting, extra tutoring and attendance "
                      "follow-up as soon as possible.")
        elif p_fail >= threshold:
            cls, tag, big = "r-med", "MEDIUM RISK", "At risk of failing"
            action = ("**Keep a close watch:** check in regularly, offer study support and review "
                      "progress after the next assessment.")
        else:
            cls, tag, big = "r-low", "LOW RISK", "Likely to PASS"
            action = "**Routine monitoring:** no extra action needed now; keep encouraging steady attendance."
        st.markdown(f"""<div class="result-card {cls}"><div class="tag">{tag}</div>
        <div class="big">{big}</div>
        <div class="sub">PASS probability <b>{res['pass_prob']:.1f}%</b> &nbsp;·&nbsp;
        FAIL probability <b>{res['fail_prob']:.1f}%</b></div></div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_gauge(res["fail_prob"], threshold), width="stretch")
        st.markdown(f'<div class="action-box">💡 {action}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">3 · Why this prediction?</div>', unsafe_allow_html=True)
    fx = explain_prediction(input_data, dataset_key, top_n=8)
    fx["label"] = [f"{LABELS.get(f, (f, ''))[0]}: "
                   f"{VALUE_NAMES.get(input_data[f], str(input_data[f]).replace('_', ' ').title()) if f in cat_cols else int(input_data[f])}"
                   for f in fx["feature"]]
    cA, cB = st.columns([3, 2], gap="large")
    with cA:
        st.plotly_chart(fig_factors(fx), width="stretch")
    with cB:
        st.plotly_chart(fig_prob_bar(res["pass_prob"], res["fail_prob"]), width="stretch")
    st.markdown('<p class="note">Red bars push this student towards FAIL, green bars towards PASS. '
                'This is a probability-based estimate from a model with modest accuracy (see Model Evaluation), '
                'not a guaranteed outcome, and it must not be used on its own to make decisions about a student.</p>',
                unsafe_allow_html=True)

# ============================================================================= HOME
elif page == "🏠  Home Dashboard":
    ensure_model(dataset_key)
    st.markdown(f"""<div class="hero"><h1>🎓 Student Performance Dashboard</h1>
    <p>Active dataset: {selected_subject_label}</p></div>""", unsafe_allow_html=True)
    n_pass = int((df["result"] == "PASS").sum())
    n_fail = int((df["result"] == "FAIL").sum())
    n_feat = df.drop(columns=["result"]).shape[1]
    m = get_metrics(dataset_key)
    acc_text = f"{m['accuracy']*100:.2f}%" if m else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("📂 Active Dataset", dataset_key)
    with c2: card("👥 Total Students", df.shape[0])
    with c3: card("🧩 Features", n_feat)
    with c4: card("🤖 Model", "Logistic Regression")
    c5, c6, c7, c8 = st.columns(4)
    with c5: card("✅ PASS Students", n_pass)
    with c6: card("❌ FAIL Students", n_fail)
    with c7: card("📊 Pass Rate", f"{n_pass/df.shape[0]*100:.1f}", "%")
    with c8: card("🎯 Test Accuracy", acc_text)
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig_pass_fail_bar(df), width="stretch")
    with col2: st.plotly_chart(fig_pass_fail_pie(df), width="stretch")

# ============================================================================= DATASET
elif page == "📊  Dataset Analysis":
    st.markdown(f'<div class="hero"><h1>📊 Dataset Analysis</h1><p>{selected_subject_label}</p></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(10), width="stretch")
    c1, c2, c3 = st.columns(3)
    with c1: card("Total Rows", raw_df.shape[0])
    with c2: card("Total Columns", raw_df.shape[1])
    with c3: card("Missing Values", int(raw_df.isnull().sum().sum()))
    st.markdown('<div class="section-header">Distributions</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_age_dist(df), width="stretch")
    with b: st.plotly_chart(fig_studytime_dist(df), width="stretch")
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_absences_dist(df), width="stretch")
    with b: st.plotly_chart(fig_gender_dist(df), width="stretch")
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_internet_dist(df), width="stretch")
    with b: st.plotly_chart(fig_famsup_dist(df), width="stretch")

# ============================================================================= VISUALIZATION
elif page == "📈  Data Visualization":
    st.markdown(f'<div class="hero"><h1>📈 Feature vs Outcome</h1><p>{selected_subject_label}</p></div>',
                unsafe_allow_html=True)
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_studytime_vs_result(df), width="stretch")
    with b: st.plotly_chart(fig_failures_vs_result(df), width="stretch")
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_absences_vs_result(df), width="stretch")
    with b: st.plotly_chart(fig_famrel_vs_result(df), width="stretch")
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_freetime_vs_result(df), width="stretch")
    with b: st.plotly_chart(fig_schoolsup_vs_result(df), width="stretch")

# ============================================================================= TRAINING
elif page == "⚙️  Model Training":
    st.markdown(f'<div class="hero"><h1>⚙️ Model Training</h1><p>{selected_subject_label}</p></div>',
                unsafe_allow_html=True)
    st.info("Logistic Regression (L-BFGS, max_iter=1000) · 80:20 stratified split · "
            "median/mode imputation · standard scaling · one-hot encoding")
    if st.button("🚀 TRAIN / RE-TRAIN MODEL", type="primary", width="stretch"):
        with st.spinner("Training model..."):
            res = train_and_save(dataset_key)
        st.success(f"Model trained on {res['train_size']} students, tested on {res['test_size']}.")
    elif os.path.exists(get_model_path(dataset_key)):
        st.success("A trained model already exists for this dataset.")

# ============================================================================= EVALUATION
elif page == "📋  Model Evaluation":
    ensure_model(dataset_key)
    st.markdown(f'<div class="hero"><h1>📋 Model Evaluation</h1><p>{selected_subject_label} · test set · '
                'positive class shown as PASS</p></div>', unsafe_allow_html=True)
    m = get_metrics(dataset_key)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: card("🎯 Accuracy", f"{m['accuracy']*100:.1f}%")
    with c2: card("🔍 Precision", f"{m['precision']:.3f}")
    with c3: card("📡 Recall", f"{m['recall']:.3f}")
    with c4: card("🛡️ Specificity", f"{m['specificity']:.3f}")
    with c5: card("⚖️ F1-Score", f"{m['f1']:.3f}")
    with c6: card("📈 ROC-AUC", f"{m['roc_auc']:.3f}")
    st.info(f"Baseline: a model that always predicts the majority class scores "
            f"**{m['baseline_accuracy']*100:.1f}%** accuracy. Accuracy alone can mislead, so also read "
            f"specificity (how well FAIL students are identified) and ROC-AUC.")
    a, b = st.columns(2)
    with a: st.plotly_chart(fig_confusion_matrix(m["cm"]), width="stretch")
    with b: st.plotly_chart(fig_roc_curve(m["fpr"], m["tpr"], m["roc_auc"]), width="stretch")
    st.plotly_chart(fig_metrics_bar(m), width="stretch")

# ============================================================================= ABOUT
elif page == "ℹ️  About Project":
    st.markdown('<div class="hero"><h1>ℹ️ About This Project</h1>'
                '<p>CIE 2 · Activity 2 · Binary classification with Logistic Regression</p></div>',
                unsafe_allow_html=True)
    st.markdown("""
**Goal:** estimate whether a student will PASS or FAIL (final grade `G3` ≥ 10) so teachers can offer support early.

**Data:** UCI *Student Performance* dataset (Cortez & Silva, 2008) — Mathematics (395 students) and Portuguese (649 students).

**Leakage prevention:** `G1`, `G2`, `G3` are removed; `Dalc` and `Walc` are removed as sensitive.

**Model:** Logistic Regression inside a scikit-learn pipeline (imputation → scaling / one-hot encoding → model).

**Caution:** the *combined* dataset contains many students who appear in both subject files, so its scores can be optimistic.
Predictions are probability-based estimates, not guarantees, and must never be the only basis for a decision about a student.
""")
