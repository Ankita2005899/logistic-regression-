import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PASS_COLOR = "#10B981"
FAIL_COLOR = "#EF4444"
BLUE = "#4F46E5"

def fig_pass_fail_bar(df: pd.DataFrame):
    counts = df["result"].value_counts().reset_index()
    counts.columns = ["Result", "Count"]
    fig = px.bar(counts, x="Result", y="Count", color="Result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 title="PASS vs FAIL Distribution", text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white", title_font_size=16)
    return fig

def fig_pass_fail_pie(df: pd.DataFrame):
    counts = df["result"].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 title="PASS vs FAIL Percentage", hole=0.4)
    fig.update_layout(title_font_size=16)
    return fig

def fig_age_dist(df: pd.DataFrame):
    fig = px.histogram(df, x="age", color="result",
                       color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                       barmode="group", nbins=10, title="Student Age Distribution by Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_studytime_dist(df: pd.DataFrame):
    fig = px.histogram(df, x="studytime", color="result",
                       color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                       barmode="group", title="Study Time Distribution by Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_absences_dist(df: pd.DataFrame):
    fig = px.box(df, x="result", y="absences", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 title="Absences Distribution by Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16, showlegend=False)
    return fig

def fig_gender_dist(df: pd.DataFrame):
    ct = df.groupby(["sex", "result"]).size().reset_index(name="count")
    ct["sex"] = ct["sex"].map({"M": "Male", "F": "Female"})
    fig = px.bar(ct, x="sex", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Gender vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_internet_dist(df: pd.DataFrame):
    ct = df.groupby(["internet", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="internet", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Internet Access vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_famsup_dist(df: pd.DataFrame):
    ct = df.groupby(["famsup", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="famsup", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Family Education Support vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_studytime_vs_result(df):
    ct = df.groupby(["studytime", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="studytime", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Study Time vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_failures_vs_result(df):
    ct = df.groupby(["failures", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="failures", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Past Failures vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_absences_vs_result(df):
    fig = px.violin(df, x="result", y="absences", color="result",
                    color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                    box=True, points="outliers", title="Absences vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16, showlegend=False)
    return fig

def fig_famrel_vs_result(df):
    ct = df.groupby(["famrel", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="famrel", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Family Relationship Quality vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_freetime_vs_result(df):
    ct = df.groupby(["freetime", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="freetime", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="Free Time vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_schoolsup_vs_result(df):
    ct = df.groupby(["schoolsup", "result"]).size().reset_index(name="count")
    fig = px.bar(ct, x="schoolsup", y="count", color="result",
                 color_discrete_map={"PASS": PASS_COLOR, "FAIL": FAIL_COLOR},
                 barmode="group", title="School Support vs Result")
    fig.update_layout(plot_bgcolor="white", title_font_size=16)
    return fig

def fig_confusion_matrix(cm: np.ndarray):
    labels = ["PASS", "FAIL"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=labels, y=labels, labels=dict(x="Predicted", y="Actual"),
                    title="Confusion Matrix Heatmap")
    fig.update_layout(title_font_size=16, coloraxis_showscale=False)
    return fig

def fig_roc_curve(fpr, tpr, roc_auc: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"Logistic Regression (AUC = {roc_auc:.3f})",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random (AUC = 0.5)",
                             line=dict(color="grey", dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", plot_bgcolor="white",
                      title_font_size=16)
    return fig

def fig_metrics_bar(metrics: dict):
    names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [metrics["accuracy"], metrics["precision"],
              metrics["recall"], metrics["f1"], metrics["roc_auc"]]
    fig = px.bar(x=names, y=values, text=[f"{v:.3f}" for v in values],
                 color=names, color_discrete_sequence=px.colors.sequential.Blues_r,
                 title="Model Performance Metrics", labels={"x": "Metric", "y": "Score"})
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.15], plot_bgcolor="white", title_font_size=16)
    return fig

def fig_prob_bar(pass_prob: float, fail_prob: float):
    fig = go.Figure(go.Bar(
        x=["PASS", "FAIL"],
        y=[pass_prob, fail_prob],
        marker_color=[PASS_COLOR, FAIL_COLOR],
        text=[f"{pass_prob:.1f}%", f"{fail_prob:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(title="Prediction Probability", yaxis_title="Probability (%)",
                      yaxis_range=[0, 115], plot_bgcolor="white", title_font_size=16,
                      showlegend=False)
    return fig


def fig_gauge(fail_prob: float, threshold: float):
    """Gauge of P(FAIL) in %, with the chosen decision threshold marked."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fail_prob,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probability of FAIL", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#1E1B4B", "thickness": 0.25},
            "steps": [
                {"range": [0, threshold * 100], "color": "#D1FAE5"},
                {"range": [threshold * 100, max(50, threshold * 100)], "color": "#FEF3C7"},
                {"range": [max(50, threshold * 100), 100], "color": "#FEE2E2"},
            ],
            "threshold": {"line": {"color": "#4F46E5", "width": 4},
                          "thickness": 0.85, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def fig_factors(factors: pd.DataFrame):
    """Horizontal bars: what pushes this student towards FAIL (red) or PASS (green)."""
    f = factors.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=f["effect"], y=f["label"], orientation="h",
        marker_color=[FAIL_COLOR if v > 0 else PASS_COLOR for v in f["effect"]],
        hovertemplate="%{y}<br>effect: %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(title="Main factors behind this prediction",
                      xaxis_title="← pushes towards PASS     pushes towards FAIL →",
                      height=60 + 38 * len(f), margin=dict(l=10, r=10, t=50, b=40),
                      plot_bgcolor="white", title_font_size=16, showlegend=False)
    fig.add_vline(x=0, line_color="#1E1B4B", line_width=1)
    return fig
