import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield — Threat Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — PROFESSIONAL DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    /* ─── Global ─── */
    .stApp {
        background: #0b0f19;
        font-family: 'IBM Plex Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: #0e1221 !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #7a8599 !important; }

    /* ─── Typography ─── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    p, span, div, label { color: #8b95a9 !important; }

    /* ─── Card ─── */
    .card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 22px 24px;
        transition: border-color 0.2s ease;
    }
    .card:hover { border-color: rgba(255,255,255,0.08); }

    /* ─── Metric Card ─── */
    .kpi-card {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 20px 22px;
    }
    .kpi-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #566072 !important;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #e2e8f0 !important;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 4px;
        color: #566072 !important;
    }
    .kpi-value.accent-teal { color: #2dd4bf !important; }
    .kpi-value.accent-red { color: #f87171 !important; }
    .kpi-value.accent-amber { color: #fbbf24 !important; }
    .kpi-value.accent-blue { color: #60a5fa !important; }

    /* ─── Section Header ─── */
    .section-hdr {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 32px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .section-hdr .label {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #c9d1e0 !important;
        letter-spacing: -0.01em;
    }
    .section-hdr .tag {
        background: rgba(45, 212, 191, 0.08);
        color: #2dd4bf !important;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ─── Small icon circles ─── */
    .ic {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .ic-teal { background: rgba(45,212,191,0.1); color: #2dd4bf !important; }
    .ic-red { background: rgba(248,113,113,0.1); color: #f87171 !important; }
    .ic-blue { background: rgba(96,165,250,0.1); color: #60a5fa !important; }
    .ic-amber { background: rgba(251,191,36,0.1); color: #fbbf24 !important; }
    .ic-purple { background: rgba(167,139,250,0.1); color: #a78bfa !important; }

    /* ─── Alert bars ─── */
    .alert-bar {
        border-radius: 8px;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        gap: 14px;
        font-size: 0.88rem;
    }
    .alert-threat {
        background: rgba(248,113,113,0.06);
        border: 1px solid rgba(248,113,113,0.15);
    }
    .alert-clear {
        background: rgba(45,212,191,0.05);
        border: 1px solid rgba(45,212,191,0.12);
    }
    .dot-pulse {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .dot-red { background: #f87171; box-shadow: 0 0 6px rgba(248,113,113,0.4); }
    .dot-green { background: #2dd4bf; box-shadow: 0 0 6px rgba(45,212,191,0.4); }

    /* ─── Tabs ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(17,24,39,0.6);
        border-radius: 8px;
        padding: 3px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #566072;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 8px 18px;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(45,212,191,0.06) !important;
        color: #2dd4bf !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ─── File uploader ─── */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        background: rgba(17,24,39,0.4) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(255,255,255,0.14) !important;
    }

    /* ─── Download button ─── */
    .stDownloadButton button {
        background: #2dd4bf !important;
        color: #0b0f19 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 9px 22px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
    }
    .stDownloadButton button:hover {
        background: #14b8a6 !important;
    }

    /* ─── Data Frame ─── */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border: 1px solid rgba(255,255,255,0.04) !important;
        border-radius: 8px;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

    /* ─── Sidebar info block ─── */
    .sb-info {
        background: rgba(17,24,39,0.5);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 14px 16px;
    }
    .sb-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
    }
    .sb-key { color: #566072 !important; font-size: 0.78rem; }
    .sb-val { color: #c9d1e0 !important; font-size: 0.78rem; font-weight: 600; }

    /* ─── Insight Block ─── */
    .insight-block {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.04);
        border-left: 3px solid #60a5fa;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .insight-block p { color: #8b95a9 !important; font-size: 0.88rem; line-height: 1.6; margin: 0; }
    .insight-block strong { color: #c9d1e0 !important; }

    hr { border-color: rgba(255,255,255,0.04) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL + SCALER + FEATURES
# ─────────────────────────────────────────────
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("features.pkl")

# ─────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,15,25,0.6)",
    font=dict(family="IBM Plex Sans, Inter, sans-serif", color="#8b95a9", size=12),
    margin=dict(l=50, r=20, t=16, b=50),
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False, title_font=dict(size=11, color="#566072")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zeroline=False, title_font=dict(size=11, color="#566072")),
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 24px 0;">
        <div style="font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700;
            color: #e2e8f0 !important; letter-spacing: -0.02em;">
            CyberShield
        </div>
        <div style="color: #566072 !important; font-size: 0.72rem; font-weight: 500;
            letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px;">
            Threat Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
        <span class="dot-pulse dot-green"></span>
        <span style="color: #2dd4bf !important; font-weight: 600; font-size: 0.8rem;">System Operational</span>
        <span style="color: #566072 !important; font-size: 0.72rem; margin-left: auto;">v2.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div style="color: #566072 !important; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Model Configuration</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-info">
        <div class="sb-row"><span class="sb-key">Algorithm</span><span class="sb-val">Random Forest</span></div>
        <div class="sb-row"><span class="sb-key">Estimators</span><span class="sb-val">100</span></div>
        <div class="sb-row"><span class="sb-key">Features</span><span class="sb-val">{len(feature_list)}</span></div>
        <div class="sb-row"><span class="sb-key">Dataset</span><span class="sb-val">CICIDS-2017</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div style="color: #566072 !important; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px;">Detection Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("Threat probability threshold", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed",
                          help="Flows above this probability are classified as threats.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div style="color: #566072 !important; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px;">Data Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 16px 0 4px 0;">
    <h1 style="font-size: 1.6rem !important; font-weight: 700 !important; margin: 0 !important;
        letter-spacing: -0.02em; color: #e2e8f0 !important;">
        Threat Detection Dashboard
    </h1>
    <p style="color: #566072 !important; font-size: 0.85rem; margin-top: 4px;">
        Network traffic analysis powered by machine learning &mdash; CICIDS-2017 pipeline
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NO FILE — EMPTY STATE
# ─────────────────────────────────────────────
if not uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 56px 40px;">
            <div style="width: 48px; height: 48px; border-radius: 10px; background: rgba(45,212,191,0.08);
                display: inline-flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2dd4bf" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0 !important; margin-bottom: 8px;">
                Upload Network Traffic Data
            </div>
            <div style="color: #566072 !important; font-size: 0.88rem; line-height: 1.7; max-width: 380px; margin: 0 auto;">
                Provide a CSV file with network flow features. The model will classify each flow
                as benign or malicious using the trained Random Forest classifier.
            </div>
            <div style="margin-top: 20px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
                <span style="background: rgba(45,212,191,0.06); color: #2dd4bf !important; padding: 4px 12px;
                    border-radius: 4px; font-size: 0.72rem; font-weight: 600;">DDoS</span>
                <span style="background: rgba(96,165,250,0.06); color: #60a5fa !important; padding: 4px 12px;
                    border-radius: 4px; font-size: 0.72rem; font-weight: 600;">Port Scan</span>
                <span style="background: rgba(167,139,250,0.06); color: #a78bfa !important; padding: 4px 12px;
                    border-radius: 4px; font-size: 0.72rem; font-weight: 600;">Web Attacks</span>
                <span style="background: rgba(251,191,36,0.06); color: #fbbf24 !important; padding: 4px 12px;
                    border-radius: 4px; font-size: 0.72rem; font-weight: 600;">Infiltration</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# PROCESS DATA
# ─────────────────────────────────────────────
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()

if "label" in df.columns:
    y_true = df["label"]
    X = df.drop("label", axis=1)
elif " Label" in df.columns:
    label_col = " Label"
    df["label"] = (df[label_col] != "BENIGN").astype(int)
    y_true = df["label"]
    X = df.drop(columns=[label_col, "label"])
else:
    y_true = None
    X = df.copy()

X = X.select_dtypes(include=[np.number])
X = X.replace([float('inf'), -float('inf')], 0)
X = X.fillna(0)
X = X.reindex(columns=feature_list, fill_value=0)
X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]
preds_thresholded = (probs >= threshold).astype(int)

df["Prediction"] = preds_thresholded
df["Threat_Probability"] = probs

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
total_samples = len(df)
attack_count = int(preds_thresholded.sum())
benign_count = total_samples - attack_count
attack_rate = (attack_count / total_samples) * 100 if total_samples > 0 else 0
avg_prob = float(probs.mean()) * 100
max_prob = float(probs.max()) * 100

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-hdr" style="margin-top: 12px;">
    <span class="ic ic-teal">S</span>
    <span class="label">Summary</span>
    <span class="tag">Live</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Flows</div>
        <div class="kpi-value">{total_samples:,}</div>
        <div class="kpi-sub">Packets analyzed</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Threats Detected</div>
        <div class="kpi-value accent-red">{attack_count:,}</div>
        <div class="kpi-sub" style="color: #f87171 !important;">{attack_rate:.1f}% of traffic</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Benign Flows</div>
        <div class="kpi-value accent-teal">{benign_count:,}</div>
        <div class="kpi-sub" style="color: #2dd4bf !important;">Clean traffic</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Mean Threat Score</div>
        <div class="kpi-value accent-blue">{avg_prob:.1f}%</div>
        <div class="kpi-sub">Average probability</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Peak Threat Score</div>
        <div class="kpi-value accent-amber">{max_prob:.1f}%</div>
        <div class="kpi-sub">Highest confidence</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ALERT
# ─────────────────────────────────────────────
if attack_count > 0:
    st.markdown(f"""
    <div class="alert-bar alert-threat">
        <span class="dot-pulse dot-red"></span>
        <div>
            <span style="color: #f87171 !important; font-weight: 600;">
                {attack_count:,} malicious flows identified
            </span>
            <span style="color: #6b4444 !important; font-size: 0.82rem;">
                &mdash; {attack_rate:.2f}% of traffic flagged at threshold {threshold:.0%}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-bar alert-clear">
        <span class="dot-pulse dot-green"></span>
        <div>
            <span style="color: #2dd4bf !important; font-weight: 600;">No threats detected</span>
            <span style="color: #3d6b5f !important; font-size: 0.82rem;">
                &mdash; all flows classified as benign
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_model, tab_features, tab_data = st.tabs([
    "Threat Analysis",
    "Model Performance",
    "Feature Intelligence",
    "Data Explorer"
])

# ═══════════ TAB 1: THREAT ANALYSIS ═══════════
with tab_overview:

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="section-hdr" style="margin-top: 8px;">
            <span class="ic ic-blue">D</span>
            <span class="label">Probability Distribution</span>
        </div>
        """, unsafe_allow_html=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=probs, nbinsx=60,
            marker=dict(color="#2dd4bf", line=dict(width=0.3, color="rgba(0,0,0,0.4)")),
            opacity=0.8,
            hovertemplate="Probability: %{x:.3f}<br>Count: %{y}<extra></extra>"
        ))
        fig_hist.add_vline(x=threshold, line_dash="dot", line_color="#fbbf24", line_width=1.5,
                          annotation_text=f"Threshold {threshold:.0%}",
                          annotation_font=dict(color="#fbbf24", size=10))
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=370,
                              xaxis_title="Threat Probability", yaxis_title="Flow Count", bargap=0.04)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("""
        <div class="section-hdr" style="margin-top: 8px;">
            <span class="ic ic-teal">C</span>
            <span class="label">Classification Breakdown</span>
        </div>
        """, unsafe_allow_html=True)

        fig_donut = go.Figure()
        fig_donut.add_trace(go.Pie(
            labels=["Benign", "Malicious"], values=[benign_count, attack_count],
            hole=0.68,
            marker=dict(colors=["#2dd4bf", "#f87171"], line=dict(width=2, color="#0b0f19")),
            textinfo="label+percent",
            textfont=dict(size=12, family="Inter", color="#c9d1e0"),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            pull=[0, 0.02]
        ))
        fig_donut.update_layout(**PLOTLY_LAYOUT)
        fig_donut.update_layout(
            xaxis=None, yaxis=None,
            height=370, showlegend=False,
            annotations=[dict(
                text=f'<b>{attack_rate:.1f}%</b><br><span style="font-size:10px;color:#566072">Threats</span>',
                x=0.5, y=0.5, font_size=20, font_color="#e2e8f0", showarrow=False, font_family="Inter"
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # Timeline
    st.markdown("""
    <div class="section-hdr">
        <span class="ic ic-purple">T</span>
        <span class="label">Threat Score Across Flows</span>
        <span class="tag">Sequential</span>
    </div>
    """, unsafe_allow_html=True)

    sample_size = min(5000, len(probs))
    indices = np.linspace(0, len(probs)-1, sample_size, dtype=int)
    probs_sampled = probs[indices]

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=list(range(len(probs_sampled))), y=probs_sampled,
        mode='lines', fill='tozeroy',
        line=dict(color="#60a5fa", width=1),
        fillcolor='rgba(96,165,250,0.05)',
        hovertemplate="Flow #%{x}<br>Score: %{y:.4f}<extra></extra>"
    ))
    fig_tl.add_hline(y=threshold, line_dash="dot", line_color="#f87171", line_width=1,
                     annotation_text=f"Threshold {threshold:.0%}",
                     annotation_font=dict(color="#f87171", size=10),
                     annotation_position="top right")
    fig_tl.update_layout(**PLOTLY_LAYOUT, height=280,
                         xaxis_title="Flow Index (sampled)", yaxis_title="Threat Probability",
                         yaxis_range=[0, 1])
    st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})


# ═══════════ TAB 2: MODEL PERFORMANCE ═══════════
with tab_model:

    if y_true is not None:
        acc = accuracy_score(y_true, preds_thresholded)
        prec = precision_score(y_true, preds_thresholded, zero_division=0)
        rec = recall_score(y_true, preds_thresholded, zero_division=0)
        f1 = f1_score(y_true, preds_thresholded, zero_division=0)

        st.markdown("""
        <div class="section-hdr" style="margin-top: 8px;">
            <span class="ic ic-teal">M</span>
            <span class="label">Evaluation Metrics</span>
            <span class="tag">Computed</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Accuracy", f"{acc:.4f}", "Overall correctness", "accent-teal", m1),
            ("Precision", f"{prec:.4f}", "Positive predictive value", "accent-blue", m2),
            ("Recall", f"{rec:.4f}", "Sensitivity", "accent-amber", m3),
            ("F1 Score", f"{f1:.4f}", "Harmonic mean", "accent-red", m4),
        ]
        for label, value, desc, accent, col in metrics:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value {accent}">{value}</div>
                    <div class="kpi-sub">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("""
            <div class="section-hdr" style="margin-top: 0;">
                <span class="ic ic-amber">C</span>
                <span class="label">Confusion Matrix</span>
            </div>
            """, unsafe_allow_html=True)

            cm = confusion_matrix(y_true, preds_thresholded)
            labels = ["Benign", "Malicious"]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels,
                text=[[f"{v:,}" for v in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=16, family="JetBrains Mono", color="#e2e8f0"),
                colorscale=[[0, '#111827'], [0.5, '#1a3a35'], [1, '#2dd4bf']],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
            ))
            fig_cm.update_layout(**PLOTLY_LAYOUT)
            fig_cm.update_layout(xaxis=dict(title="Predicted", title_font=dict(size=11, color="#566072")),
                                 yaxis=dict(title="Actual", title_font=dict(size=11, color="#566072"), autorange="reversed"),
                                 height=370, margin=dict(l=60, r=20, t=16, b=60))
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

        with col_roc:
            st.markdown("""
            <div class="section-hdr" style="margin-top: 0;">
                <span class="ic ic-blue">R</span>
                <span class="label">ROC Curve</span>
            </div>
            """, unsafe_allow_html=True)

            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', fill='tozeroy',
                line=dict(color='#2dd4bf', width=2),
                fillcolor='rgba(45,212,191,0.04)',
                name=f'AUC = {roc_auc:.4f}',
                hovertemplate="FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(color='rgba(255,255,255,0.08)', width=1, dash='dash'),
                showlegend=False, hoverinfo='skip'
            ))
            fig_roc.update_layout(**PLOTLY_LAYOUT, height=370,
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                  legend=dict(x=0.55, y=0.08, bgcolor="rgba(0,0,0,0)",
                                             font=dict(size=12, color="#2dd4bf")))
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div class="section-hdr">
            <span class="ic ic-purple">R</span>
            <span class="label">Classification Report</span>
        </div>
        """, unsafe_allow_html=True)

        report_dict = classification_report(y_true, preds_thresholded, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 50px;">
            <div style="color: #e2e8f0 !important; font-size: 1rem; font-weight: 600; margin-bottom: 8px;">
                Ground Truth Labels Unavailable
            </div>
            <div style="color: #566072 !important; font-size: 0.85rem;">
                Include a "label" column in the uploaded data to enable model evaluation metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════ TAB 3: FEATURE INTELLIGENCE ═══════════
with tab_features:

    st.markdown("""
    <div class="section-hdr" style="margin-top: 8px;">
        <span class="ic ic-teal">F</span>
        <span class="label">Feature Importance</span>
        <span class="tag">Top 15</span>
    </div>
    """, unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_list, "Importance": importances}).sort_values(by="Importance", ascending=False)
    top_n = feat_df.head(15)

    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(
        y=top_n["Feature"][::-1], x=top_n["Importance"][::-1],
        orientation='h',
        marker=dict(color=top_n["Importance"][::-1],
                    colorscale=[[0, '#1e3a4a'], [0.5, '#60a5fa'], [1, '#2dd4bf']]),
        hovertemplate="%{y}<br>Importance: %{x:.5f}<extra></extra>",
        texttemplate="%{x:.4f}", textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#566072"),
    ))
    fig_feat.update_layout(**PLOTLY_LAYOUT)
    fig_feat.update_layout(height=500, bargap=0.28,
                           xaxis_title="Importance Score",
                           margin=dict(l=200, r=80, t=16, b=50))
    st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})

    # Insight
    top3_share = feat_df.head(3)['Importance'].sum() / feat_df['Importance'].sum() * 100
    st.markdown(f"""
    <div class="insight-block">
        <p>
            The most influential feature is <strong>{feat_df.iloc[0]['Feature']}</strong>
            (score: <strong>{feat_df.iloc[0]['Importance']:.5f}</strong>).
            The top 3 features account for <strong>{top3_share:.1f}%</strong> of total model decision weight.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Correlation
    st.markdown("""
    <div class="section-hdr">
        <span class="ic ic-amber">C</span>
        <span class="label">Feature Correlation Matrix</span>
    </div>
    """, unsafe_allow_html=True)

    top_feat_names = feat_df.head(8)["Feature"].tolist()
    available_feats = [f for f in top_feat_names if f in X.columns]

    if len(available_feats) >= 2:
        corr_matrix = X[available_feats].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=available_feats, y=available_feats,
            colorscale=[[0, '#f87171'], [0.5, '#111827'], [1, '#2dd4bf']],
            zmid=0,
            texttemplate="%{z:.2f}",
            textfont=dict(size=10, family="JetBrains Mono"),
            hovertemplate="%{x} vs %{y}<br>r = %{z:.4f}<extra></extra>",
        ))
        fig_corr.update_layout(**PLOTLY_LAYOUT)
        fig_corr.update_layout(xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.03)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
                               height=400, margin=dict(l=150, r=20, t=16, b=100))
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})


# ═══════════ TAB 4: DATA EXPLORER ═══════════
with tab_data:

    st.markdown("""
    <div class="section-hdr" style="margin-top: 8px;">
        <span class="ic ic-blue">D</span>
        <span class="label">Analyzed Traffic</span>
    </div>
    """, unsafe_allow_html=True)

    filter_choice = st.radio("Filter", ["All", "Threats Only", "Benign Only"],
                             horizontal=True, label_visibility="collapsed")

    if filter_choice == "Threats Only":
        display_df = df[df["Prediction"] == 1]
    elif filter_choice == "Benign Only":
        display_df = df[df["Prediction"] == 0]
    else:
        display_df = df

    st.dataframe(
        display_df.head(500).style.background_gradient(
            subset=["Threat_Probability"], cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True, height=440
    )

    st.markdown(f'<div style="color: #566072 !important; font-size: 0.76rem; margin-top: 6px;">Showing {min(500, len(display_df)):,} of {len(display_df):,} records ({filter_choice.lower()})</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    csv_export = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Results (CSV)", data=csv_export,
                       file_name="cybershield_results.csv", mime="text/csv")

    st.markdown("""
    <div class="section-hdr">
        <span class="ic ic-purple">S</span>
        <span class="label">Statistical Summary</span>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.describe().round(4), use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.03);">
    <div style="color: #3a4050 !important; font-size: 0.72rem; font-weight: 500; letter-spacing: 0.5px;">
        CyberShield v2.0 &middot; Random Forest &middot; CICIDS-2017
    </div>
</div>
""", unsafe_allow_html=True)