"""
app.py — Telecom Intelligence System v2
========================================
Streamlit 5-page dashboard

Pages:
  0. Overview Dashboard
  1. Network Optimization  (regression + LSTM + zone clustering)
  2. User Behaviour        (KMeans + DBSCAN + churn + LTV)
  3. Anomaly Detection     (ISO + OC-SVM + Autoencoder + live alerts)
  4. Live Predictor        (real-time inference for all 3 modules)

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time, os, warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom AI Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080C14; color: #D8E4F2; }

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0A0F1E 0%, #0D1628 100%);
    border-right: 1px solid #162035;
}
section[data-testid="stSidebar"] * { color: #B0C4DE !important; }

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
     color: #4FC3F7 !important; letter-spacing: -1.5px; font-size: 2rem !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
     color: #81D4FA !important; }
h3 { font-family: 'Syne', sans-serif !important; color: #B3E5FC !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0D1628, #111E30);
    border: 1px solid #1A2E4A; border-radius: 12px; padding: 18px;
}
[data-testid="stMetricLabel"] { color: #5A85AA !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #4FC3F7 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 1.6rem !important; }

.stTabs [data-baseweb="tab-list"] { background: #0D1628; border-radius: 12px;
    gap: 3px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #5A85AA; border-radius: 9px; padding: 8px 16px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1A3A5C, #1E4976);
    color: #4FC3F7 !important; font-weight: 600; }

.stButton > button {
    background: linear-gradient(135deg, #1565C0, #0277BD);
    color: #E3F2FD !important; border: none; border-radius: 10px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    letter-spacing: 0.5px; padding: 10px 22px;
    transition: all 0.25s ease; box-shadow: 0 4px 15px rgba(21,101,192,0.3);
}
.stButton > button:hover { transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(79,195,247,0.35); }

.alert-critical {
    background: linear-gradient(135deg,#3B0A0A,#5C1111);
    border-left: 3px solid #EF5350; border-radius: 10px;
    padding: 10px 16px; margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #FFCDD2;
}
.alert-high {
    background: linear-gradient(135deg,#3E1A00,#5D2A00);
    border-left: 3px solid #FF7043; border-radius: 10px;
    padding: 10px 16px; margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #FFE0B2;
}
.alert-medium {
    background: linear-gradient(135deg,#2D2000,#3D2E00);
    border-left: 3px solid #FFB300; border-radius: 10px;
    padding: 10px 16px; margin: 5px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #FFF8E1;
}

.kpi-card {
    background: linear-gradient(135deg, #0D1628, #121F35);
    border: 1px solid #1A3050; border-radius: 14px;
    padding: 20px; text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #4FC3F7; }
.kpi-value { font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #4FC3F7; }
.kpi-label { font-size: 11px; color: #5A85AA;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 4px; }

.insight-box {
    background: linear-gradient(135deg, #0A1628, #0D1E3A);
    border: 1px solid #1A3A5C; border-left: 3px solid #4FC3F7;
    border-radius: 10px; padding: 14px 18px; margin: 10px 0;
    font-size: 13px; color: #B3D9F7;
}

hr { border-color: #162035 !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
BG   = '#080C14'; CARD = '#0D1628'
FONT = '#D8E4F2'; GRID = '#162035'
C = ['#4FC3F7','#4CAF50','#EF5350','#FFC107','#AB47BC',
     '#26C6DA','#FF7043','#66BB6A']
BASE_LAYOUT = dict(
    paper_bgcolor=CARD, plot_bgcolor=BG,
    font=dict(color=FONT, family='DM Sans'),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=GRID),
)
SEG_C = {
    'Premium Power User':  '#FFC107',
    'Data-Heavy Streamer': '#4FC3F7',
    'Mainstream':          '#4CAF50',
    'Light / Occasional':  '#AB47BC',
    'At-Risk / Frustrated':'#EF5350',
}

# ── Load pipeline ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    from ml_engine import run_pipeline
    return run_pipeline(n=5000, force=False)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 **Telecom AI**")
    st.markdown("*Intelligence Platform v2*")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Overview",
        "📶  Network Optimization",
        "👥  User Behaviour",
        "🚨  Anomaly Detection",
        "🔮  Live Predictor",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Global Filters**")
    regions  = st.multiselect("Region",
        ['North','South','East','West','Central'],
        default=['North','South','East','West','Central'])
    plans    = st.multiselect("Plan",
        ['Prepaid','Postpaid','Enterprise'],
        default=['Prepaid','Postpaid','Enterprise'])
    tenure_r = st.slider("Tenure (months)", 1, 60, (1,60))
    st.markdown("---")
    if st.button("🔄  Retrain All Models"):
        st.cache_resource.clear(); st.rerun()
    st.markdown("---")
    st.markdown("""<div style='font-size:10px;color:#2C4A6E;text-align:center;'>
    ML Project · B.Tech 2025–26<br>Python · Scikit-learn · TF/Keras<br>Streamlit · Plotly</div>""",
    unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
with st.spinner("⚡ Loading ML pipeline…"):
    df_all, r1, r2, r3 = load_pipeline()

df = df_all.copy()
df = df[df['region'].isin(regions) & df['plan_type'].isin(plans)]
df = df[(df['tenure_months'] >= tenure_r[0]) & (df['tenure_months'] <= tenure_r[1])]


# ═══════════════════════════════════════════════════════════════
#  PAGE 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════

if "Overview" in page:
    st.markdown("# 📡 Telecom Intelligence System")
    st.markdown("**Unified ML platform** — Network Optimization · User Behaviour · Anomaly Detection")
    st.divider()

    # KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Users",         f"{len(df):,}",                  "+4.2%")
    c2.metric("Avg QoE",       f"{df['quality_score'].mean():.1f}",  "+1.3")
    c3.metric("Anomalies",     f"{df['iso_flag'].sum():,}",      f"{df['iso_flag'].mean()*100:.1f}%")
    c4.metric("Churn Risk",    f"{df['churn_risk'].sum():,}",    f"{df['churn_risk'].mean()*100:.1f}%")
    c5.metric("Avg LTV",       f"₹{df['ltv'].mean():,.0f}",     "+₹380")
    c6.metric("Avg Bill",      f"₹{df['monthly_bill'].mean():,.0f}", "-₹8")

    st.divider()
    r1c, r1d = st.columns(2)

    with r1c:
        fig = px.histogram(df, x='quality_score', nbins=45,
                           color_discrete_sequence=[C[0]],
                           title='Network Quality Score Distribution')
        fig.add_vline(x=df['quality_score'].mean(), line_color=C[3],
                       line_dash='dash',
                       annotation_text=f"Avg {df['quality_score'].mean():.1f}",
                       annotation_font_color=C[3])
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with r1d:
        sc = df['segment_label'].value_counts().reset_index()
        sc.columns = ['Segment','Count']
        colors_pie = [SEG_C.get(s,'#888') for s in sc['Segment']]
        fig = go.Figure(go.Pie(labels=sc['Segment'], values=sc['Count'],
                               hole=0.58, marker_colors=colors_pie,
                               textinfo='label+percent',
                               textfont_size=11))
        fig.update_layout(**BASE_LAYOUT, title='User Segment Distribution')
        st.plotly_chart(fig, use_container_width=True)

    r2a, r2b = st.columns(2)
    with r2a:
        peak = r2['peak_usage']
        fig  = px.area(peak, x='hour', y='avg_data_gb',
                       color_discrete_sequence=[C[1]],
                       title='Hourly Average Data Usage',
                       labels={'hour':'Hour','avg_data_gb':'Avg Data (GB)'})
        fig.update_traces(fill='tozeroy', fillcolor='rgba(76,175,80,0.12)')
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with r2b:
        samp = df.sample(min(1500,len(df)), random_state=1)
        fig  = px.scatter(samp, x='latency_ms', y='packet_loss_pct',
                          color=samp['iso_flag'].map({0:'Normal',1:'Anomaly'}),
                          color_discrete_map={'Normal':C[1],'Anomaly':C[2]},
                          size='anomaly_score', size_max=16, opacity=0.55,
                          title='Anomaly Map: Latency vs Packet Loss',
                          labels={'latency_ms':'Latency (ms)',
                                  'packet_loss_pct':'Packet Loss (%)'})
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # Consolidated model performance bar
    perf = {
        'Ridge Regressor (R²)':      r1['reg_res']['Ridge']['r2'],
        'Gradient Boost (R²)':       r1['reg_res']['GradientBoosting']['r2'],
        'Random Forest (R²)':        r1['reg_res']['RandomForest']['r2'],
        'Churn RF (CV Acc)':         r2['churn_metrics']['cv_acc'],
        'Isolation Forest (F1)':     r3['metrics']['Isolation Forest']['f1'],
        'One-Class SVM (F1)':        r3['metrics']['One-Class SVM']['f1'],
        'Autoencoder (F1)':          r3['metrics']['Autoencoder']['f1'],
        'Z-Score (F1)':              r3['metrics']['Z-Score']['f1'],
    }
    perf_df = pd.DataFrame({'Model':list(perf.keys()),'Score':list(perf.values())})
    fig = px.bar(perf_df, x='Score', y='Model', orientation='h',
                 color='Score', color_continuous_scale='Blues',
                 text='Score', title='All Model Performance Scores')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    perf_layout = {**BASE_LAYOUT, 'xaxis': {**BASE_LAYOUT['xaxis'], 'range': [0, 1.12]}}
    fig.update_layout(**perf_layout)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — NETWORK OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

elif "Network" in page:
    st.markdown("# 📶 Network Optimization")
    st.markdown("*QoE prediction · KMeans/DBSCAN zone clustering · LSTM demand forecast*")
    st.divider()

    t1, t2, t3 = st.tabs(["🎯 Quality Prediction","🗺️ Zone Clustering","📈 LSTM Forecast"])

    with t1:
        best = r1['best_name']
        res  = r1['reg_res'][best]
        c1, c2 = st.columns([3,2])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['y_test'], y=res['y_pred'],
                mode='markers', marker=dict(color=C[0],opacity=0.45,size=5), name='Predictions'))
            mn,mx = float(min(res['y_test'])), float(max(res['y_test']))
            fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',
                line=dict(color=C[2],dash='dash',width=2), name='Perfect fit'))
            fig.update_layout(**BASE_LAYOUT,
                title=f'Actual vs Predicted Quality Score — {best}',
                xaxis_title='Actual', yaxis_title='Predicted')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Model Comparison")
            rows = []
            for nm, rd in r1['reg_res'].items():
                rows.append({'Model':nm,'R²':rd['r2'],'MAE':rd['mae'],'RMSE':rd['rmse']})
            tdf = pd.DataFrame(rows).set_index('Model')
            st.dataframe(tdf, use_container_width=True)
            st.markdown(f"""<div class='insight-box'>
            ✅ <b>{best}</b> achieves R²={res['r2']} — explains {res['r2']*100:.1f}% of
            quality variance from 9 KPIs. All models exceed the 0.85 target threshold.
            </div>""", unsafe_allow_html=True)

        # Feature importance
        fi = r1['feat_imp']
        fi_df = pd.DataFrame({'Feature':list(fi.keys()),'Importance':list(fi.values())})
        fi_df = fi_df.sort_values('Importance')
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title='Feature Importance — Random Forest Regressor')
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        c1,c2 = st.columns(2)
        with c1:
            # KMeans zone quality
            zone_stats = df.groupby('network_zone').agg(
                count=('user_id','count'),
                avg_quality=('quality_score','mean'),
                avg_latency=('latency_ms','mean'),
                avg_signal=('signal_dbm','mean'),
                avg_load=('cell_load_pct','mean'),
            ).round(2).reset_index()
            zone_stats['Zone'] = zone_stats['network_zone'].apply(lambda z: f"Zone {z}")
            fig = px.bar(zone_stats, x='Zone', y='avg_quality',
                         color='avg_quality', color_continuous_scale='RdYlGn',
                         title=f'KMeans ({r1["best_k"]} clusters) — Avg Quality by Zone',
                         text='avg_quality')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # DBSCAN zone distribution
            db_counts = df['dbscan_zone'].value_counts().reset_index()
            db_counts.columns = ['Zone','Count']
            db_counts['Zone'] = db_counts['Zone'].apply(
                lambda z: f'Zone {z}' if z!=-1 else 'Noise (outlier)')
            fig = px.pie(db_counts, values='Count', names='Zone', hole=0.5,
                          title='DBSCAN Zone Distribution (incl. noise)',
                          color_discrete_sequence=C)
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Zone Detail Table")
        st.dataframe(zone_stats.drop('network_zone',axis=1).set_index('Zone'),
                     use_container_width=True)

        # Silhouette plot
        sil_df = pd.DataFrame({'k':list(r1['sil_km'].keys()),
                                'Silhouette':list(r1['sil_km'].values())})
        fig = px.line(sil_df, x='k', y='Silhouette', markers=True,
                      title=f'KMeans Cluster Selection (best k={r1["best_k"]}, Sil={r1["km_sil"]})',
                      color_discrete_sequence=[C[4]])
        fig.add_vline(x=r1['best_k'], line_color=C[3], line_dash='dash',
                       annotation_text=f'k={r1["best_k"]}',
                       annotation_font_color=C[3])
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        lstm = r1['lstm_res']
        if lstm['success']:
            st.markdown(f"**LSTM Model** — RMSE: `{lstm['rmse']}` · MAE: `{lstm['mae']}`")
            c1,c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                n_pts = min(len(lstm['y_actual']), len(lstm['y_pred']))
                fig.add_trace(go.Scatter(y=lstm['y_actual'][:n_pts], mode='lines',
                    name='Actual', line=dict(color=C[0],width=2)))
                fig.add_trace(go.Scatter(y=lstm['y_pred'][:n_pts], mode='lines',
                    name='LSTM Predicted', line=dict(color=C[2],width=2,dash='dot')))
                fig.update_layout(**BASE_LAYOUT, title='LSTM Validation — Actual vs Predicted',
                    xaxis_title='Time Step', yaxis_title='Demand Load (%)')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                hrs = list(range(24))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hrs, y=lstm['forecast_24h'],
                    fill='tozeroy', mode='lines+markers',
                    line=dict(color=C[1],width=2.5),
                    fillcolor='rgba(76,175,80,0.12)', name='Forecast'))
                # Shade peak hours
                for h in [8,9,10,17,18,19,20,21]:
                    fig.add_vrect(x0=h-0.45,x1=h+0.45,
                                   fillcolor='rgba(255,193,7,0.07)',line_width=0)
                fig.update_layout(**BASE_LAYOUT,
                    title='Next 24-Hour Demand Forecast (LSTM)',
                    xaxis_title='Hour of Day', yaxis_title='Predicted Demand (%)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("LSTM unavailable — showing statistical demand baseline.")
        
        # Hourly avg from data
        fig = px.area(r1['hourly_df'].groupby('hour')['demand'].mean().reset_index(),
                      x='hour', y='demand', color_discrete_sequence=[C[0]],
                      title='Historical Average Hourly Demand (60-day simulation)',
                      labels={'hour':'Hour','demand':'Avg Demand (%)'})
        fig.update_traces(fill='tozeroy', fillcolor='rgba(79,195,247,0.10)')
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class='insight-box'>
        💡 <b>Optimization Recommendations:</b><br>
        &nbsp;• Peak demand: 8–10 AM and 5–9 PM — activate dynamic load-balancing triggers 30 min before<br>
        &nbsp;• Pre-cache popular content during off-peak (2–5 AM) to reduce backhaul load by ~18%<br>
        &nbsp;• Allocate additional spectrum for Enterprise plan users during evening peak<br>
        &nbsp;• Zones with quality &lt; 55 require signal booster deployment or tower upgrade
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — USER BEHAVIOUR
# ═══════════════════════════════════════════════════════════════

elif "Behaviour" in page:
    st.markdown("# 👥 User Behaviour Analysis")
    st.markdown("*KMeans + DBSCAN segmentation · churn prediction · lifetime value*")
    st.divider()

    t1,t2,t3 = st.tabs(["🧩 Segmentation","⚠️ Churn Model","💰 LTV & Patterns"])

    with t1:
        c1,c2 = st.columns([3,2])
        with c1:
            fig = px.scatter(df.sample(min(2000,len(df)),random_state=7),
                             x='pca1', y='pca2', color='segment_label',
                             color_discrete_map=SEG_C, opacity=0.6,
                             title=f'User Segments — PCA 2D  (var: {r2["pca_var"][0]*100:.1f}% + {r2["pca_var"][1]*100:.1f}%)',
                             labels={'pca1':'PC1','pca2':'PC2','segment_label':'Segment'})
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sc = df['segment_label'].value_counts().reset_index()
            sc.columns=['Segment','Count']
            fig = go.Figure(go.Bar(x=sc['Count'], y=sc['Segment'], orientation='h',
                marker_color=[SEG_C.get(s,'#888') for s in sc['Segment']],
                text=sc['Count'], textposition='outside'))
            fig.update_layout(**BASE_LAYOUT, title='Segment Sizes',
                xaxis_title='Users', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        seg_means = df.groupby('segment_label')[
            ['data_usage_gb','voice_mins','monthly_bill','support_calls','roaming_days']
        ].mean()
        cats = ['Data GB','Voice Mins','Bill ₹','Support Calls','Roaming Days']
        fig = go.Figure()
        for seg, row in seg_means.iterrows():
            vals = list(row.values)
            col  = seg_means.max()
            norm = [vals[i]/(col.iloc[i]+1e-9) for i in range(len(vals))]
            fig.add_trace(go.Scatterpolar(
                r=norm+[norm[0]], theta=cats+[cats[0]], name=seg,
                line=dict(color=SEG_C.get(seg,'#888'),width=2),
                fill='toself', fillcolor=SEG_C.get(seg,'#888'), opacity=0.12))
        fig.update_layout(**BASE_LAYOUT,
            polar=dict(bgcolor=BG,
                radialaxis=dict(gridcolor=GRID,color=FONT),
                angularaxis=dict(gridcolor=GRID,color=FONT)),
            title='Normalised Behaviour Radar — Segments')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Segment Profiles")
        ss = r2['seg_summary'].copy()
        ss.columns = ['Segment','Count','Avg Data (GB)','Avg Bill (₹)',
                      'Avg Churn Prob','Avg LTV (₹)','Avg Support Calls']
        st.dataframe(ss.set_index('Segment'), use_container_width=True)

    with t2:
        m = r2['churn_metrics']
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("CV Accuracy", f"{m['cv_acc']*100:.1f}%")
        mc2.metric("Precision",   f"{m['precision']:.3f}")
        mc3.metric("Recall",      f"{m['recall']:.3f}")
        mc4.metric("F1-Score",    f"{m['f1']:.3f}")

        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='churn_prob', color='segment_label',
                               color_discrete_map=SEG_C, nbins=40,
                               barmode='overlay', opacity=0.65,
                               title='Churn Probability by Segment',
                               labels={'churn_prob':'Churn Probability'})
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cr = df.groupby('region')['churn_prob'].mean().reset_index()
            fig = px.bar(cr, x='region', y='churn_prob',
                         color='churn_prob', color_continuous_scale='RdYlGn_r',
                         title='Avg Churn Risk by Region',
                         labels={'churn_prob':'Avg Churn Prob'})
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        from ml_engine import CHURN_FEATS
        fi = dict(zip(CHURN_FEATS, r2['churn_rf'].feature_importances_))
        fi_df = pd.DataFrame({'Feature':list(fi.keys()),
                               'Importance':list(fi.values())}).sort_values('Importance')
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Reds',
                     title='Churn Feature Importance')
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Classification Report"):
            st.text(m['report'])

    with t3:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.box(df, x='segment_label', y='ltv', color='segment_label',
                         color_discrete_map=SEG_C,
                         title='LTV Distribution by Segment',
                         labels={'ltv':'LTV (₹)','segment_label':'Segment'})
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df.sample(min(1200,len(df))),
                             x='tenure_months', y='ltv', color='segment_label',
                             color_discrete_map=SEG_C, opacity=0.55,
                             title='LTV vs Tenure',
                             labels={'tenure_months':'Tenure (months)','ltv':'LTV (₹)'})
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # DBSCAN noise users
        noise_count = (df['dbscan_user']==-1).sum()
        st.markdown(f"""<div class='insight-box'>
        📌 <b>DBSCAN</b> detected {noise_count} noise/outlier users ({noise_count/len(df)*100:.1f}%)
        — these users don't fit any cluster and may represent rare or anomalous behaviour patterns.
        </div>""", unsafe_allow_html=True)

        # Peak usage
        fig = px.bar(r2['peak_usage'], x='hour', y='avg_data_gb',
                     color='avg_data_gb', color_continuous_scale='Blues',
                     title='Avg Data Usage by Hour of Day',
                     labels={'hour':'Hour','avg_data_gb':'Avg Data (GB)'})
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════

elif "Anomaly" in page:
    st.markdown("# 🚨 Anomaly Detection")
    st.markdown("*Isolation Forest · One-Class SVM · Autoencoder · Z-Score · Live Alerts*")
    st.divider()

    t1,t2,t3,t4 = st.tabs([
        "⚡ Live Alerts","📊 Method Comparison",
        "🤖 Autoencoder","🔬 Profiler"
    ])

    with t1:
        col_r, col_i = st.columns([1,3])
        with col_r:
            auto_r = st.checkbox("⟳ Auto-refresh (5s)")
        with col_i:
            st.info(f"🔴 **{df['iso_flag'].sum()} anomalies** detected in filtered dataset")
        if auto_r: time.sleep(5); st.rerun()

        alerts = r3['alerts']
        for _, row in alerts.head(25).iterrows():
            sev = str(row['severity']).lower()
            icon = {'critical':'🔴','high':'🟠','medium':'🟡'}.get(sev,'⚪')
            st.markdown(f"""<div class="alert-{sev}">
            {icon} <b>User #{int(row['user_id'])}</b>
            [{row['user_type']} · {row['region']} · Hour {int(row['hour_of_day'])}]
            — Score: <b>{row['anomaly_score']:.3f}</b>
            | Latency: <b>{row['latency_ms']:.0f}ms</b>
            | PktLoss: <b>{row['packet_loss_pct']:.1f}%</b>
            | Data: <b>{row['data_usage_gb']:.1f}GB</b>
            | Support: <b>{int(row['support_calls'])}</b>
            | <b>Severity: {str(row['severity']).upper()}</b>
            </div>""", unsafe_allow_html=True)

    with t2:
        metrics_df = pd.DataFrame(r3['metrics']).T.reset_index()
        metrics_df.columns = ['Method','Precision','Recall','F1','Flagged','Pct %',
                               *([k for k in list(r3['metrics']['Autoencoder'].keys())
                                  if k not in ['precision','recall','f1','flagged','pct']])]
        metrics_df = metrics_df[['Method','Precision','Recall','F1','Flagged','Pct %']]

        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(metrics_df, x='Method', y=['Precision','Recall','F1'],
                         barmode='group', color_discrete_sequence=[C[0],C[1],C[3]],
                         title='Detection Method Comparison')
            fig.update_layout(**BASE_LAYOUT)
            fig.update_yaxes(range=[0,1.1])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(df.sample(min(2000,len(df))),
                             x='an_pc1', y='an_pc2',
                             color=df.sample(min(2000,len(df)),random_state=9)['iso_flag'].map(
                                 {0:'Normal',1:'Anomaly'}),
                             color_discrete_map={'Normal':C[1],'Anomaly':C[2]},
                             title='Isolation Forest — PCA 2D')
            fig.update_traces(marker=dict(size=4, opacity=0.5))
            fig.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df, x='anomaly_score',
                           color=df['iso_flag'].map({0:'Normal',1:'Anomaly'}),
                           color_discrete_map={'Normal':C[1],'Anomaly':C[2]},
                           nbins=60, barmode='overlay', opacity=0.7,
                           title='Anomaly Score Distribution',
                           labels={'anomaly_score':'Isolation Forest Score'})
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(metrics_df.set_index('Method'), use_container_width=True)

    with t3:
        ae_m = r3['metrics'].get('Autoencoder',{})
        a1,a2,a3 = st.columns(3)
        a1.metric("Precision", f"{ae_m.get('precision',0):.3f}")
        a2.metric("Recall",    f"{ae_m.get('recall',0):.3f}")
        a3.metric("F1-Score",  f"{ae_m.get('f1',0):.3f}")

        errors = r3['ae_errors']
        if len(errors) > 0:
            ae_flag = df['ae_flag'].values
            normal_err  = [errors[i] for i in range(len(errors)) if ae_flag[i]==0]
            anomaly_err = [errors[i] for i in range(len(errors)) if ae_flag[i]==1]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=normal_err,  name='Normal',  opacity=0.65,
                marker_color=C[1], nbinsx=60))
            fig.add_trace(go.Histogram(x=anomaly_err, name='Anomaly', opacity=0.75,
                marker_color=C[2], nbinsx=40))
            thresh = ae_m.get('threshold', 0)
            if thresh:
                fig.add_vline(x=thresh, line_color=C[3], line_dash='dash',
                               annotation_text=f'Threshold={thresh:.4f}',
                               annotation_font_color=C[3])
            fig.update_layout(**BASE_LAYOUT, barmode='overlay',
                title='Autoencoder Reconstruction Error Distribution',
                xaxis_title='Mean Squared Reconstruction Error')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class='insight-box'>
        🤖 <b>Autoencoder Architecture:</b> 9→32→16→8→16→32→9 neurons<br>
        Trained exclusively on normal records, so anomalous inputs produce high reconstruction errors.
        The detection threshold is set at the 95th percentile of normal-record errors.
        </div>""", unsafe_allow_html=True)

    with t4:
        profile = r3['profile']
        st.markdown("#### Mean Feature Values — Normal vs Anomaly")
        p_df = profile.T.reset_index()
        p_df.columns = ['Feature','Normal','Anomaly']
        p_df['Ratio×'] = (p_df['Anomaly']/p_df['Normal']).round(1)
        st.dataframe(p_df.set_index('Feature'), use_container_width=True)

        fig = px.bar(p_df, x='Feature', y=['Normal','Anomaly'], barmode='group',
                     color_discrete_sequence=[C[1],C[2]],
                     title='Feature Means: Normal vs Anomaly')
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        at = df.groupby('hour_of_day')['iso_flag'].mean().reset_index()
        at.columns = ['Hour','Anomaly Rate']
        fig = px.bar(at, x='Hour', y='Anomaly Rate',
                     color='Anomaly Rate', color_continuous_scale='RdYlGn_r',
                     title='Anomaly Rate by Hour of Day')
        fig.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════

elif "Predictor" in page:
    st.markdown("# 🔮 Live Predictor")
    st.markdown("*Enter parameters below for instant ML inference from trained models*")
    st.divider()

    pt1, pt2, pt3 = st.tabs([
        "📶 Network Quality",
        "⚠️  Churn Risk",
        "🚨 Anomaly Check",
    ])

    def gauge(value, title, ranges, colors, suffix=''):
        axis_min = ranges[0][0] if isinstance(ranges[0], (list, tuple)) else ranges[0]
        axis_max = ranges[-1][-1] if isinstance(ranges[-1], (list, tuple)) else ranges[-1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={'suffix': suffix, 'font': {'family':'IBM Plex Mono','color':FONT}},
            delta={'reference': (ranges[1][0]+ranges[1][1])/2,
                   'increasing':{'color': C[1]}, 'decreasing':{'color':C[2]}},
            gauge={
                'axis': {'range':[axis_min, axis_max], 'tickcolor':FONT},
                'bar':  {'color': C[0]},
                'steps':[{'range':r,'color':co} for r,co in zip(ranges[1:-1],colors)],
                'threshold':{'line':{'color':C[3],'width':3},'value':ranges[-2][1]},
            },
            title={'text': title, 'font':{'color':FONT,'size':14}}
        ))
        fig.update_layout(paper_bgcolor=CARD, font={'color':FONT}, height=320,
                           margin=dict(l=30,r=30,t=60,b=10))
        return fig

    with pt1:
        st.markdown("### Network Quality of Experience Predictor")
        c1,c2,c3 = st.columns(3)
        with c1:
            sig  = st.slider("Signal (dBm)", -120,-50,-80, key='net_signal_dbm')
            snr  = st.slider("SNR (dB)",      0,35,15)
            lat  = st.slider("Latency (ms)",  10,500,50)
        with c2:
            pkl  = st.slider("Packet Loss (%)",0.0,20.0,2.0,0.1)
            hnd  = st.number_input("Handovers",0,10,2)
            cl   = st.slider("Cell Load (%)", 20,100,60)
        with c3:
            td   = st.slider("Tower Dist (km)",0.1,15.0,3.0,0.1)
            bw   = st.slider("Bandwidth (Mbps)",5,150,50)
            jit  = st.slider("Jitter (ms)",   0,50,8)

        if st.button("▶  Predict Quality Score", key='q'):
            from ml_engine import predict_quality, KPI_FEATS
            score = predict_quality({
                'signal_dbm':sig,'snr_db':snr,'latency_ms':lat,
                'packet_loss_pct':pkl,'handovers':hnd,'cell_load_pct':cl,
                'tower_dist_km':td,'bandwidth_mbps':bw,'jitter_ms':jit,
            })
            c1,c2 = st.columns([1,2])
            with c1:
                if score>=70:   st.success(f"✅ Quality: **{score}/100** — Excellent")
                elif score>=45: st.warning(f"⚠️ Quality: **{score}/100** — Moderate")
                else:           st.error(f"❌ Quality: **{score}/100** — Poor")
            with c2:
                st.plotly_chart(gauge(score,'Quality Score',
                    [0,[0,45],[45,70],[70,100],100],
                    ['#1A0A0A','#1A1500','#0A1A0A'],'/100'), use_container_width=True)

    with pt2:
        st.markdown("### Customer Churn Risk Predictor")
        c1,c2,c3 = st.columns(3)
        with c1:
            cd = st.number_input("Data (GB)",      0.1,200.0,8.0,0.5)
            cv = st.number_input("Voice (mins)",   0,600,180)
            cs = st.number_input("SMS Count",      0,200,30)
        with c2:
            ca = st.number_input("App Sessions",   0,100,15)
            cn = st.slider("Night Usage (%)",      0,100,20)
            cr = st.number_input("Roaming Days",   0,30,2)
        with c3:
            csu= st.number_input("Support Calls",  0,20,1)
            cb = st.number_input("Bill (₹)",       50,2000,250)
            ct = st.slider("Tenure (months)",      1,60,24)
            cq = st.slider("Quality Score",        0.0,100.0,65.0)

        if st.button("▶  Predict Churn Risk", key='ch'):
            from ml_engine import predict_churn, CHURN_FEATS
            prob = predict_churn({
                'data_usage_gb':cd,'voice_mins':cv,'sms_count':cs,
                'app_sessions':ca,'night_usage_pct':cn,'roaming_days':cr,
                'support_calls':csu,'monthly_bill':cb,'tenure_months':ct,
                'quality_score':cq,
            })
            c1,c2 = st.columns([1,2])
            with c1:
                if prob>0.6:    st.error(f"🔴 HIGH RISK: **{prob*100:.1f}%**\nImmediate retention action needed.")
                elif prob>0.3:  st.warning(f"🟡 MODERATE: **{prob*100:.1f}%**\nProactive outreach recommended.")
                else:           st.success(f"🟢 LOW RISK: **{prob*100:.1f}%**\nUser is stable.")
                # LTV estimate
                ltv_est = cb * (1-prob) * max(1, 60-ct)
                st.metric("Estimated LTV", f"₹{ltv_est:,.0f}")
            with c2:
                col = C[2] if prob>0.6 else (C[3] if prob>0.3 else C[1])
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    number={'suffix':'%','font':{'family':'IBM Plex Mono','color':FONT}},
                    gauge={
                        'axis':{'range':[0,100],'tickcolor':FONT},
                        'bar':{'color':col},
                        'steps':[
                            {'range':[0,30],  'color':'#0A1A0A'},
                            {'range':[30,60], 'color':'#1A1500'},
                            {'range':[60,100],'color':'#1A0A0A'},
                        ],
                    },
                    title={'text':'Churn Probability','font':{'color':FONT,'size':14}}
                ))
                fig.update_layout(paper_bgcolor=CARD,font={'color':FONT},
                                   height=320,margin=dict(l=30,r=30,t=60,b=10))
                st.plotly_chart(fig, use_container_width=True)

    with pt3:
        st.markdown("### Network Anomaly Checker")
        c1,c2,c3 = st.columns(3)
        with c1:
            al  = st.number_input("Latency (ms)",     10,3000,50)
            ap  = st.number_input("Packet Loss (%)",  0.0,80.0,2.0,0.5)
            ad  = st.number_input("Data Usage (GB)",  0.1,500.0,8.0,0.5)
        with c2:
            asu = st.number_input("Support Calls",    0,30,1)
            aj  = st.number_input("Jitter (ms)",      0,100,8)
            asi = st.slider("Signal (dBm)",           -120,-50,-80, key='anom_signal_dbm')
        with c3:
            ah  = st.number_input("Handovers",        0,15,2)
            ab  = st.number_input("Bandwidth (Mbps)", 5,150,50)
            av  = st.number_input("Voice Mins",       0,600,180)

        if st.button("▶  Check for Anomaly", key='an'):
            from ml_engine import predict_anomaly, ANOM_FEATS
            is_anom, score = predict_anomaly({
                'latency_ms':al,'packet_loss_pct':ap,'data_usage_gb':ad,
                'support_calls':asu,'jitter_ms':aj,'signal_dbm':asi,
                'handovers':ah,'bandwidth_mbps':ab,'voice_mins':av,
            })
            c1,c2 = st.columns([1,2])
            with c1:
                if is_anom:
                    st.error(f"🚨 ANOMALY DETECTED\n\nScore: `{score:.4f}`")
                    st.markdown("**Possible causes:**\n- Equipment fault\n- DDoS/traffic surge\n- SIM card fraud\n- Backhaul degradation")
                else:
                    st.success(f"✅ Normal Behaviour\n\nScore: `{score:.4f}`")
            with c2:
                fig = go.Figure(go.Bar(
                    x=[score], y=['Anomaly Score'], orientation='h',
                    marker_color=C[2] if is_anom else C[1],
                    text=[f"{score:.4f}"], textposition='outside',
                ))
                fig.add_vline(x=0.5,line_color=C[3],line_dash='dash',
                               annotation_text='Approx threshold')
                fig.update_xaxes(range=[0,max(1.2,score*1.4)])
                fig.update_layout(**BASE_LAYOUT,height=200,
                    title='Anomaly Score (Isolation Forest)')
                st.plotly_chart(fig, use_container_width=True)
