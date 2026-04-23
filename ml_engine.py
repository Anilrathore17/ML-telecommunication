"""
ml_engine.py  —  Telecom Intelligence System v2
================================================
Algorithms:
  Module 1 – Network Optimization
    • Ridge Regression, Random Forest, Gradient Boosting   (quality prediction)
    • KMeans + DBSCAN                                       (zone clustering)
    • LSTM (Keras)                                          (hourly demand forecast)

  Module 2 – User Behaviour Analysis
    • KMeans + DBSCAN                                       (user segmentation)
    • Random Forest Classifier                              (churn prediction)
    • LTV estimation                                        (monetary value model)

  Module 3 – Anomaly Detection
    • Isolation Forest                                      (ensemble unsupervised)
    • One-Class SVM                                         (kernel boundary)
    • Autoencoder (Keras)                                   (deep reconstruction error)
    • Z-Score baseline
"""

import os, pickle, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                              classification_report, f1_score,
                              precision_score, recall_score, silhouette_score)

# ── paths ─────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_RAW   = os.path.join(BASE, 'data', 'telecom_dataset.csv')
DATA_RICH  = os.path.join(BASE, 'data', 'telecom_enriched.csv')
MODEL_DIR  = os.path.join(BASE, 'models')
PIPELINE_ARTIFACTS = os.path.join(MODEL_DIR, 'pipeline_artifacts.pkl')
PIPELINE_ARTIFACTS_JSON = os.path.join(MODEL_DIR, 'pipeline_artifacts.json')
os.makedirs(os.path.join(BASE,'data'),  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _file_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _required_artifact_paths():
    # Keep in sync with what training saves + what inference loads.
    return [
        DATA_RICH,
        PIPELINE_ARTIFACTS_JSON,
        os.path.join(MODEL_DIR, 'net_reg.pkl'),
        os.path.join(MODEL_DIR, 'net_km.pkl'),
        os.path.join(MODEL_DIR, 'net_scaler.pkl'),
        os.path.join(MODEL_DIR, 'churn_rf.pkl'),
        os.path.join(MODEL_DIR, 'user_km.pkl'),
        os.path.join(MODEL_DIR, 'beh_sc.pkl'),
        os.path.join(MODEL_DIR, 'iso.pkl'),
        os.path.join(MODEL_DIR, 'ocsvm.pkl'),
        os.path.join(MODEL_DIR, 'an_sc.pkl'),
        os.path.join(MODEL_DIR, 'autoencoder.keras'),
        os.path.join(MODEL_DIR, 'ae_threshold.pkl'),
        os.path.join(MODEL_DIR, 'lstm_demand.keras'),
        os.path.join(MODEL_DIR, 'lstm_scaler.pkl'),
    ]


def artifacts_available() -> bool:
    return all(_file_exists(p) for p in _required_artifact_paths())


def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _df_to_records(df: pd.DataFrame):
    return df.to_dict(orient='records')


def _records_to_df(records):
    return pd.DataFrame.from_records(records) if records is not None else pd.DataFrame()


def _strip_training_objects(r1: dict, r2: dict, r3: dict):
    """Remove large/non-essential objects so artifacts are stable and lightweight."""
    r1 = dict(r1)
    r2 = dict(r2)
    r3 = dict(r3)

    # r1: drop model objects; keep metrics + predictions
    reg_res = {}
    for name, rd in r1.get('reg_res', {}).items():
        reg_res[name] = {
            'r2': _to_jsonable(rd.get('r2')),
            'mae': _to_jsonable(rd.get('mae')),
            'rmse': _to_jsonable(rd.get('rmse')),
            'y_test': _to_jsonable(rd.get('y_test')),
            'y_pred': _to_jsonable(rd.get('y_pred')),
        }
    r1['reg_res'] = reg_res
    r1.pop('km_zone', None)
    r1.pop('df', None)
    if 'hourly_df' in r1 and isinstance(r1['hourly_df'], pd.DataFrame):
        r1['hourly_df'] = _df_to_records(r1['hourly_df'])

    # r2: drop model objects
    r2.pop('churn_rf', None)
    r2.pop('km_user', None)
    r2.pop('df', None)
    if 'seg_summary' in r2:
        r2['seg_summary'] = _df_to_records(r2['seg_summary'])
    if 'peak_usage' in r2:
        r2['peak_usage'] = _df_to_records(r2['peak_usage'])

    # r3: drop model objects
    r3.pop('iso', None)
    r3.pop('ocsvm', None)
    r3.pop('df', None)
    if 'profile' in r3 and isinstance(r3['profile'], (pd.DataFrame, pd.Series)):
        # profile is a DataFrame indexed by label; store as records plus index column
        prof = r3['profile'].reset_index().rename(columns={'index': 'Label'})
        r3['profile'] = _df_to_records(prof)
    if 'alerts' in r3 and isinstance(r3['alerts'], pd.DataFrame):
        r3['alerts'] = _df_to_records(r3['alerts'])
    if 'ae_errors' in r3:
        r3['ae_errors'] = _to_jsonable(r3['ae_errors'])

    return r1, r2, r3


def load_artifacts():
    """Fast path for production: load enriched dataset + precomputed UI metrics."""
    if not artifacts_available():
        raise FileNotFoundError("Required artifacts are missing.")
    df = pd.read_csv(DATA_RICH)
    with open(PIPELINE_ARTIFACTS_JSON, 'r', encoding='utf-8') as f:
        art = json.load(f)

    r1 = art['r1']
    r2 = art['r2']
    r3 = art['r3']

    # Restore DataFrames that app.py expects
    if isinstance(r1.get('hourly_df'), list):
        r1['hourly_df'] = _records_to_df(r1['hourly_df'])
    if isinstance(r2.get('seg_summary'), list):
        r2['seg_summary'] = _records_to_df(r2['seg_summary'])
    if isinstance(r2.get('peak_usage'), list):
        r2['peak_usage'] = _records_to_df(r2['peak_usage'])
    if isinstance(r3.get('alerts'), list):
        r3['alerts'] = _records_to_df(r3['alerts'])
    if isinstance(r3.get('profile'), list):
        prof_df = _records_to_df(r3['profile'])
        if 'Label' in prof_df.columns:
            prof_df = prof_df.set_index('Label')
        r3['profile'] = prof_df

    return df, r1, r2, r3


def train_and_save_artifacts(n=5000, force=False):
    df, r1, r2, r3 = _train_pipeline(n=n, force=force)

    # Save enriched dataset
    df.to_csv(DATA_RICH, index=False)

    # Save lightweight UI artifacts
    r1_s, r2_s, r3_s = _strip_training_objects(r1, r2, r3)
    with open(PIPELINE_ARTIFACTS_JSON, 'w', encoding='utf-8') as f:
        json.dump({'r1': r1_s, 'r2': r2_s, 'r3': r3_s}, f)
    return df, r1_s, r2_s, r3_s


def get_pipeline(n=5000, force=False):
    """Main entry used by Streamlit: load artifacts if present, otherwise train."""
    if not force and artifacts_available():
        print("[0] Loading precomputed artifacts…")
        return load_artifacts()
    print("[0] Artifacts missing or force=True — training pipeline…")
    return train_and_save_artifacts(n=n, force=True)

# ─────────────────────────────────────────────────────────────
#  SECTION 0: DATA GENERATION
# ─────────────────────────────────────────────────────────────

def generate_dataset(n=5000, seed=42):
    np.random.seed(seed)

    user_type  = np.random.choice(['Heavy','Moderate','Light','IoT'], n, p=[.20,.40,.30,.10])
    plan_type  = np.random.choice(['Prepaid','Postpaid','Enterprise'], n, p=[.45,.40,.15])
    region     = np.random.choice(['North','South','East','West','Central'], n)
    tenure     = np.random.randint(1, 60, n)
    hour       = np.random.randint(0, 24, n)

    # Network KPIs
    signal     = np.random.uniform(-120, -50, n)
    snr        = np.clip(np.random.normal(15,7,n), 0, 35)
    latency    = np.random.exponential(40,n) + 10
    pkt_loss   = np.clip(np.random.exponential(2,n), 0, 20)
    handovers  = np.random.poisson(2, n)
    cell_load  = np.random.uniform(20, 100, n)
    tower_dist = np.random.exponential(3,n) + 0.5
    bandwidth  = np.random.uniform(5, 150, n)
    jitter     = np.clip(np.random.exponential(8,n), 0, 50)

    # Behaviour
    base_data  = {'Heavy':25,'Moderate':8,'Light':2,'IoT':0.5}
    data_gb    = np.array([np.random.normal(base_data[u], base_data[u]*0.3)
                            for u in user_type]).clip(0.1)
    voice_min  = np.clip(np.random.normal(180,90,n), 0, None)
    sms        = np.random.poisson(30, n)
    sessions   = np.random.poisson(15, n)
    night_pct  = np.random.uniform(5, 60, n)
    roaming    = np.random.poisson(1.5, n)
    support    = np.random.poisson(0.8, n)
    bill       = (data_gb*3 + voice_min*0.01 + sms*0.02 +
                  np.random.normal(200,40,n)).clip(50)

    # Quality composite
    quality = (
        0.30*np.interp(signal,    [-120,-50],[0,100]) +
        0.25*np.interp(snr,       [0,35],    [0,100]) +
        0.20*np.interp(latency,   [300,10],  [0,100]) +
        0.15*np.interp(pkt_loss,  [20,0],    [0,100]) +
        0.10*np.interp(cell_load, [100,20],  [0,100]) +
        np.random.normal(0,3,n)
    ).clip(0,100)

    # Demand with realistic time-of-day pattern
    peak_flag  = np.isin(hour, [8,9,10,17,18,19,20,21])
    demand     = (cell_load + peak_flag*15 + np.random.normal(0,5,n)).clip(0,100)

    # Inject anomalies ~5%
    is_anomaly = np.zeros(n, dtype=int)
    anom_idx   = np.random.choice(n, int(0.05*n), replace=False)
    is_anomaly[anom_idx] = 1
    latency[anom_idx]  *= np.random.uniform(4,9,   len(anom_idx))
    pkt_loss[anom_idx] *= np.random.uniform(4,12,  len(anom_idx))
    data_gb[anom_idx]  *= np.random.uniform(6,18,  len(anom_idx))
    support[anom_idx]  += np.random.randint(5,15,  len(anom_idx))

    # Churn proxy
    churn = ((support > 3) | (bill > 650)).astype(int)

    df = pd.DataFrame({
        'user_id': np.arange(1,n+1),
        'user_type': user_type, 'plan_type': plan_type,
        'region': region, 'tenure_months': tenure, 'hour_of_day': hour,
        'signal_dbm': signal, 'snr_db': snr, 'latency_ms': latency,
        'packet_loss_pct': pkt_loss, 'handovers': handovers,
        'cell_load_pct': cell_load, 'tower_dist_km': tower_dist,
        'bandwidth_mbps': bandwidth, 'jitter_ms': jitter,
        'data_usage_gb': data_gb, 'voice_mins': voice_min,
        'sms_count': sms, 'app_sessions': sessions,
        'night_usage_pct': night_pct, 'roaming_days': roaming,
        'support_calls': support, 'monthly_bill': bill,
        'demand_load': demand, 'quality_score': quality,
        'churn_risk': churn, 'is_anomaly': is_anomaly,
    })
    df.to_csv(DATA_RAW, index=False)
    return df


# ── Build hourly time-series for LSTM ────────────────────────
def build_hourly_series(df, n_days=60):
    """Aggregate mean demand per hour across n_days of simulated days."""
    np.random.seed(99)
    records = []
    for day in range(n_days):
        for h in range(24):
            base = df[df['hour_of_day']==h]['demand_load'].mean()
            noise = np.random.normal(0, 3)
            weekday_bump = 5 if day % 7 < 5 else -5
            records.append({'day': day, 'hour': h,
                             'demand': float(np.clip(base + noise + weekday_bump, 0, 100))})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
#  SECTION 1: NETWORK OPTIMIZATION
# ─────────────────────────────────────────────────────────────

KPI_FEATS = ['signal_dbm','snr_db','latency_ms','packet_loss_pct',
             'handovers','cell_load_pct','tower_dist_km','bandwidth_mbps','jitter_ms']

def train_network_module(df):
    print("  [M1] Training network optimization models…")
    X = df[KPI_FEATS]; y = df['quality_score']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

    # ── 1a. Regression ────────────────────────────────────────
    regs = {
        'Ridge':    Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=150, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
    }
    reg_res = {}
    for name, m in regs.items():
        m.fit(Xtr, ytr); yp = m.predict(Xte)
        reg_res[name] = {
            'model': m, 'r2': round(r2_score(yte,yp),4),
            'mae': round(mean_absolute_error(yte,yp),3),
            'rmse': round(np.sqrt(mean_squared_error(yte,yp)),3),
            'y_test': yte.values, 'y_pred': yp,
        }
    best_name = max(reg_res, key=lambda k: reg_res[k]['r2'])

    # ── 1b. KMeans zone clustering ────────────────────────────
    scaler_kpi = StandardScaler()
    Xc = scaler_kpi.fit_transform(df[KPI_FEATS+['quality_score']])
    sil_km = {}
    for k in range(2,8):
        lbl = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xc)
        sil_km[k] = round(silhouette_score(Xc,lbl),3)
    best_k = max(sil_km, key=sil_km.get)
    km_zone = KMeans(n_clusters=best_k,random_state=42,n_init=10)
    df['network_zone'] = km_zone.fit_predict(Xc)
    km_sil = sil_km[best_k]

    # ── 1c. DBSCAN zone (alt) ─────────────────────────────────
    db_zone = DBSCAN(eps=1.2, min_samples=20)
    df['dbscan_zone'] = db_zone.fit_predict(Xc)

    # ── 1d. LSTM demand forecast ──────────────────────────────
    hourly_df = build_hourly_series(df, n_days=60)
    lstm_res  = train_lstm_forecast(hourly_df)

    # Feature importance (RF)
    rf = reg_res['RandomForest']['model']
    feat_imp = dict(zip(KPI_FEATS, rf.feature_importances_.round(4)))

    # Save
    pickle.dump(reg_res[best_name]['model'],
                open(os.path.join(MODEL_DIR,'net_reg.pkl'),'wb'))
    pickle.dump(km_zone,   open(os.path.join(MODEL_DIR,'net_km.pkl'),'wb'))
    pickle.dump(scaler_kpi,open(os.path.join(MODEL_DIR,'net_scaler.pkl'),'wb'))

    print(f"     Best regressor: {best_name}  R²={reg_res[best_name]['r2']}")
    return {
        'reg_res': reg_res, 'best_name': best_name,
        'km_zone': km_zone, 'best_k': best_k,
        'sil_km': sil_km, 'km_sil': km_sil,
        'dbscan_zones': df['dbscan_zone'].value_counts().to_dict(),
        'feat_imp': feat_imp, 'hourly_df': hourly_df,
        'lstm_res': lstm_res, 'df': df,
    }


def train_lstm_forecast(hourly_df, look_back=24):
    """Train a simple LSTM to forecast next-hour demand."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        tf.get_logger().setLevel('ERROR')

        series = hourly_df['demand'].values.astype(np.float32)
        scaler = MinMaxScaler()
        series_sc = scaler.fit_transform(series.reshape(-1,1)).flatten()

        # Build sequences
        X_seq, y_seq = [], []
        for i in range(look_back, len(series_sc)):
            X_seq.append(series_sc[i-look_back:i])
            y_seq.append(series_sc[i])
        X_seq = np.array(X_seq).reshape(-1, look_back, 1)
        y_seq = np.array(y_seq)

        split = int(len(X_seq)*0.8)
        Xtr,Xte = X_seq[:split], X_seq[split:]
        ytr,yte = y_seq[:split], y_seq[split:]

        model = Sequential([
            LSTM(64, input_shape=(look_back,1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(Xtr, ytr, epochs=20, batch_size=32,
                  validation_split=0.1, verbose=0)

        # Predictions
        yp_sc = model.predict(Xte, verbose=0).flatten()
        yp    = scaler.inverse_transform(yp_sc.reshape(-1,1)).flatten()
        yt    = scaler.inverse_transform(yte.reshape(-1,1)).flatten()

        # Forecast next 24 hours
        last_seq = series_sc[-look_back:].reshape(1,-1,1)
        forecast = []
        for _ in range(24):
            pred = model.predict(last_seq, verbose=0)[0,0]
            forecast.append(float(scaler.inverse_transform([[pred]])[0,0]))
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0,-1,0] = pred

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae  = float(mean_absolute_error(yt, yp))

        # Save model weights
        model.save(os.path.join(MODEL_DIR,'lstm_demand.keras'))
        pickle.dump(scaler, open(os.path.join(MODEL_DIR,'lstm_scaler.pkl'),'wb'))

        print(f"     LSTM trained  RMSE={rmse:.3f}  MAE={mae:.3f}")
        return {
            'success': True, 'rmse': round(rmse,3), 'mae': round(mae,3),
            'y_actual': yt.tolist(), 'y_pred': yp.tolist(),
            'forecast_24h': forecast,
            'hourly_avg': hourly_df.groupby('hour')['demand'].mean().tolist(),
        }

    except Exception as e:
        print(f"     LSTM skipped ({e})")
        # Fallback: ARIMA-like sinusoidal forecast
        hours = np.arange(24)
        avg   = hourly_df.groupby('hour')['demand'].mean().values
        return {
            'success': False, 'rmse': None, 'mae': None,
            'y_actual': [], 'y_pred': [],
            'forecast_24h': avg.tolist(),
            'hourly_avg': avg.tolist(),
        }


# ─────────────────────────────────────────────────────────────
#  SECTION 2: USER BEHAVIOUR ANALYSIS
# ─────────────────────────────────────────────────────────────

BEH_FEATS   = ['data_usage_gb','voice_mins','sms_count','app_sessions',
               'night_usage_pct','roaming_days','support_calls',
               'monthly_bill','tenure_months']
CHURN_FEATS = BEH_FEATS + ['quality_score']

SEG_MAP = {
    'premium':    'Premium Power User',
    'streamer':   'Data-Heavy Streamer',
    'at_risk':    'At-Risk / Frustrated',
    'light':      'Light / Occasional',
    'mainstream': 'Mainstream',
}

def _auto_label(row, means):
    if row['monthly_bill'] > means['monthly_bill']*1.4 and \
       row['data_usage_gb'] > means['data_usage_gb']*1.3:
        return 'Premium Power User'
    if row['data_usage_gb'] > means['data_usage_gb']*2:
        return 'Data-Heavy Streamer'
    if row['support_calls'] > means['support_calls']*2.5:
        return 'At-Risk / Frustrated'
    if row['data_usage_gb'] < means['data_usage_gb']*0.4:
        return 'Light / Occasional'
    return 'Mainstream'


def train_behaviour_module(df):
    print("  [M2] Training user behaviour models…")

    scaler_beh = StandardScaler()
    Xb = scaler_beh.fit_transform(df[BEH_FEATS])

    # ── 2a. KMeans segmentation ───────────────────────────────
    sil_km = {}
    for k in range(2,8):
        sil_km[k] = round(silhouette_score(Xb,
                    KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xb)),3)
    best_k = max(sil_km, key=sil_km.get)
    km_user = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['user_segment'] = km_user.fit_predict(Xb)
    means = df[BEH_FEATS].mean()
    df['segment_label'] = df.apply(_auto_label, axis=1, means=means)

    # ── 2b. DBSCAN (noise = rare users) ──────────────────────
    db_user = DBSCAN(eps=0.8, min_samples=15)
    df['dbscan_user'] = db_user.fit_predict(Xb)

    # ── 2c. PCA 2D ────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    Xb2 = pca.fit_transform(Xb)
    df['pca1'] = Xb2[:,0]; df['pca2'] = Xb2[:,1]
    pca_var = pca.explained_variance_ratio_.tolist()

    # ── 2d. Churn prediction ──────────────────────────────────
    Xc = df[CHURN_FEATS]; yc = df['churn_risk']
    Xct,Xce,yct,yce = train_test_split(Xc,yc,test_size=0.2,
                                        random_state=42,stratify=yc)
    churn_rf = RandomForestClassifier(n_estimators=150,
                                       class_weight='balanced',random_state=42)
    churn_rf.fit(Xct, yct)
    ycp   = churn_rf.predict(Xce)
    cv    = cross_val_score(churn_rf,Xc,yc,cv=5,scoring='accuracy').mean()
    churn_metrics = {
        'accuracy':  round((ycp==yce).mean(),4),
        'cv_acc':    round(cv,4),
        'precision': round(precision_score(yce,ycp,zero_division=0),4),
        'recall':    round(recall_score(yce,ycp,zero_division=0),4),
        'f1':        round(f1_score(yce,ycp,zero_division=0),4),
        'report':    classification_report(yce,ycp,
                         target_names=['Retain','Churn'],zero_division=0),
    }

    # ── 2e. LTV ───────────────────────────────────────────────
    churn_prob = churn_rf.predict_proba(Xc)[:,1]
    df['churn_prob'] = churn_prob
    remaining  = (1-churn_prob) * (60 - df['tenure_months']).clip(1)
    df['ltv']  = (df['monthly_bill'] * remaining).round(0)

    # Segment summary
    seg_summary = df.groupby('segment_label').agg(
        count=('user_id','count'),
        avg_data=('data_usage_gb','mean'),
        avg_bill=('monthly_bill','mean'),
        avg_churn=('churn_prob','mean'),
        avg_ltv=('ltv','mean'),
        avg_support=('support_calls','mean'),
    ).round(2).reset_index()

    # Peak hour usage
    peak = df.groupby('hour_of_day')['data_usage_gb'].mean().reset_index()
    peak.columns = ['hour','avg_data_gb']

    # Save
    pickle.dump(churn_rf,    open(os.path.join(MODEL_DIR,'churn_rf.pkl'),'wb'))
    pickle.dump(km_user,     open(os.path.join(MODEL_DIR,'user_km.pkl'), 'wb'))
    pickle.dump(scaler_beh,  open(os.path.join(MODEL_DIR,'beh_sc.pkl'),  'wb'))

    print(f"     Churn CV acc={churn_metrics['cv_acc']}  F1={churn_metrics['f1']}")
    return {
        'km_user': km_user, 'best_k': best_k, 'sil_km': sil_km,
        'churn_rf': churn_rf, 'churn_metrics': churn_metrics,
        'seg_summary': seg_summary, 'pca_var': pca_var,
        'peak_usage': peak, 'df': df,
    }


# ─────────────────────────────────────────────────────────────
#  SECTION 3: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────

ANOM_FEATS = ['latency_ms','packet_loss_pct','data_usage_gb',
              'support_calls','jitter_ms','signal_dbm',
              'handovers','bandwidth_mbps','voice_mins']

def train_anomaly_module(df):
    print("  [M3] Training anomaly detection models…")

    scaler_an = StandardScaler()
    Xa = scaler_an.fit_transform(df[ANOM_FEATS])
    gt = df['is_anomaly']

    # ── 3a. Isolation Forest ──────────────────────────────────
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso_pred   = iso.fit_predict(Xa)
    iso_scores = (-iso.score_samples(Xa))
    iso_flag   = (iso_pred==-1).astype(int)

    # ── 3b. One-Class SVM ─────────────────────────────────────
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    svm_pred = ocsvm.fit_predict(Xa[:3000])   # subset for speed
    svm_flag = np.zeros(len(df), dtype=int)
    svm_flag[:3000] = (svm_pred==-1).astype(int)

    # ── 3c. Autoencoder ───────────────────────────────────────
    ae_flag, ae_errors, ae_metrics = train_autoencoder(Xa, gt)

    # ── 3d. Z-Score baseline ──────────────────────────────────
    Xdf   = df[ANOM_FEATS]
    z     = ((Xdf - Xdf.mean()) / Xdf.std()).abs()
    z_flag= (z > 3.0).any(axis=1).astype(int)

    df['anomaly_score'] = iso_scores
    df['iso_flag']  = iso_flag
    df['svm_flag']  = svm_flag
    df['ae_flag']   = ae_flag
    df['z_flag']    = z_flag

    # Metrics
    metrics = {}
    for name, flag in [('Isolation Forest', iso_flag),
                        ('One-Class SVM',   svm_flag),
                        ('Autoencoder',     ae_flag),
                        ('Z-Score',         z_flag)]:
        metrics[name] = {
            'precision': round(precision_score(gt,flag,zero_division=0),3),
            'recall':    round(recall_score(gt,flag,zero_division=0),3),
            'f1':        round(f1_score(gt,flag,zero_division=0),3),
            'flagged':   int(flag.sum()),
            'pct':       round(flag.mean()*100,1),
        }
        metrics[name].update(ae_metrics if name=='Autoencoder' else {})

    # PCA for 2D vis
    pca = PCA(n_components=2, random_state=42)
    Xa2 = pca.fit_transform(Xa)
    df['an_pc1'] = Xa2[:,0]; df['an_pc2'] = Xa2[:,1]

    # Profile
    profile = df.groupby('iso_flag')[
        ['latency_ms','packet_loss_pct','data_usage_gb','support_calls']
    ].mean().round(2)
    profile.index = ['Normal','Anomaly']

    # Top alerts
    alerts = (df[df['iso_flag']==1]
              .sort_values('anomaly_score',ascending=False)
              .head(30)[['user_id','user_type','region',
                          'anomaly_score','latency_ms',
                          'packet_loss_pct','data_usage_gb',
                          'support_calls','hour_of_day']]
              .reset_index(drop=True))
    alerts['severity'] = pd.cut(alerts['anomaly_score'],
                                  bins=3,labels=['Medium','High','Critical'])

    # Save
    pickle.dump(iso,       open(os.path.join(MODEL_DIR,'iso.pkl'),      'wb'))
    pickle.dump(ocsvm,     open(os.path.join(MODEL_DIR,'ocsvm.pkl'),    'wb'))
    pickle.dump(scaler_an, open(os.path.join(MODEL_DIR,'an_sc.pkl'),    'wb'))

    for name, m in metrics.items():
        print(f"     {name:<20} P={m['precision']}  R={m['recall']}  F1={m['f1']}")

    return {
        'iso': iso, 'ocsvm': ocsvm,
        'metrics': metrics, 'profile': profile,
        'alerts': alerts, 'ae_errors': ae_errors, 'df': df,
    }


def train_autoencoder(Xa, gt):
    """Keras Autoencoder for reconstruction-error anomaly detection."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel('ERROR')

        n_feats = Xa.shape[1]

        # Train only on normal records
        normal_mask = (gt == 0)
        Xa_normal   = Xa[normal_mask]

        ae = Sequential([
            Dense(32, activation='relu', input_shape=(n_feats,)),
            Dense(16, activation='relu'),
            Dense(8,  activation='relu'),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(n_feats, activation='linear'),
        ])
        ae.compile(optimizer='adam', loss='mse')
        ae.fit(Xa_normal, Xa_normal,
               epochs=40, batch_size=64,
               validation_split=0.1,
               callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
               verbose=0)

        # Reconstruction errors on full set
        Xr = ae.predict(Xa, verbose=0)
        recon_error = np.mean((Xa - Xr)**2, axis=1)

        # Threshold: 95th percentile of normal errors
        threshold = np.percentile(recon_error[normal_mask], 95)
        ae_flag   = (recon_error > threshold).astype(int)

        p = precision_score(gt, ae_flag, zero_division=0)
        r = recall_score(gt, ae_flag, zero_division=0)
        f = f1_score(gt, ae_flag, zero_division=0)

        ae.save(os.path.join(MODEL_DIR, 'autoencoder.keras'))
        pickle.dump(float(threshold),
                    open(os.path.join(MODEL_DIR,'ae_threshold.pkl'),'wb'))

        return ae_flag, recon_error.tolist(), {
            'precision': round(p,3), 'recall': round(r,3), 'f1': round(f,3),
            'threshold': round(threshold,5),
        }

    except Exception as e:
        print(f"     Autoencoder skipped ({e})")
        return np.zeros(len(gt),dtype=int), np.zeros(len(gt)).tolist(), {
            'precision':0.0,'recall':0.0,'f1':0.0,'threshold':0.0
        }


# ─────────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def _train_pipeline(n=5000, force=False):
    # Avoid Unicode box-drawing chars (can fail on cp1252 consoles)
    print("\n== Telecom Intelligence Pipeline v2 ==")

    if os.path.exists(DATA_RAW) and not force:
        print("[0] Loading existing dataset…")
        df = pd.read_csv(DATA_RAW)
    else:
        print(f"[0] Generating dataset ({n} records)…")
        df = generate_dataset(n=n)
    print(f"    Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    print("[1] Network Optimization…")
    r1 = train_network_module(df); df = r1['df']

    print("[2] User Behaviour…")
    r2 = train_behaviour_module(df); df = r2['df']

    print("[3] Anomaly Detection…")
    r3 = train_anomaly_module(df); df = r3['df']

    print("All models serialized to models/")
    return df, r1, r2, r3


# Backwards-compatible name (used by older code / scripts)
def run_pipeline(n=5000, force=False):
    return get_pipeline(n=n, force=force)


# ─────────────────────────────────────────────────────────────
#  INFERENCE HELPERS (used by app.py)
# ─────────────────────────────────────────────────────────────

def predict_quality(inputs: dict) -> float:
    m = pickle.load(open(os.path.join(MODEL_DIR,'net_reg.pkl'),'rb'))
    row = pd.DataFrame([[inputs[f] for f in KPI_FEATS]], columns=KPI_FEATS)
    return round(float(m.predict(row)[0]),1)

def predict_churn(inputs: dict) -> float:
    m = pickle.load(open(os.path.join(MODEL_DIR,'churn_rf.pkl'),'rb'))
    row = pd.DataFrame([[inputs[f] for f in CHURN_FEATS]], columns=CHURN_FEATS)
    return round(float(m.predict_proba(row)[0][1]),3)

def predict_anomaly(inputs: dict):
    iso    = pickle.load(open(os.path.join(MODEL_DIR,'iso.pkl'),   'rb'))
    scaler = pickle.load(open(os.path.join(MODEL_DIR,'an_sc.pkl'), 'rb'))
    row    = pd.DataFrame([[inputs[f] for f in ANOM_FEATS]], columns=ANOM_FEATS)
    Xs     = scaler.transform(row)
    flag   = iso.predict(Xs)[0] == -1
    score  = float(-iso.score_samples(Xs)[0])
    return flag, round(score,4)


if __name__ == '__main__':
    df, r1, r2, r3 = run_pipeline(n=5000, force=True)
    print(f"\nBest regressor : {r1['best_name']}  R2={r1['reg_res'][r1['best_name']]['r2']}")
    print(f"LSTM RMSE      : {r1['lstm_res']['rmse']}")
    print(f"Churn F1       : {r2['churn_metrics']['f1']}")
    print(f"ISO F1         : {r3['metrics']['Isolation Forest']['f1']}")
    print(f"Autoencoder F1 : {r3['metrics']['Autoencoder']['f1']}")
