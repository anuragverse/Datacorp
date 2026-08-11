import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from thefuzz import fuzz
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import re

# ---- Optional heavy libraries (all FREE, no API, no cost) ----
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Datacorp | AI-Powered Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. ENHANCED STYLING
# ============================================================
st.markdown("""
<style>
    :root {
        --dc-primary: #6f88a8;
        --dc-primary-dark: #4f6888;
        --dc-accent: #7f9b8b;
        --dc-text: #1f2937;
        --dc-muted: #6b7280;
        --dc-border: #d9e0e7;
        --dc-surface: #ffffff;
        --dc-soft: #f5f7f9;
        --dc-success: #2f7d5a;
        --dc-warning: #b7791f;
        --dc-danger: #c05656;
    }

    html { scroll-behavior: smooth; }
    #MainMenu, footer, header { visibility: hidden; }

    .stApp {
        background: #f4f6f8;
        color: var(--dc-text);
    }

    .block-container {
        max-width: 1450px;
        padding: 1rem 2rem 2.5rem 2rem;
    }

    h1, h2, h3, h4 { color: var(--dc-text) !important; letter-spacing: -0.02em; }
    p, li, label, .stMarkdown { color: #374151; }

    /* Appearance switch */
    .dc-theme-caption {
        text-align: right;
        color: var(--dc-muted) !important;
        font-size: 0.76rem;
        margin-top: -4px;
        margin-bottom: 4px;
    }

    /* Hero */
    .dc-hero {
        position: relative;
        overflow: hidden;
        padding: 24px 28px;
        border-radius: 11px;
        margin: 0 0 16px 0;
        color: var(--dc-text);
        background: linear-gradient(135deg, #eaf2fb 0%, #f2f0fa 55%, #edf5f1 100%);
        border: 1px solid #cfd9e3;
        box-shadow: 0 10px 24px rgba(31, 41, 55, 0.10);
    }

    .dc-hero::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -60px;
        top: -95px;
        border-radius: 50%;
        background: rgba(127,155,139,0.12);
    }

    .dc-hero-top {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .dc-logo {
        width: 44px;
        height: 44px;
        flex: 0 0 44px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: rgba(255,255,255,0.86);
        border: 1px solid #cbd7e2;
        box-shadow: 0 4px 10px rgba(31,41,55,0.08);
        font-size: 1.35rem;
    }

    .dc-hero h1 {
        margin: 0;
        color: #26384a !important;
        font-size: 2.05rem;
        line-height: 1.1;
        font-weight: 800;
    }

    .dc-hero p {
        position: relative;
        z-index: 1;
        margin: 7px 0 0;
        color: #5f6f7f !important;
        font-size: 0.96rem;
    }

    .dc-hero-badge {
        position: relative;
        z-index: 1;
        display: inline-block;
        margin-top: 13px;
        padding: 5px 9px;
        border: 1px solid #c8d4df;
        border-radius: 7px;
        background: rgba(255,255,255,0.62);
        color: #526477;
        font-size: 0.75rem;
        font-weight: 650;
    }

    /* Section labels */
    .dc-section-label {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin: 3px 0 7px;
        padding: 5px 9px;
        border-radius: 7px;
        background: #eaf0f5;
        color: var(--dc-primary-dark);
        font-size: 0.75rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.045em;
    }

    .dc-upload-card,
    .dc-welcome-card {
        border: 1px solid var(--dc-border);
        border-radius: 10px;
        padding: 16px 18px;
        background: var(--dc-surface);
        box-shadow: 0 7px 18px rgba(31,41,55,0.07);
    }

    .dc-welcome-card {
        background: linear-gradient(135deg, #ffffff 0%, #f4f8fb 100%);
    }

    .dc-welcome-card h2 { margin: 0 0 5px; font-size: 1.3rem; }
    .dc-welcome-card p { margin: 0; color: var(--dc-muted) !important; }

    .dc-feature {
        height: 100%;
        padding: 14px;
        border: 1px solid var(--dc-border);
        border-radius: 9px;
        background: var(--dc-surface);
        box-shadow: 0 5px 14px rgba(31,41,55,0.055);
    }

    .dc-feature-title { font-weight: 700; margin-bottom: 4px; color: var(--dc-text); }
    .dc-feature-text { color: var(--dc-muted); font-size: 0.84rem; line-height: 1.45; }

    /* Workflow */
    .workflow-nav {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 9px;
        margin: 10px 0 9px;
    }

    .workflow-link {
        min-height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 9px 10px;
        border-radius: 9px;
        border: 1px solid var(--dc-border);
        text-decoration: none !important;
        font-size: 0.82rem;
        font-weight: 700;
        line-height: 1.2;
        text-align: center;
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
        box-sizing: border-box;
    }

    .workflow-link:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 16px rgba(31,41,55,0.10);
        text-decoration: none !important;
    }

    .workflow-link.completed {
        background: #edf7f1;
        color: #2f6f52 !important;
        border-color: #bcdac8;
    }

    .workflow-link.current {
        background: #e7eef7;
        color: #425d79 !important;
        border-color: #9fb4ca;
        box-shadow: 0 6px 15px rgba(79,104,136,0.12);
    }

    .workflow-link.pending {
        background: #ffffff;
        color: #697586 !important;
    }

    .workflow-link.disabled {
        opacity: .45;
        cursor: not-allowed;
        pointer-events: none;
        background: #eef1f4;
    }

    .workflow-anchor { scroll-margin-top: 28px; height: 0; overflow: hidden; }

    /* Progress */
    .dc-progress-wrap {
        padding: 10px 13px;
        border: 1px solid var(--dc-border);
        border-radius: 10px;
        background: var(--dc-surface);
        box-shadow: 0 5px 14px rgba(31,41,55,0.055);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7f9bb8, #8eaa99);
    }

    /* Metrics */
    .metric-card {
        background: var(--dc-surface);
        border: 1px solid var(--dc-border);
        border-radius: 10px;
        padding: 15px 16px;
        box-shadow: 0 6px 15px rgba(31,41,55,0.06);
    }

    .quality-score {
        font-size: 2.55rem;
        line-height: 1;
        font-weight: 800;
        color: #5c7894;
    }

    /* Issue cards */
    .issue-card-critical, .issue-card-warning, .issue-card-success {
        border-radius: 9px;
        padding: 12px 14px;
        margin: 6px 0;
    }

    .issue-card-critical { background: #fff5f5; color: #9b3d3d; border: 1px solid #f0caca; border-left: 3px solid #c05656; }
    .issue-card-warning { background: #fffaf0; color: #8a631d; border: 1px solid #ead9ad; border-left: 3px solid #b7791f; }
    .issue-card-success { background: #f0f8f4; color: #2f6f52; border: 1px solid #c4dfd0; border-left: 3px solid #4c8b6b; }

    /* Inputs / buttons */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 40px;
        border-radius: 8px;
        border: 1px solid #cfd7df;
        font-weight: 650;
        transition: all .16s ease;
        box-shadow: 0 2px 7px rgba(31,41,55,0.035);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 13px rgba(31,41,55,0.09);
        border-color: #aebdca;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #b9c8d6;
        border-radius: 9px;
        padding: 4px;
        background: #f9fbfc;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 8px !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        padding: 7px 11px;
    }

    details[data-testid="stExpander"] {
        border: 1px solid var(--dc-border);
        border-radius: 9px;
        background: var(--dc-surface);
        box-shadow: 0 3px 9px rgba(31,41,55,0.035);
    }

    .dc-footer {
        padding-top: 9px;
        text-align: center;
        color: #8b96a3;
        font-size: 0.76rem;
    }

    .metric-label { color: #718096; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
    .metric-value { color: #26384a; font-size: 1.35rem; font-weight: 800; margin-top: 4px; }
    .before-after { display: flex; align-items: center; gap: 7px; margin-top: 5px; color: #6b7280; font-size: 0.95rem; }
    .before-after strong { color: #2f6f52; }

    .stDataFrame { border: 1px solid var(--dc-border); border-radius: 9px; overflow: hidden; }

    /* Dark mode overrides */
    body:has(.dc-dark-mode) .stApp {
        background: #151a20;
        color: #e5e7eb;
    }

    body:has(.dc-dark-mode) h1,
    body:has(.dc-dark-mode) h2,
    body:has(.dc-dark-mode) h3,
    body:has(.dc-dark-mode) h4,
    body:has(.dc-dark-mode) p,
    body:has(.dc-dark-mode) li,
    body:has(.dc-dark-mode) label,
    body:has(.dc-dark-mode) .stMarkdown {
        color: #e5e7eb !important;
    }

    body:has(.dc-dark-mode) .dc-hero {
        background: linear-gradient(135deg, #253442 0%, #2d3948 55%, #2d4038 100%);
        border-color: #3d4b59;
        box-shadow: 0 12px 26px rgba(0,0,0,0.28);
    }

    body:has(.dc-dark-mode) .dc-logo {
        background: #202831;
        border-color: #465463;
    }

    body:has(.dc-dark-mode) .dc-hero h1 { color: #f3f6f8 !important; }
    body:has(.dc-dark-mode) .dc-hero p { color: #c3cdd6 !important; }
    body:has(.dc-dark-mode) .dc-hero-badge {
        background: rgba(255,255,255,0.06);
        border-color: #4b5a68;
        color: #d5dde4;
    }

    body:has(.dc-dark-mode) .dc-section-label {
        background: #24303a;
        color: #b8c9d8;
    }

    body:has(.dc-dark-mode) .dc-upload-card,
    body:has(.dc-dark-mode) .dc-welcome-card,
    body:has(.dc-dark-mode) .dc-feature,
    body:has(.dc-dark-mode) .dc-progress-wrap,
    body:has(.dc-dark-mode) .metric-card,
    body:has(.dc-dark-mode) details[data-testid="stExpander"] {
        background: #1c232b;
        border-color: #34404b;
        box-shadow: 0 7px 18px rgba(0,0,0,0.20);
    }

    body:has(.dc-dark-mode) .dc-welcome-card {
        background: linear-gradient(135deg, #1d252d 0%, #202a31 100%);
    }

    body:has(.dc-dark-mode) .dc-feature-text,
    body:has(.dc-dark-mode) .dc-welcome-card p,
    body:has(.dc-dark-mode) .dc-theme-caption,
    body:has(.dc-dark-mode) .metric-label {
        color: #aeb9c4 !important;
    }

    body:has(.dc-dark-mode) .metric-value { color: #edf2f7; }
    body:has(.dc-dark-mode) .before-after { color: #aeb9c4; }
    body:has(.dc-dark-mode) .before-after strong { color: #a9d3bd; }

    body:has(.dc-dark-mode) .workflow-link.pending {
        background: #1c232b;
        color: #aab5c0 !important;
        border-color: #34404b;
    }

    body:has(.dc-dark-mode) .workflow-link.current {
        background: #293b4b;
        color: #d4e1ec !important;
        border-color: #536b80;
    }

    body:has(.dc-dark-mode) .workflow-link.completed {
        background: #1d3329;
        color: #a9d3bd !important;
        border-color: #315842;
    }

    body:has(.dc-dark-mode) .workflow-link.disabled {
        background: #20262d;
    }

    body:has(.dc-dark-mode) div[data-testid="stFileUploader"] {
        background: #1a2128;
        border-color: #4a5967;
    }

    body:has(.dc-dark-mode) .stButton > button,
    body:has(.dc-dark-mode) .stDownloadButton > button {
        background: #202831;
        color: #e5e7eb;
        border-color: #465360;
    }

    body:has(.dc-dark-mode) .stButton > button:hover,
    body:has(.dc-dark-mode) .stDownloadButton > button:hover {
        border-color: #687989;
        background: #26313b;
    }

    body:has(.dc-dark-mode) [data-baseweb="select"] > div,
    body:has(.dc-dark-mode) [data-baseweb="input"] > div {
        background: #202831;
        color: #e5e7eb;
        border-color: #465360;
    }

    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .workflow-nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .dc-hero { padding: 20px; }
    }

    @media (max-width: 560px) {
        .workflow-nav { grid-template-columns: 1fr; }
        .dc-hero h1 { font-size: 1.7rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. CONSTANTS
# ============================================================
MAX_ROWS = 150000
MAX_FILE_SIZE_MB = 50
MAX_HISTORY = 15
ML_SAMPLE_CAP = 20000

# ============================================================
# 4. SESSION STATE + WORKFLOW MANAGEMENT
# ============================================================
def init_state():
    defaults = {
        'current_df': None,
        'original_df': None,
        'active_dataset': None,
        'history': [],
        'redo_stack': [],
        'pipeline_log': [],
        'user_mode': 'beginner',  # beginner, intermediate, expert
        'workflow_step': 0,  # Current step in workflow
        'completed_steps': [],
        '_last_nav_step': None,
        'data_quality_score': 0,
        'initial_quality_score': 0,
        'issues_detected': [],
        'fuzzy_matches': None,
        'anomaly_result': None,
        'ml_summary': None,
        'ml_importance': None,
        'anomaly_summary': None,
        'pdf_report': None,
        'show_tutorial': True,
        'cleaning_recipe': [],
        'sample_data_loaded': None,
        'confirm_reset': False,
        'analyst_result': None,
        'recommendations': [],
        'issue_filter': 'All Issues',
        'session_started': datetime.now(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def calculate_quality_score(df):
    """Calculate data quality score (0-100)"""
    if df is None or df.empty:
        return 0
    
    score = 100
    
    # Missing values penalty
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    score -= min(missing_pct, 30)
    
    # Duplicate penalty
    dup_pct = (df.duplicated().sum() / df.shape[0]) * 100
    score -= min(dup_pct, 20)
    
    # Data type consistency check
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col].dropna().head(100))
                score -= 2  # Looks like number but stored as text
            except:
                pass
    
    return max(0, min(100, score))


def detect_issues(df):
    """Auto-detect data quality issues"""
    issues = []
    
    if df is None or df.empty:
        return issues
    
    # Critical issues
    missing_cols = df.columns[df.isnull().any()].tolist()
    for col in missing_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            issues.append({
                'severity': 'critical',
                'type': 'missing_values',
                'column': col,
                'detail': f'{missing_pct:.1f}% missing values',
                'action': 'drop_or_fill',
                'auto_fixable': True
            })
        elif missing_pct > 5:
            issues.append({
                'severity': 'warning',
                'type': 'missing_values',
                'column': col,
                'detail': f'{missing_pct:.1f}% missing values',
                'action': 'fill',
                'auto_fixable': True
            })
    
    # Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = (dup_count / len(df)) * 100
        issues.append({
            'severity': 'warning' if dup_pct < 10 else 'critical',
            'type': 'duplicates',
            'column': 'All columns',
            'detail': f'{dup_count:,} duplicate rows ({dup_pct:.1f}%)',
            'action': 'remove',
            'auto_fixable': True
        })
    
    # Data type mismatches
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(100)
        if len(sample) > 0:
            # Check if looks like number
            try:
                pd.to_numeric(sample)
                issues.append({
                    'severity': 'warning',
                    'type': 'datatype',
                    'column': col,
                    'detail': 'Stored as text but looks like numbers',
                    'action': 'convert_numeric',
                    'auto_fixable': True
                })
            except:
                pass
            
            # Check if looks like date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'dt', 'timestamp']):
                try:
                    pd.to_datetime(sample.head(10))
                    issues.append({
                        'severity': 'warning',
                        'type': 'datatype',
                        'column': col,
                        'detail': 'Stored as text but looks like dates',
                        'action': 'convert_datetime',
                        'auto_fixable': True
                    })
                except:
                    pass
    
    # Outlier detection (numeric columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)][col].count()
        if outliers > 0:
            outlier_pct = (outliers / len(df)) * 100
            if outlier_pct > 5:
                issues.append({
                    'severity': 'info',
                    'type': 'outliers',
                    'column': col,
                    'detail': f'{outliers:,} outliers detected ({outlier_pct:.1f}%)',
                    'action': 'review',
                    'auto_fixable': False
                })
    
    return issues


def auto_fix_issues(df, issues):
    """Auto-fix common issues"""
    fixed_df = df.copy()
    actions_taken = []
    
    for issue in issues:
        if not issue['auto_fixable']:
            continue
        
        try:
            if issue['type'] == 'missing_values':
                col = issue['column']
                if pd.api.types.is_numeric_dtype(fixed_df[col]):
                    fixed_df[col] = fixed_df[col].fillna(fixed_df[col].median())
                    actions_taken.append(f"Filled '{col}' with median")
                else:
                    fixed_df[col] = fixed_df[col].fillna(fixed_df[col].mode()[0] if not fixed_df[col].mode().empty else "")
                    actions_taken.append(f"Filled '{col}' with mode")
            
            elif issue['type'] == 'duplicates':
                before = len(fixed_df)
                fixed_df = fixed_df.drop_duplicates()
                removed = before - len(fixed_df)
                actions_taken.append(f"Removed {removed:,} duplicate rows")
            
            elif issue['type'] == 'datatype':
                col = issue['column']
                if issue['action'] == 'convert_numeric':
                    fixed_df[col] = pd.to_numeric(fixed_df[col], errors='coerce')
                    actions_taken.append(f"Converted '{col}' to numeric")
                elif issue['action'] == 'convert_datetime':
                    fixed_df[col] = pd.to_datetime(fixed_df[col], errors='coerce')
                    actions_taken.append(f"Converted '{col}' to datetime")
        
        except Exception as e:
            st.warning(f"Could not auto-fix {issue['column']}: {e}")
    
    return fixed_df, actions_taken


def dataset_profile(df):
    if df is None or df.empty:
        return {}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    dates = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric': len(numeric),
        'categorical': len(categorical),
        'dates': len(dates),
        'missing': missing,
        'duplicates': duplicates,
        'memory_mb': memory_mb,
    }


def quality_breakdown(df):
    if df is None or df.empty:
        return {'missing_pct': 0, 'duplicate_pct': 0, 'outlier_count': 0, 'type_issues': 0}
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_pct = df.isnull().sum().sum() / total_cells * 100
    duplicate_pct = df.duplicated().sum() / max(len(df), 1) * 100
    outlier_count = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) >= 4:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_count += int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    type_issues = 0
    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(100)
        if len(sample) >= 5:
            try:
                pd.to_numeric(sample)
                type_issues += 1
            except Exception:
                pass
    return {
        'missing_pct': missing_pct,
        'duplicate_pct': duplicate_pct,
        'outlier_count': outlier_count,
        'type_issues': type_issues,
    }


def build_recommendations(df, issues):
    recommendations = []
    if df is None or df.empty:
        return recommendations
    for issue in issues:
        if issue['type'] == 'missing_values':
            col = issue['column']
            if pd.api.types.is_numeric_dtype(df[col]):
                action = 'Fill missing values with the median'
            else:
                action = 'Fill missing values with the most frequent value'
            recommendations.append({
                'priority': 'High' if issue['severity'] == 'critical' else 'Medium',
                'column': col,
                'issue': issue['detail'],
                'recommendation': action,
            })
        elif issue['type'] == 'duplicates':
            recommendations.append({
                'priority': 'High',
                'column': 'All columns',
                'issue': issue['detail'],
                'recommendation': 'Remove exact duplicate rows',
            })
        elif issue['type'] == 'datatype':
            recommendations.append({
                'priority': 'Medium',
                'column': issue['column'],
                'issue': issue['detail'],
                'recommendation': 'Convert the column to the detected data type',
            })
        elif issue['type'] == 'outliers':
            recommendations.append({
                'priority': 'Review',
                'column': issue['column'],
                'issue': issue['detail'],
                'recommendation': 'Review outliers before removing or capping them',
            })
    return recommendations[:12]


def local_analyst(df):
    if df is None or df.empty:
        return []
    insights = []
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if numeric:
        variances = df[numeric].var(numeric_only=True).sort_values(ascending=False)
        if not variances.empty:
            top = variances.index[0]
            insights.append(f"'{top}' has the highest variance among numeric columns.")
        correlations = df[numeric].corr(numeric_only=True)
        pairs=[]
        for i, a in enumerate(numeric):
            for b in numeric[i+1:]:
                val=correlations.loc[a,b]
                if pd.notna(val): pairs.append((abs(val), val, a, b))
        if pairs:
            _, val, a, b=max(pairs)
            if abs(val) >= 0.7:
                insights.append(f"'{a}' and '{b}' are strongly correlated (r={val:.2f}).")
    if categorical:
        for col in categorical[:3]:
            nunique=df[col].nunique(dropna=True)
            if nunique <= 15 and nunique > 1:
                top_value=df[col].value_counts(dropna=True).index[0] if not df[col].dropna().empty else None
                if top_value is not None:
                    share=df[col].value_counts(normalize=True, dropna=True).iloc[0]*100
                    insights.append(f"'{col}' has {nunique} distinct values; '{top_value}' is the most common ({share:.1f}% of non-null rows).")
    missing=df.isnull().sum().sort_values(ascending=False)
    if not missing.empty and missing.iloc[0] > 0:
        col=missing.index[0]
        insights.append(f"'{col}' has the most missing values ({int(missing.iloc[0]):,}).")
    dup=int(df.duplicated().sum())
    if dup:
        insights.append(f"The dataset contains {dup:,} exact duplicate rows that should be reviewed.")
    if not insights:
        insights.append('No major statistical pattern was detected from the current dataset.')
    return insights[:6]


def log_entry(action, detail, shape):
    return {
        'Step': len(st.session_state.pipeline_log) + 1,
        'Action': action,
        'Detail': detail,
        'Time': datetime.now().strftime('%H:%M:%S'),
        'Shape': f"{shape[0]:,} × {shape[1]}"
    }


def apply_change(new_df, action, detail=""):
    """Central mutation function"""
    st.session_state.history.append(
        (st.session_state.current_df.copy(), list(st.session_state.pipeline_log))
    )
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history.pop(0)
    st.session_state.redo_stack = []
    st.session_state.current_df = new_df
    st.session_state.pipeline_log.append(log_entry(action, detail, new_df.shape))
    st.session_state.data_quality_score = calculate_quality_score(new_df)
    st.session_state.issues_detected = detect_issues(new_df)
    st.session_state.recommendations = build_recommendations(new_df, st.session_state.issues_detected)

    # Any successful data transformation completes the cleaning step.
    if 2 not in st.session_state.completed_steps:
        st.session_state.completed_steps.append(2)
    st.session_state.workflow_step = max(st.session_state.workflow_step, 3)


def do_undo():
    if st.session_state.history:
        st.session_state.redo_stack.append(
            (st.session_state.current_df.copy(), list(st.session_state.pipeline_log))
        )
        df_prev, log_prev = st.session_state.history.pop()
        st.session_state.current_df = df_prev
        st.session_state.pipeline_log = log_prev
        st.session_state.data_quality_score = calculate_quality_score(df_prev)


def do_redo():
    if st.session_state.redo_stack:
        st.session_state.history.append(
            (st.session_state.current_df.copy(), list(st.session_state.pipeline_log))
        )
        df_next, log_next = st.session_state.redo_stack.pop()
        st.session_state.current_df = df_next
        st.session_state.pipeline_log = log_next
        st.session_state.data_quality_score = calculate_quality_score(df_next)

def mark_download_complete():
    if 4 not in st.session_state.completed_steps:
        st.session_state.completed_steps.append(4)

    st.session_state.workflow_step = 4

# ============================================================
# 6. CACHED FUNCTIONS
# ============================================================
@st.cache_data(ttl=1800, max_entries=6, show_spinner=" Reading file...")
def parse_file(file_bytes: bytes, file_name: str):
    bio = BytesIO(file_bytes)
    if file_name.endswith('.csv'):
        return pd.read_csv(bio)
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(bio)
    elif file_name.endswith('.json'):
        return pd.read_json(bio)
    return None


@st.cache_data(ttl=600, max_entries=10, show_spinner=False)
def cached_describe(df: pd.DataFrame):
    return df.describe(include='all').fillna('-')


@st.cache_data(ttl=600, max_entries=10, show_spinner=False)
def cached_corr(df_num: pd.DataFrame):
    return df_num.corr()


# ============================================================
# 7. WORKFLOW STEPS DEFINITION
# ============================================================
WORKFLOW_STEPS = [
    {
        'id': 0,
        'name': 'Upload Data',
        'icon': '',
        'description': 'Upload your CSV, Excel, or JSON files'
    },
    {
        'id': 1,
        'name': 'Check Quality',
        'icon': '',
        'description': 'Auto-detect issues in your data'
    },
    {
        'id': 2,
        'name': 'Clean Data',
        'icon': '',
        'description': 'Fix missing values, duplicates, and errors'
    },
    {
        'id': 3,
        'name': 'Analyze & Visualize',
        'icon': '',
        'description': 'Create charts and run ML models'
    },
    {
        'id': 4,
        'name': 'Download Results',
        'icon': '',
        'description': 'Export cleaned data and reports'
    }
]


# ============================================================
# 8. HEADER, APPEARANCE & MODE SELECTOR
# ============================================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Small, unobtrusive appearance switch above the main header.
theme_col1, theme_col2 = st.columns([8.5, 1.5])
with theme_col2:
    st.toggle(" Dark mode", key="dark_mode")

# Marker used by the CSS :has() selector to apply dark-mode styling.
if st.session_state.dark_mode:
    st.markdown("<div class='dc-dark-mode' style='display:none'></div>", unsafe_allow_html=True)

st.markdown("""
<div class="dc-hero">
    <div class="dc-hero-top">
        <div class="dc-logo">📊</div>
        <div>
            <h1>Welcome to Datacorp!</h1>
            <p>AI-powered data cleaning, quality analysis and visualization.</p>
        </div>
    </div>
    <span class="dc-hero-badge">Private • Session-based • CSV / Excel / JSON</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='dc-section-label'> Experience</div>", unsafe_allow_html=True)
mode = st.radio(
    "Choose your experience level",
    [" Beginner (Guided)", " Intermediate (Smart Assist)", " Expert (Full Control)"],
    index=0 if st.session_state.user_mode == 'beginner' else 1 if st.session_state.user_mode == 'intermediate' else 2,
    horizontal=True,
    key='mode_selector',
    label_visibility="collapsed"
)

if "Beginner" in mode:
    st.session_state.user_mode = 'beginner'
elif "Intermediate" in mode:
    st.session_state.user_mode = 'intermediate'
else:
    st.session_state.user_mode = 'expert'

st.markdown("---")

# 9. WORKFLOW PROGRESS TRACKER
# ============================================================
if st.session_state.user_mode in ['beginner', 'intermediate']:
    has_uploaded_data = (
        st.session_state.get("current_df") is not None
        or st.session_state.get("f1") is not None
        or st.session_state.get("f2") is not None
        or st.session_state.get("f3") is not None
        or st.session_state.get("sample_data_loaded") is not None
    )

    total_steps = len(WORKFLOW_STEPS)
    current_step = st.session_state.workflow_step
    completed = st.session_state.completed_steps
    progress_pct = (len(completed) / total_steps) * 100

    st.markdown("###  Workflow")
    st.markdown("<div class='dc-progress-wrap'>", unsafe_allow_html=True)
    st.progress(progress_pct / 100)
    st.caption(f"**{len(completed)} of {total_steps} completed** · {progress_pct:.0f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    workflow_targets = {
        0: 'step-0',
        1: 'step-1',
        2: 'step-2',
        3: 'step-analysis',
        4: 'step-4',
    }

    nav_html = ["<div class='workflow-nav'>"]
    for idx, step in enumerate(WORKFLOW_STEPS):
        if idx in completed:
            state_class = "completed"
            state_icon = ""
        elif idx == current_step:
            state_class = "current"
            state_icon = str(idx + 1)
        else:
            state_class = "pending"
            state_icon = str(idx + 1)

        can_navigate = idx == 0 or has_uploaded_data

        if can_navigate:
            nav_html.append(
                f"<a class='workflow-link {state_class}' href='#{workflow_targets[idx]}' title='{step['description']}'>"
                f"<span>{state_icon} · {step['name']}</span></a>"
            )
        else:
            nav_html.append(
                f"<span class='workflow-link {state_class} disabled' title='Upload a file before opening this step'>"
                f"<span> {idx + 1} · {step['name']}</span></span>"
            )

    nav_html.append("</div>")
    st.markdown("".join(nav_html), unsafe_allow_html=True)
    st.markdown("---")

# 10. FILE UPLOAD SECTION
# ============================================================
st.markdown("<div id='step-0' class='workflow-anchor'></div>", unsafe_allow_html=True)
st.markdown("<div class='dc-section-label'>01 · Upload</div>", unsafe_allow_html=True)
st.markdown("## Upload Your Data")

# --- QUICK START GUIDE ---
with st.expander(" New user? Click here to understand the complete workflow.", expanded=False):
    st.markdown("""
**How to use the app — 5 easy steps:**

1⃣ **Upload** — Upload a CSV/Excel/JSON file below (or  try Sample Data if you do not have a file)

2⃣ **Review the Quality Dashboard** — The app will automatically identify issues in your file (missing values, duplicates, outliers)

3⃣ **Clean your data** — Click " Auto-Fix All Issues" to fix issues automatically, or use the Step 4 tools manually

4⃣ **Analyze your data** — Explore the charts, or try AI-powered prediction/anomaly detection (both are optional)

5⃣ **Download your results** — Download the cleaned CSV/Excel/PDF report. All data is automatically cleared when you close the tab, and nothing is saved.

 **Need help?** Keep scrolling — each section provides a " Next Step" to guide you through the workflow.
""")

if st.session_state.user_mode == 'beginner':
    st.caption("Start with a CSV, Excel or JSON file. Your data is processed only for this session.")

# --- SAMPLE DATA OPTION ---
sample_col1, sample_col2 = st.columns([3, 1])
with sample_col2:
    use_sample = st.button(" Try Sample Data", key="sample_data_btn", use_container_width=True)

if use_sample:
    np.random.seed(42)
    n = 300
    sample_df = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'name': [f'  Customer {i}  ' if i % 7 == 0 else f'Customer {i}' for i in range(1, n + 1)],
        'age': np.random.randint(18, 70, n).astype(float),
        'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', None], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        'purchase_amount': np.concatenate([
            np.random.normal(2000, 500, n - 10),
            np.random.normal(20000, 2000, 10)  # outliers
        ]),
        'signup_date': pd.date_range('2023-01-01', periods=n, freq='D').astype(str),
        'rating': np.random.choice([1, 2, 3, 4, 5, None], n, p=[0.05, 0.05, 0.15, 0.35, 0.35, 0.05])
    })
    sample_df.loc[sample_df.sample(20, random_state=1).index, 'age'] = None
    sample_df = pd.concat([sample_df, sample_df.iloc[:8]], ignore_index=True)  # inject duplicates
    st.session_state['sample_data_loaded'] = sample_df
    st.success(" Sample data loaded. It will appear under 'Choose dataset' — it intentionally contains nulls, duplicates, and outliers so you can try all the tools!")

st.markdown("---")

st.caption(f" **Best performance with files under {MAX_FILE_SIZE_MB}MB** (~{MAX_ROWS:,} rows). "
           f"Large files may slow down the page — if your file is very large, "
           f"try splitting it into smaller parts first.")

col1, col2, col3 = st.columns(3)
with col1:
    file1 = st.file_uploader(" File 1 (Required)", type=['csv', 'xlsx', 'json'], key="f1")
with col2:
    file2 = st.file_uploader(" File 2 (Optional)", type=['csv', 'xlsx', 'json'], key="f2")
with col3:
    file3 = st.file_uploader(" File 3 (Optional)", type=['csv', 'xlsx', 'json'], key="f3")


def load_file(file):
    try:
        size_mb = file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.warning(f" **{file.name}** is {size_mb:.1f}MB — this file is large and may slow down the page "
           f"or cause the browser to become unresponsive. Recommended: keep files under {MAX_FILE_SIZE_MB}MB.")
        df = parse_file(file.getvalue(), file.name)
        if df is None:
            return None
        if df.shape[0] > MAX_ROWS:
            st.error(f" **{file.name}** has {df.shape[0]:,} rows. Max allowed: {MAX_ROWS:,}")
            return None
        return df
    except Exception as e:
        st.error(f" Error loading {file.name}: {e}")
        return None


dataframes = {}
for f in [file1, file2, file3]:
    if f:
        df = load_file(f)
        if df is not None:
            dataframes[f.name] = df
            st.success(f" **{f.name}**: {df.shape[0]:,} rows × {df.shape[1]} columns")

if st.session_state.get('sample_data_loaded') is not None:
    dataframes[' sample_data.csv'] = st.session_state['sample_data_loaded']

# Mark step 0 as complete if files uploaded
if dataframes and 0 not in st.session_state.completed_steps:
    st.session_state.completed_steps.append(0)
    st.session_state.workflow_step = 1

if dataframes:
    st.caption(" **Next Step:** Select your file under 'Choose dataset', then review the Quality Dashboard (Step 2).")

st.markdown("---")

# ============================================================
# 11. JOIN/MERGE SECTION (ENHANCED)
# ============================================================
if len(dataframes) >= 2:
    st.markdown("###  Join Your Files (Optional)")
    
    if st.session_state.user_mode == 'beginner':
        st.info(" **Join** combines two files based on a common column (like customer ID). Think of it as VLOOKUP in Excel.")
    
    file_names = list(dataframes.keys())

    jc1, jc2 = st.columns(2)
    with jc1:
        sel_f1 = st.selectbox("First File:", file_names, key="jf1")
    with jc2:
        rem = [f for f in file_names if f != sel_f1]
        sel_f2 = st.selectbox("Second File:", rem, key="jf2")

    jc3, jc4 = st.columns(2)
    with jc3:
        jcol1 = st.selectbox(f"Join Column ({sel_f1}):", dataframes[sel_f1].columns.tolist(), key="jc1")
    with jc4:
        jcol2 = st.selectbox(f"Join Column ({sel_f2}):", dataframes[sel_f2].columns.tolist(), key="jc2")

    jtype = st.radio("Join Type:", ["Inner Join", "Left Join", "Right Join", "Outer Join"],
                     horizontal=True, key="jtype",
                     help="Inner=Only matching rows | Left=All from first file | Right=All from second | Outer=All rows")
    jmap = {"Inner Join": "inner", "Left Join": "left", "Right Join": "right", "Outer Join": "outer"}

    if st.button(" Perform Join", key="join_btn", type="primary"):
        try:
            with st.spinner(" Joining files..."):
                merged = pd.merge(dataframes[sel_f1], dataframes[sel_f2],
                                  left_on=jcol1, right_on=jcol2, how=jmap[jtype])
                st.session_state['merged_df'] = merged
                
                #  SUCCESS MESSAGE
                st.success(f" Successfully Joined! Result: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
                
                #  JOIN SUMMARY METRICS
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("File 1 Rows", f"{dataframes[sel_f1].shape[0]:,}")
                with col_m2:
                    st.metric("File 2 Rows", f"{dataframes[sel_f2].shape[0]:,}")
                with col_m3:
                    st.metric("Merged Rows", f"{merged.shape[0]:,}")
                with col_m4:
                    matched_pct = (merged.shape[0] / max(dataframes[sel_f1].shape[0], dataframes[sel_f2].shape[0])) * 100
                    st.metric("Match Rate", f"{matched_pct:.1f}%")
                
                #  DETAILED JOIN INFO
                st.info(f"""
                **Join Details:**
                - **Join Type:** {jtype}
                - **Join Column (File 1):** `{jcol1}`
                - **Join Column (File 2):** `{jcol2}`
                - **New Columns Added:** {merged.shape[1] - dataframes[sel_f1].shape[1]}
                - **Rows from File 1:** {dataframes[sel_f1].shape[0]:,}
                - **Rows from File 2:** {dataframes[sel_f2].shape[0]:,}
                - **Final Merged Rows:** {merged.shape[0]:,}
                """)
                
                #  MERGED DATA PREVIEW
                st.markdown("###  Merged Data Preview (First 50 Rows)")
                st.dataframe(merged.head(50), use_container_width=True, height=400)
                
                #  SHOW SAMPLE OF EACH SOURCE
                with st.expander(" Compare Sources (Side-by-Side)", expanded=False):
                    prev_col1, prev_col2 = st.columns(2)
                    with prev_col1:
                        st.markdown(f"**{sel_f1}** (First 10 rows)")
                        st.dataframe(dataframes[sel_f1].head(10), use_container_width=True)
                    with prev_col2:
                        st.markdown(f"**{sel_f2}** (First 10 rows)")
                        st.dataframe(dataframes[sel_f2].head(10), use_container_width=True)
                
                #  COLUMN MAPPING INFO
                with st.expander(" Column Details After Merge", expanded=False):
                    col_info = pd.DataFrame({
                        'Column Name': merged.columns,
                        'Data Type': merged.dtypes.astype(str),
                        'Non-Null Count': merged.count().values,
                        'Null Count': merged.isnull().sum().values,
                        'Source': ['File 1' if c in dataframes[sel_f1].columns else 
                                  'File 2' if c in dataframes[sel_f2].columns else 
                                  'Both' for c in merged.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
        except Exception as e:
            st.error(f" Join Failed: {e}")
            st.warning("""
            **Common Join Issues:**
            - Column data types don't match (e.g., text vs number)
            - Column values don't have exact matches
            - Column names have extra spaces
            
            **Try:**
            1. Check data types in 'Fix Data Types' section
            2. Use 'Text Cleaning' to remove extra spaces
            3. Preview both columns before joining
            """)

    st.markdown("---")

# ============================================================
# 12. MAIN ANALYSIS - DATASET SELECTOR
# ============================================================
if dataframes:
    st.markdown("### Step 2: Select Dataset to Analyze")

    options = list(dataframes.keys())
    if 'merged_df' in st.session_state:
        options.append(" Merged Data")

    selected = st.selectbox("Choose dataset:", options, key="ds_select")

    # Only reset when dataset CHANGES
    if st.session_state.active_dataset != selected or st.session_state.current_df is None:
        if selected == " Merged Data":
            base_df = st.session_state['merged_df'].copy()
        else:
            base_df = dataframes[selected].copy()
        
        st.session_state.active_dataset = selected
        st.session_state.original_df = base_df.copy()
        st.session_state.current_df = base_df
        st.session_state.history = []
        st.session_state.redo_stack = []
        st.session_state.pipeline_log = [log_entry(" Load", f"Loaded '{selected}'", base_df.shape)]
        
        # Calculate initial quality score
        st.session_state.initial_quality_score = calculate_quality_score(base_df)
        st.session_state.data_quality_score = st.session_state.initial_quality_score
        
        # Detect issues
        st.session_state.issues_detected = detect_issues(base_df)

    df = st.session_state.current_df

    # ============================================================
    # DATASET OVERVIEW
    # ============================================================
    profile = dataset_profile(df)
    breakdown = quality_breakdown(df)

    st.markdown("### Dataset Overview")
    overview_cols = st.columns(6)
    overview_items = [
        ("Rows", f"{profile['rows']:,}"),
        ("Columns", f"{profile['columns']:,}"),
        ("Missing", f"{profile['missing']:,}"),
        ("Duplicates", f"{profile['duplicates']:,}"),
        ("Numeric", f"{profile['numeric']:,}"),
        ("Memory", f"{profile['memory_mb']:.1f} MB"),
    ]
    for col, (label, value) in zip(overview_cols, overview_items):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

    ov1, ov2 = st.columns([1.35, 1])
    with ov1:
        st.markdown("#### Dataset Profile")
        profile_table = pd.DataFrame({
            'Type': ['Numeric', 'Categorical', 'Date', 'Other'],
            'Columns': [profile['numeric'], profile['categorical'], profile['dates'], max(profile['columns'] - profile['numeric'] - profile['categorical'] - profile['dates'], 0)]
        })
        st.dataframe(profile_table, use_container_width=True, hide_index=True)
    with ov2:
        st.markdown("#### Quality Signals")
        quality_table = pd.DataFrame({
            'Signal': ['Missing cells', 'Duplicate rows', 'Outlier rows', 'Type issues'],
            'Value': [f"{breakdown['missing_pct']:.1f}%", f"{breakdown['duplicate_pct']:.1f}%", f"{breakdown['outlier_count']:,}", str(breakdown['type_issues'])]
        })
        st.dataframe(quality_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ============================================================
    # 13. AUTO-DETECTION DASHBOARD (NEW FEATURE!)
    # ============================================================
    st.markdown("<div id='step-1' class='workflow-anchor'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>02 · Quality</div>", unsafe_allow_html=True)
    st.markdown("## Data Quality Dashboard")
    
    # Quality Score
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    
    with score_col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 0.9rem; color: #6c757d; margin-bottom: 5px;'>Data Quality Score</div>
            <div class='quality-score'>{st.session_state.data_quality_score:.0f}/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with score_col2:
        improvement = st.session_state.data_quality_score - st.session_state.initial_quality_score
        st.metric("Improvement", f"+{improvement:.0f}" if improvement > 0 else f"{improvement:.0f}",
                  delta=f"{improvement:.0f} points" if improvement != 0 else "No changes yet")
    
    with score_col3:
        critical_issues = len([i for i in st.session_state.issues_detected if i['severity'] == 'critical'])
        st.metric("Critical Issues", critical_issues, delta=f"-{critical_issues}" if critical_issues > 0 else "All clear!")
    
    with score_col4:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Issues detected
    if st.session_state.issues_detected:
        if 1 not in st.session_state.completed_steps:
            st.session_state.completed_steps.append(1)

        st.markdown("###  Auto-Detected Issues")
        
        # Separate by severity
        critical = [i for i in st.session_state.issues_detected if i['severity'] == 'critical']
        warnings = [i for i in st.session_state.issues_detected if i['severity'] == 'warning']
        info = [i for i in st.session_state.issues_detected if i['severity'] == 'info']
        
        # Critical issues
        if critical:
            st.markdown("####  Critical Issues (Fix Immediately)")
            for issue in critical:
                st.markdown(f"""
                <div class='issue-card-critical'>
                    <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 5px;'>
                        {issue['column']}
                    </div>
                    <div>{issue['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Warnings
        if warnings:
            st.markdown("####  Warnings (Recommended to Fix)")
            for issue in warnings:
                st.markdown(f"""
                <div class='issue-card-warning'>
                    <div style='font-size: 1.1rem; font-weight: 600; margin-bottom: 5px;'>
                        {issue['column']}
                    </div>
                    <div>{issue['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # One-click fix button
        st.markdown("<br>", unsafe_allow_html=True)
        
        auto_fixable = [i for i in st.session_state.issues_detected if i['auto_fixable']]
        
        if auto_fixable:
            fix_col1, fix_col2, fix_col3 = st.columns([2, 1, 1])
            
            with fix_col1:
                st.info(f" {len(auto_fixable)} issues can be auto-fixed with smart defaults")
            
            with fix_col2:
                if st.button(" Auto-Fix All Issues", type="primary", key="auto_fix_btn"):
                    with st.spinner(" Applying smart fixes..."):
                        fixed_df, actions = auto_fix_issues(df, auto_fixable)
                        apply_change(fixed_df, " Auto-Fix", f"Fixed {len(actions)} issues")
                        st.session_state.issues_detected = detect_issues(fixed_df)
                        
                        if 2 not in st.session_state.completed_steps:
                            st.session_state.completed_steps.append(2)
                            st.session_state.workflow_step = 3
                        
                        st.success(f" Auto-fixed {len(actions)} issues!")
                        for action in actions:
                            st.success(f"  • {action}")
                        st.rerun()
            
            with fix_col3:
                with st.expander("What will be fixed?"):
                    for issue in auto_fixable:
                        st.write(f"• {issue['column']}: {issue['detail']}")
    
    else:
        st.markdown(f"""
        <div class='issue-card-success'>
            <div style='font-size: 1.5rem; margin-bottom: 10px;'></div>
            <div style='font-size: 1.1rem; font-weight: 600;'>Excellent Data Quality!</div>
            <div style='margin-top: 5px;'>No critical issues detected. Your data looks clean.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if 1 not in st.session_state.completed_steps:
            st.session_state.completed_steps.append(1)
        if 2 not in st.session_state.completed_steps:
            st.session_state.completed_steps.append(2)
            st.session_state.workflow_step = 3

    # ============================================================
    # ISSUE EXPLORER + SMART RECOMMENDATIONS
    # ============================================================
    st.markdown("### Issue Explorer")
    issue_rows = st.session_state.issues_detected
    if issue_rows:
        issue_types = ['All Issues'] + sorted({i['type'].replace('_', ' ').title() for i in issue_rows})
        issue_filter = st.selectbox("Filter issues", issue_types, key="issue_filter_select")
        filtered = issue_rows if issue_filter == 'All Issues' else [i for i in issue_rows if i['type'].replace('_', ' ').title() == issue_filter]
        issue_df = pd.DataFrame([{
            'Severity': i['severity'].title(),
            'Column': i['column'],
            'Issue': i['detail'],
            'Action': i['action'].replace('_', ' ').title()
        } for i in filtered])
        st.dataframe(issue_df, use_container_width=True, hide_index=True)
    else:
        st.success("No data-quality issues were detected.")

    st.markdown("### Smart Cleaning Recommendations")
    recommendations = build_recommendations(df, st.session_state.issues_detected)
    st.session_state.recommendations = recommendations
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        rec_df.columns = ['Priority', 'Column', 'Detected Issue', 'Recommended Action']
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("No cleaning recommendations are currently needed.")

    st.caption("Review recommendations before applying changes. Automatic fixes use conservative statistical defaults.")
    st.markdown("---")

    # ============================================================
    # 14. UNDO/REDO + PIPELINE LOG
    # ============================================================
    st.markdown("###  Cleaning History & Controls")

    ur1, ur2, ur3, ur4 = st.columns([1, 1, 1.4, 2.6])
    with ur1:
        if st.button(f"↩ Undo ({len(st.session_state.history)})",
                     key="undo_btn", disabled=len(st.session_state.history) == 0,
                     use_container_width=True):
            do_undo()
            st.rerun()
    with ur2:
        if st.button(f"↪ Redo ({len(st.session_state.redo_stack)})",
                     key="redo_btn", disabled=len(st.session_state.redo_stack) == 0,
                     use_container_width=True):
            do_redo()
            st.rerun()
    with ur3:
        if not st.session_state.get('confirm_reset', False):
            if st.button(" Reset Everything", key="reset_btn", use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            if st.button(" Confirm Reset?", key="reset_confirm_btn", use_container_width=True):
                # Wipe all cleaning progress and derived results, go back to the originally uploaded file
                for k in ['ml_importance', 'anomaly_result', 'anomaly_summary', 'pivot_result',
                          'fuzzy_matches', 'confirm_reset', 'analyst_result']:
                    st.session_state.pop(k, None)
                st.session_state.original_df = None
                st.session_state.active_dataset = None
                st.rerun()
    with ur4:
        if st.session_state.get('confirm_reset', False):
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.caption(" This will delete all cleaning steps and results and restore the original file.")
            with rc2:
                if st.button(" Cancel", key="reset_cancel_btn", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()
        else:
            st.caption(f" {len(st.session_state.pipeline_log)} steps logged | Quality Score: {st.session_state.data_quality_score:.0f}/100")

    with st.expander(f" View All Cleaning Steps ({len(st.session_state.pipeline_log)} steps)", expanded=False):
        if st.session_state.pipeline_log:
            log_df = pd.DataFrame(st.session_state.pipeline_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            
            log_csv = log_df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Log (CSV)", log_csv,
                               file_name="cleaning_log.csv", mime="text/csv")
        else:
            st.info("No steps logged yet.")

    st.markdown("---")

    # ============================================================
    # 15. BASIC OVERVIEW (COLLAPSIBLE)
    # ============================================================
    with st.expander("Basic Data Overview", expanded=st.session_state.user_mode == 'expert'):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with m2:
            st.metric("Total Columns", df.shape[1])
        with m3:
            mem = df.memory_usage(deep=True).sum() / 1024 ** 2
            st.metric("Memory Usage", f"{mem:.2f} MB")
        with m4:
            st.metric("Data Types", df.dtypes.nunique())

        t1, t2, t3 = st.tabs(["Head", "Sample", "Info"])

        with t1:
            st.dataframe(df.head(10), use_container_width=True)
        with t2:
            st.dataframe(df.sample(min(5, df.shape[0])), use_container_width=True)
        with t3:
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Non-Null': df.count().values,
                'Dtype': df.dtypes.astype(str).values
            })
            st.dataframe(info_df, use_container_width=True)

    # ============================================================
    # 16. DATA CLEANING TOOLBOX (ENHANCED)
    # ============================================================
    st.markdown("<div id='step-2' class='workflow-anchor'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>03 · Clean</div>", unsafe_allow_html=True)
    st.markdown("## Data Cleaning Tools")
    
    if st.session_state.user_mode == 'beginner':
        st.info("Use these tools to fix specific issues manually, or use Auto-Fix above for conservative defaults.")

    tb1, tb2, tb3, tb4, tb5 = st.tabs(
        [" Missing Values", " Rename/Drop Columns", " Text Cleaning", " Find & Replace", " Fix Data Types"]
    )

    # MISSING VALUES TAB (ENHANCED WITH VALIDATION)
    with tb1:
        st.markdown("**Handle missing (null) values:**")
        miss_cols = df.columns[df.isnull().any()].tolist()

        if not miss_cols:
            st.success(" No missing values in this dataset!")
        else:
            mv1, mv2, mv3 = st.columns(3)
            with mv1:
                mv_col = st.selectbox("Select Column:", miss_cols, key="mv_col")
            with mv2:
                mv_action = st.selectbox(
                    "Action:",
                    ["Drop Rows (this column null)", "Fill with Mean", "Fill with Median",
                     "Fill with Mode", "Fill with Custom Value", "Forward Fill", "Backward Fill",
                     "Interpolate"],
                    key="mv_action"
                )
            
            custom_val = None
            validation_passed = True

            with mv3:
                if mv_action == "Fill with Custom Value":
                    custom_val = st.text_input("Custom Value:", key="mv_custom")
                    
                    #  VALIDATE INPUT
                    if custom_val == "":
                        st.warning(" Enter a value")
                        validation_passed = False
                    elif pd.api.types.is_numeric_dtype(df[mv_col]):
                        try:
                            float(custom_val)
                            st.success(f" Will fill with: {custom_val}")
                        except:
                            st.error(" Column is numeric, enter a number")
                            validation_passed = False
                    else:
                        st.success(f" Will fill with: '{custom_val}'")
                
                st.write("")
                
                # Disable button if validation failed
                if mv_action == "Fill with Custom Value":
                    apply_mv = st.button(" Apply", key="mv_apply_btn", disabled=not validation_passed)
                else:
                    apply_mv = st.button(" Apply", key="mv_apply_btn")

            if apply_mv:
                try:
                    new_df = df.copy()
                    if mv_action == "Drop Rows (this column null)":
                        new_df = new_df.dropna(subset=[mv_col])
                    elif mv_action == "Fill with Mean":
                        new_df[mv_col] = new_df[mv_col].fillna(new_df[mv_col].mean())
                    elif mv_action == "Fill with Median":
                        new_df[mv_col] = new_df[mv_col].fillna(new_df[mv_col].median())
                    elif mv_action == "Fill with Mode":
                        mode_val = new_df[mv_col].mode()
                        new_df[mv_col] = new_df[mv_col].fillna(mode_val[0] if not mode_val.empty else "")
                    elif mv_action == "Fill with Custom Value":
                        if pd.api.types.is_numeric_dtype(df[mv_col]):
                            custom_val = float(custom_val)
                        new_df[mv_col] = new_df[mv_col].fillna(custom_val)
                    elif mv_action == "Forward Fill":
                        new_df[mv_col] = new_df[mv_col].ffill()
                    elif mv_action == "Backward Fill":
                        new_df[mv_col] = new_df[mv_col].bfill()
                    elif mv_action == "Interpolate":
                        new_df[mv_col] = new_df[mv_col].interpolate()

                    apply_change(new_df, " Missing Values", f"'{mv_col}' → {mv_action}")
                    st.success(f" '{mv_col}' updated. Remaining nulls: {new_df[mv_col].isnull().sum()}")
                    st.rerun()
                except Exception as e:
                    st.error(f" Action failed: {e}")

    # RENAME/DROP COLUMNS TAB
    with tb2:
        st.markdown("**Rename a column:**")
        rc1, rc2, rc3 = st.columns([2, 2, 1])
        with rc1:
            col_rename = st.selectbox("Select Column:", df.columns.tolist(), key="rename_col")
        with rc2:
            new_name = st.text_input("New Name:", value=col_rename, key="rename_val")
        with rc3:
            st.write("")
            if st.button(" Rename", key="rename_btn"):
                if new_name and new_name != col_rename:
                    new_df = df.rename(columns={col_rename: new_name})
                    apply_change(new_df, " Rename Column", f"'{col_rename}' → '{new_name}'")
                    st.success(f" Renamed!")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Drop columns:**")
        drop_cols = st.multiselect("Select Columns to Drop:", df.columns.tolist(), key="drop_cols_ms")
        if st.button(" Drop Selected", key="drop_cols_btn"):
            if drop_cols:
                new_df = df.drop(columns=drop_cols)
                apply_change(new_df, " Drop Columns", f"Dropped: {', '.join(drop_cols)}")
                st.success(f" Dropped {len(drop_cols)} columns!")
                st.rerun()

    # TEXT CLEANING TAB
    with tb3:
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if not text_cols:
            st.info("No text/string columns found.")
        else:
            tc1, tc2 = st.columns(2)
            with tc1:
                tc_col = st.selectbox("Select Text Column:", text_cols, key="tc_col")
            with tc2:
                tc_action = st.selectbox(
                    "Cleaning Action:",
                    ["Trim Whitespace", "Lowercase", "UPPERCASE", "Title Case",
                     "Remove Special Characters", "Remove Extra Spaces"],
                    key="tc_action"
                )

            if st.button(" Apply", key="tc_apply_btn"):
                try:
                    new_df = df.copy()
                    s = new_df[tc_col].astype(str)
                    if tc_action == "Trim Whitespace":
                        s = s.str.strip()
                    elif tc_action == "Lowercase":
                        s = s.str.lower()
                    elif tc_action == "UPPERCASE":
                        s = s.str.upper()
                    elif tc_action == "Title Case":
                        s = s.str.title()
                    elif tc_action == "Remove Special Characters":
                        s = s.str.replace(r'[^A-Za-z0-9\s]', '', regex=True)
                    elif tc_action == "Remove Extra Spaces":
                        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

                    new_df[tc_col] = s
                    apply_change(new_df, " Text Cleaning", f"'{tc_col}' → {tc_action}")
                    st.success(f" Cleaned!")
                    st.rerun()
                except Exception as e:
                    st.error(f" Failed: {e}")

    # FIND & REPLACE TAB
    with tb4:
        fr_col = st.selectbox("Select Column:", df.columns.tolist(), key="fr_col")
        fr1, fr2 = st.columns(2)
        with fr1:
            find_val = st.text_input("Find:", key="fr_find")
        with fr2:
            replace_val = st.text_input("Replace With:", key="fr_replace")

        use_regex = st.checkbox("Use Regex Pattern", value=False, key="fr_regex_toggle")

        if st.button(" Apply", key="fr_apply_btn"):
            if find_val == "":
                st.warning(" Enter a value to find")
            else:
                try:
                    new_df = df.copy()
                    new_df[fr_col] = new_df[fr_col].astype(str).str.replace(
                        find_val, replace_val, regex=use_regex
                    )
                    apply_change(new_df, " Find & Replace", f"'{find_val}' → '{replace_val}'")
                    st.success(f" Replaced!")
                    st.rerun()
                except Exception as e:
                    st.error(f" Failed: {e}")

    # FIX DATA TYPES TAB (was missing — referenced by Join error message but never existed)
    with tb5:
        st.markdown("**Convert a column's data type** (e.g. text → number, text → date):")
        st.caption(" Convert a date column here to unlock the  Trend chart in Visualizations. "
                   "Also useful before Joining two files if a key column doesn't match (e.g. '101' vs 101).")

        dt1, dt2, dt3 = st.columns(3)
        with dt1:
            col_to_fix = st.selectbox("Select Column:", df.columns.tolist(), key="dtype_col")
        with dt2:
            st.markdown(f"**Current Type:** `{str(df[col_to_fix].dtype)}`")
            new_type = st.selectbox("Convert To:", ["Integer", "Float", "String", "Date"], key="new_dtype")
        with dt3:
            st.write("")
            convert_clicked = st.button(" Convert Column", key="convert_btn")

        if new_type == "Date":
            date_fmt_hint = st.text_input(
                "Date format (optional, e.g. %d-%m-%Y — leave blank to auto-detect):",
                key="date_fmt_hint"
            )

        if convert_clicked:
            try:
                new_df = df.copy()
                before_type = str(new_df[col_to_fix].dtype)
                if new_type == "Integer":
                    new_df[col_to_fix] = pd.to_numeric(new_df[col_to_fix], errors='coerce').astype('Int64')
                elif new_type == "Float":
                    new_df[col_to_fix] = pd.to_numeric(new_df[col_to_fix], errors='coerce')
                elif new_type == "String":
                    new_df[col_to_fix] = new_df[col_to_fix].astype(str)
                elif new_type == "Date":
                    fmt = date_fmt_hint.strip() if date_fmt_hint.strip() else None
                    new_df[col_to_fix] = pd.to_datetime(new_df[col_to_fix], format=fmt, errors='coerce')

                n_failed = int(new_df[col_to_fix].isnull().sum() - df[col_to_fix].isnull().sum())
                apply_change(new_df, " Convert Data Type", f"'{col_to_fix}': {before_type} → {new_type}")
                if n_failed > 0:
                    st.warning(f" Converted, but {n_failed} value(s) couldn't be parsed and became empty (NaN).")
                else:
                    st.success(f" '{col_to_fix}' converted to {new_type}!")
                st.rerun()
            except Exception as e:
                st.error(f" Conversion failed: {e}")

    # ============================================================
    # BEFORE / AFTER CLEANING SUMMARY
    # ============================================================
    st.markdown("### Cleaning Impact")
    original = st.session_state.get('original_df')
    if original is not None:
        before = dataset_profile(original)
        after = dataset_profile(df)
        impact_cols = st.columns(5)
        impact_items = [
            ('Rows', f"{before['rows']:,}", f"{after['rows']:,}"),
            ('Missing cells', f"{before['missing']:,}", f"{after['missing']:,}"),
            ('Duplicates', f"{before['duplicates']:,}", f"{after['duplicates']:,}"),
            ('Quality', f"{calculate_quality_score(original):.0f}", f"{calculate_quality_score(df):.0f}"),
            ('Cleaning steps', f"{max(len(st.session_state.pipeline_log)-1,0)}", f"{max(len(st.session_state.pipeline_log)-1,0)}")
        ]
        for col, (label, old_value, new_value) in zip(impact_cols, impact_items):
            with col:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='before-after'><span>{old_value}</span><span>→</span><strong>{new_value}</strong></div></div>", unsafe_allow_html=True)

    # ============================================================
    # LOCAL DATA ANALYST
    # ============================================================
    st.markdown("### Data Analyst")
    st.caption("Local statistical analysis based on the current dataset. No external AI service or API is required.")
    analyst_questions = [
        'Summarize this dataset',
        'What should I investigate first?',
        'Which columns look important?',
    ]
    analyst_q = st.selectbox("Choose an analysis prompt", analyst_questions, key="analyst_prompt")
    if st.button("Generate Insights", key="analyst_btn"):
        st.session_state.analyst_result = local_analyst(df)
    if st.session_state.get('analyst_result'):
        for insight in st.session_state.analyst_result:
            st.markdown(f"- {insight}")

    st.caption(" **Next Step:** Use Advanced Detection for outliers and fuzzy duplicates, then continue to analysis and export.")
    st.markdown("---")

    # ============================================================
    # 17. OUTLIER & FUZZY DETECTION (ENHANCED)
    # ============================================================
    st.markdown("##  Advanced Detection")

    ob1, ob2 = st.tabs([" Outlier Detection", " Fuzzy Duplicates"])

    # OUTLIER DETECTION TAB
    with ob1:
        out_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not out_num_cols:
            st.info("No numeric columns found.")
        else:
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                out_col = st.selectbox("Select Column:", out_num_cols, key="out_col")
            with oc2:
                out_method = st.selectbox("Method:", ["IQR", "Z-Score"], key="out_method")
            with oc3:
                if out_method == "IQR":
                    iqr_mult = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1, key="iqr_mult")
                else:
                    z_thresh = st.slider("Z-Score Threshold:", 1.0, 5.0, 3.0, 0.1, key="z_thresh")

            col_data = df[out_col].dropna()

            if out_method == "IQR":
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_mult * iqr
                upper = q3 + iqr_mult * iqr
                outlier_mask = (df[out_col] < lower) | (df[out_col] > upper)
            else:
                mean = col_data.mean()
                std = col_data.std()
                z_scores = (df[out_col] - mean) / std if std != 0 else df[out_col] * 0
                outlier_mask = z_scores.abs() > z_thresh

            outlier_mask = outlier_mask.fillna(False)
            n_outliers = int(outlier_mask.sum())

            oc4, oc5 = st.columns(2)
            with oc4:
                st.metric("Outliers Found", n_outliers)
            with oc5:
                fig = px.box(df, y=out_col, title=f'Box Plot: {out_col}', points='outliers')
                st.plotly_chart(fig, use_container_width=True)

            if n_outliers > 0:
                if st.checkbox(" Show Outliers", key="show_outliers"):
                    if n_outliers > 500:
                        st.caption(f" Showing first 500 of {n_outliers:,} outlier rows for smooth performance.")
                    st.dataframe(df[outlier_mask].head(500), use_container_width=True)

                oa1, oa2 = st.columns(2)
                with oa1:
                    if st.button(" Remove Outliers", key="remove_outliers_btn"):
                        new_df = df[~outlier_mask]
                        apply_change(new_df, " Remove Outliers", f"{n_outliers} rows removed")
                        st.success(f" Removed {n_outliers} outliers!")
                        st.rerun()
                with oa2:
                    if st.button(" Cap Outliers", key="cap_outliers_btn"):
                        new_df = df.copy()
                        if out_method == "IQR":
                            new_df[out_col] = new_df[out_col].clip(lower=lower, upper=upper)
                        else:
                            new_df[out_col] = new_df[out_col].clip(lower=mean - z_thresh * std,
                                                                   upper=mean + z_thresh * std)
                        apply_change(new_df, " Cap Outliers", f"'{out_col}' capped")
                        st.success(f" Outliers capped!")
                        st.rerun()

    # FUZZY DUPLICATES TAB (ENHANCED - PERSISTENT RESULTS)
    with ob2:
        st.info(" Finds similar but not identical values (e.g., 'Anurag Sinha' vs 'anurag  sinha')")

        fuzzy_text_cols = df.select_dtypes(include=['object']).columns.tolist()

        if not fuzzy_text_cols:
            st.info("No text columns found.")
        else:
            fz1, fz2 = st.columns(2)
            with fz1:
                fuzzy_col = st.selectbox("Text Column:", fuzzy_text_cols, key="fuzzy_col")
            with fz2:
                fuzzy_thresh = st.slider("Similarity %:", 70, 99, 90, 1, key="fuzzy_thresh")

            if st.button(" Find Fuzzy Duplicates", key="fuzzy_find_btn"):
                sample_df = df[[fuzzy_col]].dropna().reset_index()
                sample_df = sample_df.head(2000)
                values = sample_df[fuzzy_col].astype(str).tolist()
                idxs = sample_df['index'].tolist()

                #  PROGRESS BAR
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                matches = []
                seen = set()
                total = len(values)
                
                for i in range(total):
                    if i % 10 == 0:
                        progress_bar.progress(min(i / total, 1.0))
                        status_text.text(f" Comparing row {i:,} of {total:,}...")
                    
                    if i in seen:
                        continue
                    group = [i]
                    for j in range(i + 1, total):
                        if j in seen:
                            continue
                        score = fuzz.ratio(values[i], values[j])
                        if score >= fuzzy_thresh:
                            group.append(j)
                            seen.add(j)
                    if len(group) > 1:
                        seen.add(i)
                        for g in group:
                            matches.append({
                                'Group': len(matches) // max(len(group), 1) + 1,
                                'Row Index': idxs[g],
                                'Value': values[g]
                            })
                
                progress_bar.empty()
                status_text.empty()

                if matches:
                    match_df = pd.DataFrame(matches)
                    st.session_state['fuzzy_matches'] = match_df
                    st.rerun()
                else:
                    st.session_state['fuzzy_matches'] = None
                    st.success(" No fuzzy duplicates found!")

        #  SHOW RESULTS PERSISTENTLY
        if 'fuzzy_matches' in st.session_state and st.session_state['fuzzy_matches'] is not None:
            match_df = st.session_state['fuzzy_matches']
            
            st.success(f" Analysis complete!")
            
            fm1, fm2, fm3 = st.columns(3)
            with fm1:
                st.metric("Groups Found", match_df['Group'].nunique())
            with fm2:
                st.metric("Rows Involved", match_df['Row Index'].nunique())
            with fm3:
                avg = match_df.groupby('Group').size().mean()
                st.metric("Avg Group Size", f"{avg:.1f}")
            
            st.markdown("###  Fuzzy Duplicate Groups")
            st.dataframe(match_df, use_container_width=True, height=400)
            
            fuzzy_csv = match_df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Matches", fuzzy_csv,
                               file_name="fuzzy_matches.csv", mime="text/csv")
            
            if st.button(" Clear Results", key="clear_fuzzy"):
                st.session_state['fuzzy_matches'] = None
                st.rerun()

    st.markdown("---")

    # ============================================================
    # 18. ML & ANOMALY DETECTION
    # ============================================================
    st.markdown("<div id='step-3' class='workflow-anchor'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>04 · Analyze</div>", unsafe_allow_html=True)
    st.markdown("## AI-Powered Analysis")

    if not SKLEARN_OK:
        st.warning(" scikit-learn not installed. Run: `pip install scikit-learn`")
    else:
        ml1, ml2 = st.tabs([" Quick Predict", " Anomaly Detection"])

        # QUICK PREDICT TAB
        with ml1:
            st.info("Train a local machine-learning model on your data. No external API is required.")

            ml_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(df.columns) < 2:
                st.info("Need at least 2 columns.")
            else:
                mp1, mp2 = st.columns(2)
                with mp1:
                    target_col = st.selectbox(" Target (predict this):", df.columns.tolist(), key="ml_target")
                with mp2:
                    feat_options = [c for c in df.columns if c != target_col]
                    feature_cols = st.multiselect("Features (use these):", feat_options,
                                                  default=[c for c in ml_num_cols if c != target_col][:5],
                                                  key="ml_features")

                target_series = df[target_col].dropna()
                is_numeric = pd.api.types.is_numeric_dtype(target_series)
                n_unique = target_series.nunique()

                if is_numeric and n_unique > 15:
                    problem_type = "Regression"
                else:
                    problem_type = "Classification"
                st.caption(f" Detected: **{problem_type}** ({n_unique:,} unique values)")

                if st.button(" Train Model", key="ml_train_btn"):
                    if not feature_cols:
                        st.warning(" Select features")
                    else:
                        try:
                            with st.spinner(" Training..."):
                                ml_df = df[feature_cols + [target_col]].dropna()
                                if len(ml_df) > ML_SAMPLE_CAP:
                                    ml_df = ml_df.sample(ML_SAMPLE_CAP, random_state=42)

                                if len(ml_df) < 30:
                                    st.error(" Need 30+ rows")
                                else:
                                    X = pd.get_dummies(ml_df[feature_cols], drop_first=True)
                                    if X.shape[1] > 200:
                                        X = X.iloc[:, :200]
                                    y = ml_df[target_col]

                                    is_reg = problem_type == "Regression"
                                    if not is_reg:
                                        y = y.astype(str)

                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                    if is_reg:
                                        model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
                                    else:
                                        model = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)

                                    model.fit(X_train, y_train)
                                    preds = model.predict(X_test)

                                    st.markdown("###  Results")
                                    r1, r2, r3 = st.columns(3)
                                    if is_reg:
                                        r2_val = r2_score(y_test, preds)
                                        mae = mean_absolute_error(y_test, preds)
                                        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                                        with r1:
                                            st.metric("R² Score", f"{r2_val:.3f}")
                                        with r2:
                                            st.metric("MAE", f"{mae:,.2f}")
                                        with r3:
                                            st.metric("RMSE", f"{rmse:,.2f}")
                                        st.session_state['ml_summary'] = f"R²={r2_val:.3f}, MAE={mae:.2f}"
                                        
                                        fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'})
                                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                                                 y=[y_test.min(), y_test.max()],
                                                                 mode='lines', name='Perfect', line=dict(dash='dash')))
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        acc = accuracy_score(y_test, preds)
                                        with r1:
                                            st.metric("Accuracy", f"{acc*100:.1f}%")
                                        with r2:
                                            st.metric("Classes", y.nunique())
                                        with r3:
                                            st.metric("Test Rows", len(y_test))
                                        st.session_state['ml_summary'] = f"Accuracy={acc*100:.1f}%"

                                    # Feature importance
                                    imp_df = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False).head(15)
                                    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                                 title='Feature Importance', color='Importance',
                                                 text_auto='.3f')
                                    fig.update_traces(textposition='outside', cliponaxis=False)
                                    fig.update_layout(
                                        yaxis={'categoryorder': 'total ascending'},
                                        height=max(350, 40 * len(imp_df)),
                                        uniformtext_minsize=9, uniformtext_mode='hide'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.session_state['ml_importance'] = imp_df
                                    
                                    if 3 not in st.session_state.completed_steps:
                                        st.session_state.completed_steps.append(3)
                                        st.session_state.workflow_step = 4

                        except Exception as e:
                            st.error(f" Failed: {e}")

        # ANOMALY DETECTION TAB
        with ml2:
            st.info("Isolation Forest identifies unusual rows that may represent errors, rare cases, or potential anomalies.")

            an_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(an_num_cols) < 1:
                st.info("Need numeric columns")
            else:
                an1, an2 = st.columns(2)
                with an1:
                    anomaly_cols = st.multiselect("Columns:", an_num_cols, default=an_num_cols[:4], key="anomaly_cols")
                with an2:
                    contamination = st.slider("Expected Anomaly %:", 1, 20, 5, 1, key="contamination") / 100

                if st.button(" Detect", key="anomaly_btn"):
                    if not anomaly_cols:
                        st.warning(" Select columns")
                    else:
                        try:
                            with st.spinner(" Scanning..."):
                                an_data = df[anomaly_cols].dropna()
                                if len(an_data) < 20:
                                    st.error(" Need 20+ rows")
                                else:
                                    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                                    labels = iso.fit_predict(an_data)
                                    scores = iso.decision_function(an_data)
                                    st.session_state['anomaly_result'] = {
                                        'index': an_data.index[labels == -1].tolist(),
                                        'cols': anomaly_cols,
                                        'labels': labels,
                                        'scores': scores,
                                        'data_index': an_data.index.tolist(),
                                        'n_scanned': len(an_data),
                                    }
                                    st.rerun()
                        except Exception as e:
                            st.error(f" Failed: {e}")

                # Show results persistently
                ar = st.session_state.get('anomaly_result')
                if ar:
                    anomaly_idx = [i for i in ar['index'] if i in df.index]
                    n_anom = len(anomaly_idx)

                    am1, am2, am3 = st.columns(3)
                    with am1:
                        st.metric("Anomalies", f"{n_anom:,}")
                    with am2:
                        st.metric("Scanned", f"{ar['n_scanned']:,}")
                    with am3:
                        st.metric("Rate", f"{n_anom/max(ar['n_scanned'],1)*100:.1f}%")

                    st.session_state['anomaly_summary'] = f"{n_anom:,} anomalies in {ar['n_scanned']:,} rows"

                    plot_cols = [c for c in ar['cols'] if c in df.columns]
                    valid_idx = [i for i in ar['data_index'] if i in df.index]
                    if len(plot_cols) >= 2 and valid_idx:
                        plot_df = df.loc[valid_idx, plot_cols].copy()
                        plot_df['Status'] = np.where(plot_df.index.isin(anomaly_idx), 'Anomaly', 'Normal')
                        if len(plot_df) > 5000:
                            # keep all anomalies visible, sample down the normal points only
                            anomalies_part = plot_df[plot_df['Status'] == 'Anomaly']
                            normal_part = plot_df[plot_df['Status'] == 'Normal'].sample(
                                min(5000, len(plot_df[plot_df['Status'] == 'Normal'])), random_state=42
                            )
                            plot_df = pd.concat([anomalies_part, normal_part])
                            st.caption(" Showing all anomalies + a sample of normal points for smooth performance.")
                        fig = px.scatter(plot_df, x=plot_cols[0], y=plot_cols[1], color='Status',
                                         color_discrete_map={'Anomaly': 'red', 'Normal': 'lightblue'})
                        st.plotly_chart(fig, use_container_width=True)

                    if n_anom > 0:
                        with st.expander(f" View {n_anom} Anomalies"):
                            if n_anom > 500:
                                st.caption(f" Showing first 500 of {n_anom:,} anomaly rows for smooth performance.")
                            st.dataframe(df.loc[anomaly_idx].head(500), use_container_width=True)

                        if st.button(" Remove", key="rm_anomaly_btn"):
                            new_df = df.drop(index=anomaly_idx)
                            apply_change(new_df, " Remove Anomalies", f"{n_anom} rows removed")
                            st.session_state.pop('anomaly_result', None)
                            st.success(f" Removed!")
                            st.rerun()

    st.caption(" **Next Step:** Explore the charts to understand your cleaned data, then go to 'Step 5' to download your results.")
    st.markdown("---")

    # ============================================================
    # 19. VISUALIZATIONS (ENHANCED)
    # ============================================================
    st.markdown("<div id='step-analysis' class='workflow-anchor'></div>", unsafe_allow_html=True)
    st.markdown("### Visual Analytics Workspace")

    if st.session_state.user_mode == 'beginner':
        st.info(" Explore your data with interactive charts")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    v1, v2, v3, v4, v5, v6 = st.tabs(
        ["Numeric", "Categorical", "Correlation", "Trend", "Pivot Table", "Geo Map"]
    )

    with v1:
        if num_cols:
            sel_num = st.selectbox("Column:", num_cols, key="num_viz")
            vc1, vc2 = st.columns(2)
            with vc1:
                fig = px.histogram(df, x=sel_num, nbins=30, title=f'Distribution: {sel_num}', text_auto=True)
                fig.update_traces(textposition='outside', textfont_size=9, cliponaxis=False)
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)
            with vc2:
                fig = px.box(df, y=sel_num, title=f'Box Plot: {sel_num}')
                st.plotly_chart(fig, use_container_width=True)

            if len(num_cols) >= 2:
                st.markdown("**Scatter Plot:**")
                sc1, sc2 = st.columns(2)
                with sc1:
                    x_axis = st.selectbox("X:", num_cols, key="scatter_x")
                with sc2:
                    y_axis = st.selectbox("Y:", [c for c in num_cols if c != x_axis], key="scatter_y")

                scatter_plot_df = df
                if len(df) > 5000:
                    scatter_plot_df = df.sample(5000, random_state=42)
                    st.caption(f" Showing a random sample of 5,000 of {len(df):,} rows for smooth performance.")

                fig = px.scatter(scatter_plot_df, x=x_axis, y=y_axis, title=f'{x_axis} vs {y_axis}')
                st.plotly_chart(fig, use_container_width=True)
                if 3 not in st.session_state.completed_steps:
                    st.session_state.completed_steps.append(3)
                    st.session_state.workflow_step = 4
        else:
            st.info("No numeric columns")

    with v2:
        if cat_cols:
            sel_cat = st.selectbox("Column:", cat_cols, key="cat_viz")
            vc = df[sel_cat].value_counts().head(15).reset_index()
            vc.columns = [sel_cat, 'Count']
            fig = px.bar(vc, x=sel_cat, y='Count', title=f'Top 15: {sel_cat}', color='Count', text_auto=True)
            fig.update_traces(textposition='outside', textfont_size=10, cliponaxis=False)
            fig.update_layout(
                xaxis_tickangle=-40,
                uniformtext_minsize=8, uniformtext_mode='hide',
                margin=dict(b=120)
            )
            st.plotly_chart(fig, use_container_width=True)
            if 3 not in st.session_state.completed_steps:
                st.session_state.completed_steps.append(3)
                st.session_state.workflow_step = 4
        else:
            st.info("No categorical columns")

    with v3:
        if len(num_cols) >= 2:
            corr = cached_corr(df[num_cols])
            n_c = len(num_cols)
            fig = px.imshow(corr, text_auto='.2f', title='Correlation Heatmap', color_continuous_scale='RdBu_r')
            fig.update_traces(textfont_size=max(8, 14 - n_c // 2))
            fig.update_layout(height=max(400, 35 * n_c))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 2+ numeric columns")

    with v4:
        if date_cols and num_cols:
            dc1, dc2 = st.columns(2)
            with dc1:
                x_date = st.selectbox("Date:", date_cols, key="trend_x")
            with dc2:
                y_val = st.selectbox("Value:", num_cols, key="trend_y")
            trend_df = df.dropna(subset=[x_date, y_val]).sort_values(x_date)
            fig = px.line(trend_df, x=x_date, y=y_val, title=f'{y_val} Over Time')
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(" Time Series Decomposition (Trend / Seasonality / Residual)"):
                st.caption("Breaks the series into Trend, Seasonal pattern, and Residual noise. "
                           "Needs enough data points to detect a repeating pattern.")
                period_guess = st.number_input(
                    "Seasonal Period (e.g. 7 = weekly, 12 = monthly, 365 = yearly):",
                    min_value=2, value=7, step=1, key="decomp_period"
                )
                if st.button("Run Decomposition", key="decomp_btn"):
                    try:
                        ts = trend_df.set_index(x_date)[y_val]
                        inferred = pd.infer_freq(ts.index)
                        ts = ts.asfreq(inferred or 'D')
                        ts = ts.interpolate()

                        if len(ts) < 2 * period_guess:
                            st.warning(f" Need at least {2 * period_guess} data points for period={period_guess}. "
                                       f"This dataset has {len(ts)}.")
                        else:
                            result = seasonal_decompose(ts, model='additive', period=int(period_guess))
                            decomp_df = pd.DataFrame({
                                'Date': ts.index,
                                'Trend': result.trend.values,
                                'Seasonal': result.seasonal.values,
                                'Residual': result.resid.values
                            })
                            st.plotly_chart(px.line(decomp_df, x='Date', y='Trend', title='Trend Component'),
                                             use_container_width=True)
                            st.plotly_chart(px.line(decomp_df, x='Date', y='Seasonal', title='Seasonal Component'),
                                             use_container_width=True)
                            st.plotly_chart(px.scatter(decomp_df, x='Date', y='Residual', title='Residual (Noise)'),
                                             use_container_width=True)
                    except Exception as e:
                        st.error(f" Decomposition failed: {e}. Try a different period, or check for gaps in dates.")
        else:
            st.info("Need date + numeric column. Convert a text column to Date type in "
                     "' Fix Data Types' (Step 4) above.")

    with v5:
        st.markdown("**Build a Pivot Table** (group data and aggregate values):")
        pv1, pv2, pv3 = st.columns(3)
        with pv1:
            pivot_rows = st.multiselect("Rows (group by):", df.columns.tolist(), key="pivot_rows")
        with pv2:
            pivot_cols_sel = st.multiselect("Columns (optional split by):",
                                             [c for c in cat_cols if c not in pivot_rows], key="pivot_cols")
        with pv3:
            pivot_val = st.selectbox("Value Column (to aggregate):", num_cols if num_cols else df.columns.tolist(),
                                      key="pivot_val")

        agg_func = st.selectbox("Aggregation:", ["sum", "mean", "count", "min", "max", "median"], key="pivot_agg")

        if st.button(" Generate Pivot Table", key="pivot_btn"):
            if not pivot_rows:
                st.warning(" Select at least one column for Rows.")
            else:
                try:
                    pivot_result = pd.pivot_table(
                        df, index=pivot_rows,
                        columns=pivot_cols_sel if pivot_cols_sel else None,
                        values=pivot_val, aggfunc=agg_func, fill_value=0
                    )
                    st.session_state['pivot_result'] = pivot_result
                    st.dataframe(pivot_result, use_container_width=True)

                    pivot_csv = pivot_result.to_csv().encode('utf-8')
                    st.download_button(" Download Pivot Table (CSV)", pivot_csv,
                                        file_name="pivot_table.csv", mime="text/csv", key="pivot_dl_btn")
                except Exception as e:
                    st.error(f" Pivot failed: {e}")

    with v6:
        st.markdown("**Geo Map** (plots countries/regions on a world map — no API key needed):")
        st.caption(" Works best with a column containing full country names (e.g. 'India', 'United States').")

        geo_col = st.selectbox("Select Location Column (country names):",
                                cat_cols if cat_cols else df.columns.tolist(), key="geo_col")
        geo_val_col = st.selectbox("Value Column (color intensity, optional):",
                                    ["Count of Rows"] + num_cols, key="geo_val_col")

        if st.button(" Generate Map", key="geo_btn"):
            try:
                if geo_val_col == "Count of Rows":
                    geo_data = df[geo_col].value_counts().reset_index()
                    geo_data.columns = [geo_col, 'Value']
                else:
                    geo_data = df.groupby(geo_col)[geo_val_col].sum().reset_index()
                    geo_data.columns = [geo_col, 'Value']

                fig = px.choropleth(
                    geo_data, locations=geo_col, locationmode='country names',
                    color='Value', color_continuous_scale='Blues', title='Geographic Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(" If the map looks empty, your column may not match standard country names.")
            except Exception as e:
                st.error(f" Map generation failed: {e}")

    st.caption(" **Next Step:** Done! You can now download your cleaned file in CSV or Excel format below — "
           "all data will be automatically cleared when you close the tab.")
    st.markdown("---")

    # ============================================================
    # 20. DOWNLOAD SECTION (ENHANCED)
    # ============================================================
    st.markdown("<div id='step-4' class='workflow-anchor'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dc-section-label'>05 · Export</div>", unsafe_allow_html=True)
    st.markdown("## Download Results")
    
    if st.session_state.user_mode == 'beginner':
        st.info(" Download your cleaned data before closing this tab - nothing is saved on our servers!")

    final_df = st.session_state.current_df

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        csv_out = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
    " Download CSV",
    csv_out,
    file_name=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
    use_container_width=True,
    on_click=mark_download_complete
)

    with dl2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Data')
        st.download_button(
    " Download Excel",
    buffer.getvalue(),
    file_name=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
    on_click=mark_download_complete
)

    with dl3:
        pipeline_text = "\n".join(f"{s['Step']}. {s['Action']} - {s['Detail']}" for s in st.session_state.pipeline_log)
        report = f"""DATA CLEANING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Quality Score: {st.session_state.data_quality_score:.0f}/100
Improvement: +{st.session_state.data_quality_score - st.session_state.initial_quality_score:.0f}

STEPS APPLIED:
{pipeline_text}

FINAL DATA:
Rows: {final_df.shape[0]:,}
Columns: {final_df.shape[1]}
Missing Values: {final_df.isnull().sum().sum():,}
"""
        st.download_button(
    " Download Report",
    report,
    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
    use_container_width=True,
    on_click=mark_download_complete
)

    # PDF Report
    if FPDF_OK:
        st.markdown("###  Professional PDF Report")
        
        def _safe(text):
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF"
                u"\U0001F680-\U0001F6FF"
                u"\U0001F1E0-\U0001F1FF"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            cleaned = emoji_pattern.sub('', str(text))
            try:
                cleaned.encode('latin-1')
                return cleaned
            except:
                return cleaned.encode('latin-1', 'replace').decode('latin-1')

        def build_pdf():
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_margins(10, 15, 10)
            pdf.add_page()

            def pdf_text(text, height=6):
                """Render a full-width paragraph safely.

                Always reset to the left margin and use the effective page width.
                CHAR wrapping also prevents fpdf2 from failing on long unbroken
                strings such as column names, filenames, or generated values.
                """
                value = _safe(text).replace('\x00', '')
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    pdf.epw,
                    height,
                    value,
                    wrapmode='CHAR',
                    new_x='LMARGIN',
                    new_y='NEXT'
                )
            
            # Header
            pdf.set_fill_color(79, 104, 136)
            pdf.rect(0, 0, 210, 40, 'F')
            pdf.set_font('Helvetica', 'B', 22)
            pdf.set_text_color(255, 255, 255)
            pdf.set_y(12)
            pdf.cell(0, 10, 'DATACORP DATA QUALITY REPORT', align='C')
            pdf.ln(20)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 8, _safe(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}"), align='C')
            pdf.ln(14)

            # Quality Score
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, '1. Quality Summary')
            pdf.ln(10)
            pdf.set_font('Helvetica', '', 10)
            
            summary = [
                ("Initial Quality Score", f"{st.session_state.initial_quality_score:.0f}/100"),
                ("Final Quality Score", f"{st.session_state.data_quality_score:.0f}/100"),
                ("Improvement", f"+{st.session_state.data_quality_score - st.session_state.initial_quality_score:.0f}"),
                ("Total Rows", f"{final_df.shape[0]:,}"),
                ("Total Columns", str(final_df.shape[1])),
                ("Missing Values", f"{final_df.isnull().sum().sum():,}"),
            ]
            
            for label, val in summary:
                pdf.set_fill_color(243, 244, 246)
                pdf.cell(70, 8, _safe(label), border=1, fill=True)
                pdf.cell(60, 8, _safe(val), border=1)
                pdf.ln(8)

            # Executive insights
            pdf.ln(6)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, '2. Key Insights')
            pdf.ln(10)
            pdf.set_font('Helvetica', '', 9)
            for insight in local_analyst(final_df)[:5]:
                pdf_text('- ' + insight)

            # Pipeline
            pdf.ln(6)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.cell(0, 10, '3. Cleaning Steps')
            pdf.ln(10)
            pdf.set_font('Helvetica', '', 9)
            for s in st.session_state.pipeline_log:
                action = ''.join(ch for ch in s['Action'] if ord(ch) < 256).strip()
                line = f"Step {s['Step']} [{s['Time']}] {action}: {s['Detail']}"
                pdf_text(line)

            return bytes(pdf.output())

        if st.button(" Generate PDF", key="pdf_gen", type="primary"):
            with st.spinner("Building PDF..."):
                st.session_state['pdf_report'] = build_pdf()
            st.success(" PDF ready!")

        if st.session_state.get('pdf_report'):
            st.download_button(
    " Download PDF",
    st.session_state['pdf_report'],
    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
    mime="application/pdf",
    use_container_width=True,
    on_click=mark_download_complete
)
    
else:
    st.markdown("### Start with a dataset")
    feat1, feat2, feat3 = st.columns(3)
    with feat1:
        st.markdown("""
        <div class="dc-feature">
            <div class="dc-feature-title"> Detect issues</div>
            <div class="dc-feature-text">Missing values, duplicates, type problems and outliers.</div>
        </div>
        """, unsafe_allow_html=True)
    with feat2:
        st.markdown("""
        <div class="dc-feature">
            <div class="dc-feature-title"> Clean data</div>
            <div class="dc-feature-text">Smart fixes, transformations, undo/redo and fuzzy matching.</div>
        </div>
        """, unsafe_allow_html=True)
    with feat3:
        st.markdown("""
        <div class="dc-feature">
            <div class="dc-feature-title">Analyze</div>
            <div class="dc-feature-text">Charts, predictions, anomaly detection and reports.</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(" Session-only processing · Nothing stored on servers · CSV · Excel · JSON · Up to 150,000 rows")

# Session status
if st.session_state.get('current_df') is not None:
    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.caption(f"Session dataset: {st.session_state.active_dataset}")
    with sc2:
        st.caption(f"Cleaning steps: {max(len(st.session_state.pipeline_log)-1, 0)}")
    with sc3:
        st.caption(f"Current quality: {st.session_state.data_quality_score:.0f}/100")

# Footer
st.markdown("<div class='dc-footer'>Private session · No data storage · © 2026 Datacorp</div>", unsafe_allow_html=True)
