"""Datacorp - clean, analyze and export your data."""
import re
from datetime import datetime
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Datacorp", page_icon="📊", layout="wide")
st.markdown("""<style>
    footer {visibility: hidden;}
    .block-container {padding-top: 4.5rem; max-width: 1300px;}
</style>""", unsafe_allow_html=True)

# ============================================================
# Constants
# ============================================================
MAX_ROWS = 150_000
MAX_HISTORY = 20
ML_ROW_CAP = 20_000
FUZZY_ROW_CAP = 2_000
PLOT_ROW_CAP = 5_000
SAMPLE_NAME = "sample_data.csv"
PAGES = ["Overview", "Clean", "Analyze", "Export"]
NUMBER, DATE = [np.number], ["datetime", "datetimetz"]
TEXT = ["object", "string"]  # pandas 3 stores text as 'str', which 'string' selects
SEVERITY = {"critical": "🔴 Critical", "warning": "🟠 Warning", "info": "🔵 Info"}

TEXT_OPS = {
    "Trim whitespace": lambda s: s.str.strip(),
    "Lowercase": lambda s: s.str.lower(),
    "Uppercase": lambda s: s.str.upper(),
    "Title case": lambda s: s.str.title(),
    "Remove special characters": lambda s: s.str.replace(r"[^\w\s]|_", "", regex=True),
    "Collapse extra spaces": lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip(),
}
FILLERS = {
    "Mean": lambda s, v: s.fillna(s.mean()),
    "Median": lambda s, v: s.fillna(s.median()),
    "Most common value": lambda s, v: s.fillna(s.mode().iloc[0] if not s.mode().empty else ""),
    "Custom value": lambda s, v: s.fillna(float(v) if pd.api.types.is_numeric_dtype(s) else v),
    "Forward fill": lambda s, v: s.ffill(),
    "Backward fill": lambda s, v: s.bfill(),
    "Interpolate": lambda s, v: s.interpolate(),
}
CASTS = {
    "Integer": lambda s, fmt: pd.to_numeric(s, errors="coerce").astype("Int64"),
    "Decimal": lambda s, fmt: pd.to_numeric(s, errors="coerce"),
    "Text": lambda s, fmt: as_text(s),
    "Date": lambda s, fmt: pd.to_datetime(s, format=fmt or None, errors="coerce"),
}


# ============================================================
# Data helpers
# ============================================================
def cols_of(df, kind):
    return df.select_dtypes(include=kind).columns.tolist()


def looks_numeric(series):
    """True if the first non-null values of a text column are all numbers."""
    sample = series.dropna().head(100)
    return len(sample) > 0 and bool(pd.to_numeric(sample, errors="coerce").notna().all())


def iqr_bounds(series, mult=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    return q1 - mult * (q3 - q1), q3 + mult * (q3 - q1)


def outlier_count(series):
    lower, upper = iqr_bounds(series)
    return int(((series < lower) | (series > upper)).sum())


def as_text(series):
    """Convert to strings while keeping missing values missing."""
    return series.astype(str).where(series.notna())


def with_col(df, col, values):
    out = df.copy()
    out[col] = values
    return out


def to_excel_bytes(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()


def detect_issues(df):
    """List data-quality problems. `kind` tells auto_fix how to handle each one."""
    issues = []
    if df.empty:
        return issues

    def add(severity, kind, column, issue, fix):
        issues.append({"severity": severity, "column": column, "issue": issue, "fix": fix, "kind": kind})

    for col in df.columns[df.isnull().any()]:
        pct = df[col].isnull().mean() * 100
        fix = "Fill with median" if pd.api.types.is_numeric_dtype(df[col]) else "Fill with most common value"
        if pct > 50:
            add("critical", "missing", col, f"{pct:.1f}% missing", fix)
        elif pct > 5:
            add("warning", "missing", col, f"{pct:.1f}% missing", fix)

    dups = int(df.duplicated().sum())
    if dups:
        pct = dups / len(df) * 100
        add("warning" if pct < 10 else "critical", "duplicates", "All columns",
            f"{dups:,} duplicate rows ({pct:.1f}%)", "Remove duplicate rows")

    for col in cols_of(df, TEXT):
        if looks_numeric(df[col]):
            add("warning", "numeric", col, "Stored as text but looks like numbers", "Convert to number")
        elif any(word in col.lower() for word in ("date", "time")) and df[col].notna().any():
            try:
                pd.to_datetime(df[col].dropna().head(10))
                add("warning", "date", col, "Stored as text but looks like dates", "Convert to date")
            except (ValueError, TypeError):
                pass

    for col in cols_of(df, NUMBER):
        count = outlier_count(df[col])
        if count / len(df) > 0.05:
            add("info", "outliers", col, f"{count:,} outliers ({count / len(df) * 100:.1f}%)", "Review in Clean → Outliers")
    return issues


def auto_fix(df, issues):
    """Apply the safe fixes. Returns the new DataFrame and a list of what was done."""
    df, actions = df.copy(), []
    for issue in issues:
        col, kind = issue["column"], issue["kind"]
        try:
            if kind == "missing":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = FILLERS["Most common value"](df[col], None)
                actions.append(f"Filled '{col}'")
            elif kind == "duplicates":
                before = len(df)
                df = df.drop_duplicates()
                actions.append(f"Removed {before - len(df):,} duplicate rows")
            elif kind == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                actions.append(f"'{col}' → number")
            elif kind == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")
                actions.append(f"'{col}' → date")
        except Exception as e:
            actions.append(f"Skipped '{col}' ({e})")
    return df, actions


def insights(df):
    """A few plain-language observations about the data."""
    found = []
    numeric = cols_of(df, NUMBER)
    if len(numeric) >= 2:
        corr = df[numeric].corr()
        pairs = [(corr.loc[a, b], a, b) for i, a in enumerate(numeric) for b in numeric[i + 1:] if pd.notna(corr.loc[a, b])]
        if pairs:
            r, a, b = max(pairs, key=lambda p: abs(p[0]))
            if abs(r) >= 0.7:
                found.append(f"'{a}' and '{b}' are strongly correlated (r = {r:.2f}).")
    for col in cols_of(df, TEXT + ["category"])[:3]:
        shares = df[col].value_counts(normalize=True)
        if 1 < len(shares) <= 15:
            found.append(f"'{col}' has {len(shares)} distinct values; '{shares.index[0]}' is the most common ({shares.iloc[0] * 100:.0f}%).")
    missing = df.isnull().sum()
    if missing.max() > 0:
        found.append(f"'{missing.idxmax()}' has the most missing values ({missing.max():,}).")
    return found or ["No notable patterns found."]


@st.cache_data(ttl=1800, max_entries=6, show_spinner="Reading file...")
def parse_file(data: bytes, name: str) -> pd.DataFrame:
    reader = {"csv": pd.read_csv, "xlsx": pd.read_excel, "json": pd.read_json}[name.lower().rsplit(".", 1)[-1]]
    df = reader(BytesIO(data))
    df.columns = df.columns.astype(str)
    nested = [c for c in df.columns[df.dtypes == object] if df[c].map(lambda v: isinstance(v, (list, dict))).any()]
    return df.astype({c: str for c in nested})


@st.cache_data
def make_sample():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "name": [f"  Customer {i}  " if i % 7 == 0 else f"Customer {i}" for i in range(1, n + 1)],
        "age": rng.integers(18, 70, n).astype(float),
        "city": rng.choice(["Mumbai", "Delhi", "Bangalore", "Chennai", None], n, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        "purchase_amount": np.concatenate([rng.normal(2000, 500, n - 10), rng.normal(20000, 2000, 10)]),  # 10 outliers
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "rating": rng.choice([1, 2, 3, 4, 5, np.nan], n, p=[0.05, 0.05, 0.15, 0.35, 0.35, 0.05]),
    })
    df.loc[df.sample(20, random_state=1).index, "age"] = None
    return pd.concat([df, df.iloc[:8]], ignore_index=True)  # 8 duplicate rows


# ============================================================
# Cleaning history (every step is a snapshot, so undo/redo just moves a pointer)
# ============================================================
ss = st.session_state


def snapshot(df, action, detail=""):
    issues = detect_issues(df)
    missing = int(df.isnull().sum().sum())
    dups = 0 if df.empty else int(df.duplicated().sum())
    score = 0 if df.empty else max(0, 100 - min(missing / df.size * 100, 30) - min(dups / len(df) * 100, 20)
                                   - 2 * sum(i["kind"] == "numeric" for i in issues))
    return {"df": df, "Action": action, "Detail": detail, "Time": datetime.now().strftime("%H:%M:%S"),
            "Shape": f"{df.shape[0]:,} × {df.shape[1]}", "issues": issues, "score": score,
            "missing": missing, "duplicates": dups}


def commit(df, action, detail=""):
    ss.steps = ss.steps[:ss.pos + 1] + [snapshot(df, action, detail)]
    if len(ss.steps) > MAX_HISTORY + 1:
        del ss.steps[1]  # keep the original, drop the oldest change
    ss.pos = len(ss.steps) - 1
    ss.flash = f"{action}: {detail}" if detail else action


def attempt(action, detail, build):
    """Run build() -> new DataFrame, then commit it (or show the error)."""
    try:
        new_df = build()
    except Exception as e:
        st.error(f"{action} failed: {e}")
        return
    commit(new_df, action, detail)
    st.rerun()


# ============================================================
# Clean page: one small function per tool
# ============================================================
def tool_missing(df):
    cols = df.columns[df.isnull().any()].tolist()
    if not cols:
        return st.success("No missing values.")
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Column", cols)
    method = c2.selectbox("Method", ["Drop rows"] + list(FILLERS))
    value = c3.text_input("Value") if method == "Custom value" else ""
    if st.button("Apply", type="primary", disabled=method == "Custom value" and not value):
        attempt("Missing values", f"{col}: {method}",
                lambda: df.dropna(subset=[col]) if method == "Drop rows" else with_col(df, col, FILLERS[method](df[col], value)))


def tool_duplicates(df):
    count = int(df.duplicated().sum())
    st.metric("Exact duplicate rows", f"{count:,}")
    if st.button("Remove duplicates", type="primary", disabled=count == 0):
        attempt("Remove duplicates", f"{count:,} rows", df.drop_duplicates)


def tool_fuzzy(df):
    cols = cols_of(df, TEXT)
    if not cols:
        return st.info("No text columns.")
    c1, c2 = st.columns(2)
    col = c1.selectbox("Text column", cols)
    threshold = c2.slider("Similarity (%)", 70, 99, 90)
    st.caption(f"Compares the first {FUZZY_ROW_CAP:,} non-empty values. Case and extra spaces are ignored.")
    if not st.button("Find similar values", type="primary"):
        return
    values = df[col].dropna().astype(str).head(FUZZY_ROW_CAP)
    texts, names = values.tolist(), [" ".join(t.lower().split()) for t in values]
    seen, rows, groups = set(), [], 0
    with st.spinner("Comparing values..."):
        for i in range(len(texts)):
            if i in seen:
                continue
            group = [i] + [j for j in range(i + 1, len(texts)) if j not in seen and fuzz.ratio(names[i], names[j]) >= threshold]
            if len(group) > 1:
                groups += 1
                seen.update(group)
                rows += [{"Group": groups, "Row": values.index[g], "Value": texts[g]} for g in group]
    if not rows:
        return st.success("No similar values found.")
    result = pd.DataFrame(rows)
    st.metric("Groups found", groups)
    st.dataframe(result, hide_index=True, width="stretch")
    st.download_button("Download matches (CSV)", partial(result.to_csv, index=False), file_name="fuzzy_matches.csv",
                       mime="text/csv", on_click="ignore")


def tool_columns(df):
    st.markdown("**Rename a column**")
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    col = c1.selectbox("Column", df.columns.tolist(), key="rename_col")
    new = c2.text_input("New name", value=col, key=f"rename_{col}")
    if c3.button("Rename"):
        if new and new not in df.columns:
            attempt("Rename column", f"{col} → {new}", lambda: df.rename(columns={col: new}))
        else:
            st.warning("Enter a new name that no other column uses.")
    st.markdown("**Drop columns**")
    drop = st.multiselect("Columns to drop", df.columns.tolist())
    if st.button("Drop selected", disabled=not drop or len(drop) == df.shape[1]):
        attempt("Drop columns", ", ".join(drop), lambda: df.drop(columns=drop))


def tool_text(df):
    cols = cols_of(df, TEXT)
    if not cols:
        return st.info("No text columns.")
    c1, c2 = st.columns(2)
    col, op = c1.selectbox("Column", cols), c2.selectbox("Action", list(TEXT_OPS))
    if st.button("Apply", type="primary"):
        attempt("Text cleaning", f"{col}: {op}", lambda: with_col(df, col, TEXT_OPS[op](as_text(df[col]))))


def tool_replace(df):
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Column", df.columns.tolist())
    find, repl = c2.text_input("Find"), c3.text_input("Replace with")
    regex = st.checkbox("Treat 'Find' as a regex pattern")
    if st.button("Replace", type="primary", disabled=not find):
        attempt("Find & replace", f"{col}: '{find}' → '{repl}'",
                lambda: with_col(df, col, as_text(df[col]).str.replace(find, repl, regex=regex)))


def tool_dtype(df):
    c1, c2, c3 = st.columns(3)
    col = c1.selectbox("Column", df.columns.tolist())
    c1.caption(f"Current type: `{df[col].dtype}`")
    kind = c2.selectbox("Convert to", list(CASTS))
    fmt = c3.text_input("Date format (optional)", placeholder="%d-%m-%Y") if kind == "Date" else ""
    if st.button("Convert", type="primary"):
        try:
            new_df = with_col(df, col, CASTS[kind](df[col], fmt))
        except Exception as e:
            return st.error(f"Conversion failed: {e}")
        lost = int(new_df[col].isna().sum() - df[col].isna().sum())
        note = f" ({lost} unparseable value(s) became empty)" if lost > 0 else ""
        commit(new_df, "Change type", f"{col}: {df[col].dtype} → {kind}{note}")
        st.rerun()


def tool_outliers(df):
    cols = cols_of(df, NUMBER)
    if not cols:
        return st.info("No numeric columns.")
    c1, c2, c3 = st.columns(3)
    col, method = c1.selectbox("Column", cols), c2.selectbox("Method", ["IQR", "Z-score"])
    data = df[col].dropna()
    if method == "IQR":
        lower, upper = iqr_bounds(data, c3.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1))
    else:
        z = c3.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
        lower, upper = data.mean() - z * data.std(), data.mean() + z * data.std()
    mask = ((df[col] < lower) | (df[col] > upper)).fillna(False)
    count = int(mask.sum())
    left, right = st.columns([1, 2])
    left.metric("Outliers found", f"{count:,}")
    left.caption(f"Normal range: {lower:,.2f} to {upper:,.2f}")
    right.plotly_chart(px.box(df, x=col, points="outliers", height=200), width="stretch")
    if count:
        with st.expander("Show outlier rows"):
            st.dataframe(df[mask].head(500), width="stretch")
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("Remove outliers"):
            attempt("Remove outliers", f"{col}: {count:,} rows", lambda: df[~mask])
        if b2.button("Cap outliers"):
            attempt("Cap outliers", f"{col}: clipped to normal range", lambda: with_col(df, col, df[col].clip(lower, upper)))


def tool_merge(df):
    others = [n for n in ss.sources if n != ss.name]
    other = st.selectbox("Merge with", others)
    right = ss.sources[other]
    c1, c2, c3 = st.columns(3)
    left_key = c1.selectbox("Key in this data", df.columns.tolist())
    right_key = c2.selectbox(f"Key in {other}", right.columns.tolist())
    how = c3.selectbox("Join type", ["inner", "left", "right", "outer"],
                       help="inner: only matching rows · left: all rows of this data · right: all rows of the other file · outer: everything")
    st.caption(f"{df[left_key].isin(right[right_key]).mean() * 100:.0f}% of rows in this data have a match in {other}.")
    if st.button("Merge", type="primary"):
        attempt("Merge", f"with {other} on {left_key} = {right_key} ({how})",
                lambda: df.merge(right, left_on=left_key, right_on=right_key, how=how))


# ============================================================
# Analyze page: one small function per view
# ============================================================
def view_distribution(df):
    cols = cols_of(df, NUMBER)
    if not cols:
        return st.info("No numeric columns.")
    col = st.selectbox("Column", cols)
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}"), width="stretch")
    right.plotly_chart(px.box(df, y=col, title=f"Box plot of {col}"), width="stretch")


def view_scatter(df):
    cols = cols_of(df, NUMBER)
    if len(cols) < 2:
        return st.info("Needs at least two numeric columns.")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", cols)
    y = c2.selectbox("Y axis", [c for c in cols if c != x])
    if len(df) > PLOT_ROW_CAP:
        st.caption(f"Showing a random sample of {PLOT_ROW_CAP:,} of {len(df):,} rows.")
        df = df.sample(PLOT_ROW_CAP, random_state=42)
    st.plotly_chart(px.scatter(df, x=x, y=y, title=f"{x} vs {y}"), width="stretch")


def view_categories(df):
    cols = cols_of(df, TEXT + ["category"])
    if not cols:
        return st.info("No text columns.")
    col = st.selectbox("Column", cols)
    counts = df[col].value_counts().head(15).rename_axis(col).reset_index(name="Count")
    st.plotly_chart(px.bar(counts, x=col, y="Count", text_auto=True, title=f"Top 15 values of {col}"), width="stretch")


def view_correlation(df):
    cols = cols_of(df, NUMBER)
    if len(cols) < 2:
        return st.info("Needs at least two numeric columns.")
    fig = px.imshow(df[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    height=max(400, 40 * len(cols)))
    st.plotly_chart(fig, width="stretch")


def view_trend(df):
    dates, nums = [c for c in cols_of(df, DATE) if df[c].notna().any()], cols_of(df, NUMBER)
    if not dates or not nums:
        return st.info("Needs a date column and a numeric column. Convert text to dates in Clean → Change data type.")
    c1, c2 = st.columns(2)
    x, y = c1.selectbox("Date column", dates), c2.selectbox("Value column", nums)
    data = df.dropna(subset=[x, y]).sort_values(x)
    st.plotly_chart(px.line(data, x=x, y=y, title=f"{y} over time"), width="stretch")
    if not st.toggle("Show trend / seasonality breakdown"):
        return
    period = int(st.number_input("Seasonal period (7 = weekly, 12 = monthly)", min_value=2, value=7))
    try:
        series = data.groupby(x)[y].mean()
        series = series.asfreq(pd.infer_freq(series.index) or "D").interpolate()
        if len(series) < 2 * period:
            return st.warning(f"Needs at least {2 * period} data points for this period; found {len(series)}.")
        parts = seasonal_decompose(series, model="additive", period=period)
    except Exception as e:
        return st.error(f"Breakdown failed: {e}")
    long = pd.DataFrame({"Trend": parts.trend, "Seasonal": parts.seasonal, "Residual": parts.resid}) \
        .rename_axis("Date").reset_index().melt("Date", var_name="Component")
    fig = px.line(long, x="Date", y="value", facet_row="Component", height=600)
    fig.update_yaxes(matches=None, title="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, width="stretch")


def view_pivot(df):
    c1, c2, c3, c4 = st.columns(4)
    rows = c1.multiselect("Rows", df.columns.tolist())
    columns = c2.multiselect("Columns (optional)", [c for c in cols_of(df, TEXT) if c not in rows])
    value = c3.selectbox("Values", cols_of(df, NUMBER) or df.columns.tolist())
    agg = c4.selectbox("Aggregate", ["sum", "mean", "count", "min", "max", "median"])
    if not rows:
        return st.info("Pick at least one column for Rows.")
    try:
        table = pd.pivot_table(df, index=rows, columns=columns or None, values=value, aggfunc=agg, fill_value=0)
    except Exception as e:
        return st.error(f"Pivot failed: {e}")
    st.dataframe(table, width="stretch")
    st.download_button("Download pivot table (CSV)", partial(table.to_csv), file_name="pivot_table.csv",
                       mime="text/csv", on_click="ignore")


def view_map(df):
    st.caption("Plots country names (e.g. 'India', 'United States') on a world map.")
    c1, c2 = st.columns(2)
    place = c1.selectbox("Country column", cols_of(df, TEXT) or df.columns.tolist())
    value = c2.selectbox("Color by", ["Count of rows"] + cols_of(df, NUMBER))
    if value == "Count of rows":
        data = df[place].value_counts().rename_axis(place).reset_index(name="Value")
    else:
        data = df.groupby(place, as_index=False)[value].sum().rename(columns={value: "Value"})
    st.plotly_chart(px.choropleth(data, locations=place, locationmode="country names", color="Value",
                                  color_continuous_scale="Blues"), width="stretch")


def view_predict(df):
    c1, c2 = st.columns(2)
    target = c1.selectbox("Predict this column", df.columns.tolist())
    options = [c for c in df.columns if c != target]
    features = c2.multiselect("Using these columns", options, key=f"features_{target}",
                              default=[c for c in cols_of(df, NUMBER) if c != target][:5])
    is_reg = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 15
    c1.caption(f"Detected: {'regression' if is_reg else 'classification'} ({df[target].nunique():,} unique values)")
    if not st.button("Train model", type="primary", disabled=not features):
        return
    data = df[features + [target]].dropna()
    if len(data) < 30:
        return st.error("Needs at least 30 complete rows.")
    try:
        with st.spinner("Training..."):
            data = data.sample(min(len(data), ML_ROW_CAP), random_state=42)
            X = pd.get_dummies(data[features], drop_first=True).iloc[:, :200]
            y = data[target] if is_reg else data[target].astype(str)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = (RandomForestRegressor if is_reg else RandomForestClassifier)(n_estimators=60, random_state=42, n_jobs=-1)
            pred = model.fit(X_train, y_train).predict(X_test)
    except Exception as e:
        return st.error(f"Training failed: {e}")
    m1, m2, m3 = st.columns(3)
    if is_reg:
        m1.metric("R² score", f"{r2_score(y_test, pred):.3f}", border=True)
        m2.metric("MAE", f"{mean_absolute_error(y_test, pred):,.2f}", border=True)
        m3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):,.2f}", border=True)
        fig = px.scatter(x=y_test, y=pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs predicted")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line_dash="dash")
        st.plotly_chart(fig, width="stretch")
    else:
        m1.metric("Accuracy", f"{accuracy_score(y_test, pred) * 100:.1f}%", border=True)
        m2.metric("Classes", y.nunique(), border=True)
        m3.metric("Test rows", len(y_test), border=True)
    top = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).nlargest(15, "Importance")
    st.plotly_chart(px.bar(top, x="Importance", y="Feature", orientation="h", title="What drives the prediction")
                    .update_layout(yaxis={"categoryorder": "total ascending"}), width="stretch")


@st.cache_data(show_spinner="Scanning for anomalies...")
def find_anomalies(data, contamination):
    flags = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1).fit_predict(data)
    return data.index[flags == -1]


def view_anomalies(df):
    nums = cols_of(df, NUMBER)
    if not nums:
        return st.info("No numeric columns.")
    c1, c2 = st.columns(2)
    cols = c1.multiselect("Columns to scan", nums, default=nums[:4])
    share = c2.slider("Expected anomalies (%)", 1, 20, 5) / 100
    data = df[cols].dropna()
    if not cols or len(data) < 20:
        return st.info("Select columns with at least 20 complete rows.")
    found = find_anomalies(data, share)
    m1, m2, m3 = st.columns(3)
    m1.metric("Anomalies", f"{len(found):,}", border=True)
    m2.metric("Rows scanned", f"{len(data):,}", border=True)
    m3.metric("Rate", f"{len(found) / len(data) * 100:.1f}%", border=True)
    if len(cols) >= 2:
        plot = data[cols[:2]].assign(Status=np.where(data.index.isin(found), "Anomaly", "Normal"))
        normal = plot[plot["Status"] == "Normal"]
        plot = pd.concat([plot[plot["Status"] == "Anomaly"], normal.sample(min(len(normal), PLOT_ROW_CAP), random_state=42)])
        st.plotly_chart(px.scatter(plot, x=cols[0], y=cols[1], color="Status",
                                   color_discrete_map={"Anomaly": "#c05656", "Normal": "#9fb4ca"}), width="stretch")
    if len(found):
        with st.expander("Show anomalous rows"):
            st.dataframe(df.loc[found].head(500), width="stretch")
        if st.button("Remove anomalies"):
            attempt("Remove anomalies", f"{len(found):,} rows", lambda: df.drop(index=found))


# ============================================================
# Pages
# ============================================================
def page_overview(df, cur):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Data quality")
        st.progress(cur["score"] / 100, text=f"Quality score: {cur['score']:.0f} / 100")
        issues = cur["issues"]
        if not issues:
            st.success("No issues detected. Your data looks clean.")
        else:
            table = pd.DataFrame(issues).drop(columns="kind")
            table["severity"] = table["severity"].map(SEVERITY)
            st.dataframe(table.rename(columns=str.title), hide_index=True, width="stretch")
            fixable = sum(i["kind"] != "outliers" for i in issues)
            if fixable and st.button(f"Auto-fix {fixable} issue(s)", type="primary"):
                fixed, actions = auto_fix(df, issues)
                commit(fixed, "Auto-fix", "; ".join(actions))
                st.rerun()
    with right:
        st.subheader("Insights")
        for line in insights(df):
            st.markdown(f"- {line}")
    st.subheader("Data preview")
    st.dataframe(df.head(100), width="stretch")
    with st.expander("Column details and statistics"):
        st.dataframe(pd.DataFrame({"Type": df.dtypes.astype(str), "Non-null": df.count(), "Unique": df.nunique()}),
                     width="stretch")
        if cols_of(df, NUMBER):
            st.dataframe(df[cols_of(df, NUMBER)].describe().T, width="stretch")


def page_clean(df, cur):
    tools = {
        "Missing values": tool_missing,
        "Exact duplicates": tool_duplicates,
        "Similar values (fuzzy)": tool_fuzzy,
        "Rename / drop columns": tool_columns,
        "Text cleaning": tool_text,
        "Find & replace": tool_replace,
        "Change data type": tool_dtype,
        "Outliers": tool_outliers,
    }
    if len(ss.sources) > 1:
        tools["Merge files"] = tool_merge
    tool = st.selectbox("Choose a tool", list(tools), key="tool")
    with st.container(border=True):
        tools[tool](df)


def page_analyze(df, cur):
    views = {
        "Distribution": view_distribution,
        "Scatter plot": view_scatter,
        "Categories": view_categories,
        "Correlation": view_correlation,
        "Trend over time": view_trend,
        "Pivot table": view_pivot,
        "World map": view_map,
        "Predict (machine learning)": view_predict,
        "Anomaly detection": view_anomalies,
    }
    view = st.selectbox("Choose an analysis", list(views), key="view")
    with st.container(border=True):
        views[view](df)


def pdf_safe(text):
    text = re.sub("[\U0001F000-\U0001FFFF☀-➿]", "", text).replace("→", "->")
    return text.encode("latin-1", "replace").decode("latin-1")


def report_pdf(text):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_fill_color(79, 104, 136)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(9)
    pdf.cell(0, 10, "DATACORP REPORT", align="C")
    pdf.set_y(38)
    pdf.set_text_color(0, 0, 0)
    for line in pdf_safe(text).splitlines():
        pdf.set_font("Helvetica", "B" if line.isupper() else "", 10)
        pdf.multi_cell(pdf.epw, 6, line or " ", wrapmode="CHAR", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def page_export(df, cur):
    first = ss.steps[0]
    history = ss.steps[1:ss.pos + 1]
    steps = "\n".join(f"{n}. {s['Action']} - {s['Detail']}" for n, s in enumerate(history, 1)) or "No changes made."
    report = f"""DATACORP DATA CLEANING REPORT
Generated: {datetime.now():%Y-%m-%d %H:%M}
Dataset: {ss.name}

QUALITY
Initial score: {first['score']:.0f} / 100
Final score: {cur['score']:.0f} / 100 ({cur['score'] - first['score']:+.0f})

FINAL DATA
Rows: {len(df):,}
Columns: {df.shape[1]}
Missing values: {cur['missing']:,}
Duplicate rows: {cur['duplicates']:,}

KEY INSIGHTS
""" + "\n".join(f"- {line}" for line in insights(df)) + f"\n\nSTEPS APPLIED\n{steps}\n"

    st.subheader("Download")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    c1, c2, c3, c4 = st.columns(4)
    # Files are built when you click, not on every rerun.
    c1.download_button("Cleaned data (CSV)", partial(df.to_csv, index=False), file_name=f"cleaned_{stamp}.csv",
                       mime="text/csv", on_click="ignore", icon=":material/download:", width="stretch")
    c2.download_button("Cleaned data (Excel)", partial(to_excel_bytes, df), file_name=f"cleaned_{stamp}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       on_click="ignore", icon=":material/download:", width="stretch")
    c3.download_button("Report (TXT)", report, file_name=f"report_{stamp}.txt", mime="text/plain",
                       on_click="ignore", icon=":material/description:", width="stretch")
    c4.download_button("Report (PDF)", partial(report_pdf, report), file_name=f"report_{stamp}.pdf",
                       mime="application/pdf", on_click="ignore", icon=":material/picture_as_pdf:", width="stretch")

    st.subheader("Before and after")
    st.dataframe(pd.DataFrame({
        "": ["Rows", "Missing cells", "Duplicate rows", "Quality score"],
        "Before": [f"{len(first['df']):,}", f"{first['missing']:,}", f"{first['duplicates']:,}", f"{first['score']:.0f}"],
        "After": [f"{len(df):,}", f"{cur['missing']:,}", f"{cur['duplicates']:,}", f"{cur['score']:.0f}"],
    }), hide_index=True, width="stretch")

    st.subheader("Cleaning history")
    st.dataframe(pd.DataFrame(ss.steps[:ss.pos + 1])[["Action", "Detail", "Time", "Shape"]],
                 hide_index=True, width="stretch")


# ============================================================
# App
# ============================================================
if msg := ss.pop("flash", None):
    st.toast(msg)

with st.sidebar:
    st.title("📊 Datacorp")
    st.caption("Clean, analyze and export your data.")
    files = st.file_uploader("Upload data", type=["csv", "xlsx", "json"], accept_multiple_files=True)
    st.toggle("Use sample data", key="use_sample", help="A small made-up dataset with missing values, duplicates and outliers.")

sources, keys = {}, {}
for f in files:
    try:
        df = parse_file(f.getvalue(), f.name)
        if len(df) > MAX_ROWS:
            raise ValueError(f"{len(df):,} rows; the limit is {MAX_ROWS:,}")
        sources[f.name], keys[f.name] = df, f.file_id
    except Exception as e:
        st.sidebar.error(f"Could not load {f.name}: {e}")
if ss.use_sample:
    sources[SAMPLE_NAME], keys[SAMPLE_NAME] = make_sample(), "sample"

if not sources:
    ss.pop("active", None)
    st.title("Welcome to Datacorp")
    st.markdown("##### Find problems in your data, fix them in a few clicks, and export the result.")
    for col, (title, text) in zip(st.columns(3), [
        ("1 · Upload", "Add a CSV, Excel or JSON file in the sidebar, or switch on the sample data."),
        ("2 · Clean", "See what is wrong, auto-fix it, or use the cleaning tools. Every step can be undone."),
        ("3 · Analyze and export", "Explore charts, predictions and anomalies, then download the data and a report."),
    ]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
    st.caption(f"Up to {MAX_ROWS:,} rows. Your data stays in this session and is never stored.")
    st.stop()

ss.sources = sources
with st.sidebar:
    name = st.selectbox("Dataset", list(sources)) if len(sources) > 1 else next(iter(sources))
if ss.get("active") != (name, keys[name]):
    ss.active, ss.name = (name, keys[name]), name
    ss.steps, ss.pos = [snapshot(sources[name], "Load", f"Loaded '{name}'")], 0

with st.sidebar:
    st.divider()
    st.subheader("History")
    c1, c2 = st.columns(2)
    if c1.button("Undo", disabled=ss.pos == 0, icon=":material/undo:", width="stretch"):
        ss.pos -= 1
        st.rerun()
    if c2.button("Redo", disabled=ss.pos == len(ss.steps) - 1, icon=":material/redo:", width="stretch"):
        ss.pos += 1
        st.rerun()
    if st.button("Reset to original", disabled=ss.pos == 0, icon=":material/restart_alt:", width="stretch"):
        commit(ss.steps[0]["df"], "Reset", "Restored the original data")
        st.rerun()
    st.caption(f"{ss.pos} change(s) applied")

first, cur = ss.steps[0], ss.steps[ss.pos]
df = cur["df"]


def kpi(column, label, now, before, color="off", fmt="{:,.0f}"):
    diff = now - before
    column.metric(label, fmt.format(now), f"{diff:+,.0f}" if diff else None, delta_color=color, border=True)


k = st.columns(5)
kpi(k[0], "Rows", len(df), len(first["df"]))
kpi(k[1], "Columns", df.shape[1], first["df"].shape[1])
kpi(k[2], "Missing cells", cur["missing"], first["missing"], "inverse")
kpi(k[3], "Duplicate rows", cur["duplicates"], first["duplicates"], "inverse")
kpi(k[4], "Quality score", cur["score"], first["score"], "normal")

page = st.segmented_control("Page", PAGES, default=PAGES[0], key="page", label_visibility="collapsed") or PAGES[0]
{"Overview": page_overview, "Clean": page_clean, "Analyze": page_analyze, "Export": page_export}[page](df, cur)