import io
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Google Sheets Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Zj6XC0_BCfxvJR6e-GAM1Mhi9Gf5JH4aPGr1OPr3Tbg/edit?usp=sharing"
DEFAULT_GID = "0"


# =========================
# STYLING
# =========================
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(32, 201, 151, 0.06), transparent 24%),
                radial-gradient(circle at 90% 5%, rgba(13, 110, 253, 0.07), transparent 22%),
                #f7f9fc;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1739 0%, #102653 100%);
        }

        [data-testid="stSidebar"] * {
            color: #f5f7fb;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #101828 !important;
        }

        .hero {
            background: linear-gradient(120deg, #0b1739 0%, #173b75 58%, #087f8c 100%);
            border-radius: 22px;
            padding: 28px 30px;
            margin-bottom: 20px;
            box-shadow: 0 14px 40px rgba(11, 23, 57, 0.16);
        }

        .hero h1 {
            color: white;
            font-size: 2rem;
            line-height: 1.2;
            margin: 0 0 6px 0;
            letter-spacing: -0.02em;
        }

        .hero p {
            color: rgba(255,255,255,0.82);
            margin: 0;
            font-size: 0.98rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e8edf5;
            padding: 14px 16px;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(16, 24, 40, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #667085;
        }

        div[data-testid="stMetricValue"] {
            color: #101828;
        }

        .section-title {
            font-size: 1.14rem;
            font-weight: 750;
            color: #172b4d;
            margin: 0.3rem 0 0.7rem 0;
        }

        .info-card {
            background: white;
            border: 1px solid #e8edf5;
            border-radius: 16px;
            padding: 15px 18px;
            box-shadow: 0 4px 16px rgba(16, 24, 40, 0.04);
        }

        .small-muted {
            color: #667085;
            font-size: 0.88rem;
        }

        div[data-testid="stDataFrame"] {
            background: white;
            border-radius: 14px;
            overflow: hidden;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: white;
            border-radius: 10px 10px 0 0;
            padding-left: 16px;
            padding-right: 16px;
        }

        hr {
            border-color: #e8edf5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def extract_sheet_id(url_or_id: str) -> str:
    """Extract Google Sheet ID from a URL, or accept a raw ID."""
    value = (url_or_id or "").strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", value):
        return value

    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", value)
    if not match:
        raise ValueError("Không tìm thấy Google Sheet ID trong đường dẫn.")
    return match.group(1)


def format_number(value):
    if value is None or pd.isna(value):
        return "—"

    value = float(value)
    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"{value/1_000_000_000:,.2f} tỷ"
    if abs_value >= 1_000_000:
        return f"{value/1_000_000:,.2f} triệu"
    if abs_value >= 1_000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove unnamed/fully empty columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # Clean column names and make duplicates unique
    cleaned = []
    seen = {}
    for col in df.columns:
        name = re.sub(r"\s+", " ", str(col)).strip() or "Cột"
        seen[name] = seen.get(name, 0) + 1
        cleaned.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    df.columns = cleaned

    # Clean object cells
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x
        )

    return df.reset_index(drop=True)


def try_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Convert text-like numeric columns intelligently.
    Supports values such as:
    1,234.56 | 1.234,56 | 1 234 567 | 2.500.000 đ | 35%
    """
    if series.dtype != "object":
        return series

    s = series.astype("string").str.strip()
    non_empty = s.dropna()
    non_empty = non_empty[non_empty != ""]

    if len(non_empty) == 0:
        return series

    cleaned = (
        s.str.replace(r"[₫đĐ$€£¥%]", "", regex=True)
         .str.replace("\u00a0", "", regex=False)
         .str.replace(" ", "", regex=False)
    )

    # Detect common Vietnamese thousands separator: 2.500.000
    dot_thousands_ratio = cleaned.str.match(r"^-?\d{1,3}(\.\d{3})+$", na=False).mean()
    comma_thousands_ratio = cleaned.str.match(r"^-?\d{1,3}(,\d{3})+$", na=False).mean()

    if dot_thousands_ratio > 0.45:
        cleaned = cleaned.str.replace(".", "", regex=False)
        cleaned = cleaned.str.replace(",", ".", regex=False)
    elif comma_thousands_ratio > 0.45:
        cleaned = cleaned.str.replace(",", "", regex=False)
    else:
        # Mixed formats: infer by last separator
        def standardize_number(x):
            if x is pd.NA or x is None:
                return x
            x = str(x)

            if "." in x and "," in x:
                if x.rfind(",") > x.rfind("."):
                    x = x.replace(".", "").replace(",", ".")
                else:
                    x = x.replace(",", "")
            elif "," in x:
                # Treat comma as decimal only if 1-2 digits follow it
                if re.match(r"^-?\d+,\d{1,2}$", x):
                    x = x.replace(",", ".")
                else:
                    x = x.replace(",", "")
            return x

        cleaned = cleaned.map(standardize_number)

    numeric = pd.to_numeric(cleaned, errors="coerce")
    success_ratio = numeric.notna().sum() / max(len(non_empty), 1)

    # Avoid converting ID / code-like columns too aggressively
    unique_ratio = non_empty.nunique(dropna=True) / max(len(non_empty), 1)
    if success_ratio >= 0.82 and not (
        unique_ratio > 0.95
        and non_empty.str.len().median() >= 8
        and non_empty.str.match(r"^\d+$", na=False).mean() > 0.9
    ):
        return numeric

    return series


def try_convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_keywords = (
        "ngày", "date", "time", "tháng", "month", "năm", "year",
        "created", "updated", "checkin", "checkout", "booking"
    )

    for col in df.columns:
        if df[col].dtype != "object":
            continue

        col_lower = str(col).lower()
        sample = df[col].dropna().astype(str).head(200)

        if len(sample) == 0:
            continue

        keyword_hint = any(k in col_lower for k in date_keywords)
        pattern_hint = sample.str.contains(
            r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", regex=True
        ).mean() >= 0.55

        if keyword_hint or pattern_hint:
            converted = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if converted.notna().mean() >= 0.65:
                df[col] = converted

    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_google_sheet(sheet_url_or_id: str, gid: str) -> pd.DataFrame:
    sheet_id = extract_sheet_id(sheet_url_or_id)
    gid = str(gid).strip() or "0"

    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/"
        f"gviz/tq?tqx=out:csv&gid={gid}"
    )

    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    text = response.text

    # Private / unavailable Sheets commonly return HTML instead of CSV
    if "text/html" in content_type or "<html" in text[:500].lower():
        raise PermissionError(
            "Google Sheet chưa cho phép truy cập công khai hoặc GID không hợp lệ. "
            "Hãy đặt quyền chia sẻ thành “Anyone with the link / Bất kỳ ai có đường liên kết – Viewer”."
        )

    df = pd.read_csv(io.StringIO(text))
    df = normalize_columns(df)

    for col in df.columns:
        df[col] = try_convert_numeric(df[col])

    df = try_convert_dates(df)

    return df


def get_column_groups(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    date_cols = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols + date_cols
    ]
    return numeric_cols, date_cols, categorical_cols


def apply_filters(df, date_cols, categorical_cols):
    filtered = df.copy()

    st.sidebar.markdown("### 🔎 Bộ lọc dữ liệu")

    # Date filter
    if date_cols:
        date_col = st.sidebar.selectbox(
            "Cột thời gian",
            ["Không lọc"] + date_cols,
            index=1 if date_cols else 0,
        )

        if date_col != "Không lọc":
            valid_dates = filtered[date_col].dropna()
            if not valid_dates.empty:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()

                selected_range = st.sidebar.date_input(
                    "Khoảng thời gian",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )

                if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
                    start_date, end_date = selected_range
                    mask = (
                        filtered[date_col].dt.date.ge(start_date)
                        & filtered[date_col].dt.date.le(end_date)
                    )
                    filtered = filtered[mask | filtered[date_col].isna()]

    # Category filters: show only manageable-cardinality columns
    filterable = []
    for col in categorical_cols:
        nunique = filtered[col].nunique(dropna=True)
        if 1 < nunique <= 80:
            filterable.append(col)

    selected_filter_cols = st.sidebar.multiselect(
        "Thêm bộ lọc danh mục",
        options=filterable,
        default=filterable[:2],
        help="Chọn các cột muốn dùng làm bộ lọc.",
    )

    for col in selected_filter_cols:
        options = sorted(
            filtered[col].dropna().astype(str).unique().tolist(),
            key=lambda x: x.lower(),
        )

        chosen = st.sidebar.multiselect(
            f"Lọc: {col}",
            options=options,
            default=[],
            placeholder="Tất cả",
        )

        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]

    # Search filter
    search_text = st.sidebar.text_input(
        "Tìm kiếm toàn bảng",
        placeholder="Nhập từ khóa...",
    )
    if search_text:
        text_mask = filtered.astype(str).apply(
            lambda col: col.str.contains(
                search_text, case=False, na=False, regex=False
            )
        ).any(axis=1)
        filtered = filtered[text_mask]

    return filtered


def chart_container(fig, key=None):
    fig.update_layout(
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial"),
        legend_title_text="",
        hoverlabel=dict(namelength=-1),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# =========================
# SIDEBAR - DATA SOURCE
# =========================
st.sidebar.markdown("## 📊 Analytics Dashboard")
st.sidebar.caption("Google Sheets → Streamlit → Plotly")

sheet_url = st.sidebar.text_input(
    "Google Sheet URL / Sheet ID",
    value=DEFAULT_SHEET_URL,
)

gid = st.sidebar.text_input(
    "GID của tab cần đọc",
    value=DEFAULT_GID,
    help=(
        "Mỗi tab trong Google Sheet có một GID. "
        "Mở tab cần dùng và lấy số sau `gid=` trên URL. "
        "Tab đầu tiên thường là 0."
    ),
)

refresh = st.sidebar.button("🔄 Làm mới dữ liệu", use_container_width=True)
if refresh:
    st.cache_data.clear()


# =========================
# LOAD DATA
# =========================
try:
    with st.spinner("Đang tải dữ liệu từ Google Sheets..."):
        df = load_google_sheet(sheet_url, gid)
except Exception as exc:
    st.error("Không thể đọc Google Sheet.")
    st.code(str(exc))
    st.info(
        "Kiểm tra: (1) Sheet được chia sẻ quyền xem qua liên kết, "
        "(2) URL đúng, (3) GID đúng với tab cần đọc."
    )
    st.stop()

if df.empty:
    st.warning("Google Sheet hiện không có dữ liệu để phân tích.")
    st.stop()

numeric_cols, date_cols, categorical_cols = get_column_groups(df)
filtered_df = apply_filters(df, date_cols, categorical_cols)


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>Google Sheets Analytics Dashboard</h1>
        <p>Dashboard tương tác • Tự nhận diện dữ liệu • Bộ lọc động • Plotly charts • Xuất dữ liệu</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# KPI SECTION
# =========================
st.markdown('<div class="section-title">Tổng quan dữ liệu</div>', unsafe_allow_html=True)

metric_col = None
if numeric_cols:
    metric_col = st.selectbox(
        "Chỉ tiêu chính",
        numeric_cols,
        index=0,
        label_visibility="collapsed",
        help="Chọn cột số dùng cho các KPI tổng hợp.",
    )

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Số dòng", f"{len(filtered_df):,}")
k2.metric("Số cột", f"{len(filtered_df.columns):,}")
k3.metric(
    "Dữ liệu trống",
    f"{filtered_df.isna().sum().sum():,}",
)

if metric_col:
    k4.metric(f"Tổng {metric_col}", format_number(filtered_df[metric_col].sum()))
    k5.metric(
        f"Trung bình {metric_col}",
        format_number(filtered_df[metric_col].mean()),
    )
else:
    k4.metric("Cột số", len(numeric_cols))
    k5.metric("Cột thời gian", len(date_cols))

st.caption(
    f"Đang hiển thị **{len(filtered_df):,}/{len(df):,} dòng** sau khi áp dụng bộ lọc."
)


# =========================
# DASHBOARD TABS
# =========================
tab_overview, tab_time, tab_category, tab_relation, tab_quality, tab_data = st.tabs(
    [
        "📈 Tổng quan",
        "🗓️ Theo thời gian",
        "🏷️ Theo danh mục",
        "🔬 Phân tích sâu",
        "🧹 Chất lượng dữ liệu",
        "📋 Dữ liệu",
    ]
)


# -------------------------
# OVERVIEW
# -------------------------
with tab_overview:
    left, right = st.columns(2)

    with left:
        st.markdown("#### Phân bố chỉ tiêu")
        if numeric_cols:
            hist_col = st.selectbox(
                "Cột số",
                numeric_cols,
                index=numeric_cols.index(metric_col) if metric_col in numeric_cols else 0,
                key="hist_col",
            )
            fig = px.histogram(
                filtered_df,
                x=hist_col,
                nbins=30,
                marginal="box",
                title=f"Phân bố {hist_col}",
            )
            chart_container(fig, "hist")
        else:
            st.info("Chưa phát hiện cột số.")

    with right:
        st.markdown("#### Top danh mục")
        cat_candidates = [
            c for c in categorical_cols
            if 1 < filtered_df[c].nunique(dropna=True) <= 100
        ]

        if cat_candidates:
            cat_col = st.selectbox(
                "Danh mục",
                cat_candidates,
                key="overview_cat",
            )

            count_df = (
                filtered_df[cat_col]
                .fillna("(Trống)")
                .astype(str)
                .value_counts()
                .head(12)
                .rename_axis(cat_col)
                .reset_index(name="Số lượng")
            )

            fig = px.bar(
                count_df.sort_values("Số lượng"),
                x="Số lượng",
                y=cat_col,
                orientation="h",
                text_auto=True,
                title=f"Top 12 theo {cat_col}",
            )
            chart_container(fig, "overview_bar")
        else:
            st.info("Chưa có cột danh mục phù hợp để vẽ biểu đồ.")

    if numeric_cols:
        st.markdown("#### Thống kê mô tả")
        stats = filtered_df[numeric_cols].describe().T
        stats = stats.rename(
            columns={
                "count": "Số lượng",
                "mean": "Trung bình",
                "std": "Độ lệch chuẩn",
                "min": "Nhỏ nhất",
                "25%": "Q1",
                "50%": "Trung vị",
                "75%": "Q3",
                "max": "Lớn nhất",
            }
        )
        st.dataframe(
            stats.style.format("{:,.2f}"),
            use_container_width=True,
        )


# -------------------------
# TIME ANALYSIS
# -------------------------
with tab_time:
    if not date_cols:
        st.info("Chưa phát hiện cột ngày/thời gian.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            time_col = st.selectbox("Cột thời gian", date_cols, key="time_col")

        with c2:
            value_option = ["Số bản ghi"] + numeric_cols
            time_value = st.selectbox("Chỉ tiêu", value_option, key="time_value")

        with c3:
            freq_map = {
                "Ngày": "D",
                "Tuần": "W",
                "Tháng": "MS",
                "Quý": "QS",
                "Năm": "YS",
            }
            freq_label = st.selectbox(
                "Tần suất",
                list(freq_map.keys()),
                index=2,
                key="time_freq",
            )

        temp = filtered_df.dropna(subset=[time_col]).copy()

        if temp.empty:
            st.warning("Không có dữ liệu ngày hợp lệ sau bộ lọc.")
        else:
            temp["_period"] = temp[time_col].dt.to_period(
                {"D": "D", "W": "W", "MS": "M", "QS": "Q", "YS": "Y"}[
                    freq_map[freq_label]
                ]
            ).dt.start_time

            if time_value == "Số bản ghi":
                trend = (
                    temp.groupby("_period")
                    .size()
                    .reset_index(name="Số bản ghi")
                )
                y_col = "Số bản ghi"
            else:
                agg_method = st.radio(
                    "Phép tổng hợp",
                    ["Tổng", "Trung bình", "Trung vị"],
                    horizontal=True,
                )
                agg_func = {
                    "Tổng": "sum",
                    "Trung bình": "mean",
                    "Trung vị": "median",
                }[agg_method]

                trend = (
                    temp.groupby("_period")[time_value]
                    .agg(agg_func)
                    .reset_index()
                )
                y_col = time_value

            fig = px.line(
                trend,
                x="_period",
                y=y_col,
                markers=True,
                title=f"{y_col} theo {freq_label.lower()}",
            )
            fig.update_xaxes(title="")
            chart_container(fig, "time_series")

            # Optional categorical breakdown
            time_cat_candidates = [
                c for c in categorical_cols
                if 1 < temp[c].nunique(dropna=True) <= 15
            ]

            if time_cat_candidates:
                breakdown = st.selectbox(
                    "Phân tách theo danh mục (tùy chọn)",
                    ["Không"] + time_cat_candidates,
                    key="time_breakdown",
                )

                if breakdown != "Không":
                    top_values = (
                        temp[breakdown].astype(str).value_counts().head(8).index
                    )
                    t2 = temp[temp[breakdown].astype(str).isin(top_values)].copy()

                    if time_value == "Số bản ghi":
                        trend2 = (
                            t2.groupby(["_period", breakdown])
                            .size()
                            .reset_index(name="Số bản ghi")
                        )
                        y2 = "Số bản ghi"
                    else:
                        agg_func = {
                            "Tổng": "sum",
                            "Trung bình": "mean",
                            "Trung vị": "median",
                        }[agg_method]
                        trend2 = (
                            t2.groupby(["_period", breakdown])[time_value]
                            .agg(agg_func)
                            .reset_index()
                        )
                        y2 = time_value

                    fig = px.line(
                        trend2,
                        x="_period",
                        y=y2,
                        color=breakdown,
                        markers=True,
                        title=f"Xu hướng {y2} theo {breakdown}",
                    )
                    fig.update_xaxes(title="")
                    chart_container(fig, "time_breakdown_chart")


# -------------------------
# CATEGORY ANALYSIS
# -------------------------
with tab_category:
    cat_candidates = [
        c for c in categorical_cols
        if 1 < filtered_df[c].nunique(dropna=True) <= 200
    ]

    if not cat_candidates:
        st.info("Chưa có cột danh mục phù hợp.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            cat_dim = st.selectbox(
                "Chiều phân tích",
                cat_candidates,
                key="cat_dimension",
            )
        with c2:
            cat_metric = st.selectbox(
                "Chỉ tiêu",
                ["Số bản ghi"] + numeric_cols,
                key="cat_metric",
            )
        with c3:
            top_n = st.slider("Top N", 5, 30, 12)

        temp = filtered_df.copy()
        temp[cat_dim] = temp[cat_dim].fillna("(Trống)").astype(str)

        if cat_metric == "Số bản ghi":
            grouped = (
                temp.groupby(cat_dim)
                .size()
                .reset_index(name="Số bản ghi")
            )
            value_col = "Số bản ghi"
        else:
            cat_agg = st.radio(
                "Tổng hợp",
                ["Tổng", "Trung bình"],
                horizontal=True,
                key="cat_agg",
            )
            agg_func = "sum" if cat_agg == "Tổng" else "mean"
            grouped = (
                temp.groupby(cat_dim)[cat_metric]
                .agg(agg_func)
                .reset_index()
            )
            value_col = cat_metric

        grouped = grouped.sort_values(value_col, ascending=False).head(top_n)

        left, right = st.columns([1.35, 1])

        with left:
            fig = px.bar(
                grouped.sort_values(value_col),
                x=value_col,
                y=cat_dim,
                orientation="h",
                text_auto=".3s",
                title=f"Top {top_n} {cat_dim}",
            )
            chart_container(fig, "cat_bar")

        with right:
            pie_df = grouped.head(10)
            fig = px.pie(
                pie_df,
                names=cat_dim,
                values=value_col,
                hole=0.56,
                title="Cơ cấu Top 10",
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            chart_container(fig, "cat_pie")


# -------------------------
# DEEP ANALYSIS
# -------------------------
with tab_relation:
    if len(numeric_cols) >= 2:
        st.markdown("#### Tương quan giữa các biến số")
        corr = filtered_df[numeric_cols].corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            title="Ma trận tương quan",
        )
        chart_container(fig, "corr_heatmap")

        st.markdown("#### Scatter / phân tích mối quan hệ")

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("Trục X", numeric_cols, index=0, key="scatter_x")
        with c2:
            y_col = st.selectbox(
                "Trục Y",
                numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0,
                key="scatter_y",
            )
        with c3:
            scatter_color_options = ["Không"] + [
                c for c in categorical_cols
                if 1 < filtered_df[c].nunique(dropna=True) <= 15
            ]
            scatter_color = st.selectbox(
                "Màu theo danh mục",
                scatter_color_options,
                key="scatter_color",
            )

        fig = px.scatter(
            filtered_df,
            x=x_col,
            y=y_col,
            color=None if scatter_color == "Không" else scatter_color,
            hover_data=[
                c for c in filtered_df.columns
                if c not in [x_col, y_col]
            ][:5],
            opacity=0.75,
            title=f"{y_col} so với {x_col}",
        )
        chart_container(fig, "scatter")

        # Box plot
        box_cat_candidates = [
            c for c in categorical_cols
            if 1 < filtered_df[c].nunique(dropna=True) <= 20
        ]
        if box_cat_candidates:
            st.markdown("#### Phân bố chỉ tiêu theo nhóm")
            b1, b2 = st.columns(2)
            with b1:
                box_y = st.selectbox("Chỉ tiêu số", numeric_cols, key="box_y")
            with b2:
                box_x = st.selectbox("Nhóm", box_cat_candidates, key="box_x")

            fig = px.box(
                filtered_df,
                x=box_x,
                y=box_y,
                points="outliers",
                title=f"Phân bố {box_y} theo {box_x}",
            )
            chart_container(fig, "box")
    elif len(numeric_cols) == 1:
        st.info("Cần tối thiểu 2 cột số để phân tích tương quan/scatter.")
    else:
        st.info("Chưa phát hiện cột số để phân tích sâu.")


# -------------------------
# DATA QUALITY
# -------------------------
with tab_quality:
    st.markdown("#### Tình trạng dữ liệu")

    quality = pd.DataFrame({
        "Cột": filtered_df.columns,
        "Kiểu dữ liệu": [str(filtered_df[c].dtype) for c in filtered_df.columns],
        "Số giá trị": [filtered_df[c].notna().sum() for c in filtered_df.columns],
        "Số trống": [filtered_df[c].isna().sum() for c in filtered_df.columns],
        "% trống": [
            filtered_df[c].isna().mean() * 100 for c in filtered_df.columns
        ],
        "Số giá trị duy nhất": [
            filtered_df[c].nunique(dropna=True) for c in filtered_df.columns
        ],
    }).sort_values("% trống", ascending=False)

    q1, q2, q3 = st.columns(3)
    q1.metric("Dòng trùng lặp", f"{filtered_df.duplicated().sum():,}")
    q2.metric(
        "Cột có dữ liệu trống",
        f"{(filtered_df.isna().sum() > 0).sum():,}",
    )
    q3.metric(
        "Tỷ lệ ô trống",
        f"{filtered_df.isna().mean().mean() * 100:.2f}%",
    )

    st.dataframe(
        quality.style.format({"% trống": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        quality.head(20).sort_values("% trống"),
        x="% trống",
        y="Cột",
        orientation="h",
        text_auto=".1f",
        title="Top cột có tỷ lệ dữ liệu trống cao",
    )
    chart_container(fig, "missing_chart")


# -------------------------
# DATA TABLE
# -------------------------
with tab_data:
    st.markdown("#### Bảng dữ liệu sau bộ lọc")

    selected_columns = st.multiselect(
        "Chọn cột hiển thị",
        filtered_df.columns.tolist(),
        default=filtered_df.columns.tolist(),
    )

    display_df = filtered_df[selected_columns] if selected_columns else filtered_df

    st.dataframe(
        display_df,
        use_container_width=True,
        height=540,
        hide_index=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Tải CSV đã lọc",
        data=csv_bytes,
        file_name="google_sheet_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================
# SIDEBAR - DATA INFO
# =========================
st.sidebar.divider()
st.sidebar.markdown("### ℹ️ Thông tin nguồn")
st.sidebar.write(f"**Dòng gốc:** {len(df):,}")
st.sidebar.write(f"**Dòng sau lọc:** {len(filtered_df):,}")
st.sidebar.write(f"**Cột:** {len(df.columns):,}")
st.sidebar.write(f"**Cột số:** {len(numeric_cols):,}")
st.sidebar.write(f"**Cột ngày:** {len(date_cols):,}")

st.sidebar.caption(
    "Dữ liệu được cache 5 phút. Nhấn “Làm mới dữ liệu” để lấy dữ liệu mới ngay."
)
