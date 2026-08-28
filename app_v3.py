
import io
import re
import unicodedata
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# GOOGLE SHEETS BI DASHBOARD - V3
# ============================================================
st.set_page_config(
    page_title="Google Sheets BI Dashboard V3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "V3.0"
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1Zj6XC0_BCfxvJR6e-GAM1Mhi9Gf5JH4aPGr1OPr3Tbg/edit?usp=sharing"


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0b1739 0%,#112b5f 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f7f9fc;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #101828 !important;
    }

    .hero {
        border-radius: 22px;
        padding: 28px 30px;
        margin-bottom: 18px;
        background: linear-gradient(120deg,#0b1739 0%,#174a8b 58%,#078a91 100%);
        box-shadow: 0 14px 38px rgba(11,23,57,.16);
    }

    .hero h1 {
        color: white;
        margin: 0 0 6px 0;
        font-size: 2rem;
    }

    .hero p {
        color: rgba(255,255,255,.82);
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e7ecf3;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 4px 18px rgba(16,24,40,.045);
    }

    .version-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: #dff7ea;
        color: #166534;
        font-size: 12px;
        font-weight: 700;
        margin-left: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def extract_sheet_id(url_or_id: str) -> str:
    value = (url_or_id or "").strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", value):
        return value

    match = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]+)", value)
    if not match:
        raise ValueError("Không tìm thấy Google Sheet ID trong đường dẫn.")

    return match.group(1)


def strip_accents(text: str) -> str:
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower()


def make_unique_columns(columns) -> List[str]:
    seen = {}
    output = []

    for col in columns:
        name = re.sub(r"\s+", " ", str(col)).strip()
        if not name or name.lower().startswith("unnamed"):
            name = "Cột"

        seen[name] = seen.get(name, 0) + 1
        output.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    return output


def detect_header_row(raw_df: pd.DataFrame, max_scan: int = 15) -> int:
    """
    Tự tìm dòng tiêu đề hợp lý:
    - nhiều ô có nội dung
    - có nhiều giá trị text ngắn
    - ưu tiên các dòng có nhiều cột khác nhau
    """
    max_scan = min(max_scan, len(raw_df))
    best_row = 0
    best_score = -1

    for i in range(max_scan):
        row = raw_df.iloc[i]
        values = row.dropna().astype(str).str.strip()
        values = values[values != ""]

        if len(values) == 0:
            continue

        non_empty = len(values)
        unique = values.nunique()
        textish = sum(not re.fullmatch(r"[-+]?\d+([.,]\d+)?", v) for v in values)

        score = non_empty * 3 + unique + textish

        if score > best_score:
            best_score = score
            best_row = i

    return best_row + 1  # 1-based for UI


def looks_like_id_column(col_name: str, series: pd.Series) -> bool:
    name = strip_accents(col_name)

    keywords = [
        "id", "ma", "code", "mst", "cccd", "cmnd",
        "so dien thoai", "phone", "hoa don", "invoice",
    ]

    if any(k in name for k in keywords):
        return True

    sample = series.dropna().astype(str).str.strip()

    if sample.empty:
        return False

    digits_ratio = sample.str.fullmatch(r"\d+").mean()
    median_len = sample.str.len().median()
    unique_ratio = sample.nunique() / max(len(sample), 1)

    return digits_ratio > 0.9 and median_len >= 8 and unique_ratio > 0.9


def parse_numeric_series(series: pd.Series, col_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    if looks_like_id_column(col_name, series):
        return series

    s = series.astype("string").str.strip()

    non_empty = s.dropna()
    non_empty = non_empty[non_empty != ""]
    if non_empty.empty:
        return series

    def parse_one(x):
        if pd.isna(x):
            return np.nan

        x = str(x).strip()

        if x in {"", "-", "--", "—", "N/A", "NA", "nan", "None"}:
            return np.nan

        negative = x.startswith("(") and x.endswith(")")
        x = x.strip("()")

        x = re.sub(r"[₫đĐ$€£¥%]", "", x)
        x = x.replace("\u00a0", "").replace(" ", "")
        x = re.sub(r"[^0-9,.\-+]", "", x)

        if not x:
            return np.nan

        if "." in x and "," in x:
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif x.count(".") >= 2:
            x = x.replace(".", "")
        elif x.count(",") >= 2:
            x = x.replace(",", "")
        elif "." in x:
            left, right = x.rsplit(".", 1)
            if len(right) == 3 and left.replace("-", "").isdigit():
                x = left + right
        elif "," in x:
            left, right = x.rsplit(",", 1)
            if len(right) == 3 and left.replace("-", "").isdigit():
                x = left + right
            else:
                x = left + "." + right

        try:
            value = float(x)
            return -value if negative else value
        except Exception:
            return np.nan

    parsed = s.map(parse_one)
    source_count = non_empty.shape[0]
    success_ratio = parsed.notna().sum() / max(source_count, 1)

    if success_ratio >= 0.65:
        return pd.to_numeric(parsed, errors="coerce")

    return series


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    keywords = [
        "ngay", "date", "time", "thang", "month",
        "nam", "year", "created", "updated",
        "checkin", "checkout",
    ]

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        name = strip_accents(col)
        sample = df[col].dropna().astype(str).head(300)

        if sample.empty:
            continue

        keyword_hint = any(k in name for k in keywords)
        pattern_hint = sample.str.contains(
            r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b",
            regex=True,
        ).mean() >= 0.4

        if keyword_hint or pattern_hint:
            converted = pd.to_datetime(
                df[col],
                errors="coerce",
                dayfirst=True,
            )

            ratio = converted.notna().sum() / max(df[col].notna().sum(), 1)

            if ratio >= 0.55:
                df[col] = converted

    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    df.columns = make_unique_columns(df.columns)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: re.sub(r"\s+", " ", x).strip()
            if isinstance(x, str)
            else x
        )

    for col in df.columns:
        df[col] = parse_numeric_series(df[col], col)

    df = parse_date_columns(df)

    return df.reset_index(drop=True)


def get_groups(df):
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


def format_number(value):
    if value is None or pd.isna(value):
        return "—"

    value = float(value)
    av = abs(value)

    if av >= 1_000_000_000:
        return f"{value/1_000_000_000:,.2f} tỷ"
    if av >= 1_000_000:
        return f"{value/1_000_000:,.2f} triệu"
    if av >= 1_000:
        return f"{value:,.0f}"

    return f"{value:,.2f}"


def show_chart(fig, key):
    fig.update_layout(
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ============================================================
# GOOGLE SHEETS WORKBOOK
# ============================================================
@st.cache_data(ttl=300, show_spinner=False)
def download_workbook(sheet_url_or_id: str) -> bytes:
    sheet_id = extract_sheet_id(sheet_url_or_id)

    url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/export?format=xlsx"
    )

    response = requests.get(url, timeout=45, allow_redirects=True)
    response.raise_for_status()

    data = response.content
    content_type = response.headers.get("content-type", "").lower()

    if "text/html" in content_type or data[:100].lstrip().lower().startswith(b"<html"):
        raise PermissionError(
            "Google Sheet chưa cho phép truy cập công khai. "
            "Hãy đặt quyền: Bất kỳ ai có đường liên kết → Người xem."
        )

    return data


@st.cache_data(ttl=300, show_spinner=False)
def get_sheet_names(workbook_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(workbook_bytes), engine="openpyxl")
    return xls.sheet_names


@st.cache_data(ttl=300, show_spinner=False)
def preview_raw_sheet(workbook_bytes: bytes, sheet_name: str):
    return pd.read_excel(
        io.BytesIO(workbook_bytes),
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )


@st.cache_data(ttl=300, show_spinner=False)
def read_sheet(workbook_bytes: bytes, sheet_name: str, header_row: int):
    df = pd.read_excel(
        io.BytesIO(workbook_bytes),
        sheet_name=sheet_name,
        header=header_row - 1,
        engine="openpyxl",
    )
    return normalize_dataframe(df)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 📊 Google Sheets BI Dashboard")
st.sidebar.caption(f"Phiên bản {APP_VERSION} • Multi-sheet • Auto Header")

sheet_url = st.sidebar.text_input(
    "Google Sheet URL / Sheet ID",
    value=DEFAULT_SHEET_URL,
)

if st.sidebar.button("🔄 Làm mới dữ liệu", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD WORKBOOK
# ============================================================
try:
    with st.spinner("Đang tải toàn bộ Google Sheet..."):
        workbook_bytes = download_workbook(sheet_url)
        sheet_names = get_sheet_names(workbook_bytes)
except Exception as exc:
    st.error("Không thể tải Google Sheet.")
    st.code(str(exc))
    st.stop()

st.sidebar.markdown("### 🗂️ Worksheet")
selected_sheet = st.sidebar.selectbox(
    "Chọn tab dữ liệu",
    options=sheet_names,
)

raw_preview = preview_raw_sheet(workbook_bytes, selected_sheet)

auto_header = detect_header_row(raw_preview)

st.sidebar.markdown("### 🧠 Nhận diện bảng")
use_auto_header = st.sidebar.checkbox(
    "Tự nhận diện dòng tiêu đề",
    value=True,
)

if use_auto_header:
    header_row = auto_header
    st.sidebar.success(f"Đã nhận diện dòng tiêu đề: {header_row}")
else:
    header_row = st.sidebar.number_input(
        "Dòng tiêu đề",
        min_value=1,
        max_value=min(30, max(len(raw_preview), 1)),
        value=auto_header,
        step=1,
    )

try:
    df = read_sheet(
        workbook_bytes,
        selected_sheet,
        int(header_row),
    )
except Exception as exc:
    st.error(f"Không thể đọc tab {selected_sheet}.")
    st.code(str(exc))
    st.stop()

if df.empty:
    st.warning("Worksheet không có dữ liệu.")
    st.stop()

numeric_cols, date_cols, categorical_cols = get_groups(df)


# ============================================================
# FILTERS
# ============================================================
filtered_df = df.copy()

st.sidebar.markdown("### 🔎 Bộ lọc")

if date_cols:
    date_col = st.sidebar.selectbox(
        "Cột thời gian",
        ["Không lọc"] + date_cols,
    )

    if date_col != "Không lọc":
        valid = filtered_df[date_col].dropna()

        if not valid.empty:
            date_range = st.sidebar.date_input(
                "Khoảng thời gian",
                value=(valid.min().date(), valid.max().date()),
            )

            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                d1, d2 = date_range
                mask = (
                    filtered_df[date_col].dt.date.ge(d1)
                    & filtered_df[date_col].dt.date.le(d2)
                )
                filtered_df = filtered_df[mask]

filterable = [
    c for c in categorical_cols
    if 1 < filtered_df[c].nunique(dropna=True) <= 100
]

chosen_filter_cols = st.sidebar.multiselect(
    "Bộ lọc danh mục",
    options=filterable,
    default=filterable[:2],
)

for col in chosen_filter_cols:
    options = sorted(
        filtered_df[col].dropna().astype(str).unique().tolist()
    )

    chosen = st.sidebar.multiselect(
        f"Lọc: {col}",
        options=options,
    )

    if chosen:
        filtered_df = filtered_df[
            filtered_df[col].astype(str).isin(chosen)
        ]

search_text = st.sidebar.text_input(
    "Tìm kiếm toàn bảng",
    placeholder="Nhập từ khóa...",
)

if search_text:
    mask = filtered_df.astype(str).apply(
        lambda col: col.str.contains(
            search_text,
            case=False,
            na=False,
            regex=False,
        )
    ).any(axis=1)

    filtered_df = filtered_df[mask]


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <h1>
            Google Sheets BI Dashboard
            <span class="version-badge">{APP_VERSION}</span>
        </h1>
        <p>
            Worksheet: <b>{selected_sheet}</b> •
            Header dòng {header_row} •
            {len(filtered_df):,}/{len(df):,} dòng sau bộ lọc
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW METRICS
# ============================================================
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Số dòng", f"{len(filtered_df):,}")
m2.metric("Số cột", f"{len(df.columns):,}")
m3.metric("Cột số", f"{len(numeric_cols):,}")
m4.metric("Cột ngày", f"{len(date_cols):,}")
m5.metric("Cột danh mục", f"{len(categorical_cols):,}")

if len(df.columns) <= 1:
    st.error(
        "App vẫn chỉ nhận diện 1 cột. "
        "Hãy tắt 'Tự nhận diện dòng tiêu đề' và chọn đúng dòng chứa tên 11 cột."
    )


# ============================================================
# KPI
# ============================================================
st.markdown("### KPI tổng quan")

if numeric_cols:
    selected_kpis = st.multiselect(
        "Chọn các KPI cần hiển thị",
        options=numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))],
        max_selections=8,
    )

    if selected_kpis:
        cols = st.columns(min(4, len(selected_kpis)))

        for i, col in enumerate(selected_kpis):
            cols[i % len(cols)].metric(
                f"Tổng {col}",
                format_number(filtered_df[col].sum()),
            )
else:
    st.info("Chưa phát hiện cột số.")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Tổng quan",
        "🗓️ Xu hướng",
        "🏷️ Danh mục",
        "🔬 Tương quan",
        "🧹 Chất lượng dữ liệu",
        "📋 Dữ liệu",
    ]
)


# ============================================================
# TAB 1
# ============================================================
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        if numeric_cols:
            hist_col = st.selectbox(
                "Chỉ tiêu phân bố",
                numeric_cols,
                key="hist_col",
            )

            fig = px.histogram(
                filtered_df,
                x=hist_col,
                nbins=30,
                marginal="box",
                title=f"Phân bố {hist_col}",
            )
            show_chart(fig, "hist")
        else:
            st.info("Không có cột số để vẽ histogram.")

    with c2:
        cats = [
            c for c in categorical_cols
            if 1 < filtered_df[c].nunique(dropna=True) <= 100
        ]

        if cats:
            cat_col = st.selectbox(
                "Danh mục",
                cats,
                key="overview_cat",
            )

            count_df = (
                filtered_df[cat_col]
                .fillna("(Trống)")
                .astype(str)
                .value_counts()
                .head(15)
                .rename_axis(cat_col)
                .reset_index(name="Số lượng")
            )

            fig = px.bar(
                count_df.sort_values("Số lượng"),
                x="Số lượng",
                y=cat_col,
                orientation="h",
                text_auto=True,
                title=f"Top {cat_col}",
            )

            show_chart(fig, "overview_cat_chart")


# ============================================================
# TAB 2
# ============================================================
with tab2:
    if not date_cols:
        st.info("Chưa phát hiện cột ngày.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            time_col = st.selectbox(
                "Cột thời gian",
                date_cols,
                key="time_col",
            )

        with c2:
            metric = st.selectbox(
                "Chỉ tiêu",
                ["Số bản ghi"] + numeric_cols,
                key="time_metric",
            )

        with c3:
            freq_label = st.selectbox(
                "Chu kỳ",
                ["Ngày", "Tuần", "Tháng", "Quý", "Năm"],
                index=2,
            )

        freq_map = {
            "Ngày": "D",
            "Tuần": "W",
            "Tháng": "M",
            "Quý": "Q",
            "Năm": "Y",
        }

        temp = filtered_df.dropna(subset=[time_col]).copy()

        if not temp.empty:
            temp["_period"] = (
                temp[time_col]
                .dt.to_period(freq_map[freq_label])
                .dt.start_time
            )

            if metric == "Số bản ghi":
                trend = (
                    temp.groupby("_period")
                    .size()
                    .reset_index(name="Số bản ghi")
                )
                y = "Số bản ghi"
            else:
                trend = (
                    temp.groupby("_period")[metric]
                    .sum()
                    .reset_index()
                )
                y = metric

            fig = px.line(
                trend,
                x="_period",
                y=y,
                markers=True,
                title=f"{y} theo {freq_label.lower()}",
            )

            show_chart(fig, "trend")


# ============================================================
# TAB 3
# ============================================================
with tab3:
    cats = [
        c for c in categorical_cols
        if 1 < filtered_df[c].nunique(dropna=True) <= 300
    ]

    if not cats:
        st.info("Không có cột danh mục phù hợp.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            cat_col = st.selectbox(
                "Phân nhóm theo",
                cats,
                key="cat_analysis_col",
            )

        with c2:
            metric = st.selectbox(
                "Chỉ tiêu",
                ["Số bản ghi"] + numeric_cols,
                key="cat_analysis_metric",
            )

        with c3:
            top_n = st.slider(
                "Top N",
                5,
                30,
                12,
            )

        temp = filtered_df.copy()
        temp[cat_col] = temp[cat_col].fillna("(Trống)").astype(str)

        if metric == "Số bản ghi":
            grouped = (
                temp.groupby(cat_col)
                .size()
                .reset_index(name="Số bản ghi")
            )
            value_col = "Số bản ghi"
        else:
            grouped = (
                temp.groupby(cat_col)[metric]
                .sum()
                .reset_index()
            )
            value_col = metric

        grouped = grouped.sort_values(
            value_col,
            ascending=False,
        ).head(top_n)

        left, right = st.columns([1.35, 1])

        with left:
            fig = px.bar(
                grouped.sort_values(value_col),
                x=value_col,
                y=cat_col,
                orientation="h",
                text_auto=".3s",
                title=f"Top {top_n} {cat_col}",
            )
            show_chart(fig, "category_bar")

        with right:
            fig = px.pie(
                grouped.head(10),
                names=cat_col,
                values=value_col,
                hole=.55,
                title="Cơ cấu Top 10",
            )
            show_chart(fig, "category_pie")


# ============================================================
# TAB 4
# ============================================================
with tab4:
    if len(numeric_cols) < 2:
        st.info("Cần ít nhất 2 cột số.")
    else:
        corr_cols = st.multiselect(
            "Chọn biến số",
            numeric_cols,
            default=numeric_cols[:min(10, len(numeric_cols))],
            max_selections=15,
        )

        if len(corr_cols) >= 2:
            corr = filtered_df[corr_cols].corr(numeric_only=True)

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
                title="Ma trận tương quan",
            )

            show_chart(fig, "corr")


# ============================================================
# TAB 5
# ============================================================
with tab5:
    quality = pd.DataFrame(
        {
            "Cột": filtered_df.columns,
            "Kiểu dữ liệu": [
                str(filtered_df[c].dtype)
                for c in filtered_df.columns
            ],
            "Số trống": [
                filtered_df[c].isna().sum()
                for c in filtered_df.columns
            ],
            "% trống": [
                filtered_df[c].isna().mean() * 100
                for c in filtered_df.columns
            ],
            "Số giá trị duy nhất": [
                filtered_df[c].nunique(dropna=True)
                for c in filtered_df.columns
            ],
        }
    )

    st.dataframe(
        quality.style.format({"% trống": "{:.2f}%"}),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 6
# ============================================================
with tab6:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=560,
        hide_index=True,
    )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Tải CSV đã lọc",
        data=csv_data,
        file_name=f"{selected_sheet}_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# DEBUG / VERIFICATION
# ============================================================
st.sidebar.divider()
st.sidebar.markdown("### ✅ Kiểm tra phiên bản")
st.sidebar.success(f"Đang chạy: {APP_VERSION}")
st.sidebar.write(f"**Worksheet:** {selected_sheet}")
st.sidebar.write(f"**Header dòng:** {header_row}")
st.sidebar.write(f"**Số cột:** {len(df.columns)}")
st.sidebar.write(f"**Cột số:** {len(numeric_cols)}")
st.sidebar.write(f"**Cột ngày:** {len(date_cols)}")

with st.sidebar.expander("Tên các cột đã đọc"):
    for i, col in enumerate(df.columns, start=1):
        st.write(f"{i}. {col}")
