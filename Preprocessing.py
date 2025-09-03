# DataPreprocessing.py
import re
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _parse_cpu_fields(s):
    if pd.isna(s):
        return "Unknown", "Unknown", 0.0
    s = str(s)
    tokens = s.split()
    brand = tokens[0] if tokens else "Unknown"
    model = " ".join(tokens[1:3]) if len(tokens) >= 3 else (" ".join(tokens[1:]) if len(tokens) >= 2 else "Unknown")
    m = re.search(r'(\d+(?:\.\d+)?)\s*GHz', s, flags=re.I)
    speed = float(m.group(1)) if m else 0.0
    return brand, model, speed


def _parse_gpu_fields(s):
    if pd.isna(s):
        return "Unknown", "Unknown"
    s = str(s)
    tokens = s.split()
    brand = tokens[0] if tokens else "Unknown"
    model = " ".join(tokens[1:]) if len(tokens) >= 2 else "Unknown"
    return brand, model


def _to_float_from_text(x):
    if pd.isna(x):
        return None
    s = str(x)
    s = re.sub(r'[^\d\.]+', '', s)
    if s == '':
        return None
    try:
        return float(s)
    except Exception:
        return None


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cpu_source_col = None
    if "Cpu" in df.columns:
        cpu_source_col = "Cpu"
    elif "Cpu_brand" in df.columns:
        sample_val = str(df["Cpu_brand"].dropna().astype(str).head(1).values[0]) if df["Cpu_brand"].notna().any() else ""
        if any(k in sample_val for k in ["Core", "Ryzen", "GHz", "i3", "i5", "i7", "i9"]):
            cpu_source_col = "Cpu_brand"

    if cpu_source_col:
        parsed = df[cpu_source_col].apply(_parse_cpu_fields)
        df["Cpu_brand"] = parsed.apply(lambda t: t[0])
        df["Cpu_model"] = parsed.apply(lambda t: t[1])
        df["Cpu_speed"] = parsed.apply(lambda t: t[2])
    else:
        if "Cpu_brand" not in df.columns:
            df["Cpu_brand"] = "Unknown"
        if "Cpu_model" not in df.columns:
            df["Cpu_model"] = "Unknown"
        if "Cpu_speed" not in df.columns:
            df["Cpu_speed"] = 0.0

    gpu_source_col = None
    if "Gpu" in df.columns:
        gpu_source_col = "Gpu"
    elif "Gpu_brand" in df.columns:
        sample_val = str(df["Gpu_brand"].dropna().astype(str).head(1).values[0]) if df["Gpu_brand"].notna().any() else ""
        if len(sample_val.split()) > 1:
            gpu_source_col = "Gpu_brand"

    if gpu_source_col:
        parsed = df[gpu_source_col].apply(_parse_gpu_fields)
        df["Gpu_brand"] = parsed.apply(lambda t: t[0])
        df["Gpu_model"] = parsed.apply(lambda t: t[1])
    else:
        if "Gpu_brand" not in df.columns:
            df["Gpu_brand"] = "Unknown"
        if "Gpu_model" not in df.columns:
            df["Gpu_model"] = "Unknown"

    if "OpSys" in df.columns and "Os" not in df.columns:
        df["Os"] = df["OpSys"].fillna("Unknown")
    elif "Os" in df.columns:
        df["Os"] = df["Os"].fillna("Unknown")
    else:
        df["Os"] = "Unknown"

    return df


def preprocess_data(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df = feature_engineering(df)

    numeric_text_candidates = ["Weight", "Ram", "HDD", "SSD", "Ppi", "Price", "Screen_Size", "ScreenSize"]
    for col in numeric_text_candidates:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].apply(_to_float_from_text)

    for bcol in ["TouchScreen", "Ips"]:
        if bcol in df.columns:
            df[bcol] = pd.to_numeric(df[bcol], errors="coerce")

    base_numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != "Price":
            base_numeric_cols.append(col)

    for col in base_numeric_cols:
        median_val = df[col].median(skipna=True)
        df[col] = df[col].fillna(median_val)

    categorical_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    to_scale = [c for c in base_numeric_cols if c not in ["Price"]]
    if to_scale:
        df[to_scale] = scaler.fit_transform(df[to_scale])

    if "Price" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Price"]):
            df["Price"] = df["Price"].apply(_to_float_from_text)
        df = df.dropna(subset=["Price"])

    df = df.fillna(0)
    numeric_cols = to_scale.copy()

    return df, label_encoders, scaler, numeric_cols


def app():
    st.title("Data Preprocessing & Feature Engineering")

    path = r"laptop.csv"
    try:
        raw_df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File not found at: {path}")
        return

    st.subheader("Original Data Sample")
    st.dataframe(raw_df.head(10))

    st.markdown("Missing Values (Original)")
    st.dataframe(raw_df.isnull().sum())

    processed_df, label_encoders, scaler, numeric_cols = preprocess_data(raw_df)

    st.subheader("Processed Data Sample")
    st.dataframe(processed_df.head(10))

    st.markdown("Missing Values After Processing")
    st.dataframe(processed_df.drop(columns=["Price"] if "Price" in processed_df.columns else []).isnull().sum())

    st.session_state["df_original"] = raw_df
    st.session_state["df_processed"] = processed_df
    st.session_state["label_encoders"] = label_encoders
    st.session_state["scaler"] = scaler
    st.session_state["numeric_cols"] = numeric_cols

    csv = processed_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Processed CSV",
        data=csv,
        file_name="processed_laptop_data.csv",
        mime="text/csv",
    )
