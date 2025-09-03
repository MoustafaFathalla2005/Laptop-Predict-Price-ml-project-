import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Preprocessing import preprocess_data   


def remove_outliers_iqr(df, target_col="Price"):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if col == target_col:
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


def app():
    st.title("Laptop Price Prediction (Regression)")

    path = "laptop.csv"
    try:
        raw_df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        return

    processed_df, encoders, scaler, numeric_cols = preprocess_data(raw_df)

    model_choice = st.selectbox("Choose Regression Model:", ["Random Forest Regressor", "Linear Regression"])

    if model_choice == "Linear Regression":
        before_rows = processed_df.shape[0]
        processed_df = remove_outliers_iqr(processed_df, target_col="Price")
        after_rows = processed_df.shape[0]
        st.info(f"Outlier removal applied: {before_rows - after_rows} rows removed.")

    X = processed_df.drop("Price", axis=1)
    y = processed_df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    else:
        model = LinearRegression()

    with st.spinner("Training model... Please wait."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmpe = np.sqrt(np.mean(((y_test - y_pred) / y_test) ** 2)) * 100

    st.subheader("Regression Performance :")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("RMPE (%)", f"{rmpe:.2f}")

    # --- Prediction Form (using original CSV columns) ---
    st.subheader("Predict Laptop Price ")
    with st.form("regression_form"):
        col1, col2 = st.columns(2)

        with col1:
            company = st.selectbox("Company", raw_df["Company"].dropna().unique())
            typename = st.selectbox("TypeName", raw_df["TypeName"].dropna().unique())
            cpu_brand = st.selectbox("CPU Brand", raw_df["Cpu_brand"].dropna().unique())
            gpu_brand = st.selectbox("GPU Brand", raw_df["Gpu_brand"].dropna().unique())
            os = st.selectbox("Operating System", raw_df["Os"].dropna().unique())

        with col2:
            ram = st.number_input("RAM (GB)", min_value=2.0, max_value=64.0, value=8.0)
            weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5)
            ppi = st.number_input("PPI", min_value=50.0, max_value=500.0, value=150.0)
            hdd = st.number_input("HDD (GB)", min_value=0.0, max_value=2000.0, value=0.0)
            ssd = st.number_input("SSD (GB)", min_value=0.0, max_value=2000.0, value=256.0)

        submitted = st.form_submit_button("Predict Price ")

        if submitted:
            with st.spinner("Predicting laptop price..."):
                input_data = {
                    "Company": [company],
                    "TypeName": [typename],
                    "Ram": [ram],
                    "Weight": [weight],
                    "TouchScreen": [0],   # تقدر تخليها input لو عايز
                    "Ips": [0],           # تقدر تخليها input
                    "Ppi": [ppi],
                    "Cpu_brand": [cpu_brand],
                    "HDD": [hdd],
                    "SSD": [ssd],
                    "Gpu_brand": [gpu_brand],
                    "Os": [os]
                }

                input_df = pd.DataFrame(input_data)

                # Apply encoders
                for col, le in encoders.items():
                    if col in input_df.columns:
                        if input_df[col].iloc[0] in le.classes_:
                            input_df[col] = le.transform(input_df[col])
                        else:
                            input_df[col] = 0

                # ترتيب الأعمدة زي ما الموديل متوقع
                input_df = input_df.reindex(columns=X.columns, fill_value=0)

                # Scaling للأعمدة الرقمية
                if numeric_cols:
                    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

                prediction = model.predict(input_df)[0]

            st.markdown(f"""
                <div style="background-color:#f0f8ff; padding:30px; border-radius:15px; 
                            margin:10px; text-align:center; 
                            box-shadow:2px 2px 12px rgba(0,0,0,0.2);">
                    <h2 style="color:#1f77b4;">🎊 Predicted Laptop Price 🎊</h2>
                    <h1 style="color:#2ca02c;">💰 {prediction:,.2f} $</h1>
                </div>
            """, unsafe_allow_html=True)

            if st.checkbox("Show processed input data"):
                st.dataframe(input_df)
