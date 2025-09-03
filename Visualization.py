import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# from utils import preprocess_data   
import io


# --- Streamlit App ---
def app():
    st.title("💻 Laptop Data Visualization")

    # --- Load dataset ---
    path = r"laptop.csv"
    try:
        raw_df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f" File not found at: {path}")
        return

    # --- Preprocess ---
    # df, label_encoders, scaler, numeric_cols = preprocess_data(raw_df)
    df=raw_df
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    theme = st.session_state.get("theme", "plotly")

    # --- Show Data ---
    st.subheader(" Data Overview")
    num_rows = st.slider("Select number of rows to display:",
                         min_value=5, max_value=len(df), value=10, step=5)
    st.dataframe(df.head(num_rows))

    # --- Show Data Info ---
    categorical_cols = raw_df.select_dtypes(include=['object']).columns.tolist()
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.code(info_str, language='text')

    # --- Heatmap ---
    if len(numeric_cols) > 1:
        st.markdown("###  Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols + (["Price"] if "Price" in df.columns else [])].corr(),
                    annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- Tabs for plots ---
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Scatter", "📊 Histogram", "📦 Box", "📈 Line"])

    # ================= Scatter Plot =================
    with tab1:
        st.subheader("Scatter Plot")
        if len(numeric_cols) >= 2:
            x = st.selectbox("X axis", numeric_cols, key="scatter_x")
            y = st.selectbox("Y axis", numeric_cols, key="scatter_y")
            color_user = st.selectbox("Color (optional)", [None] + categorical_cols, key="scatter_color")

            scatter_df = pd.DataFrame({x: df[x], y: df[y].fillna(0)})
            if color_user:
                scatter_df[color_user] = raw_df[color_user]  # نستخدم الأصل عشان labels

            fig = px.scatter(scatter_df, x=x, y=y, color=color_user, template=theme,
                             title=f"Scatter: {x} vs {y}")
            st.plotly_chart(fig, use_container_width=True)

    # ================= Histogram =================
    with tab2:
        st.subheader("Histogram")
        input_hist = st.selectbox("Column", numeric_cols, key="hist")
        bins = st.slider("Bins", 5, 100, 30, step=5)
        st.markdown(f"**choosen bins :** {bins}")
        fig = px.histogram(df, x=input_hist, nbins=bins, template=theme,
                        title=f"Histogram of {input_hist}")
        st.plotly_chart(fig, use_container_width=True)

    # ================= Box Plot =================
    with tab3:
        st.subheader("Box Plot")
        selected_columns = st.multiselect(
            "Select 1 or 2 numeric columns for Box Plot:",
            numeric_cols,
            default=numeric_cols[:1])
        color_box = st.selectbox("Group by (optional)", [None] + categorical_cols, key="box_color")

        if selected_columns:
            col_layout = st.columns(len(selected_columns))
            for i, col_name in enumerate(selected_columns):
                if color_box:
                    fig = px.box(df, y=col_name, color=raw_df[color_box],
                                 title=f"Box Plot of {col_name}", template=theme)
                else:
                    fig = px.box(df, y=col_name,
                                 title=f"Box Plot of {col_name}", template=theme)
                col_layout[i].plotly_chart(fig, use_container_width=True)

    # ================= Line Plot =================
    with tab4:
        if len(numeric_cols) >= 2:
            st.subheader("Line Plot")
            x_line = st.selectbox("X axis", numeric_cols, key="line_x")
            y_line = st.selectbox("Y axis", numeric_cols, key="line_y")

            line_df = pd.DataFrame({x_line: df[x_line], y_line: df[y_line].fillna(0)})

            fig = px.line(line_df, x=x_line, y=y_line, template=theme,
                          title=f"Line: {x_line} vs {y_line}")
            st.plotly_chart(fig, use_container_width=True)

    # ================= Categorical Plots =================
    if len(categorical_cols) > 0:
        tab5, tab6 = st.tabs(["📊 Count Plot", "🥧 Pie Chart"])
        with tab5:
            input_count = st.selectbox("Column", categorical_cols, key="count")
            input_hue = st.selectbox("Hue (optional)", [None] + categorical_cols, key="hue")
            fig = px.histogram(raw_df, x=input_count, color=input_hue, template=theme,
                               title=f"Count of {input_count}")
            st.plotly_chart(fig, use_container_width=True)

        with tab6:
            input_pie = st.selectbox("Column", categorical_cols, key="pie")
            fig = px.pie(raw_df, names=input_pie, template=theme,
                         title=f"Pie of {input_pie}")
            st.plotly_chart(fig, use_container_width=True)
