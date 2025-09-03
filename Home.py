import streamlit as st
from PIL import Image

def app():
    st.markdown(
        """
        <h1 style="text-align:center; color:#1f77b4;">💻 Laptop ML Project</h1>
        <h3 style="text-align:center; color:grey;">Laptop Data Analysis & Price Prediction</h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            background-color:#f9f9f9;
            padding:20px;
            border-radius:12px;
            box-shadow:2px 2px 10px rgba(0,0,0,0.1);
            text-align:center;">
            <p style="font-size:16px; line-height:1.6; color:#333;">
            This project analyzes laptop datasets and builds machine learning models 
            to <b>predict laptop prices</b> based on features like 
            <span style="color:#1f77b4;">RAM</span>, 
            <span style="color:#1f77b4;">Weight</span>, 
            <span style="color:#1f77b4;">CPU</span>, 
            <span style="color:#1f77b4;">GPU</span>, and more.  
            </p>
            <p style="font-size:15px; color:#444;">
             Explore, preprocess, visualize, and predict using this interactive Streamlit app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        image = Image.open(r"OIP.webp")  
        st.image(image, caption="Laptop Dataset Analysis", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Image not found. Please check the path or upload a valid image.")

    # st.sidebar.success("📌 Use the sidebar to navigate through different sections.")
