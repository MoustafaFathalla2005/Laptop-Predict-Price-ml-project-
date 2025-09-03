import streamlit as st
import Home
import Visualization
import Preprocessing
import Modeling

PAGES = {
    "🏠 Home": Home,
    "📊 Data Visualization": Visualization,
    "🛠️ Data Preprocessing": Preprocessing,
    "🤖 Model Training & Prediction": Modeling,
}

st.set_page_config(
    page_title="moustafa",
    page_icon="💻",
    layout="wide",
)

with st.sidebar:
    st.markdown(
        """
        <h1 style='text-align: center; color: #1f77b4;'>💻 Laptop App</h1>
        <hr>
        """,
        unsafe_allow_html=True,
    )
    selected_page = st.selectbox(" Choose Page :", list(PAGES.keys()))
    st.markdown(
        """
        <hr>
        <div style="text-align:center; font-size:13px; color:grey;">
        Made with ❤️ by Moustafa--تيم قاعدة الرجالة 
        </div>
        """,
        unsafe_allow_html=True,
    )

PAGES[selected_page].app()
