import streamlit as st
from app.src.pages import PAGE_MAP

if __name__ == "__main__":
    current_page = st.sidebar.radio("Go To", list(PAGE_MAP))
    PAGE_MAP[current_page]().write()
