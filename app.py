import streamlit as st
import pandas as pd
import os
import datetime
import stock_news_page
import paper_trading  # Import the paper trading module


# Sidebar for navigation
st.sidebar.title("Navigation")

# Radio button for selecting the chart type (placed in the sidebar)
page = st.sidebar.radio("Choose a chart", ["Paper Trading", "Stock News"])

# Navigation logic based on the selected option in the sidebar
if   page == "Paper Trading":
    paper_trading.app()
elif page == "Stock News":
    stock_news_page.app()  # Call the paper trading page function