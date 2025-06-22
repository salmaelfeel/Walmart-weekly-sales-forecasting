import streamlit as st
import pickle
import pandas as pd
from prophet.plot import plot_plotly

# Load the saved model
with open('prophet_model.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

st.title("Walmart Weekly Sales Forecast")

# Get forecast horizon from user
periods = st.number_input("Weeks to forecast:", min_value=1, max_value=52, value=10)

if st.button("Generate Forecast"):
    # Make future df
    future = prophet_model.make_future_dataframe(periods=periods, freq='W')
    
    # Predict
    forecast = prophet_model.predict(future)
    
    # Plot
    st.write(plot_plotly(prophet_model, forecast))
    
    # Show forecast data
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
