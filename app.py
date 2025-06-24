import streamlit as st
import pickle
import pandas as pd

# Gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fee63a , #ffffff, #0c77e1);
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved Prophet model
with open('prophet_model_full.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

# Titles and description
st.markdown(
    "<h1 style='text-align: center; font-family: Arial;'>Walmart Weekly Sales Forecaster☀️📈</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; font-family: Georgia; font-style: italic;'>"
    "AI-powered tool that delivers accurate, multi-step weekly sales forecasts to support smarter business decisions"
    "</h3><br><hr>",
    unsafe_allow_html=True
)

# Forecast input
st.markdown("**How many future weeks would you like to forecast ?**")
user_input = st.number_input("", min_value=1, value=5)
total_periods = user_input

# Forecast button
if st.button("✨ Generate Forecast"):
    last_train_date = prophet_model.history['ds'].max()
    st.write("Training data ends at:", last_train_date)

    total_periods =  user_input
    future = prophet_model.make_future_dataframe(periods=total_periods, freq='W', include_history=False)
    forecast = prophet_model.predict(future)

    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2)
    forecast_table = forecast_table.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Sales',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })
    forecast_table['Date'] = pd.to_datetime(forecast_table['Date']).dt.date

    st.markdown("### **Forecast Data**")
    st.markdown(
        f"""
        <div style="height:400px; overflow-y: auto;">
            {forecast_table.to_html(index=False)}
        </div>
        """,
        unsafe_allow_html=True
    )

