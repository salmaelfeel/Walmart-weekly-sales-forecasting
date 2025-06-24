import pickle
import pandas as pd

# Load model
with open('prophet_model.pkl', 'rb') as f:
    prophet_model = pickle.load(f)

# Generate future dataframe with enough periods
periods = 31  # 40 generated 10 new
future = prophet_model.make_future_dataframe(periods=periods, freq='W')

# Predict
forecast = prophet_model.predict(future)

# Check forecast range
print("Forecast covers:")
print(forecast['ds'].min(), "to", forecast['ds'].max())

# Filter predictions after 2012-10-26
forecast_filtered = forecast[forecast['ds'] > '2012-10-26']

# Prepare table
forecast_table = forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2)
forecast_table = forecast_table.rename(columns={
    'ds': 'Date',
    'yhat': 'Predicted Sales',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
})

# Print table
if forecast_table.empty:
    print("⚠ No forecast generated beyond 2012-10-26 — increase periods!")
else:
    print(forecast_table.to_string(index=False))
