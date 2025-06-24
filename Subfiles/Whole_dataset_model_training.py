import pickle
import pandas as pd
from prophet import Prophet

# Read dataset
data = pd.read_csv(r'D:\ITI\24) Time Series\Project\Walmart-weekly-sales-forecasting\Walmart_department_27.csv')

# Print columns to verify
print("Columns:", data.columns)

# Rename columns as needed
data = data.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

# Ensure date is datetime
data['ds'] = pd.to_datetime(data['ds'])

# Train model
model = Prophet()
model.fit(data)

# Save model
with open('prophet_model_full.pkl', 'wb') as f:
    pickle.dump(model, f)

print(" New model trained and saved as 'prophet_model_full.pkl'")
