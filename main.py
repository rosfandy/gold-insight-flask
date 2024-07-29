from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Load the model and scalers
model = load_model('model/gold_model.h5')
scaler_X = joblib.load('model/scaler_X.pkl')
scaler_y = joblib.load('model/scaler_y.pkl')

# Path to the persistent CSV file
persistent_csv_path = 'data/predicted_gold_price.csv'

# Function to create sequences for the model
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

@app.route('/')
def index():
    return "Gold Price Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the month and year from the form
    month = int(request.json['month'])
    year = int(request.json['year'])

    # Set start and end dates for predictions
    start_of_month = datetime(year, month, 1)
    end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    dates = pd.date_range(start_of_month, end_of_month)

    # Check if the persistent CSV file exists
    if os.path.exists(persistent_csv_path):
        # Load the persistent CSV if it exists
        df = pd.read_csv(persistent_csv_path, delimiter=',', decimal='.')
    else:
        # Load the original data if the persistent CSV does not exist
        df = pd.read_csv('data/gold_price.csv', delimiter=',', decimal='.')

    df['Periode'] = pd.to_datetime(df['Periode'], infer_datetime_format=True)
    df.set_index('Periode', inplace=True)
    df.sort_index(inplace=True)

    # Create a DataFrame to hold predictions
    predictions_df = pd.DataFrame(index=dates, columns=['Gold price', 'Minyak', 'Suku bunga', 'Inflasi'])

    # Start with the last 60 days of data from the dataframe
    last_60_days = df[-60:]

    # Iterate over the dates and predict one day at a time
    for current_date in predictions_df.index:
        # Select the last 60 days of data
        data = last_60_days[['Minyak', 'Suku bunga', 'Inflasi']].values
        X_scaled = scaler_X.transform(data)
        X_sequence = create_sequences(X_scaled, 60)

        # Predict the next day's price
        y_pred_scaled = model.predict(X_sequence[-1].reshape(1, 60, 3))
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predictions_df.loc[current_date, 'Gold price'] = y_pred[0, 0]

        # Dynamically update the values for 'Minyak', 'Suku bunga', and 'Inflasi' (for demonstration purposes)
        # In a real scenario, you'd use a more sophisticated method to predict these values
        predictions_df.loc[current_date, 'Minyak'] = last_60_days['Minyak'].iloc[-1] * (1 + np.random.normal(0, 0.01))
        predictions_df.loc[current_date, 'Suku bunga'] = last_60_days['Suku bunga'].iloc[-1] * (1 + np.random.normal(0, 0.01))
        predictions_df.loc[current_date, 'Inflasi'] = last_60_days['Inflasi'].iloc[-1] * (1 + np.random.normal(0, 0.01))

        # Add the predicted price to the DataFrame
        new_row = pd.DataFrame({
            'Periode': current_date,
            'Gold price': predictions_df.loc[current_date, 'Gold price'],
            'Minyak': predictions_df.loc[current_date, 'Minyak'],
            'Suku bunga': predictions_df.loc[current_date, 'Suku bunga'],
            'Inflasi': predictions_df.loc[current_date, 'Inflasi']
        }, index=[current_date])
        new_row.set_index('Periode', inplace=True)
        last_60_days = pd.concat([last_60_days, new_row]).tail(60)

    # Save only the last 60 days to the persistent CSV
    last_60_days.to_csv(persistent_csv_path, index=True)

    # Convert predictions to JSON format
    predictions_json = predictions_df.reset_index().to_dict(orient='records')

    return jsonify(predictions_json)

if __name__ == '__main__':
    app.run(debug=True)
