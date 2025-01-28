from flask import Flask, render_template, request
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

app = Flask(__name__)

API_KEY = 'TE2D1BAI2J73EUU2'
models = {
    'aapl': load_model('model_aapl.keras'),
    'msft': load_model('model_msft.keras'),
    'tsla': load_model('model_tsla.keras')
}
scalers = {
    'aapl': joblib.load('scaler_aapl.joblib'),
    'msft': joblib.load('scaler_msft.joblib'),
    'tsla': joblib.load('scaler_tsla.joblib')
}

def get_apidata(company):
    ts = TimeSeries(API_KEY, output_format='pandas')
    df1 = ts.get_daily(company, outputsize='full')
    df = pd.DataFrame(df1[0])
    df.sort_index(inplace=True)
    Df = df[['4. close']]
    last_60_days = Df[-60:].values
    last_60_days_scaled = scalers[company].transform(last_60_days)
    x_test = []
    x_test.append(last_60_days_scaled)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction = ''
    if request.method == 'POST':
        company = request.form['company']
        test_data = get_apidata(company)
        model = models[company]
        pred = model.predict(test_data)
        st_value = float(scalers[company].inverse_transform(pred))
        prediction = f"{company.upper()}: ${round(st_value, 2)}"

    return render_template('index2.html', message=prediction)

@app.route('/')
def index():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)
