'''
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from joblib import dump

# Replace with your Alpha Vantage API key
API_KEY = ' TE2D1BAI2J73EUU2'

# Function to train and save model
def train_and_save_model(stock_symbol, model_name, scaler_name):
    ts = TimeSeries(API_KEY, output_format='pandas')
    
    try:
        df1 = ts.get_daily(stock_symbol, outputsize='full')
    except ValueError as e:
        print(f"Error retrieving data for {stock_symbol}: {e}")
        return
    
    df = pd.DataFrame(df1[0])
    df.sort_index(inplace=True)
    Df = df[['4. close']]
    dataset = Df.values

    training_data_len = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    model.save(model_name)
    dump(scaler, scaler_name)

# Update stock symbols as needed
train_and_save_model('AAPL', 'model_aapl.keras', 'scaler_aapl.joblib')
train_and_save_model('MSFT', 'model_msft.keras', 'scaler_msft.joblib')
train_and_save_model('TSLA', 'model_tsla.keras', 'scaler_tsla.joblib')
train_and_save_model('META', 'model_meta.keras', 'scaler_meta.joblib')  # Updated from 'FB' to 'META'
'''
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from joblib import dump

# Replace with your Alpha Vantage API key
API_KEY = ' TE2D1BAI2J73EUU2'

# Function to train and save model
def train_and_save_model(stock_symbol, model_name, scaler_name):
    ts = TimeSeries(API_KEY, output_format='pandas')
    
    try:
        df1 = ts.get_daily(stock_symbol, outputsize='full')
        df1.head()
    except ValueError as e:
        print(f"Error retrieving data for {stock_symbol}: {e}")
        return
    
    df = pd.DataFrame(df1[0])
    df.sort_index(inplace=True)
    Df = df[['4. close']]
    dataset = Df.values

    training_data_len = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    model.save(model_name)
    dump(scaler, scaler_name)

# Update stock symbols as needed
train_and_save_model('AAPL', 'model_aapl.keras', 'scaler_aapl.joblib')
train_and_save_model('MSFT', 'model_msft.keras', 'scaler_msft.joblib')
train_and_save_model('TSLA', 'model_tsla.keras', 'scaler_tsla.joblib')
train_and_save_model('META', 'model_meta.keras', 'scaler_meta.joblib') 
