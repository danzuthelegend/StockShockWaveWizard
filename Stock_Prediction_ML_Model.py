
### STOCK SHOCKWAVE WIZARD:  A STOCK MARKET FORECASTER!

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from pmdarima import auto_arima
import seaborn as sborn
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import warnings
from statsmodels.tsa.arima.model import ARIMA

yf.pdr_override()
error_found = False
# Function to get and preprocess stock data
# def get_stock_data(company):
#     while True:
#             ticker = yf.Ticker(company)
#             if not ticker.info or 'quoteType' not in ticker.info:
#                 st.error(f"Stock ticker '{company}' does not exist on Yahoo Finance. Please enter a valid stock ticker symbol.")
#                 #company = st.text_input("Company Stock Ticker: ", key="retry_input")
#                 # if st.button("Retry", key="retry_button"):
#                 #     get_stock_data(company)
#                 error_found = True
#                 break
#     end = datetime.now()
#     start = datetime(end.year - 10, end.month, end.day)
#     dframe = pdr.get_data_yahoo(company, start=start, end=end)
#     dframe['SMA250'] = dframe['Adj Close'].rolling(250).mean()
#     dframe['SMA100'] = dframe['Adj Close'].rolling(100).mean()
#     dframe['SMA50'] = dframe['Adj Close'].rolling(50).mean()
#     dframe['EMA50'] = dframe['Adj Close'].ewm(span=50, adjust=False).mean()
#     dframe['RSI10'] = RSIIndicator(dframe['Adj Close'], window=10).rsi()
#     dframe['RSI14'] = RSIIndicator(dframe['Adj Close'], window=14).rsi()
#     ema_12 = dframe['Adj Close'].ewm(span=12, adjust=False).mean()
#     ema_26 = dframe['Adj Close'].ewm(span=26, adjust=False).mean()
#     dframe['MACD_Line'] = ema_12 - ema_26
#     dframe['Signal_Line'] = dframe['MACD_Line'].ewm(span=9, adjust=False).mean()
#     dframe['MACD_Histogram'] = dframe['MACD_Line'] - dframe['Signal_Line']
#     dframe = dframe.dropna()
#     return dframe



# Function to train ARIMA model
def train_arima_model(train_data, training_vals_array):
    best_parameter_model = auto_arima(train_data['Adj Close'], exog=training_vals_array, seasonal=False, stepwise=True, suppress_warnings=True)
    p_val, d_val, q_val = best_parameter_model.order
    arima_model = ARIMA(train_data['Adj Close'], exog=training_vals_array, order=(p_val, d_val, q_val))
    best_ARIMA_result = arima_model.fit()
    return best_ARIMA_result

# Function to evaluate ARIMA model
def evaluate_arima_model(best_ARIMA_result, test_data, training_size):
    start_index = training_size
    end_index = len(test_data) + training_size - 1
    arima_predictions = best_ARIMA_result.predict(start=start_index, end=end_index,
                                                  exog=test_data[['EMA50', 'RSI10', 'RSI14', 'SMA250', 'SMA100', 'SMA50', 'MACD_Line', 'Signal_Line', 'MACD_Histogram']],
                                                  dynamic=False)
    rmse = np.sqrt(mean_squared_error(test_data['Adj Close'], arima_predictions))
    return arima_predictions, rmse

# Function to train LSTM model
def train_lstm_model(residuals_scaled, time_step):
    residuals_gen = TimeseriesGenerator(residuals_scaled, residuals_scaled, length=time_step, batch_size=1)  ##-> generates sequences from the normalized data, which is crucial for maintaiing its temporal order.
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.40))
    model.add(LSTM(units=50))
    model.add(Dropout(0.40))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(residuals_gen, epochs=15, verbose=1)
    return model

# Function to predict 'RESIDUALS' using LSTM model --> (which would then be added to the predictions by the ARIMA model)
def predict_lstm_model(model, residuals_scaled, time_step):
    residuals_gen = TimeseriesGenerator(residuals_scaled, residuals_scaled, length=time_step, batch_size=1)
    lstm_predictions = model.predict(residuals_gen)
    return lstm_predictions

# Function to tune noise parameters for ARIMA model --> DEAL WITH OVERFITTING!
def tune_noise_parameters(model, train_endog, train_exog, test_endog, test_exog, future_steps, noise_means, noise_stds, seed):
    best_params = None
    best_mse = float("inf")
    best_mae = float("inf")
    best_predictions = None

    for mean in noise_means:
        for std in noise_stds:
            mse, mae, predictions = evaluate_model(model, train_endog, train_exog, test_endog, test_exog, future_steps, mean, std, seed)
            if mse < best_mse and mae < best_mae:
                best_mse = mse
                best_mae = mae
                best_params = (mean, std)
                best_predictions = predictions

    return best_params, best_mse, best_mae, best_predictions

# Function to evaluate model performance
def evaluate_model(model, train_endog, train_exog, test_endog, test_exog, future_steps, mean, std, seed):
    np.random.seed(seed)
    last_known_exog = test_exog.iloc[-1].values.reshape(1, -1)
    future_exog = np.tile(last_known_exog, (future_steps, 1))
    future_exog += np.random.normal(mean, std, future_exog.shape)
    forecast = model.get_forecast(steps=future_steps, exog=future_exog)
    predictions = forecast.predicted_mean
    mse = mean_squared_error(test_endog[:future_steps], predictions)
    mae = mean_absolute_error(test_endog[:future_steps], predictions)
    return mse, mae, predictions

# Main function
def main():
    
    import base64
    import random
    
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Let's display a randomly chosen background photo!
    #image_paths = ['./photo1.jpg', './photo2.jpg', './photo3.jpg', './photo4.jpg', './photo5.jpg','./photo6.jpg', './photo7.jpg','./photo8.jpg',
     #              './photo9.jpg', './photo10.jpg', './photo11.jpg','./photo12.jpg']
    
    
    image_paths = ['./background_photos/AdobeStock1.jpeg', './background_photos/AdobeStock2.jpeg', './background_photos/AdobeStock3.jpeg', './background_photos/AdobeStock4.jpeg','./background_photos/AdobeStock5.jpeg',
                   './background_photos/AdobeStock6.jpeg','./background_photos/AdobeStock7.jpeg','./background_photos/AdobeStock8.jpeg','./background_photos/AdobeStock9.jpeg','./background_photos/AdobeStock10.jpeg', './background_photos/AdobeStock11.jpeg',
                   './background_photos/AdobeStock12.jpeg', './background_photos/AdobeStock13.jpeg', './background_photos/AdobeStock14.jpeg',]

    # Randomly select an image
    selected_image_path = random.choice(image_paths)
    # Convert the image to base64
    img_base64 = get_base64_of_bin_file(selected_image_path)

    # CSS to add the background image
    page_background_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    h1 {{
        text-align: center;
        font-family: 'Great Vibes', cursive;
        color: #FF8C00;
    }}
    h4 {{
    text-align: center;
    font-family: 'Lobster', sans-serif;
    color: #8B0000;
    }}
    p {{
        text-align: center;
        font-family: 'Roboto', sans-serif;
        color: #32CD32;
    }}
    div[data-baseweb="input"] > div {{
        width: 300px;
        margin: 0 auto;
    }}
    input {{
        font-size: 20px !important;
    }}
    label {{
        font-size: 24px !important;
        font-family: 'Lobster', sans-serif;
    }}
    </style>
    """

    # Apply the CSS
    st.markdown(page_background_img, unsafe_allow_html=True)

    st.markdown("<h1>Stock ShockWave Wizard</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Creators: 'Shahmeer Sajid' & 'Danyal Valika'</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: black; font-size: 20px; font-weight: bold;'>Welcome to the ultimate stock prediction platform!</p>", unsafe_allow_html=True)
        
    st.markdown("<p style='font-size: 25px; font-weight: 900; color: black;'>Please press 'Enter' after typing the stock ticker to search if the stock ticker exists on 'Yahoo Finance'.</p>", unsafe_allow_html=True)

    

    st.markdown("""
    <style>
    .custom-label {
        font-size: 24px !important;
        font-weight: bold !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Wrap the text input in a container with the custom class
    container = st.container()
    with container:
        st.markdown("<p style='font-size: 27px; font-weight: 900; color: black;'>Company Stock Ticker: </p>", unsafe_allow_html=True)
        #st.markdown('<p class="custom-label">Company Stock Ticker:</p>', unsafe_allow_html=True)
        
        company = st.text_input("", key="initial_input")
    
    
    
    ticker = yf.Ticker(company)
    if not ticker.info or 'quoteType' not in ticker.info:
        # st.error(f"Stock ticker '{company}' does not exist on Yahoo Finance. Please enter a valid stock ticker symbol.")
        st.markdown("<p style='font-size: 23px; font-weight: 900; color: black;'>The entered Stock Ticker does not exist on Yahoo Finance. Please enter a valid stock ticker.</p>", unsafe_allow_html=True)

        error_found = True
    else:

        st.markdown("<p style='font-size: 23px; font-weight: 900; color: black;'>Stock Ticker succesfully found! Please press 'Predict' to continue.</p>", unsafe_allow_html=True)

        error_found = False
    if st.button("Predict", key="predict_button") and (error_found == False):       
   

        end = datetime.now()
        start = datetime(end.year - 10, end.month, end.day)
        dframe = pdr.get_data_yahoo(company, start=start, end=end)
        dframe = pdr.get_data_yahoo(company, start=start, end=end)
        dframe['SMA250'] = dframe['Adj Close'].rolling(250).mean()
        dframe['SMA100'] = dframe['Adj Close'].rolling(100).mean()
        dframe['SMA50'] = dframe['Adj Close'].rolling(50).mean()
        dframe['EMA50'] = dframe['Adj Close'].ewm(span=50, adjust=False).mean()
        dframe['RSI10'] = RSIIndicator(dframe['Adj Close'], window=10).rsi()
        dframe['RSI14'] = RSIIndicator(dframe['Adj Close'], window=14).rsi()
        ema_12 = dframe['Adj Close'].ewm(span=12, adjust=False).mean()
        ema_26 = dframe['Adj Close'].ewm(span=26, adjust=False).mean()
        dframe['MACD_Line'] = ema_12 - ema_26
        dframe['Signal_Line'] = dframe['MACD_Line'].ewm(span=9, adjust=False).mean()
        dframe['MACD_Histogram'] = dframe['MACD_Line'] - dframe['Signal_Line']
        dframe = dframe.dropna()
    
        st.markdown(f"<h3 style='font-family: Georgia, serif; color: #8B0000;'>*Last 50 Days* Data for '{company}'</h3>", unsafe_allow_html=True)
        st.write(dframe.tail(50))
    
        fig, ax = plt.subplots()
        ax.plot(dframe.index, dframe['Adj Close'], label='Closing Price', color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price/USD")
        ax.set_title(f"Adj Close Stock Price trend for '{company}'")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        correlation = dframe.corr()
        sborn.heatmap(correlation, annot=True, ax=ax)
        ax.set_title(f"Correlation matrix for '{company}'")
        st.pyplot(fig)
        

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dframe['Adj Close'], 'b', label="Adj Close")
        ax.plot(dframe['SMA250'], 'r', label="MA_250days")
        ax.plot(dframe['SMA100'], 'g', label="MA_100days")
        ax.plot(dframe['SMA50'], 'm', label="MA_50days")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price/USD")
        ax.set_title(f"STOCK PRICES MOVING AVERAGES FOR '{company}'")
        ax.legend()
        st.pyplot(fig)

        mean_value = dframe['Adj Close'].mean()
        dframe['Adj Close'] = dframe['Adj Close'].fillna(mean_value)
        training_size = int(len(dframe) * 0.80)
        test_size = len(dframe) - training_size
        train_data = dframe[:training_size]
        test_data = dframe[training_size:]
        original_length = len(dframe)

        training_vals_array = train_data[['EMA50', 'RSI10', 'RSI14', 'SMA250','SMA100','SMA50', 'MACD_Line', 'Signal_Line','MACD_Histogram']].values
        best_ARIMA_result = train_arima_model(train_data, training_vals_array)
        arima_predictions, rmse_arima = evaluate_arima_model(best_ARIMA_result, test_data, training_size)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dframe.index[training_size:], test_data['Adj Close'], label='Actual')
        ax.plot(dframe.index[training_size:], arima_predictions, label='Predicted')
        ax.set_title(f'ARIMA Model - Actual vs Predicted (Validation Set) for: {company}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adj Close Price')
        ax.legend()
        st.pyplot(fig)

        adj_close_reset = test_data['Adj Close'].reset_index(drop=True)
        adj_close_reset.index = pd.RangeIndex(start=training_size, stop=len(dframe), step=1)
        residuals = adj_close_reset - arima_predictions

        scaler = MinMaxScaler(feature_range=(0, 1))
        residuals_scaled = scaler.fit_transform(np.array(residuals).reshape(-1, 1))
        time_step = 25

        lstm_model = train_lstm_model(residuals_scaled, time_step)
        lstm_predictions = predict_lstm_model(lstm_model, residuals_scaled, time_step)
        lstm_predictions = scaler.inverse_transform(lstm_predictions)

        aligned_arima_predictions = arima_predictions[-len(lstm_predictions):]

        combined_predictions = aligned_arima_predictions + lstm_predictions.flatten()

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(test_data.index[time_step:], adj_close_reset[time_step:], label="Actual Previous 'Adj Close' Values", color="r")
        ax.plot(test_data.index[time_step:], aligned_arima_predictions, label="ARIMA Predictions", color="b")
        ax.plot(test_data.index[time_step:], combined_predictions, label="Combined Hybrid Predictions", color="y")
        ax.set_xlabel("Index")
        ax.set_ylabel("Price")
        ax.set_title(f"Combined Hybrid LSTM + ARIMA Stock Predictions for '{company}'")
        ax.legend()
        st.pyplot(fig)

        train_endog = train_data['Adj Close']
        train_exog = train_data[['EMA50', 'RSI10', 'RSI14', 'SMA250','SMA100','SMA50', 'MACD_Line', 'Signal_Line', 'MACD_Histogram']]
        test_endog = test_data['Adj Close']
        test_exog = test_data[['EMA50', 'RSI10', 'RSI14', 'SMA250','SMA100','SMA50', 'MACD_Line', 'Signal_Line', 'MACD_Histogram']]
        noise_means = np.linspace(-0.5, 1.0, 5)
        noise_stds = np.linspace(0.5, 1.7, 5)
        random_seed = 40
        future_steps = 10
        best_params, best_mse, best_mae, best_predictions = tune_noise_parameters(best_ARIMA_result, train_endog, train_exog, test_endog, test_exog, future_steps, noise_means, noise_stds, random_seed)

        arima_future_predictions = np.maximum(best_predictions, 0)

        last_residuals = residuals_scaled[-time_step:]
        future_residuals = []
        
        ## Predicting future residuals, using the LSTM model we created
        for i in range(future_steps):
            input_seq = last_residuals.reshape(1, time_step, 1)
            pred_residual = lstm_model.predict(input_seq)
            future_residuals.append(pred_residual[0, 0])
            last_residuals = np.append(last_residuals[1:], pred_residual)
        future_residuals = scaler.inverse_transform(np.array(future_residuals).reshape(-1, 1)).flatten()

        combined_future_predictions = arima_future_predictions + future_residuals

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(test_data.index[time_step:], adj_close_reset[time_step:], label="Actual Previous 'Adj Close' Values", color="r", alpha=0.99)

        # Create a range for future predictions starting from the end of the test data
        future_indices = np.arange(len(test_data), len(test_data) + len(combined_future_predictions))

        # # Plot the future predictions
        # ax.plot(future_indices, combined_future_predictions, label="Next 10 Day 'Adj Close' Predictions", color="b", alpha=0.99)

        # ax.set_xlabel("Index")
        # ax.set_ylabel("Price/USD")
        # ax.set_title(f"Previous 'Adj Close' Stock Trend and 10 Day Predictions for: {company}")
        # ax.legend()
        # ax.grid(True)
        # st.pyplot(fig)

        current_date = datetime.now()
        future_dates = [current_date + timedelta(days=i) for i in range(1, future_steps + 1)]
        predicted_data = pd.DataFrame({
            'Date': future_dates,
            'Predicted Adj Close': combined_future_predictions
        })
        
        st.markdown(f"<h3 style='font-family: Georgia, serif; color: #000000;'>Next 10 Day 'Adj Close' Stock Price Prediction for: {company}</h3>", unsafe_allow_html=True)
        
        html_table = predicted_data.to_html(index=False)
        html_table = html_table.replace('<table border="1" class="dataframe">', 
        '<table border="1" class="dataframe" style="color:black; font-weight:bold;">')

        # Display the styled HTML table
        st.write(html_table, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
