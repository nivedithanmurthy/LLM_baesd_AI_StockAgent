import pandas as pd
import yfinance as yf
import openai
import backtrader as bt
import time
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up OpenAI API key
openai.api_key = 'your_secret_api_key_here'

# Define the list of Yahoo Finance tickers
tickers = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'BABA', 'V', 'JPM', 'JNJ', 'WMT', 'PG', 'DIS',
    'HD', 'MA', 'PYPL', 'BAC', 'XOM', 'VZ', 'PFE', 'KO', 'MRK', 'INTC', 'CSCO', 'ORCL', 'ABT', 'PEP', 'NKE', 'MCD'
]

# Function to get stock recommendations from ChatGPT with retry logic and batch processing
def get_stock_recommendations(tickers):
    prompt = f"""
    You are a stock market expert. Based on the historical performance and current market conditions, which of the following stocks would you recommend investing in for the next quarter: {', '.join(tickers)}? Please provide your reasoning.
    """
    
    max_retries = 5
    initial_delay = 10  # initial delay in seconds
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a stock market expert."},
                          {"role": "user", "content": prompt}],
                max_tokens=300  # Increase max tokens if needed
            )
            return response.choices[0].message['content'].strip()
        except openai.error.RateLimitError as e:
            delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limit error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("Failed to get a response from OpenAI API after multiple attempts.")

# Download historical data for tickers from Yahoo Finance
def download_data(tickers):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start='2010-01-01', end='2023-12-31')
        if not data.empty:
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            stock_data[ticker] = data[['open', 'high', 'low', 'close', 'volume']]
    return stock_data

# Get stock recommendations
recommendations = get_stock_recommendations(tickers)
print("Stock Recommendations from ChatGPT:")
print(recommendations)

# Download historical data for the tickers
stock_data = download_data(tickers)

# Create a DataFrame for backtrader
class PandasData(bt.feeds.PandasData):
    lines = ('close',)
    params = (('datetime', None), ('close', -1))

# Define a strategy with the best MA and RSI indicators
class BestChatGPTStrategy(bt.Strategy):
    params = (
        ('ma_period', 50),  # Best Moving Average period
        ('rsi_period', 5),  # Best RSI period
        ('rsi_overbought', 70),  # RSI overbought threshold
        ('rsi_oversold', 30)  # RSI oversold threshold
    )

    def __init__(self):
        self.recommendations = recommendations
        self.dataclose = {data._name: data.close for data in self.datas}
        self.ma = {data._name: bt.indicators.SimpleMovingAverage(data.close, period=self.params.ma_period) for data in self.datas}
        self.rsi = {data._name: bt.indicators.RelativeStrengthIndex(data.close, period=self.params.rsi_period) for data in self.datas}
        self.actual_prices = []
        self.predicted_prices = []

    def next(self):
        for data in self.datas:
            ticker = data._name
            if ticker in self.recommendations:
                actual_price = self.dataclose[ticker][0]
                
                # Ensure the moving average and RSI are valid
                if len(data) >= max(self.params.ma_period, self.params.rsi_period):
                    # Predict the price using the moving average
                    predicted_price = self.ma[ticker][0]

                    if not self.getposition(data).size:  # not in the market
                        if self.dataclose[ticker][0] > self.ma[ticker][0] and self.rsi[ticker][0] < self.params.rsi_oversold:
                            print(f"Buying {ticker} at {self.dataclose[ticker][0]}")
                            self.buy(data=data, size=10)  # buy 10 shares
                            self.actual_prices.append(actual_price)
                            self.predicted_prices.append(predicted_price)
                    else:
                        if self.rsi[ticker][0] > self.params.rsi_overbought:
                            print(f"Selling {ticker} at {self.dataclose[ticker][0]}")
                            self.sell(data=data, size=10)  # sell 10 shares
                            self.actual_prices.append(actual_price)
                            self.predicted_prices.append(predicted_price)

# Initialize the Cerebro engine
cerebro = bt.Cerebro()

# Split the data into training and testing sets
training_data = {}
testing_data = {}
buffer_period = '2022-12-01'  # Add a buffer period to ensure indicators have enough data

for ticker, df in stock_data.items():
    training_data[ticker] = df.loc[:'2022-12-31']
    testing_data[ticker] = df.loc[buffer_period:]

# Add the training data feeds
for ticker, df in training_data.items():
    df.index.name = 'datetime'
    data = PandasData(dataname=df)
    cerebro.adddata(data, name=ticker)

# Add the strategy with the best parameters
cerebro.addstrategy(BestChatGPTStrategy)

# Set the initial cash
cerebro.broker.setcash(100000.0)

# Run the backtest on training data
print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
strategies = cerebro.run()
print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

# Extract actual and predicted prices from the strategy for training data
strategy = strategies[0]
training_actual_prices = strategy.actual_prices
training_predicted_prices = strategy.predicted_prices

# Initialize the Cerebro engine for testing data
cerebro = bt.Cerebro()
cerebro.broker.setcash(100000.0)

# Add the testing data feeds
for ticker, df in testing_data.items():
    df.index.name = 'datetime'
    data = PandasData(dataname=df)
    cerebro.adddata(data, name=ticker)

# Add the strategy with the best parameters
cerebro.addstrategy(BestChatGPTStrategy)

# Run the backtest on testing data
print("Starting Portfolio Value (Testing): %.2f" % cerebro.broker.getvalue())
strategies = cerebro.run()
print("Final Portfolio Value (Testing): %.2f" % cerebro.broker.getvalue())

# Extract actual and predicted prices from the strategy for testing data
strategy = strategies[0]
testing_actual_prices = strategy.actual_prices
testing_predicted_prices = strategy.predicted_prices

# Calculate error metrics for training data
training_mae = mean_absolute_error(training_actual_prices, training_predicted_prices)
training_mse = mean_squared_error(training_actual_prices, training_predicted_prices)
training_rmse = np.sqrt(training_mse)
training_mape = np.mean(np.abs((np.array(training_actual_prices) - np.array(training_predicted_prices)) / np.array(training_actual_prices))) * 100
training_r2 = r2_score(training_actual_prices, training_predicted_prices)

# Print error metrics for training data
print("Training Data Error Metrics:")
print(f"Mean Absolute Error (MAE): {training_mae}")
print(f"Mean Squared Error (MSE): {training_mse}")
print(f"Root Mean Squared Error (RMSE): {training_rmse}")
print(f"Mean Absolute Percent Error (MAPE): {training_mape}%")
print(f"R-squared (R²): {training_r2}")

# Calculate error metrics for testing data
testing_mae = mean_absolute_error(testing_actual_prices, testing_predicted_prices)
testing_mse = mean_squared_error(testing_actual_prices, testing_predicted_prices)
testing_rmse = np.sqrt(testing_mse)
testing_mape = np.mean(np.abs((np.array(testing_actual_prices) - np.array(testing_predicted_prices)) / np.array(testing_actual_prices))) * 100
testing_r2 = r2_score(testing_actual_prices, testing_predicted_prices)

# Print error metrics for testing data
print("Testing Data Error Metrics:")
print(f"Mean Absolute Error (MAE): {testing_mae}")
print(f"Mean Squared Error (MSE): {testing_mse}")
print(f"Root Mean Squared Error (RMSE): {testing_rmse}")
print(f"Mean Absolute Percent Error (MAPE): {testing_mape}%")
print(f"R-squared (R²): {testing_r2}")

# Plot the results for testing data
#try:
    #cerebro.plot()
#except Exception as e:
    #print(f"Plotting failed: {e}")

# Print recommendations
print("Final Stock Recommendations:")
print(recommendations)
