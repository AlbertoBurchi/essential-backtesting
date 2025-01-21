import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#%%

# Download data from yfinance
ticker = 'MMM'  # Example ticker
start_date = '2020-01-01'
end_date = '2025-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Compute short and long SMAs
short_window = 14#50
long_window = 50#200

data['short_SMA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['long_SMA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

#%%
# Implement golden cross strategy
data['Signal'] = 0
data['Position'] = 0
data['Signal'] = np.where(data['short_SMA'] > data['long_SMA'], 1, 0)
data['Operation'] = data['Signal'].diff()
data['Position'] = data['Signal'].shift(1).fillna(0)
#%%

# Plot the closing price and SMAs
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['short_SMA'], label='SHORT SMA')
plt.plot(data['long_SMA'], label='LONG SMA')

# Plot buy signals
plt.plot(data[data['Operation'] == 1].index,
         data['short_SMA'][data['Operation'] == 1],
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
plt.plot(data[data['Operation'] == -1].index,
         data['short_SMA'][data['Operation'] == -1],
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title(f'{ticker} Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Display the data with signals
print(data.tail())

#%% Backtest the strategy

#%% Backtest the strategy
# Calculate daily returns
data['Daily Return'] = data['Close'].pct_change()

# Calculate strategy returns
data['Strategy Return'] = data['Daily Return'] * data['Position'].shift(1)

# Calculate cumulative returns
data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(data['Cumulative Market Return'], label='Market Return')
plt.plot(data['Cumulative Strategy Return'], label='Strategy Return')
plt.title(f'{ticker} Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Display the final cumulative returns
print(f"Final Cumulative Market Return: {data['Cumulative Market Return'].iloc[-1]:.2f}")
print(f"Final Cumulative Strategy Return: {data['Cumulative Strategy Return'].iloc[-1]:.2f}")


