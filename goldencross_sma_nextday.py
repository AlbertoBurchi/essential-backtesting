import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Download data from yfinance
ticker = 'NKE'  # Example ticker
start_date = '2020-01-01'
end_date = '2025-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Compute short and long SMAs
short_window = 50
long_window = 200

data['short_SMA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['long_SMA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

#%%
# Implement golden cross strategy
data['Signal'] = 0
data['Position'] = 0
data['Signal'] = np.where(data['short_SMA'] > data['long_SMA'], 1, 0)
data['Operation'] = data['Signal'].diff()

# Open long position at the open price of the next day after the signal
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
#%%
# Backtest the strategy
# Calculate daily returns based on open prices
data['Daily Return'] = np.log(data['Open'] / data['Open'].shift(1))

# Calculate strategy returns
data['Strategy Return'] = data['Daily Return'] * data['Position']
# Calculate cumulative returns
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Buy and hold strategy
data['Buy and Hold Return'] = (1 + data['Daily Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(data['Cumulative Strategy Return'], label='Golden Cross Strategy Return')
plt.plot(data['Buy and Hold Return'], label='Buy and Hold Return')
plt.title(f'{ticker} Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

#%%
# Calculate performance metrics
# Display the final cumulative returns
print(f"Final Cumulative Strategy Return: {data['Cumulative Strategy Return'].iloc[-1]:.2f}")
print(f"Final Buy and Hold Return: {data['Buy and Hold Return'].iloc[-1]:.2f}")

# Calculate performance metrics
# Sharpe Ratio
sharpe_ratio_strategy = data['Strategy Return'].mean() / data['Strategy Return'].std() * np.sqrt(252)
print(f"Sharpe Ratio (Strategy): {sharpe_ratio_strategy:.2f}")

sharpe_ratio_bh = data['Daily Return'].mean() / data['Daily Return'].std() * np.sqrt(252)
print(f"Sharpe Ratio (Buy and Hold): {sharpe_ratio_bh:.2f}")

# Maximum Drawdown
rolling_max_strategy = data['Cumulative Strategy Return'].cummax()
drawdown_strategy = data['Cumulative Strategy Return'] / rolling_max_strategy - 1
max_drawdown_strategy = drawdown_strategy.min()
print(f"Maximum Drawdown (Strategy): {max_drawdown_strategy:.2%}")

rolling_max_bh = data['Buy and Hold Return'].cummax()
drawdown_bh = data['Buy and Hold Return'] / rolling_max_bh - 1
max_drawdown_bh = drawdown_bh.min()
print(f"Maximum Drawdown (Buy and Hold): {max_drawdown_bh:.2%}")

# Annualized Return
annualized_return_strategy = data['Strategy Return'].mean() * 252
print(f"Annualized Return (Strategy): {annualized_return_strategy:.2%}")

annualized_return_bh = data['Daily Return'].mean() * 252
print(f"Annualized Return (Buy and Hold): {annualized_return_bh:.2%}")