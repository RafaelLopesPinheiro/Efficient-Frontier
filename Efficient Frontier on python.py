

# # Libraries

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import yfinance as yf



# # Getting Data

#setting start & end date
end = dt.datetime(2022,1,1)
start = dt.datetime(2016,1,1)

#tickers choosed, MUST BE IN ALPHABETICAL ORDER.
stocks = ['AAPL','AMZN','GOOG','NVDA']

#Getting data from Yahoo Finance
df = yf.Tickers(stocks)
df = df.history(start=start, end=end)['Close']
df.sort_index(inplace=True)
df.head()


# creating percentage returns from daily data
returns = df.pct_change()

# creating mean returns and covariance matrix from daily data
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

#setting number of portfolios runs with random weights
num_portfolios = 100000

#set up array to hold results
#increasing the number of arrays to hold the weights values for each stock
results = np.zeros((4+len(stocks)-1, num_portfolios))

for i in range(num_portfolios):
    weights = np.array(np.random.random(4))
    #rebalance weights to sum to 1
    weights /= np.sum(weights)

    #calculate portfolio return and volatility
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    #keep the results on the results array
    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    #store Sharpe Ratio: formulation (return/ volatility) - risk free rate, where risk is zero to this case
    results[2,i] = results[0,i]/results[1,i]
    #iterate through the weight vector and keep data on result array
    for j in range(len(weights)):
        results[j+3,i] = weights[j]


#keep results and convert into DataFrame
results_df = pd.DataFrame(results.T,columns=['Return','Volatility','Sharpe',stocks[0], stocks[1], stocks[2], stocks[3]])

results_df


#locate the portfolio with highest sharpe
max_sharpe = results_df.iloc[results_df['Sharpe'].idxmax()]
#locate the portfolio with lowest std deviation
min_deviation = results_df.iloc[results_df['Volatility'].idxmin()]


#plotting the efficient frontier
plt.figure(figsize=(15, 6))
plt.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap='coolwarm')
plt.xlabel('Volatility')
plt.ylabel('Expected Returns')
plt.colorbar()

#plot a mark on the highest sharp ratio
plt.scatter(max_sharpe[1],max_sharpe[0], marker = (4,1,0), color='black', s=700)
#plot a mark on the lowest std deviation portfolio
plt.scatter(min_deviation[1],min_deviation[0], marker = (4,1,0), color='g', s=700)
plt.show()


# # Results
# the portfolio with max sharpe ratio
print(max_sharpe)

# the portfolio with min standard deviation
print(min_deviation)


### Disclaimer

# this isn't the best type of strategy to use because not consider some points like costs of transactions, impact of the portfolio size in the market(liquidity) and should analyze more other metrics together.

# It's theoretical and not a recomendation.

#Past performance is no guarantee of future results.
