import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime as dt
pd.set_option("display.max_rows", 100)


def get_clean_data(stock_list, start, end, period='1D'): # Download and clean data from yahoo finance
    data = yf.Tickers(stock_list)
    data = data.history(start=start, end=end, period=period)['Close']
    data.sort_index(inplace=True)
    data.ffill(axis=0, inplace=True)
    return data


def efficient_frontier(cov_matrix, mean_daily_returns, num_portfolios, risk_free_rate=0):
    for i in range(num_portfolios):
        weights = np.array(np.random.random(len(cov_matrix)))  # Create a array with random weights from 0.0 to 1.0
        weights /= np.sum(weights)  # Divide by the sum of weights to make sum of array equals to 1.0

        portfolio_return = np.sum(mean_daily_returns *  weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized std deviation (volatility)
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = (results[0, i] - risk_free_rate)/ results[1, i]  # Sharpe ratio formula
        
        for j in range(len(weights)):
            results[j+3, i] = weights[j]
 

def plot_efficient_frontier(results_df, max_sharpe, min_vol):
    plt.figure(figsize=(15, 6))
    plt.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap='YlGnBu')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Returns')
    plt.title("Efficient Frontier")
    plt.colorbar()
    
    plt.scatter(max_sharpe[1],max_sharpe[0], marker = (4,1,0), color='red', s=400, label='Max Sharpe ratio')
    plt.scatter(min_vol[1],min_vol[0], marker = (4,1,0), color='black', s=400, label='Min volatility')
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    end = dt.datetime(2022,1,1)
    start = dt.datetime(2015,1,1)
    stocks = ['AAPL','GOOG','AMZN','NVDA']
    df = get_clean_data(stocks, start, end)

    returns = df.pct_change()
    cov_matrix = returns.cov()
    mean_daily_returns = returns.mean()
    num_portfolios = 10000
    
    results = np.zeros((len(stocks) + 3, num_portfolios))  # Need +3 columns to sharpe, expected returns and volatility
    efficient_frontier(cov_matrix, mean_daily_returns, num_portfolios, 0.05)

    col_names = ["Return", "Volatility", "Sharpe"] + stocks.copy()
    results_df = pd.DataFrame(results.T)
    results_df.columns = [name for name in col_names]

    max_sharpe = results_df.iloc[results_df['Sharpe'].idxmax()]  # Return the row of max_sharpe ratio
    min_volatility = results_df.iloc[results_df['Volatility'].idxmin()]  # Return the row with minimum volatility 
    print('-'*70)
    print(f"Portfolio with max sharpe ratio in {num_portfolios} simulations\n{max_sharpe}")
    print('-'*70)
    print(f"Portfolio with min volatility in {num_portfolios} simulations\n{min_volatility}")  
    
    plot_efficient_frontier(results_df, max_sharpe, min_volatility)