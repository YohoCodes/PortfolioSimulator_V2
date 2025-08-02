import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

class MonteCarlo:
    """
    How to use:
    1. Initialize the class with the tickers, start date, and end date.
    2. Run the simulation using the run_sim method.
    """

    def __init__(self, tickers, start_date=dt.date.today()-dt.timedelta(days=365), end_date=dt.date.today()):
        self.tickers = tickers
        self.end_date = end_date
        self.stock_data = yf.download(tickers, start=start_date, end=end_date)
        self.stock_data.index = pd.to_datetime(self.stock_data.index)

        # Check for missing data after start date
        if self.stock_data.isna().values.any():
            # Find the latest index of a row that contains a NaN value
            nan_rows = self.stock_data.isna().any(axis=1)
            self.start_date = nan_rows[::-1].idxmax()
            print('Missing row data!\n')
            print(f'Using {self.start_date} as start date')
            self.stock_data = self.stock_data.loc[self.start_date:]
        else:
            self.start_date = start_date

    def get_data(self):
        return self.stock_data
    
    def get_info(self):
        print(f"Tickers: {self.tickers}")
        print(f"Start Date: {self.start_date}")
        print(f"End Date: {self.end_date}")

    def calc_mean(self, by='Close'):
        returns = self.stock_data.xs(by, level='Price', axis=1).pct_change()
        return returns.mean()

    def calc_covariance_matrix(self, by='Close'):
        returns = self.stock_data.xs(by, level='Price', axis=1).pct_change()
        return returns.cov()

    def gen_weights(self):
        weights = np.random.random(len(self.tickers))
        weights /= np.sum(weights)
        return weights

    def run_sim(self, initial_investment=10000, n_sims=100, n_days=365, by='Close', weights=None):
        """
        If you want to set the weights, you need to pass in the weights as a list of floats.
        If you want to use the random weights, you can pass in None.
        """
        meanReturns = self.calc_mean(by=by)
        covMatrix = self.calc_covariance_matrix(by=by)
        meanMatrix = np.full(shape=(n_days, len(self.tickers)), fill_value=meanReturns).T
        portfolio_sims = np.full(shape=(n_days, n_sims), fill_value=0.0)

        if weights == None:
            weights = self.gen_weights()

        for sim in range(n_sims):
            # Z is a matrix of random normal variables
            Z = np.random.normal(size=(n_days, len(weights)))
            # L is the lower triangular matrix of the covariance matrix cholesky decomposition
            L = np.linalg.cholesky(covMatrix)
            daily_returns = meanMatrix + np.inner(L, Z)

            portfolio_sims[:, sim] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initial_investment

        self.portfolio_sims = portfolio_sims

        print(portfolio_sims)

        return portfolio_sims

    def plot_sim(self):
        plt.plot(self.portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        plt.title('Monte Carlo Simulation')
        plt.show()

if __name__ == '__main__':

    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start = pd.to_datetime('2000-01-01')

    sim = MonteCarlo(stocks, start)
    sim.run_sim()
    sim.plot_sim()
