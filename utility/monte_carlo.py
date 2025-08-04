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

    def __init__(self, tickers, start_date=dt.datetime.now()-dt.timedelta(days=5*365.25), end_date=dt.datetime.now()):
        self.tickers = tickers

        # Convert to string format for yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self.end_date = end_date
        self.stock_data = yf.download(tickers, start=start_str, end=end_str)
        self.stock_data.index = pd.to_datetime(self.stock_data.index)

        # Check for missing data after start date
        if self.stock_data.isna().values.any():
            # Find the latest index of a row that contains a NaN value
            nan_rows = self.stock_data.isna().any(axis=1)
            print(nan_rows)
            self.start_date = nan_rows[::-1].idxmax()
            print(self.start_date)
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

    def calculate_annual_rate(self, initial_investment, final_investment, days_passed):
        # Calculate the total return
        total_return = final_investment / initial_investment - 1
        
        # Convert days to years
        years = days_passed / 365.25
        
        # Calculate the effective annual rate using the compound interest formula
        # (1 + r)^n = final_value / initial_value
        # r = (final_value / initial_value)^(1/n) - 1
        annual_rate = (final_investment / initial_investment) ** (1 / years) - 1

        self.annual_rate = annual_rate
        self.total_return = total_return
        self.years = years
    
        return annual_rate

    def run_sim(self, initial_investment=10000, n_sims=100, n_days=365, by='Close', weights=None):
        """
        If you want to set the weights, you need to pass in the weights as a list of floats.
        If you want to use the random weights, you can pass in None.
        """
        meanReturns = self.calc_mean(by=by)
        covMatrix = self.calc_covariance_matrix(by=by)
        self.initial_investment = initial_investment
        self.n_sims = n_sims
        self.n_days = n_days

        self.meanReturns = meanReturns
        self.covMatrix = covMatrix
        meanMatrix = np.full(shape=(n_days, len(self.tickers)), fill_value=meanReturns).T
        portfolio_sims = np.full(shape=(n_days, n_sims), fill_value=0.0)

        if weights == None:
            weights = self.gen_weights()
        
        self.weights = weights

        for sim in range(n_sims):
            # Z is a matrix of random normal variables
            Z = np.random.normal(size=(n_days, len(weights)))
            # L is the lower triangular matrix of the covariance matrix cholesky decomposition
            L = np.linalg.cholesky(covMatrix)
            daily_returns = meanMatrix + np.inner(L, Z)

            portfolio_sims[:, sim] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initial_investment

        self.portfolio_sims = portfolio_sims
        self.mean_final_investment = portfolio_sims[-1].mean()
        self.sd_final_investment = portfolio_sims[-1].std()

        # Calculate the annualized return
        self.calculate_annual_rate(self.initial_investment, self.mean_final_investment, self.n_days)

        print("\n")
        print(50*"*")
        print("INITIAL CONDITIONS")
        print(50*"*")
        print(f"Initial Investment: {self.initial_investment}")
        print(f"Years Simulated: {self.years}")
        for (ticker, weight) in zip(self.tickers, self.weights):
            print(f"Weight of {ticker}: {weight}")
        print(50*"*")

        print("\n")

        print(50*"*")
        print("FINAL CONDITIONS")
        print(50*"*")
        print(f"Mean Final Investment: {self.mean_final_investment}")
        print(f"SD Final Investment: {self.sd_final_investment}")
        print(f"Mean Annual Rate: {self.annual_rate}")
        print(f"Mean Total Return: {self.total_return}")
        print(50*"*")

        print('\n')
        print("Displaying portfolio simulations as a matrix. Each row is a day, each column is a simulation.")
        print("\n")
        print(portfolio_sims)
        print("\n")

        return portfolio_sims

    def daily_avg(self):
        return np.mean(self.portfolio_sims, axis=1)

    def plot_sim(self):
        # Plot the daily return for each simulation
        plt.plot(self.portfolio_sims, alpha = 0.7,linewidth=1)
        
        # Plot the average daily return of each simulation
        mean_line = np.mean(self.portfolio_sims, axis=1)
        plt.plot(mean_line, color='red', linewidth=3)

        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        plt.title('Monte Carlo Simulation')
        plt.show()

if __name__ == '__main__':

    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    sim = MonteCarlo(stocks)
    sim.run_sim()
    #sim.plot_sim()
