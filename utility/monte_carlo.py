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

    def __init__(self, tickers, sample_period_in_yrs=15):
        
        self.tickers = tickers
        self.sample_period_in_yrs = sample_period_in_yrs

        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=sample_period_in_yrs*365.25)

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
            print(f'Truncating...\nNew start date: {self.start_date}')
            self.stock_data = self.stock_data.loc[self.start_date:]
            self.truncated = True
        else:
            self.start_date = start_date
            self.truncated = False

    def get_data(self):
        return self.stock_data
    
    def get_info(self):
        print(f"Tickers: {self.tickers}")
        print(f"Sample Period: {self.sample_period_in_yrs} years")
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

    def calculate_annual_rate(self, initial_investment, final_investment, yrs_passed):
        # Calculate the total return
        total_return = final_investment / initial_investment - 1
        
        # Calculate the effective annual rate using the compound interest formula
        # (1 + r)^n = final_value / initial_value
        # r = (final_value / initial_value)^(1/n) - 1
        annual_rate = (final_investment / initial_investment) ** (1 / yrs_passed) - 1

        return annual_rate, total_return

    def run_sim(self, initial_investment=10000, n_sims=200, n_days=100, by='Close', weights=None, log_return=False, vol_drag=False):
        """
        If you want to set the weights, you need to pass in the weights as a list of floats.
        If you want to use the random weights, you can pass in None.
        
        log_return: If True, uses log-normal simulation (geometric Brownian motion)
        vol_drag: If True, adjusts mean returns for volatility drag (subtracts 0.5*σ²)
        """
        meanReturns = self.calc_mean(by=by)
        covMatrix = self.calc_covariance_matrix(by=by)
        self.initial_investment = initial_investment
        self.n_sims = n_sims
        self.n_days = n_days
        self.years = n_days / 365.25

        self.meanReturns = meanReturns
        self.covMatrix = covMatrix

        # Apply volatility drag adjustment if requested
        if vol_drag:
            # Calculate volatility drag: 0.5 * variance for each asset
            volatility_drag = 0.5 * np.diag(covMatrix)
            adjusted_mean_returns = meanReturns - volatility_drag
            print(f"Volatility drag applied. Original mean returns: {meanReturns.values}")
            print(f"Adjusted mean returns: {adjusted_mean_returns.values}")
        else:
            adjusted_mean_returns = meanReturns

        # meanMatrix is a matrix where each row contains the mean daily returns for each stock, 
        # repeated for each day in the simulation.
        meanMatrix = np.full(shape=(n_days, len(self.tickers)), fill_value=adjusted_mean_returns).T
        portfolio_sims = np.full(shape=(n_days, n_sims), fill_value=0.0)

        if weights == None:
            weights = self.gen_weights()
        
        self.weights = weights

        for sim in range(n_sims):
            # Z is a matrix of random normal variables
            Z = np.random.normal(size=(n_days, len(weights)))
            # L is the lower triangular matrix of the covariance matrix cholesky decomposition
            L = np.linalg.cholesky(covMatrix)
            
            if log_return:
                # Log-normal simulation (geometric Brownian motion)
                # Generate log-returns, then convert to arithmetic returns
                log_returns = meanMatrix + np.inner(L, Z)
                daily_returns = np.exp(log_returns) - 1
            else:
                # Standard arithmetic return simulation
                daily_returns = meanMatrix + np.inner(L, Z)

            portfolio_sims[:, sim] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initial_investment

        self.portfolio_sims = portfolio_sims
        self.mean_final_investment = portfolio_sims[-1].mean()
        self.median_final_investment = np.median(portfolio_sims[-1])
        self.sd_final_investment = portfolio_sims[-1].std()

        # Calculate the annualized return
        annual_rate, total_return = self.calculate_annual_rate(
            self.initial_investment,
            self.mean_final_investment,
            self.years
        )
        self.mean_annual_rate = annual_rate
        self.mean_total_return = total_return

        annual_rate, total_return = self.calculate_annual_rate(
            self.initial_investment,
            self.median_final_investment,
            self.years
        )
        self.median_annual_rate = annual_rate
        self.median_total_return = total_return

        print("\n")
        print(50*"*")
        print("HISTORICAL MEASUREMENTS")
        print(50*"*")

        if self.truncated:
            print("Due to insufficient data, the sample period was truncated to start at the first valid row.")
            print(f"Sample Period: {self.sample_period_in_yrs} years")
            print(f"Sampled from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
        else:
            print(f"Sample Period: {self.sample_period_in_yrs} years")
            print(f"Sampled from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
        
        print("Mean Returns:")
        for (ticker, meanReturn) in zip(self.meanReturns.index, self.meanReturns):
            print(f"    {ticker} - {meanReturn*100}%")
        print(50*"*")
        
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
        print(f"Median Final Investment: {self.median_final_investment}")
        print(f"SD Final Investment: {self.sd_final_investment}")
        print(f"Mean Annual Rate: {self.mean_annual_rate*100}%")
        print(f"Median Annual Rate: {self.median_annual_rate*100}%")
        print(f"Mean Total Return: {self.mean_total_return*100}%")
        print(f"Median Total Return: {self.median_total_return*100}%")
        print(50*"*")
        
        print('\n')
        print("Displaying portfolio simulations as a matrix. Each row is a day, each column is a simulation.")
        print("\n")
        print(portfolio_sims)
        print("\n")
        print(50*"*")

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
