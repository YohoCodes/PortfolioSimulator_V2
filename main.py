from utility import MonteCarlo
import pandas as pd

# Example usage
if __name__ == "__main__":
    # Define some stock tickers
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Create a Monte Carlo simulation instance
    mc = MonteCarlo(stocks)
    
    # Run simulation with default settings
    mc.run_sim()
