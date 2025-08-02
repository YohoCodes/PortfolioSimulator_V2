# Monte Carlo Portfolio Simulation

A Python implementation of Monte Carlo simulation for portfolio analysis and risk assessment using historical stock data.

## Overview

This project provides a `MonteCarlo` class that performs Monte Carlo simulations on stock portfolios to analyze potential future returns and assess investment risk. The simulation uses historical price data from Yahoo Finance to model realistic market behavior.

## Features

- **Historical Data Integration**: Automatically downloads stock data from Yahoo Finance
- **Portfolio Simulation**: Runs multiple Monte Carlo simulations to model potential outcomes
- **Risk Analysis**: Calculates mean returns and covariance matrices for risk assessment
- **Flexible Weighting**: Supports both random portfolio weights and custom weight assignments
- **Visualization**: Plots simulation results for easy interpretation
- **Missing Data Handling**: Automatically adjusts for missing historical data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MonteCarlo2
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install pandas numpy matplotlib yfinance
```

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Data visualization
- `yfinance`: Yahoo Finance data download
- `datetime`: Date handling

## Usage

### Basic Example

```python
import datetime as dt
from monte_carlo import MonteCarlo

# Define stock tickers and date range
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = dt.date(2020, 1, 1)
end_date = dt.date.today()

# Initialize simulation
sim = MonteCarlo(stocks, start_date, end_date)

# Run simulation
portfolio_sims = sim.run_sim(
    initial_investment=10000,  # Starting investment amount
    n_sims=100,                # Number of simulations
    n_days=365,                # Simulation period in days
    by='Close'                 # Use closing prices
)

# Visualize results
sim.plot_sim()
```

### Advanced Usage

```python
# Custom portfolio weights
custom_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Must sum to 1.0

# Run simulation with custom weights
portfolio_sims = sim.run_sim(
    initial_investment=50000,
    n_sims=1000,
    n_days=252,  # One trading year
    weights=custom_weights
)

# Get simulation information
sim.get_info()

# Access raw stock data
stock_data = sim.get_data()
```

## Class Methods

### `__init__(tickers, start_date, end_date)`
Initializes the Monte Carlo simulation with stock data.

**Parameters:**
- `tickers` (list): List of stock ticker symbols
- `start_date` (datetime): Start date for historical data (default: 1 year ago)
- `end_date` (datetime): End date for historical data (default: today)

### `run_sim(initial_investment, n_sims, n_days, by, weights)`
Runs the Monte Carlo simulation.

**Parameters:**
- `initial_investment` (float): Starting portfolio value (default: 10000)
- `n_sims` (int): Number of simulation runs (default: 100)
- `n_days` (int): Simulation period in days (default: 365)
- `by` (str): Price type to use ('Open', 'High', 'Low', 'Close', 'Adj Close') (default: 'Close')
- `weights` (list): Portfolio weights (default: None for random weights)

**Returns:**
- `numpy.ndarray`: Matrix of simulation results (days × simulations)

### `plot_sim()`
Plots all simulation paths.

### `get_data()`
Returns the raw stock data.

### `get_info()`
Prints simulation configuration information.

### `calc_mean(by)`
Calculates mean returns for each stock.

### `calc_covariance_matrix(by)`
Calculates the covariance matrix of returns.

### `gen_weights()`
Generates random portfolio weights that sum to 1.0.

## Mathematical Background

The simulation uses the following approach:

1. **Data Collection**: Downloads historical price data from Yahoo Finance
2. **Return Calculation**: Computes daily returns and their statistics
3. **Cholesky Decomposition**: Uses the covariance matrix to generate correlated random returns
4. **Portfolio Simulation**: Applies weights and compounds returns over time
5. **Monte Carlo Sampling**: Repeats the process multiple times to generate a distribution of outcomes

The simulation assumes:
- Returns follow a multivariate normal distribution
- Historical correlations and volatilities persist into the future
- No transaction costs or taxes

## Output Interpretation

The simulation generates a matrix where:
- **Rows**: Represent time periods (days)
- **Columns**: Represent different simulation runs
- **Values**: Portfolio values at each time point

The resulting plot shows multiple possible portfolio value paths, helping to visualize:
- Potential upside and downside scenarios
- Portfolio volatility
- Risk of loss over different time horizons

## Limitations

- **Historical Assumption**: Assumes past performance predicts future behavior
- **Normal Distribution**: Assumes returns follow a normal distribution
- **No Market Regime Changes**: Doesn't account for structural market changes
- **No External Factors**: Ignores economic, political, or company-specific events

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project.

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions. 