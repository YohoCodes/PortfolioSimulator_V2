# Monte Carlo Portfolio Simulation

A Python implementation of Monte Carlo simulation for portfolio analysis and risk assessment using historical stock data from Yahoo Finance.

## Overview

This project provides a `MonteCarlo` class that performs Monte Carlo simulations on stock portfolios to analyze potential future returns and assess investment risk. The simulation uses historical price data from Yahoo Finance to model realistic market behavior with support for both arithmetic and log-normal return simulations.

## Features

- **Historical Data Integration**: Automatically downloads stock data from Yahoo Finance
- **Portfolio Simulation**: Runs multiple Monte Carlo simulations to model potential outcomes
- **Risk Analysis**: Calculates mean returns, covariance matrices, and volatility drag adjustments
- **Flexible Weighting**: Supports both random portfolio weights and custom weight assignments
- **Multiple Simulation Types**: Supports both arithmetic and log-normal (geometric Brownian motion) simulations
- **Volatility Drag Adjustment**: Optional adjustment for volatility drag effects
- **Visualization**: Plots simulation results for easy interpretation
- **Missing Data Handling**: Automatically adjusts for missing historical data
- **Comprehensive Reporting**: Detailed output of historical measurements, initial conditions, and final results

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
pip install -r requirements.txt
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
from utility import MonteCarlo

# Define stock tickers
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Create a Monte Carlo simulation instance
mc = MonteCarlo(stocks, sample_period_in_yrs=15)

# Run simulation with default settings
mc.run_sim()
mc.plot_sim()
```

### Advanced Usage

```python
# Custom portfolio weights
custom_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Must sum to 1.0

# Run simulation with custom parameters
portfolio_sims = mc.run_sim(
    initial_investment=50000,    # Starting investment amount
    n_sims=1000,                 # Number of simulations
    n_days=252,                  # Simulation period in days (one trading year)
    weights=custom_weights,      # Custom portfolio weights
    log_return=True,             # Use log-normal simulation
    vol_drag=True                # Apply volatility drag adjustment
)

# Get simulation information
mc.get_info()

# Access raw stock data
stock_data = mc.get_data()
```

## Class Methods

### `__init__(tickers, sample_period_in_yrs=15)`
Initializes the Monte Carlo simulation with stock data.

**Parameters:**
- `tickers` (list): List of stock ticker symbols
- `sample_period_in_yrs` (int): Number of years of historical data to use (default: 15)

### `run_sim(initial_investment, n_sims, n_days, by, weights, log_return, vol_drag)`
Runs the Monte Carlo simulation.

**Parameters:**
- `initial_investment` (float): Starting portfolio value (default: 10000)
- `n_sims` (int): Number of simulation runs (default: 200)
- `n_days` (int): Simulation period in days (default: 100)
- `by` (str): Price type to use ('Open', 'High', 'Low', 'Close', 'Adj Close') (default: 'Close')
- `weights` (list): Portfolio weights (default: None for random weights)
- `log_return` (bool): If True, uses log-normal simulation (default: False)
- `vol_drag` (bool): If True, adjusts mean returns for volatility drag (default: True)

**Returns:**
- `numpy.ndarray`: Matrix of simulation results (days × simulations)

### `plot_sim()`
Plots all simulation paths with the mean path highlighted in red.

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

### `daily_avg()`
Returns the average portfolio value across all simulations for each day.

## Mathematical Background

The simulation uses the following approach:

1. **Data Collection**: Downloads historical price data from Yahoo Finance for the specified period
2. **Return Calculation**: Computes daily returns and their statistics
3. **Cholesky Decomposition**: Uses the covariance matrix to generate correlated random returns
4. **Volatility Drag Adjustment**: Optionally adjusts mean returns by subtracting 0.5 × variance
5. **Portfolio Simulation**: Applies weights and compounds returns over time
6. **Monte Carlo Sampling**: Repeats the process multiple times to generate a distribution of outcomes

### Simulation Types

**Arithmetic Return Simulation (default):**
- Uses standard arithmetic returns
- Daily returns = mean + correlated random component

**Log-Normal Simulation:**
- Uses geometric Brownian motion
- Generates log-returns, then converts to arithmetic returns
- More realistic for long-term simulations

### Volatility Drag

When `vol_drag=True`, the simulation adjusts mean returns by subtracting 0.5 × variance for each asset. This accounts for the fact that volatility reduces compound returns over time.

## Output Interpretation

The simulation provides comprehensive output including:

### Historical Measurements
- Sample period and date range
- Mean returns for each stock (original and adjusted if volatility drag is applied)
- Data truncation warnings if necessary

### Initial Conditions
- Initial investment amount
- Simulation period in years
- Portfolio weights for each stock

### Final Conditions
- Mean and median final investment values
- Standard deviation of final values
- Annualized return rates (mean and median)
- Total return percentages (mean and median)

### Visualization
The plot shows:
- Multiple possible portfolio value paths (light lines)
- Average portfolio path (red line)
- Portfolio value on y-axis, days on x-axis

## Limitations

- **Historical Assumption**: Assumes past performance predicts future behavior
- **Normal Distribution**: Assumes returns follow a normal distribution
- **No Market Regime Changes**: Doesn't account for structural market changes
- **No External Factors**: Ignores economic, political, or company-specific events
- **No Transaction Costs**: Assumes frictionless trading
- **No Rebalancing**: Portfolio weights remain constant throughout simulation

## Known Issues

- Mathematical logic for daily returns may need verification (see ToDo.txt)
- Method to optimize mean return by adjusting portfolio weighting is planned

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project.

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions. 