# 🧬 Genetic Algorithm Trading Strategy Optimizer

An advanced trading strategy optimizer using genetic algorithms with **robust anti-overfitting protections** based on academic research.

## 🎯 Overview

This project implements a genetic algorithm to discover profitable trading strategies while preventing overfitting through:

- ✅ **Train/Validation/Test Split** - Separate data for each phase
- ✅ **Walk-Forward Analysis** - Tests consistency across time periods
- ✅ **Complexity Penalty** - Favors simpler, more generalizable strategies
- ✅ **Multi-Objective Fitness** - Balances return, risk, and robustness
- ✅ **Theory-Based Constraints** - Uses indicators with academic evidence

## 📚 Theoretical Foundation

Based on empirical research:

### Tier 1 Evidence (Strong)
- **Momentum** - Moskowitz, Ooi & Pedersen (2012): "Time Series Momentum"
- **Trend Following** - Hurst, Ooi & Pedersen (2017): "A Century of Evidence on Trend-Following"

### Tier 2 Evidence (Moderate)
- **EMA Crossover** - Works in trending markets
- **Bollinger Bands** - Mean reversion in ranging markets
- **ATR** - For risk management and position sizing

### Anti-Overfitting
- **Sullivan, Timmermann & White (1999)** - "Data-Snooping, Technical Trading Rule Performance"
- **White's Reality Check** - Statistical correction for multiple testing

## 🚀 Features

### Genetic Algorithm
- Tournament selection
- Uniform crossover
- Adaptive mutation
- Elitism (keeps best strategies)
- Population diversity tracking

### Technical Indicators
- Momentum (Time-Series)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- EMA Crossover

### Backtesting Engine
- Realistic cost modeling (commission + slippage)
- Position management
- Stop loss / Take profit support
- Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)

### Validation Framework
- Train/Val/Test split (50/25/25)
- Walk-forward analysis
- Degradation metrics
- Overfitting detection

### Visualization
- **CLI Interface**: Full command-line tool
- **Streamlit Web App**: Interactive web interface
- Equity curves
- Drawdown analysis
- Evolution progress
- Trade distribution

## 📦 Installation

### Prerequisites
- Python 3.9+
- FMP API Key (free at [financialmodelingprep.com](https://financialmodelingprep.com))

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/evolution_analysis.git
cd evolution_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your FMP_API_KEY
```

## 🎮 Usage

### Option 1: Command Line Interface

```bash
python main.py --api-key YOUR_API_KEY --symbol AAPL
```

**Arguments:**
- `--api-key`: Your FMP API key (required)
- `--symbol`: Trading symbol (default: AAPL)
- `--config`: Path to config file (default: config/config.yaml)
- `--skip-validation`: Skip validation step
- `--skip-walkforward`: Skip walk-forward analysis

### Option 2: Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Interactive parameter tuning
- Real-time evolution progress
- Visual analysis and charts
- Strategy comparison
- Validation reports

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Genetic Algorithm
```yaml
genetic:
  population_size: 100      # Number of strategies per generation
  generations: 50           # Number of generations
  mutation_rate: 0.15       # Probability of mutation
  crossover_rate: 0.7       # Probability of crossover
  complexity_penalty: 0.05  # Penalty for complex strategies
```

### Data Split
```yaml
split:
  train_ratio: 0.50        # 50% for training
  validation_ratio: 0.25   # 25% for validation
  test_ratio: 0.25         # 25% for testing (never touched until final)
```

### Walk-Forward
```yaml
walk_forward:
  enabled: true
  train_window_months: 12  # Train on 12 months
  test_window_months: 3    # Test on 3 months
  step_months: 3           # Move forward 3 months
```

### Fitness Function
```yaml
fitness:
  sharpe_weight: 0.30          # Weight for Sharpe ratio
  total_return_weight: 0.30    # Weight for total return
  max_drawdown_weight: 0.20    # Weight for drawdown
  win_rate_weight: 0.10        # Weight for win rate
  profit_factor_weight: 0.10   # Weight for profit factor
  min_trades: 30               # Minimum trades required
```

## 📊 Output

### Files Generated
```
output/
├── {SYMBOL}_strategy_{timestamp}/
│   ├── equity_curve.png
│   ├── drawdown.png
│   ├── trades.png
│   └── walk_forward.png
└── {SYMBOL}_evolution.png
```

### Metrics Reported
- **Returns**: Total return, annualized return
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown, volatility
- **Trades**: Number of trades, win rate, profit factor, avg win/loss
- **Degradation**: Train→Val→Test performance decline

## 🛡️ Anti-Overfitting Protections

### 1. Train/Validation/Test Split
```python
# Data is split chronologically
Train:      50% (2019-2021) ← GA optimizes here
Validation: 25% (2022-2023) ← Select best strategies here
Test:       25% (2024)      ← Final evaluation (never touched before)
```

### 2. Walk-Forward Analysis
Tests strategy across multiple time periods to ensure consistency:
```python
Period 1: Train 2019-2020 → Test 2021
Period 2: Train 2020-2021 → Test 2022
Period 3: Train 2021-2022 → Test 2023
...
```

### 3. Complexity Penalty
```python
fitness = raw_fitness - (complexity_penalty * num_parameters)
```
Simpler strategies are preferred, as they generalize better.

### 4. Minimum Requirements
- Minimum 30 trades required
- Minimum Sharpe ratio threshold
- Rejects strategies with insufficient data

### 5. Realistic Costs
- Commission: 0.1% per trade (configurable)
- Slippage: 0.05% per trade (configurable)
- No look-ahead bias in signals

## 📈 Expected Results

### Realistic Expectations
```
Train Sharpe:      2.5  ← Optimistic (some overfitting expected)
Validation Sharpe: 1.8  ← Expected degradation ~28%
Test Sharpe:       1.5  ← Further degradation ~17%
Live Trading:      1.2  ← Includes real-world frictions
```

**This is NORMAL and ACCEPTABLE.**

### Red Flags (Overfitting)
- ❌ Train: 5.0, Validation: 0.5 (>80% degradation)
- ❌ Test Sharpe < 0 (losing strategy)
- ❌ >50% return/year in backtest (unrealistic)

## ⚠️ Important Warnings

### Before Live Trading
1. ✅ Review validation metrics carefully
2. ✅ Check degradation levels (<50% is good)
3. ✅ Verify walk-forward consistency
4. ✅ **PAPER TRADE for 3-6 months minimum**
5. ✅ Monitor live performance vs backtest

### Limitations
- Past performance ≠ future results
- Market regimes change (re-optimize every 6-12 months)
- Transaction costs matter more in reality
- Slippage increases with position size
- Strategies decay over time

## 🧪 Example Run

```bash
python main.py --api-key YOUR_KEY --symbol AAPL

# Output:
╔═══════════════════════════════════════════════════════════════╗
║  Genetic Algorithm Trading Strategy Optimizer                 ║
║  With Anti-Overfitting Protections                            ║
╚═══════════════════════════════════════════════════════════════╝

📈 Target Symbol: AAPL

====================================================================
STEP 1: DATA ACQUISITION
====================================================================
✅ Fetched 1500 data points

====================================================================
STEP 2: DATA PREPARATION (Anti-Overfitting Protection #1)
====================================================================
Train: 750 rows (50.0%) - 2019-01-01 to 2021-12-31
Validation: 375 rows (25.0%) - 2022-01-01 to 2023-06-30
Test: 375 rows (25.0%) - 2023-07-01 to 2024-12-31

====================================================================
STEP 3: GENETIC ALGORITHM EVOLUTION
====================================================================
🧬 Starting Genetic Algorithm Evolution
Population: 100, Generations: 50
...
```

## 📚 References

1. Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and Selling Losers"
2. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). "Time Series Momentum"
3. Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). "A Century of Evidence on Trend-Following"
4. Sullivan, R., Timmermann, A., & White, H. (1999). "Data-Snooping, Technical Trading Rule Performance"
5. Park, C. H., & Irwin, S. H. (2007). "What Do We Know About the Profitability of Technical Analysis?"

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- No guarantee of profitability
- Always paper trade first
- Use at your own risk
- Past performance does not predict future results

## 👨‍💻 Author

Built with ❤️ for the quantitative trading community

## 🙏 Acknowledgments

- Financial Modeling Prep for API access
- Academic researchers for theoretical foundations
- Open source community for tools and libraries

---

**Happy Trading! 📈 (But please backtest responsibly)**
