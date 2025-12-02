"""
Streamlit Web Interface for Genetic Algorithm Trading Strategy Optimizer

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fmp_client import FMPClient
from src.data.data_loader import DataLoader
from src.genetic.evolution import GeneticAlgorithm
from src.validation.validator import StrategyValidator
from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import PerformanceMetrics


# Page config
st.set_page_config(
    page_title="Trading Strategy Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧬 Genetic Algorithm Trading Strategy Optimizer</h1>', unsafe_allow_html=True)
st.markdown("### Find optimal trading strategies with anti-overfitting protections")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'evolution_complete' not in st.session_state:
    st.session_state.evolution_complete = False


def load_config():
    """Load configuration."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# API Key
api_key = st.sidebar.text_input("FMP API Key", type="password", help="Get your free API key from financialmodelingprep.com")

# Symbol selection
symbol = st.sidebar.text_input("Trading Symbol", value="AAPL", help="Stock symbol to analyze")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# Genetic Algorithm Parameters
st.sidebar.subheader("🧬 GA Parameters")
population_size = st.sidebar.slider("Population Size", 20, 200, 100)
generations = st.sidebar.slider("Generations", 10, 100, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.05, 0.30, 0.15, 0.05)

# Backtesting Parameters
st.sidebar.subheader("💰 Backtesting")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
commission = st.sidebar.number_input("Commission (%)", value=0.1, step=0.05) / 100
slippage = st.sidebar.number_input("Slippage (%)", value=0.05, step=0.01) / 100


# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Evolution", "🏆 Best Strategies", "✅ Validation", "📈 Analysis"])


# Tab 1: Data Loading and Evolution
with tab1:
    st.header("1. Data Acquisition & Evolution")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Load Data & Start Evolution", type="primary", disabled=not api_key):
            if not api_key:
                st.error("Please enter your FMP API key in the sidebar")
            else:
                with st.spinner("Loading data..."):
                    try:
                        # Load data
                        client = FMPClient(api_key=api_key)
                        data = client.get_historical_prices(
                            symbol=symbol,
                            from_date=start_date.strftime("%Y-%m-%d"),
                            to_date=end_date.strftime("%Y-%m-%d")
                        )

                        st.session_state.data = data
                        st.session_state.symbol = symbol
                        st.session_state.data_loaded = True

                        st.success(f"✅ Loaded {len(data)} data points for {symbol}")

                        # Prepare data
                        data_loader = DataLoader()
                        data = data_loader.clean_data(data)

                        # Split data
                        train_data, val_data, test_data = data_loader.train_val_test_split(data)
                        st.session_state.train_data = train_data
                        st.session_state.val_data = val_data
                        st.session_state.test_data = test_data

                        # Show split info
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Train Set", f"{len(train_data)} days")
                        with col_b:
                            st.metric("Validation Set", f"{len(val_data)} days")
                        with col_c:
                            st.metric("Test Set", f"{len(test_data)} days")

                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        st.session_state.data_loaded = False

        if st.session_state.data_loaded:
            st.info("✅ Data loaded. Now run evolution below.")

            if st.button("🧬 Run Genetic Algorithm", type="primary"):
                with st.spinner("Evolving strategies... This may take a few minutes"):
                    try:
                        # Create config
                        config = {
                            'genetic': {
                                'population_size': population_size,
                                'generations': generations,
                                'mutation_rate': mutation_rate,
                                'crossover_rate': 0.7,
                                'elitism': 0.1,
                                'tournament_size': 5,
                                'complexity_penalty': 0.05
                            },
                            'backtesting': {
                                'initial_capital': initial_capital,
                                'commission': commission,
                                'slippage': slippage
                            },
                            'fitness': {
                                'sharpe_weight': 0.30,
                                'total_return_weight': 0.30,
                                'max_drawdown_weight': 0.20,
                                'win_rate_weight': 0.10,
                                'profit_factor_weight': 0.10,
                                'min_trades': 30
                            },
                            'strategy': {
                                'indicators': ['momentum', 'rsi', 'macd', 'bollinger', 'ema_cross'],
                                'parameters': {
                                    'momentum': {'lookback_min': 20, 'lookback_max': 200},
                                    'rsi': {'period_min': 10, 'period_max': 20, 'oversold_min': 25, 'oversold_max': 35, 'overbought_min': 65, 'overbought_max': 75},
                                    'macd': {'fast_min': 8, 'fast_max': 16, 'slow_min': 20, 'slow_max': 30, 'signal_min': 7, 'signal_max': 11},
                                    'bollinger': {'period_min': 15, 'period_max': 25, 'std_min': 1.5, 'std_max': 2.5},
                                    'ema_cross': {'fast_min': 20, 'fast_max': 50, 'slow_min': 100, 'slow_max': 200}
                                }
                            }
                        }

                        # Run GA
                        ga = GeneticAlgorithm(config)

                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        best_strategy = ga.evolve(st.session_state.train_data, verbose=False)

                        progress_bar.progress(100)
                        status_text.text("Evolution complete!")

                        st.session_state.ga = ga
                        st.session_state.best_strategy = best_strategy
                        st.session_state.config = config
                        st.session_state.evolution_complete = True

                        st.success(f"🏆 Evolution complete! Best fitness: {best_strategy.fitness:.4f}")

                        # Plot evolution history
                        history = ga.get_evolution_history()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=history['generation'], y=history['best_fitness'],
                                                mode='lines+markers', name='Best Fitness'))
                        fig.add_trace(go.Scatter(x=history['generation'], y=history['avg_fitness'],
                                                mode='lines+markers', name='Avg Fitness'))
                        fig.update_layout(title="Evolution Progress", xaxis_title="Generation", yaxis_title="Fitness")
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error during evolution: {str(e)}")
                        st.exception(e)

    with col2:
        st.info("""
        ### 📚 How it works:

        1. **Load Data**: Fetches historical price data from FMP
        2. **Split Data**: Divides into train/val/test (50/25/25)
        3. **Evolution**: Genetic algorithm finds optimal strategies
        4. **Validation**: Tests on unseen data to prevent overfitting

        ### 🛡️ Protections:
        - ✅ Train/val/test split
        - ✅ Complexity penalty
        - ✅ Multi-objective fitness
        - ✅ Minimum trade requirements
        """)


# Tab 2: Best Strategies
with tab2:
    st.header("🏆 Best Strategies")

    if st.session_state.evolution_complete:
        ga = st.session_state.ga
        top_strategies = ga.get_best_strategies(n=10)

        st.subheader("Top 10 Strategies from Evolution")

        # Create comparison table
        comparison_data = []
        for i, strategy in enumerate(top_strategies):
            comparison_data.append({
                'Rank': i + 1,
                'Fitness': f"{strategy.fitness:.4f}",
                'Entry Indicator': strategy.genes['entry_indicator_type'],
                'Exit Indicator': strategy.genes['exit_indicator_type'],
                'Complexity': strategy.get_complexity(),
                'Position Size': f"{strategy.genes['position_size']:.2f}",
                'Use Stop Loss': '✅' if strategy.genes['use_stop_loss'] else '❌'
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)

        # Show best strategy details
        st.subheader("🥇 Best Strategy Details")
        best = st.session_state.best_strategy

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fitness", f"{best.fitness:.4f}")
        with col2:
            st.metric("Complexity", best.get_complexity())
        with col3:
            st.metric("Position Size", f"{best.genes['position_size']:.1%}")

        st.json({
            'Entry Indicator': {
                'Type': best.genes['entry_indicator_type'],
                'Parameters': best.genes['entry_params']
            },
            'Exit Indicator': {
                'Type': best.genes['exit_indicator_type'],
                'Parameters': best.genes['exit_params']
            },
            'Risk Management': {
                'Use Stop Loss': best.genes['use_stop_loss'],
                'Stop Loss %': best.genes['stop_loss_pct']
            }
        })

    else:
        st.warning("⚠️ Please run the evolution first in the 'Data & Evolution' tab")


# Tab 3: Validation
with tab3:
    st.header("✅ Validation & Overfitting Detection")

    if st.session_state.evolution_complete:
        if st.button("🔍 Run Full Validation"):
            with st.spinner("Validating strategies..."):
                try:
                    validator = StrategyValidator(st.session_state.config)

                    # Validate best strategy
                    validation_result = validator.validate_full(
                        st.session_state.best_strategy,
                        st.session_state.train_data,
                        st.session_state.val_data,
                        st.session_state.test_data
                    )

                    st.session_state.validation_result = validation_result

                    # Display results
                    st.subheader("📊 Performance Across Data Splits")

                    splits_data = []
                    for split_name in ['train', 'validation', 'test']:
                        result = validation_result[split_name]
                        metrics = result['metrics']
                        splits_data.append({
                            'Split': split_name.capitalize(),
                            'Fitness': f"{result['fitness']:.4f}",
                            'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                            'Total Return': f"{metrics.get('total_return', 0):.2f}%",
                            'Max Drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                            'Win Rate': f"{metrics.get('win_rate', 0):.2f}%",
                            'Num Trades': metrics.get('num_trades', 0)
                        })

                    st.dataframe(pd.DataFrame(splits_data), use_container_width=True)

                    # Degradation analysis
                    st.subheader("⚠️ Fitness Degradation")
                    deg = validation_result['degradation']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Train → Validation", f"{deg['train_to_val']:.1f}%",
                                 delta=f"{-deg['train_to_val']:.1f}%", delta_color="inverse")
                    with col2:
                        st.metric("Validation → Test", f"{deg['val_to_test']:.1f}%",
                                 delta=f"{-deg['val_to_test']:.1f}%", delta_color="inverse")
                    with col3:
                        st.metric("Train → Test", f"{deg['train_to_test']:.1f}%",
                                 delta=f"{-deg['train_to_test']:.1f}%", delta_color="inverse")

                    # Overfitting assessment
                    if validation_result['is_overfit']:
                        st.error("❌ WARNING: Potential overfitting detected!")
                        st.warning("""
                        High degradation or poor test performance suggests the strategy may not generalize well.
                        Consider:
                        - Simplifying the strategy
                        - Increasing training data
                        - Adjusting complexity penalty
                        """)
                    else:
                        st.success("✅ Strategy appears robust with reasonable degradation levels")

                except Exception as e:
                    st.error(f"Validation error: {str(e)}")
                    st.exception(e)
    else:
        st.warning("⚠️ Please run the evolution first")


# Tab 4: Analysis & Charts
with tab4:
    st.header("📈 Detailed Analysis")

    if st.session_state.evolution_complete:
        if st.button("📊 Generate Full Analysis"):
            with st.spinner("Running backtest and generating charts..."):
                try:
                    backtest_engine = BacktestEngine(
                        initial_capital=st.session_state.config['backtesting']['initial_capital'],
                        commission=st.session_state.config['backtesting']['commission'],
                        slippage=st.session_state.config['backtesting']['slippage']
                    )

                    best = st.session_state.best_strategy

                    # Run backtest on full data
                    result = backtest_engine.run(
                        data=st.session_state.data.copy(),
                        entry_indicator=best.get_entry_indicator(),
                        exit_indicator=best.get_exit_indicator(),
                        position_size=best.genes['position_size'],
                        use_stop_loss=best.genes['use_stop_loss'],
                        stop_loss_pct=best.genes['stop_loss_pct']
                    )

                    metrics = result['metrics']

                    # Display key metrics
                    st.subheader("📊 Performance Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                    with col2:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
                    with col4:
                        st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", metrics['num_trades'])
                    with col2:
                        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    with col3:
                        st.metric("Avg Win", f"${metrics.get('avg_win', 0):,.0f}")
                    with col4:
                        st.metric("Avg Loss", f"${metrics.get('avg_loss', 0):,.0f}")

                    # Equity curve
                    st.subheader("💰 Equity Curve")
                    equity_curve = result['equity_curve']

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=equity_curve['date'], y=equity_curve['equity'],
                                            mode='lines', name='Equity', fill='tozeroy'))
                    fig.update_layout(title=f"{st.session_state.symbol} - Equity Curve",
                                     xaxis_title="Date", yaxis_title="Equity ($)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Drawdown
                    st.subheader("📉 Drawdown")
                    equity = equity_curve['equity']
                    running_max = equity.expanding().max()
                    drawdown = (equity - running_max) / running_max * 100

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=equity_curve['date'], y=drawdown,
                                            mode='lines', name='Drawdown', fill='tozeroy',
                                            line=dict(color='red')))
                    fig.update_layout(title="Drawdown Over Time",
                                     xaxis_title="Date", yaxis_title="Drawdown (%)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Trade analysis
                    if not result['trades'].empty:
                        st.subheader("📈 Trade Analysis")
                        trades = result['trades']

                        col1, col2 = st.columns(2)

                        with col1:
                            # Returns distribution
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=trades['return_pct'], nbinsx=30,
                                                      name='Returns'))
                            fig.update_layout(title="Distribution of Trade Returns",
                                            xaxis_title="Return (%)", yaxis_title="Frequency")
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Win/Loss pie
                            winners = len(trades[trades['net_pnl'] > 0])
                            losers = len(trades[trades['net_pnl'] < 0])

                            fig = go.Figure(data=[go.Pie(labels=['Winners', 'Losers'],
                                                        values=[winners, losers],
                                                        marker=dict(colors=['#2ecc71', '#e74c3c']))])
                            fig.update_layout(title="Win/Loss Ratio")
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.exception(e)
    else:
        st.warning("⚠️ Please run the evolution first")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Genetic Algorithm Trading Strategy Optimizer</p>
    <p>⚠️ <strong>Disclaimer:</strong> Past performance does not guarantee future results. Always paper trade before using real capital.</p>
</div>
""", unsafe_allow_html=True)
