"""
Main Script for Genetic Algorithm Trading Strategy Optimizer

Run this script to execute the full optimization pipeline.
"""

import yaml
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.data.fmp_client import FMPClient
from src.data.data_loader import DataLoader
from src.genetic.evolution import GeneticAlgorithm
from src.validation.validator import StrategyValidator
from src.utils.reporting import ReportGenerator
from src.backtesting.metrics import PerformanceMetrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm Trading Strategy Optimizer'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='FMP API key'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Trading symbol (overrides config)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step'
    )
    parser.add_argument(
        '--skip-walkforward',
        action='store_true',
        help='Skip walk-forward analysis'
    )

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  Genetic Algorithm Trading Strategy Optimizer                 ║
    ║  With Anti-Overfitting Protections                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Load configuration
    config = load_config(args.config)

    # Override symbol if provided
    if args.symbol:
        config['data']['symbol'] = args.symbol

    symbol = config['data']['symbol']
    print(f"\n📈 Target Symbol: {symbol}")

    # Step 1: Fetch Data
    print(f"\n{'='*70}")
    print("STEP 1: DATA ACQUISITION")
    print(f"{'='*70}")

    client = FMPClient(
        api_key=args.api_key,
        base_url=config['api']['base_url'],
        rate_limit_calls=config['api']['rate_limit_calls'],
        rate_limit_period=config['api']['rate_limit_period']
    )

    print(f"Fetching historical data for {symbol}...")
    data = client.get_historical_prices(
        symbol=symbol,
        from_date=config['data']['start_date'],
        to_date=config['data']['end_date']
    )

    print(f"✅ Fetched {len(data)} data points")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")

    # Step 2: Data Preparation and Splitting
    print(f"\n{'='*70}")
    print("STEP 2: DATA PREPARATION (Anti-Overfitting Protection #1)")
    print(f"{'='*70}")

    data_loader = DataLoader()
    data = data_loader.clean_data(data)

    train_data, val_data, test_data = data_loader.train_val_test_split(
        data,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['validation_ratio'],
        test_ratio=config['split']['test_ratio']
    )

    # Step 3: Genetic Algorithm Evolution
    print(f"\n{'='*70}")
    print("STEP 3: GENETIC ALGORITHM EVOLUTION")
    print(f"{'='*70}")

    ga = GeneticAlgorithm(config)
    best_strategy = ga.evolve(train_data, verbose=True)

    print(f"\n🏆 Best Strategy Found:")
    print(f"   Entry:  {best_strategy.genes['entry_indicator_type']}")
    print(f"   Exit:   {best_strategy.genes['exit_indicator_type']}")
    print(f"   Fitness: {best_strategy.fitness:.4f}")
    print(f"   Complexity: {best_strategy.get_complexity()}")

    # Step 4: Validation
    if not args.skip_validation:
        print(f"\n{'='*70}")
        print("STEP 4: VALIDATION (Anti-Overfitting Check)")
        print(f"{'='*70}")

        validator = StrategyValidator(config)

        # Get top 10 strategies
        top_strategies = ga.get_best_strategies(n=10)

        # Evaluate on validation data
        val_results = ga.evaluate_on_validation(top_strategies, val_data, verbose=True)

        # Select best strategy based on validation
        best_on_validation = val_results[0]['strategy']

        # Full validation report
        full_validation = validator.validate_full(
            best_on_validation,
            train_data,
            val_data,
            test_data
        )

        validator.print_validation_report(full_validation)

        # Use validated strategy
        final_strategy = best_on_validation
    else:
        final_strategy = best_strategy
        full_validation = None

    # Step 5: Walk-Forward Analysis (Optional)
    wf_result = None
    if not args.skip_walkforward and config['walk_forward']['enabled']:
        print(f"\n{'='*70}")
        print("STEP 5: WALK-FORWARD ANALYSIS (Anti-Overfitting Protection #2)")
        print(f"{'='*70}")

        wf_result = validator.walk_forward_validation(
            final_strategy,
            data,
            train_months=config['walk_forward']['train_window_months'],
            test_months=config['walk_forward']['test_window_months'],
            step_months=config['walk_forward']['step_months']
        )

        print(f"\n📊 Walk-Forward Results:")
        print(f"   Periods tested: {wf_result['num_periods']}")
        print(f"   Mean test fitness: {wf_result['consistency']['mean_test_fitness']:.4f}")
        print(f"   Consistency score: {wf_result['consistency']['consistency_score']:.4f}")
        print(f"   Is consistent: {'✅ Yes' if wf_result['is_consistent'] else '❌ No'}")

    # Step 6: Final Backtest and Reporting
    print(f"\n{'='*70}")
    print("STEP 6: FINAL BACKTEST AND REPORTING")
    print(f"{'='*70}")

    from src.backtesting.engine import BacktestEngine

    backtest_engine = BacktestEngine(
        initial_capital=config['backtesting']['initial_capital'],
        commission=config['backtesting']['commission'],
        slippage=config['backtesting']['slippage']
    )

    # Run on full data for visualization
    final_result = backtest_engine.run(
        data=data.copy(),
        entry_indicator=final_strategy.get_entry_indicator(),
        exit_indicator=final_strategy.get_exit_indicator(),
        position_size=final_strategy.genes['position_size'],
        use_stop_loss=final_strategy.genes['use_stop_loss'],
        stop_loss_pct=final_strategy.genes['stop_loss_pct']
    )

    # Print final metrics
    PerformanceMetrics.print_metrics(
        final_result['metrics'],
        title=f"Final Strategy Performance - {symbol}"
    )

    # Generate reports
    if config['reporting']['save_plots']:
        print(f"\n📊 Generating visual reports...")
        reporter = ReportGenerator(output_dir='output')

        reporter.generate_full_report(
            strategy_name=f"{symbol}_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            backtest_result=final_result,
            validation_result=full_validation,
            wf_result=wf_result
        )

        # Plot evolution progress
        history = ga.get_evolution_history()
        reporter.plot_evolution_progress(
            history,
            save_path=f"{symbol}_evolution.png"
        )

        print(f"✅ Reports saved to 'output/' directory")

    # Step 7: Final Recommendations
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*70}")

    print(f"\n✅ Optimization Complete!")
    print(f"\n📌 Next Steps:")
    print(f"   1. Review validation metrics (especially degradation)")
    print(f"   2. Check walk-forward consistency")
    print(f"   3. Analyze generated plots in 'output/' directory")
    print(f"   4. Consider paper trading for 3-6 months")
    print(f"   5. Monitor performance vs backtest expectations")

    print(f"\n⚠️  Important Reminders:")
    print(f"   • Past performance ≠ future results")
    print(f"   • High train/val degradation = potential overfitting")
    print(f"   • Always paper trade before using real capital")
    print(f"   • Market regimes change - re-optimize periodically")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
