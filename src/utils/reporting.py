"""
Reporting and Visualization

Generate charts and reports for strategy analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path


class ReportGenerator:
    """Generate reports and visualizations for trading strategies."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve.

        Args:
            equity_curve: DataFrame with date and equity columns
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(equity_curve['date'], equity_curve['equity'], linewidth=2, label='Equity')
        ax.fill_between(equity_curve['date'], equity_curve['equity'],
                        alpha=0.3)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown",
        save_path: Optional[str] = None
    ):
        """
        Plot drawdown over time.

        Args:
            equity_curve: DataFrame with date and equity columns
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        equity = equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        ax.fill_between(equity_curve['date'], drawdown, 0,
                        alpha=0.3, color='red', label='Drawdown')
        ax.plot(equity_curve['date'], drawdown, color='darkred', linewidth=2)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_evolution_progress(
        self,
        history: pd.DataFrame,
        title: str = "Evolution Progress",
        save_path: Optional[str] = None
    ):
        """
        Plot genetic algorithm evolution progress.

        Args:
            history: DataFrame with generation statistics
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Fitness over generations
        ax1.plot(history['generation'], history['best_fitness'],
                label='Best Fitness', linewidth=2, marker='o')
        ax1.plot(history['generation'], history['avg_fitness'],
                label='Avg Fitness', linewidth=2, marker='s', alpha=0.7)
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title(f'{title} - Fitness', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Diversity over generations
        ax2.plot(history['generation'], history['diversity'],
                linewidth=2, marker='o', color='green')
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Diversity', fontsize=12)
        ax2.set_title(f'{title} - Population Diversity', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_trade_distribution(
        self,
        trades: pd.DataFrame,
        title: str = "Trade Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of trade returns.

        Args:
            trades: DataFrame with trade records
            title: Plot title
            save_path: Optional path to save plot
        """
        if trades.empty:
            print("No trades to plot")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram of returns
        ax1.hist(trades['return_pct'], bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.set_xlabel('Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Win/Loss pie chart
        winners = len(trades[trades['net_pnl'] > 0])
        losers = len(trades[trades['net_pnl'] < 0])

        colors = ['#2ecc71', '#e74c3c']
        ax2.pie([winners, losers], labels=['Winners', 'Losers'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_validation_comparison(
        self,
        comparison_df: pd.DataFrame,
        title: str = "Train vs Validation Performance",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of train vs validation performance.

        Args:
            comparison_df: DataFrame with train and validation metrics
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ['fitness', 'sharpe', 'return_pct', 'max_dd_pct']
        titles = ['Fitness', 'Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)']

        for ax, metric, subtitle in zip(axes.flat, metrics, titles):
            if metric in comparison_df.columns:
                top_10 = comparison_df.head(10)
                x = range(len(top_10))

                ax.bar(x, top_10[metric], alpha=0.7)
                ax.set_xlabel('Strategy Rank', fontsize=10)
                ax.set_ylabel(subtitle, fontsize=10)
                ax.set_title(subtitle, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_walk_forward_results(
        self,
        wf_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Plot walk-forward analysis results.

        Args:
            wf_results: Walk-forward results dictionary
            save_path: Optional path to save plot
        """
        periods = wf_results['periods']
        df = pd.DataFrame(periods)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Train vs Test fitness
        x = range(len(df))
        width = 0.35

        ax1.bar([i - width/2 for i in x], df['train_fitness'],
               width, label='Train Fitness', alpha=0.7)
        ax1.bar([i + width/2 for i in x], df['test_fitness'],
               width, label='Test Fitness', alpha=0.7)

        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('Walk-Forward Analysis: Train vs Test Fitness',
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Degradation
        ax2.plot(x, df['degradation'], marker='o', linewidth=2, color='red')
        ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')

        ax2.set_xlabel('Period', fontsize=12)
        ax2.set_ylabel('Degradation (%)', fontsize=12)
        ax2.set_title('Fitness Degradation by Period', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add consistency score
        consistency = wf_results['consistency']
        plt.figtext(0.99, 0.01,
                   f"Consistency Score: {consistency['consistency_score']:.3f}",
                   ha='right', fontsize=10, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_full_report(
        self,
        strategy_name: str,
        backtest_result: Dict[str, Any],
        validation_result: Optional[Dict[str, Any]] = None,
        wf_result: Optional[Dict[str, Any]] = None
    ):
        """
        Generate complete report with all plots.

        Args:
            strategy_name: Name for the strategy
            backtest_result: Backtest results
            validation_result: Optional validation results
            wf_result: Optional walk-forward results
        """
        print(f"\n📊 Generating full report for: {strategy_name}")

        # Create subdirectory for this strategy
        strategy_dir = self.output_dir / strategy_name.replace(' ', '_')
        strategy_dir.mkdir(exist_ok=True)

        # Equity curve
        if not backtest_result['equity_curve'].empty:
            self.plot_equity_curve(
                backtest_result['equity_curve'],
                title=f"{strategy_name} - Equity Curve",
                save_path=f"{strategy_name.replace(' ', '_')}/equity_curve.png"
            )
            plt.close()

            # Drawdown
            self.plot_drawdown(
                backtest_result['equity_curve'],
                title=f"{strategy_name} - Drawdown",
                save_path=f"{strategy_name.replace(' ', '_')}/drawdown.png"
            )
            plt.close()

        # Trade distribution
        if not backtest_result['trades'].empty:
            self.plot_trade_distribution(
                backtest_result['trades'],
                title=f"{strategy_name} - Trade Analysis",
                save_path=f"{strategy_name.replace(' ', '_')}/trades.png"
            )
            plt.close()

        # Walk-forward results
        if wf_result:
            self.plot_walk_forward_results(
                wf_result,
                save_path=f"{strategy_name.replace(' ', '_')}/walk_forward.png"
            )
            plt.close()

        print(f"✅ Report saved to: {strategy_dir}")
