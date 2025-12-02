"""
Position and Portfolio Management
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import pandas as pd


@dataclass
class Position:
    """Represents a trading position."""

    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    direction: int  # 1 for long, -1 for short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def entry_value(self) -> float:
        """Total value at entry."""
        return abs(self.shares * self.entry_price)

    def current_value(self, current_price: float) -> float:
        """Current position value."""
        return self.shares * current_price

    def pnl(self, current_price: float) -> float:
        """Profit/loss at current price."""
        return (current_price - self.entry_price) * self.shares

    def pnl_percent(self, current_price: float) -> float:
        """Profit/loss percentage."""
        return ((current_price / self.entry_price) - 1) * 100 * self.direction

    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if self.stop_loss is None:
            return False

        if self.direction == 1:  # Long position
            return current_price <= self.stop_loss
        else:  # Short position
            return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit is hit."""
        if self.take_profit is None:
            return False

        if self.direction == 1:  # Long position
            return current_price >= self.take_profit
        else:  # Short position
            return current_price <= self.take_profit


class PositionManager:
    """Manages trading positions and portfolio."""

    def __init__(self, initial_capital: float, commission: float = 0.001, slippage: float = 0.0005):
        """
        Initialize position manager.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions: list[Position] = []
        self.closed_positions: list[dict] = []
        self.equity_curve = []

    @property
    def current_position(self) -> Optional[Position]:
        """Get current open position (assumes single position at a time)."""
        return self.positions[0] if self.positions else None

    @property
    def is_flat(self) -> bool:
        """Check if no positions are open."""
        return len(self.positions) == 0

    def calculate_transaction_cost(self, price: float, shares: float) -> float:
        """
        Calculate total transaction costs.

        Args:
            price: Trade price
            shares: Number of shares

        Returns:
            Total cost including commission and slippage
        """
        value = abs(price * shares)
        commission_cost = value * self.commission
        slippage_cost = value * self.slippage
        return commission_cost + slippage_cost

    def open_position(
        self,
        symbol: str,
        date: datetime,
        price: float,
        direction: int,
        position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            date: Entry date
            price: Entry price
            direction: 1 for long, -1 for short
            position_size: Fraction of capital to use (0-1)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            True if position opened successfully
        """
        # Can't open if already in a position
        if not self.is_flat:
            return False

        # Calculate shares to buy
        available_capital = self.cash * position_size
        shares = (available_capital / price) * direction

        # Calculate costs
        cost = self.calculate_transaction_cost(price, shares)

        # Check if we have enough cash
        required_capital = abs(shares * price) + cost
        if required_capital > self.cash:
            return False

        # Create position
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
            shares=shares,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        # Update cash (only pay for long positions, short positions give us cash)
        if direction == 1:  # Long: we pay
            self.cash -= (abs(shares * price) + cost)
        else:  # Short: we receive (minus costs)
            self.cash += (abs(shares * price) - cost)

        # Add to positions
        self.positions.append(position)

        return True

    def close_position(self, date: datetime, price: float, reason: str = "signal") -> Optional[dict]:
        """
        Close current position.

        Args:
            date: Exit date
            price: Exit price
            reason: Reason for closing (signal, stop_loss, take_profit)

        Returns:
            Trade record dictionary or None
        """
        if self.is_flat:
            return None

        position = self.current_position

        # Calculate P&L
        gross_pnl = position.pnl(price)
        cost = self.calculate_transaction_cost(price, position.shares)
        net_pnl = gross_pnl - cost

        # Update cash based on position type
        if position.direction == 1:  # Closing long: receive proceeds
            self.cash += (abs(position.shares) * price) - cost
        else:  # Closing short: pay to buy back
            self.cash -= (abs(position.shares) * price) + cost
            # Also add the P&L from the short
            self.cash += gross_pnl

        # Create trade record
        trade = {
            'symbol': position.symbol,
            'entry_date': position.entry_date,
            'exit_date': date,
            'entry_price': position.entry_price,
            'exit_price': price,
            'shares': position.shares,
            'direction': position.direction,
            'gross_pnl': gross_pnl,
            'cost': cost,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / position.entry_value) * 100,
            'reason': reason
        }

        # Record closed position
        self.closed_positions.append(trade)

        # Remove from positions
        self.positions.remove(position)

        return trade

    def update_equity(self, date: datetime, current_price: float):
        """
        Update equity curve.

        Args:
            date: Current date
            current_price: Current market price
        """
        # Calculate total equity
        position_value = 0
        if not self.is_flat:
            position_value = self.current_position.current_value(current_price)

        total_equity = self.cash + position_value

        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'cash': self.cash,
            'position_value': position_value
        })

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)

    def get_trades(self) -> pd.DataFrame:
        """Get all closed trades as DataFrame."""
        if not self.closed_positions:
            return pd.DataFrame()
        return pd.DataFrame(self.closed_positions)

    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if not self.equity_curve:
            return 0.0
        final_equity = self.equity_curve[-1]['equity']
        return ((final_equity / self.initial_capital) - 1) * 100

    @property
    def num_trades(self) -> int:
        """Number of closed trades."""
        return len(self.closed_positions)

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.equity_curve = []
