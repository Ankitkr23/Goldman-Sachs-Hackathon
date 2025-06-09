import pandas as pd
import numpy as np

class AutomatedMarketMaking:
    def __init__(self, tick_size=0.1, lot_size=2):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.reset_simulator()
        
        # Strategy parameters
        self.max_inventory = 20  # Maximum inventory allowed
        self.volatility_window = 50  # Window for volatility calculation
        self.risk_aversion = 0.1  # Risk aversion parameter (gamma)
        self.transaction_cost = 1.0  # Transaction cost parameter (k)
        self.time_horizon = 1.0  # Time horizon for position unwinding (T)

    def reset_simulator(self):
        self.inventory = 0
        self.active_bid = None
        self.active_ask = None
        self.valid_from = None
        self.mid_prices = []  # Store mid prices for volatility calculation

    def update_quote(self, timestamp, bid_price, ask_price):
        # Post or update your quote at timestamp It takes effect at t+1
        self.active_bid = bid_price
        self.active_ask = ask_price
        self.valid_from = timestamp + 1

    def process_trades(self, timestamp, trades_at_t):
        # Process all public trades at timestamp Returns updated inventory
        if self.valid_from is None or timestamp < self.valid_from:
            return self.inventory

        filled = False

        # sellside fill against your bid
        sells = trades_at_t[trades_at_t.side == 'sell']
        if self.active_bid is not None and not sells.empty:
            if self.active_bid >= sells.price.max():
                self.inventory += self.lot_size
                self.active_bid = None
                filled = True

        # buyside fill against your ask
        buys = trades_at_t[trades_at_t.side == 'buy']
        if self.active_ask is not None and not buys.empty:
            if self.active_ask <= buys.price.min():
                self.inventory -= self.lot_size
                self.active_ask = None
                filled = True

        if filled:
            # deactivate until next update
            self.valid_from = float('inf')

        return self.inventory

    def round_to_tick(self, price):
        """Round price to nearest tick size"""
        return round(price / self.tick_size) * self.tick_size

    def get_orderbook_data(self, ob_df, t):
        """Get latest orderbook data up to timestamp t"""
        ob_snapshot = ob_df[ob_df.timestamp <= t].tail(1)
        
        if ob_snapshot.empty:
            return None, None, None
        
        latest = ob_snapshot.iloc[0]
        
        # Get bid and ask prices - handle different column naming conventions
        bid_col = 'bid_price_1' if 'bid_price_1' in latest.index else 'bid_1_price'
        ask_col = 'ask_price_1' if 'ask_price_1' in latest.index else 'ask_1_price'
        
        best_bid = latest[bid_col]
        best_ask = latest[ask_col]
        mid_price = (best_bid + best_ask) / 2
        
        return best_bid, best_ask, mid_price
    
    def estimate_volatility(self):
        """Estimate volatility from mid price history"""
        if len(self.mid_prices) < 2:
            return 0.01  # Default volatility if not enough data
        
        # Calculate returns using mid price changes
        returns = np.diff(self.mid_prices[-self.volatility_window:])
        volatility = np.std(returns) if len(returns) > 0 else 0.01
        
        # Scale volatility to be more reasonable
        return max(volatility, 0.01)

    def calculate_price_trend(self, tr_df, current_time, lookback=10):
        """Calculate price trend from recent trades"""
        # Get trades before current timestamp
        past_trades = tr_df[tr_df.timestamp < current_time].tail(lookback)
        
        if len(past_trades) < 2:
            return 0
        
        # Calculate weighted average price trend (most recent trades have higher weight)
        prices = past_trades['price'].values
        weights = np.linspace(0.5, 1.0, len(prices))
        trend = np.average(np.diff(prices), weights=weights[1:]) if len(prices) > 1 else 0
        return trend

    def strategy(self, ob_df, tr_df, inventory, t):
        """Implement market making strategy"""
        # Get current order book data
        best_bid, best_ask, mid_price = self.get_orderbook_data(ob_df, t)
        
        if mid_price is None:
            return None, None
            
        # Store mid price for volatility calculation
        self.mid_prices.append(mid_price)
        
        # Calculate volatility
        sigma = self.estimate_volatility()
        
        # Avellaneda-Stoikov market making model parameters
        gamma = self.risk_aversion
        k = self.transaction_cost
        T = self.time_horizon
        
        # Calculate reservation price adjustment based on inventory
        inventory_skew = inventory * gamma * sigma**2 * T / 2
        
        # Calculate optimal spread
        delta = (1 / gamma) * np.log(1 + gamma / k)
        spread = gamma * sigma**2 * T + 2 * delta
        half_spread = spread / 2
        
        # Calculate price trend and incorporate it
        trend = self.calculate_price_trend(tr_df, t)
        trend_factor = trend * 0.3  # Scale trend impact
        
        # Calculate quotes
        bid = mid_price - half_spread - inventory_skew + trend_factor
        ask = mid_price + half_spread - inventory_skew + trend_factor
        
        # Round to tick size
        bid = self.round_to_tick(bid)
        ask = self.round_to_tick(ask)
        
        # Make sure bid < ask
        if bid >= ask:
            ask = bid + self.tick_size
            
        # Inventory risk management
        inventory_limit = self.max_inventory
        
        # If inventory is near the limit, adjust quotes to reduce risk
        if inventory + self.lot_size > inventory_limit * 0.8:
            # When inventory is high, be more aggressive on asks and more passive on bids
            bid = min(bid, best_bid - self.tick_size * 2)  # More passive bid
            ask = min(ask, best_ask - self.tick_size)      # More aggressive ask
            
            # If at max inventory, don't place a bid
            if inventory + self.lot_size >= inventory_limit:
                bid = None
                
        elif inventory - self.lot_size < -inventory_limit * 0.8:
            # When short inventory is high, be more aggressive on bids and more passive on asks
            bid = max(bid, best_bid + self.tick_size)      # More aggressive bid
            ask = max(ask, best_ask + self.tick_size * 2)  # More passive ask
            
            # If at max short inventory, don't place an ask
            if inventory - self.lot_size <= -inventory_limit:
                ask = None
        
        return bid, ask

    def run(self, ob_df, tr_df):
        self.reset_simulator()
        quotes = []

        all_ts = sorted(ob_df.timestamp.unique())
        for t in all_ts:
            trades_t = tr_df[tr_df.timestamp == t]
            inv = self.process_trades(t, trades_t)

            bid, ask = self.strategy(ob_df, tr_df, inv, t)

            self.update_quote(t, bid, ask)

            quotes.append({
                'timestamp': t,
                'bid_price': bid,
                'ask_price': ask
            })

        return pd.DataFrame(quotes)

if __name__ == "__main__":
    ob_obj = pd.read_csv(input().strip())
    tr_obj = pd.read_csv(input().strip())
    
    #pick top 3k timestamps
    ob_obj = ob_obj.head(3000)
    tr_obj = tr_obj.head(3000)

    amm = AutomatedMarketMaking(tick_size=0.1, lot_size=2)

    df_submission = amm.run(ob_obj, tr_obj)
    df_submission.to_csv('submission.csv', index=False)
