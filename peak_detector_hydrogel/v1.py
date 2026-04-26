import json
import warnings
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, Order, TradingState, Symbol

# Suppress RankWarning which occurs if polyfit encounters ill-conditioned data in flat markets
warnings.simplefilter('ignore', np.RankWarning)

class Trader:
    # =========================================================================
    # HYPER-PARAMETERS
    # =========================================================================
    TARGET_PRODUCT = "HYDROGEL_PACK"
    MAX_POS = 200

    WINDOW_SIZE = 20
    POLY_DEGREE = 5
    ROOT_TOLERANCE = 1.2
    CONCAVITY_THRESH = 1.3
    COOLDOWN = 10
    SMA_WINDOW_SIZE = 300
    QUALITY_WINDOW = 20

    OUTLIER_DELTA_THRESH = 25
    OUTLIER_DELTA_MAX = 50
    # =========================================================================

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        
        # 1. State Deserialization
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                history = data.get("history", [])
                maxima_qualities = data.get("maxima_qualities", [])
                minima_qualities = data.get("minima_qualities", [])
                last_max = data.get("last_max", -999)
                last_min = data.get("last_min", -999)
                # NEW: Load pending orders from previous ticks
                pending_qty = data.get("pending_qty", 0) 
            except Exception:
                history, maxima_qualities, minima_qualities, last_max, last_min, pending_qty = [], [], [], -999, -999, 0
        else:
            history, maxima_qualities, minima_qualities, last_max, last_min, pending_qty = [], [], [], -999, -999, 0

        current_tick = state.timestamp // 100

        # 2. Process Current Tick
        if self.TARGET_PRODUCT in state.order_depths:
            depth = state.order_depths[self.TARGET_PRODUCT]
            position = state.position.get(self.TARGET_PRODUCT, 0)
            
            best_ask = min(depth.sell_orders.keys()) if len(depth.sell_orders) > 0 else None
            best_bid = max(depth.buy_orders.keys()) if len(depth.buy_orders) > 0 else None
            
            if best_ask is not None and best_bid is not None:
                current_price = (best_ask + best_bid) / 2.0
                history.append(current_price)
                
                if len(history) > self.SMA_WINDOW_SIZE:
                    history.pop(0)

                orders: List[Order] = []

                # 3. Core Logic Signal Generation 
                # (We only generate new signals if we aren't currently trying to clear a massive backlog)
                if len(history) >= self.WINDOW_SIZE and pending_qty == 0:
                    
                    current_sma = np.mean(history[-self.SMA_WINDOW_SIZE:]) if len(history) >= self.SMA_WINDOW_SIZE else np.mean(history)
                    
                    y = np.array(history[-self.WINDOW_SIZE:])
                    x = np.arange(self.WINDOW_SIZE)
                    
                    poly = np.polyfit(x, y, self.POLY_DEGREE)
                    p_d1 = np.polyder(poly, 1) 
                    p_d2 = np.polyder(poly, 2) 
                    
                    current_x = self.WINDOW_SIZE - 1
                    
                    roots = np.roots(p_d1)
                    real_roots = roots[np.isreal(roots)].real
                    
                    for r in real_roots:
                        if (current_x - self.ROOT_TOLERANCE <= r <= current_x + 1e-5):
                            val_d2 = np.polyval(p_d2, r)
                            
                            # --- MAXIMA (PEAK DETECTED) -> SELL SHORT ---
                            if val_d2 < -self.CONCAVITY_THRESH and (current_tick - last_max > self.COOLDOWN):
                                if current_price > current_sma:
                                    quality = current_price - current_sma
                                    avg_recent_max = np.mean(maxima_qualities[-self.QUALITY_WINDOW:]) if len(maxima_qualities) > 0 else 0
                                    
                                    if quality > avg_recent_max and quality >= self.OUTLIER_DELTA_THRESH:
                                        ratio = (quality - self.OUTLIER_DELTA_THRESH) / (self.OUTLIER_DELTA_MAX - self.OUTLIER_DELTA_THRESH)
                                        ratio = max(0.0, min(1.0, ratio))
                                        commit_fraction = 0.5 + (0.5 * ratio)
                                        
                                        remaining_limit = -self.MAX_POS - position
                                        trade_qty = int(remaining_limit * commit_fraction)
                                        
                                        if trade_qty < 0:
                                            # OVERWRITE PENDING QTY INSTEAD OF APPENDING ORDER
                                            pending_qty = trade_qty
                                            
                                        maxima_qualities.append(quality)
                                        last_max = current_tick
                                        break 
                                        
                            # --- MINIMA (TROUGH DETECTED) -> BUY LONG ---
                            elif val_d2 > self.CONCAVITY_THRESH and (current_tick - last_min > self.COOLDOWN):
                                if current_price < current_sma:
                                    quality = current_sma - current_price
                                    avg_recent_min = np.mean(minima_qualities[-self.QUALITY_WINDOW:]) if len(minima_qualities) > 0 else 0
                                    
                                    if quality > avg_recent_min and quality >= self.OUTLIER_DELTA_THRESH:
                                        ratio = (quality - self.OUTLIER_DELTA_THRESH) / (self.OUTLIER_DELTA_MAX - self.OUTLIER_DELTA_THRESH)
                                        ratio = max(0.0, min(1.0, ratio)) 
                                        commit_fraction = 0.5 + (0.5 * ratio)
                                        
                                        remaining_limit = self.MAX_POS - position
                                        trade_qty = int(remaining_limit * commit_fraction)
                                        
                                        if trade_qty > 0:
                                            # OVERWRITE PENDING QTY INSTEAD OF APPENDING ORDER
                                            pending_qty = trade_qty
                                            
                                        minima_qualities.append(quality)
                                        last_min = current_tick
                                        break

                # 4. ORDER BOOK SWEEPER (Execute pending quantities level by level)
                if pending_qty > 0:
                    # We want to BUY. Verify we won't breach MAX_POS first.
                    allowed_to_buy = self.MAX_POS - position
                    executable_qty = min(pending_qty, allowed_to_buy)
                    
                    if executable_qty > 0:
                        # Walk the ASK book (sorted lowest price to highest)
                        for ask_price in sorted(depth.sell_orders.keys()):
                            # sell_orders are represented as negative volumes in Prosperity
                            available_vol = -depth.sell_orders[ask_price] 
                            
                            if available_vol > 0:
                                take_vol = min(executable_qty, available_vol)
                                orders.append(Order(self.TARGET_PRODUCT, ask_price, take_vol))
                                executable_qty -= take_vol
                                
                            if executable_qty == 0:
                                break
                                
                        # Save whatever we couldn't execute for the next tick
                        pending_qty = executable_qty
                    else:
                        pending_qty = 0 # Limits breached, kill the pending order

                elif pending_qty < 0:
                    # We want to SELL. Verify we won't breach -MAX_POS first.
                    allowed_to_sell = -self.MAX_POS - position
                    # Both pending and allowed are negative, so we use max() to restrict it
                    executable_qty = max(pending_qty, allowed_to_sell) 
                    
                    if executable_qty < 0:
                        qty_to_sell = abs(executable_qty)
                        
                        # Walk the BID book (sorted highest price to lowest)
                        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                            available_vol = depth.buy_orders[bid_price]
                            
                            if available_vol > 0:
                                take_vol = min(qty_to_sell, available_vol)
                                orders.append(Order(self.TARGET_PRODUCT, bid_price, -take_vol))
                                qty_to_sell -= take_vol
                                
                            if qty_to_sell == 0:
                                break
                                
                        # Restore the negative sign for serialization
                        pending_qty = -qty_to_sell 
                    else:
                        pending_qty = 0

                if len(orders) > 0:
                    result[self.TARGET_PRODUCT] = orders

        # 5. Serialize state for next tick (Including the new pending_qty)
        trader_state = {
            "history": history,
            "maxima_qualities": maxima_qualities[-self.QUALITY_WINDOW:], 
            "minima_qualities": minima_qualities[-self.QUALITY_WINDOW:],
            "last_max": last_max,
            "last_min": last_min,
            "pending_qty": pending_qty  # Persistent Execution State
        }
        
        traderData = json.dumps(trader_state)
        conversions = 0
        
        return result, conversions, traderData
