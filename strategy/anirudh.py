import json, math
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def run(self, state: TradingState):
        result = {}
        try: td = json.loads(state.traderData)
        except: td = {}
        
        PRODUCT_CONFIGS = {
            "GALAXY_SOUNDS_SOLAR_FLAMES": {
                "base": 11000,
                "pos_limit": 10,
                "grid_limit": 10,
                "tiers": [
                    (200, 0.30),
                    (500, 0.80),
                    (1000, 1.00)
                ]
            },
            "GALAXY_SOUNDS_DARK_MATTER": {
                "base": 10250,
                "pos_limit": 10,
                "grid_limit": 10,
                "tiers": [
                    (200, 0.30),
                    (500, 0.80),
                    (1000, 1.00)
                ]
            },
            "SNACKPACK_RASPBERRY": {
                "base": 10100,
                "pos_limit": 10,
                "grid_limit": 10,
                "tiers": [
                    (200, 0.80),
                    (300, 1.00)
                ]
            },
            "SNACKPACK_VANILLA": {
                "base": 10100,
                "pos_limit": 10,
                "grid_limit": 10,
                "tiers": [
                    (200, 0.80),
                    (300, 1.00)
                ]
            },
            "SNACKPACK_CHOCOLATE": {
                "base": 9800,
                "pos_limit": 10,
                "grid_limit": 10,
                "tiers": [
                    (200, 0.80),
                    (300, 1.00)
                ]
            }
        }
        
        alpha = 2 / (300 + 1)
        
        for product, config in PRODUCT_CONFIGS.items():
            if product not in state.order_depths:
                continue
                
            depth = state.order_depths[product]
            buys, sells = depth.buy_orders, depth.sell_orders
            if not buys or not sells:
                continue
                
            best_bid = max(buys.keys())
            best_ask = min(sells.keys())
            mid = (best_bid + best_ask) / 2
            
            ema_key = f"ema_{product}"
            ema = td.get(ema_key, None)
            if ema is None: ema = mid
            else: ema = mid * alpha + ema * (1 - alpha)
            td[ema_key] = ema
            
            bid_vol = sum(q for _, q in sorted(buys.items(), reverse=True)[:5])
            ask_vol = sum(abs(q) for _, q in sorted(sells.items())[:5])
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            
            fair = ema + obi * 1.5
            pos = state.position.get(product, 0)
            
            pos_limit = config["pos_limit"]
            GRID_LIMIT = config["grid_limit"]
            
            orders = []
            sim_bid_pos = pos
            sim_ask_pos = pos
            
            # --- END OF DAY RISK REDUCTION ---
            if state.timestamp >= 900000:
                # Halve the allowed operational limits
                pos_limit = pos_limit // 2
                GRID_LIMIT = GRID_LIMIT // 2
                
                # Liquidate merely the excess "ASAP" to the moving average
                if pos > pos_limit:
                    dump_qty = pos - pos_limit
                    out_px = max(best_ask - 1, math.ceil(ema))
                    orders.append(Order(product, int(out_px), -dump_qty))
                    sim_ask_pos -= dump_qty
                elif pos < -pos_limit:
                    dump_qty = abs(pos + pos_limit) # amount to buy to return to -pos_limit
                    out_px = min(best_bid + 1, math.floor(ema))
                    orders.append(Order(product, int(out_px), dump_qty))
                    sim_bid_pos += dump_qty
            
            MM_LIMIT = pos_limit - GRID_LIMIT
            base_px = config["base"]
            
            for dist, frac in config["tiers"]:
                target_cumulative = int(GRID_LIMIT * frac)
                
                # Place Bids
                bid_allowance = target_cumulative - sim_bid_pos
                if bid_allowance > 0:
                    orders.append(Order(product, base_px - dist, bid_allowance))
                    sim_bid_pos += bid_allowance
                    
                # Place Asks
                ask_allowance = sim_ask_pos - (-target_cumulative)
                if ask_allowance > 0:
                    orders.append(Order(product, base_px + dist, -ask_allowance))
                    sim_ask_pos -= ask_allowance

            grid_bid_volume = sim_bid_pos - pos
            grid_ask_volume = pos - sim_ask_pos
            
            bid_cap = pos_limit - pos
            ask_cap = pos_limit + pos
            
            mm_bid_qty = min(MM_LIMIT, bid_cap - grid_bid_volume)
            mm_ask_qty = min(MM_LIMIT, ask_cap - grid_ask_volume)
            
            skewed_fair = fair - pos * 0.05
            
            if mm_bid_qty > 0:
                b_px = min(best_bid + 1, math.floor(skewed_fair) - 1)
                orders.append(Order(product, int(b_px), mm_bid_qty))
                
            if mm_ask_qty > 0:
                a_px = max(best_ask - 1, math.ceil(skewed_fair) + 1)
                orders.append(Order(product, int(a_px), -mm_ask_qty))
                
            result[product] = orders
        
        # ── SHORT-ONLY STRATEGY FOR DECAYING PRODUCTS ──
        SHORT_PRODUCTS = ["PEBBLES_XS", "MICROCHIP_OVAL", "UV_VISOR_AMBER"]
        for sp in SHORT_PRODUCTS:
            if sp in state.order_depths:
                depth = state.order_depths[sp]
                if depth.buy_orders:
                    best_bid = max(depth.buy_orders.keys())
                    pos = state.position.get(sp, 0)
                    
                    # Send a large short order; the exchange will automatically truncate 
                    # it to the actual position limit for the product.
                    if pos > -10: 
                        if sp not in result:
                            result[sp] = []
                        result[sp].append(Order(sp, best_bid, -10))

        # ── SHORT-ONLY STRATEGY FOR DECAYING PRODUCTS ──
        LONG_PRODUCTS = ["OXYGEN_SHAKE_GARLIC", "MICROCHIP_SQUARE"]
        for sp in LONG_PRODUCTS:
            if sp in state.order_depths:
                depth = state.order_depths[sp]
                if depth.buy_orders:
                    best_ask = min(depth.sell_orders.keys())
                    pos = state.position.get(sp, 0)
                    
                    # Send a large short order; the exchange will automatically truncate 
                    # it to the actual position limit for the product.
                    if pos < 10: 
                        if sp not in result:
                            result[sp] = []
                        result[sp].append(Order(sp, best_ask, 10))


            
        return result, 0, json.dumps(td)