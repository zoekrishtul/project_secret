"""Capacity-aware pricing against a seller whose current offer is unknown."""

from collections import deque
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


class Agent:
    def __init__(self, agent_number, params=None):
        self.number = agent_number
        self.opponent = 1 - agent_number
        self.offers = deque(maxlen=12)
        self.previous_opponent_stock = 0
        self.previous_offer = None
        self.previous_opponent_sold = False
        self.price_ratios = {False: deque(maxlen=6), True: deque(maxlen=6)}

        helpers = Path(__file__).with_name("helper_files_here")
        with (helpers / "xgb_model.pkl").open("rb") as model_file:
            self.model = pickle.load(model_file)
        self.values = np.load(helpers / "bellman_values_xgb.npy")
        self.grid = np.arange(0.0, 501.0, 5.0)

    def action(self, obs):
        buyer, last_sale, _, inventories, remaining = obs
        stock = int(inventories[self.number])
        opponent_stock = inventories[self.opponent]

        opponent_offer = None
        opponent_sold = False
        if last_sale is not None and self.previous_opponent_stock > 0:
            winner, prices = last_sale
            actual_offer = prices[self.opponent]
            if np.isfinite(actual_offer) and actual_offer >= 0:
                opponent_offer = float(actual_offer)
                if self.previous_offer is not None and self.previous_offer > 0:
                    self.price_ratios[self.previous_opponent_sold].append(
                        opponent_offer / self.previous_offer
                    )
                self.offers.append(opponent_offer)
                opponent_sold = winner == self.opponent
        self.previous_offer = opponent_offer
        self.previous_opponent_sold = opponent_sold
        self.previous_opponent_stock = opponent_stock

        if stock <= 0:
            return float(self.grid[-1])

        time_index = min(max(int(remaining) - 1, 0), self.values.shape[0] - 1)
        stock_index = min(stock, self.values.shape[1] - 1)
        cost = self.values[time_index, stock_index] - self.values[time_index, stock_index - 1]

        prices = self.grid
        win_probability = 1.0
        # Repeated markdowns after a lost sale can create a price war.
        # Against that pattern, use a slower price cap from recent offers.
        lost_sale_ratios = self.price_ratios[False]
        reactive = (
            len(lost_sale_ratios) >= 3
            and np.ptp(lost_sale_ratios) < 1e-6
            and 0 < lost_sale_ratios[-1] < 0.999
        )
        if opponent_stock > 0 and self.offers and not reactive:
            forecasts = np.asarray(self.offers)
            ratios = self.price_ratios[opponent_sold]
            if opponent_offer is not None and len(ratios) >= 3 and np.ptp(ratios) < 1e-6:
                forecasts = np.array([opponent_offer * np.median(ratios)])

            # One cent below each observed or predicted offer is also a candidate.
            undercuts = np.clip(forecasts - 0.01, 0.0, self.grid[-1])
            prices = np.unique(np.concatenate((self.grid, undercuts)))
            win_probability = (
                (prices[:, None] < forecasts).mean(axis=1)
                + 0.5 * (prices[:, None] == forecasts).mean(axis=1)
            )

        features = pd.DataFrame(
            np.broadcast_to(buyer, (len(prices), 3)),
            columns=["Covariate1", "Covariate2", "Covariate3"],
        )
        features["price_item"] = prices
        purchase_probability = self.model.predict_proba(features)[:, 1]
        score = (prices - cost) * purchase_probability * win_probability
        if np.max(score) <= 0:
            return float(self.grid[-1])

        price = float(prices[np.argmax(score)])
        if opponent_stock > 0 and reactive:
            price_cap = float(np.mean(list(self.offers)[-10:])) - 1.0
            if price_cap > cost:
                price = min(price, price_cap)
        return price
