"""ORIE 5355 Part 2: demand-based, opponent-adaptive, inventory-aware pricing.

Required artifact:
    agents/helper_files_here/xgb_model.pkl

Optional artifact:
    agents/helper_files_here/bellman_values_xgb.npy

Uses the fitted Part 1 purchase model and current/past public observations only.
It does not access hidden valuations or simulator internals for training.
"""

import os
import pickle
from collections import deque

import numpy as np
import pandas as pd


class Agent(object):
    """Competitive pricing agent with online opponent-price estimation."""

    def __init__(self, agent_number, params={}):
        self.this_agent_number = int(agent_number)
        self.opponent_number = 1 - self.this_agent_number
        self.project_part = params.get("project_part", 2)

        helper_dir = os.path.join(os.path.dirname(__file__), "helper_files_here")
        model_path = os.path.join(helper_dir, "xgb_model.pkl")

        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

        value_path = os.path.join(helper_dir, "bellman_values_xgb.npy")
        self.V = np.load(value_path) if os.path.isfile(value_path) else None

        self.base_grid = np.arange(5.0, 500.01, 5.0)
        self.history = deque(maxlen=100)  # (previous buyer, opponent price)
        self.prev_customer = None
        self.prev_opp_stock = None
        self.last_seen_price = None
        self.prior_competitor_prices = np.array(
            [35.0, 50.0, 75.0, 100.0, 140.0, 200.0, 300.0],
            dtype=float,
        )

    def _record_previous_round(self, last_sale):
        """Associate the last observed opponent price with the previous buyer."""
        if self.prev_customer is None or self.prev_opp_stock is None:
            return
        if self.prev_opp_stock <= 0:
            return

        try:
            opponent_price = float(last_sale[1][self.opponent_number])
        except (TypeError, IndexError, KeyError, ValueError):
            return

        if np.isfinite(opponent_price) and opponent_price >= 0.0:
            self.history.append((self.prev_customer.copy(), opponent_price))
            self.last_seen_price = opponent_price

    def _candidate_prices(self, opponent_stock):
        """Construct base-grid prices and prices near recently observed rivals."""
        prices = list(self.base_grid)

        if opponent_stock > 0 and self.history:
            for _, opponent_price in list(self.history)[-35:]:
                if 0.01 < opponent_price <= 500.0:
                    prices.extend(
                        [round(opponent_price - 0.01, 2), round(opponent_price, 2)]
                    )

        return np.unique(np.asarray(prices, dtype=float))

    def _demand(self, buyer, candidate_prices):
        """Estimate purchase probability for the current buyer at each price."""
        buyer = np.asarray(buyer, dtype=float).reshape(1, 3)
        features = pd.DataFrame(
            np.repeat(buyer, len(candidate_prices), axis=0),
            columns=["Covariate1", "Covariate2", "Covariate3"],
        )
        features["price_item"] = candidate_prices

        probabilities = np.asarray(
            self.model.predict_proba(features)[:, 1], dtype=float
        )
        return np.clip(
            np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )

    def _opponent_win_chance(self, buyer, prices, opponent_stock):
        """Estimate P(rival price > ours) + 0.5 * P(exact tie)."""