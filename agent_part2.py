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
        if opponent_stock <= 0:
            return np.ones(len(prices), dtype=float)

        if not self.history:
            rivals = self.prior_competitor_prices.copy()
            weights = np.ones(len(rivals), dtype=float)
        else:
            recent = list(self.history)
            rivals = np.array([item[1] for item in recent], dtype=float)
            historical_buyers = np.array([item[0] for item in recent], dtype=float)

            ages = np.arange(len(recent) - 1, -1, -1, dtype=float)
            weights = np.power(0.965, ages)

            # Softly prefer comparable buyer covariates while keeping weights finite.
            current_buyer = np.asarray(buyer, dtype=float).reshape(1, 3)
            scales = np.array([1.0, 3.0, 25.0], dtype=float)
            scaled_diff = np.clip(
                (historical_buyers - current_buyer) / scales,
                -1e6,
                1e6,
            )
            distances = np.sum(scaled_diff ** 2, axis=1)
            weights *= 0.35 + 0.65 * np.exp(-0.5 * distances)

            # A modest generic prior reduces overconfidence early in a matchup.
            prior_weight = max(0.0, 6.0 - len(recent))
            if prior_weight > 0.0:
                rivals = np.concatenate((rivals, self.prior_competitor_prices))
                prior_weights = np.full(
                    len(self.prior_competitor_prices),
                    prior_weight / len(self.prior_competitor_prices),
                    dtype=float,
                )
                weights = np.concatenate((weights, prior_weights))

        # Clean and normalize before matrix operations to avoid invalid results.
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 0.0)
        weight_sum = float(np.sum(weights))

        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.full(len(rivals), 1.0 / len(rivals), dtype=float)
        else:
            weights = weights / weight_sum

        greater = rivals[None, :] > prices[:, None]
        tied = rivals[None, :] == prices[:, None]
        win_matrix = greater.astype(float) + 0.5 * tied.astype(float)
        win_probability = np.sum(win_matrix * weights[None, :], axis=1)

        return np.clip(win_probability, 0.0, 1.0)

    def _shadow_cost(self, time_remaining, my_stock, opponent_stock):
        """Approximate marginal inventory value from the Part 1 Bellman table."""
        if my_stock <= 0 or time_remaining <= 1:
            return 0.0

        if self.V is None or self.V.ndim != 2 or self.V.shape[1] < 2:
            return 0.0

        t_index = min(max(time_remaining - 1, 0), self.V.shape[0] - 1)
        stock_index = min(max(my_stock, 1), self.V.shape[1] - 1)
        basic = float(self.V[t_index, stock_index] - self.V[t_index, stock_index - 1])

        if not np.isfinite(basic):
            return 0.0
        basic = max(0.0, basic)

        if opponent_stock <= 0:
            return basic

        ratio = min(opponent_stock / max(time_remaining, 1), 1.0)
        competition_factor = 0.60 + 0.40 * (1.0 - ratio)
        return basic * competition_factor

    def action(self, obs):
        """Return one nonnegative finite price for the next customer."""
        buyer, last_sale, profits, inventories, time_until_replenish = obs

        buyer = np.asarray(buyer, dtype=float)
        my_stock = max(0, int(inventories[self.this_agent_number]))
        opponent_stock = max(0, int(inventories[self.opponent_number]))
        time_remaining = max(1, int(time_until_replenish))

        self._record_previous_round(last_sale)
        self.prev_customer = buyer.copy()
        self.prev_opp_stock = opponent_stock

        if my_stock <= 0:
            return 500.0

        prices = self._candidate_prices(opponent_stock)
        purchase_probabilities = self._demand(buyer, prices)
        win_probabilities = self._opponent_win_chance(
            buyer, prices, opponent_stock
        )
        shadow_cost = self._shadow_cost(
            time_remaining, my_stock, opponent_stock
        )

        objective = purchase_probabilities * win_probabilities * (
            prices - shadow_cost
        )
        objective = np.nan_to_num(
            objective, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
        )
        chosen_price = float(prices[int(np.argmax(objective))])

        if not np.isfinite(chosen_price) or chosen_price < 0.0:
            return 500.0
        return chosen_price
