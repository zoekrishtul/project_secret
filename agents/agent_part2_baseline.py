
import os
import pickle
import numpy as np
import pandas as pd


class Agent:
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number
        self.opponent_number = 1 - agent_number

        self.project_part = params["project_part"]
        self.remaining_inventory = params["inventory_limit"]
        self.inventory_replenish = params["inventory_replenish"]

        helper = os.path.join(
            os.path.dirname(__file__),
            "helper_files_here"
        )

        with open(os.path.join(helper, "xgb_model.pkl"), "rb") as f:
            self.trained_model = pickle.load(f)

        self.V = np.load(
            os.path.join(helper, "bellman_values_xgb.npy")
        )

        self.price_grid = np.linspace(0, 500, 101)

        # Previous competitor-learning approach
        self.opponent_price_history = []

    def action(self, obs):
        buyer, last_sale, state, inventories, time_until_replenish = obs

        s = int(inventories[self.this_agent_number])
        t = max(1, min(20, int(time_until_replenish)))

        # Record the opponent's previous offered price
        try:
            opponent_price = float(
                last_sale[1][self.opponent_number]
            )

            if np.isfinite(opponent_price):
                self.opponent_price_history.append(opponent_price)
                self.opponent_price_history = (
                    self.opponent_price_history[-100:]
                )
        except (TypeError, ValueError, IndexError):
            pass

        if s <= 0:
            return 500.0

        # XGBoost demand predictions
        features = pd.DataFrame(
            np.tile(buyer, (len(self.price_grid), 1)),
            columns=["Covariate1", "Covariate2", "Covariate3"]
        )
        features["price_item"] = self.price_grid

        probability = self.trained_model.predict_proba(features)[:, 1]

        # Bellman inventory opportunity cost
        opportunity_cost = self.V[t-1, s] - self.V[t-1, s-1]

        # Original Bellman price
        bellman_score = probability * (
            self.price_grid - opportunity_cost
        )
        bellman_price = self.price_grid[np.argmax(bellman_score)]

        # Competitor price distribution
        win_probability = np.ones(len(self.price_grid))

        if (
            inventories[self.opponent_number] > 0
            and len(self.opponent_price_history) >= 10
        ):
            recent_prices = np.array(
                self.opponent_price_history[-60:]
            )

            lower_price = (
                self.price_grid[:, None] < recent_prices[None, :]
            ).mean(axis=1)

            equal_price = (
                self.price_grid[:, None] == recent_prices[None, :]
            ).mean(axis=1)

            win_probability = lower_price + 0.5 * equal_price

        # Never discount over 15% below Bellman price
        minimum_price = max(
            opportunity_cost,
            0.85 * bellman_price
        )

        score = (
            probability
            * win_probability
            * (self.price_grid - opportunity_cost)
        )

        score = np.where(
            self.price_grid >= minimum_price,
            score,
            -np.inf
        )

        return float(self.price_grid[np.argmax(score)])
