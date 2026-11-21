import random
import pickle
import os
import numpy as np
import pandas as pd


'''
This template serves as a starting point for your Part 2 agent .
'''


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## inventory is reset to a uniformly random integer from 7 to 20 every 20 customers.
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        ### useful if you want to use a more complex price prediction model
        ### do not rename agent_part2.py or the helper_files_here folder
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### custom classes must also be importable when the saved model is loaded
        ### refer to 'model_training/purchase_model_baseline.ipynb' for a quick tutorial on how to use pickle
        self.filename = os.path.join(os.path.dirname(__file__), 'helper_files_here', 'xgb_model.pkl')
        with open(self.filename, 'rb') as file:
            self.trained_model = pickle.load(file)
        self.price_grid = np.linspace(0, 500, 101)  # Baseline grid, not a pricing restriction.
        self.V = np.load(os.path.join(os.path.dirname(__file__),"helper_files_here", "bellman_values_xgb.npy")
)

        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        self.opponent_number = 1 - agent_number  # index for opponent

        self.opponent_history = []
        self.last_buyer = None
        self.last_opponent_stocked = False

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)

        ### keep track of the prices that were offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]
        opponent_last_prices = last_sale[1][self.opponent_number]

        ### keep track of the profit for this agent after the last sale
        my_current_profit = state[self.this_agent_number]
        opponent_current_profit = state[self.opponent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]
        opponent_inventory = inventories[self.opponent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round
        pass

    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''

        buyer, last_sale, state, inventories, time_until_replenish = obs

        #associate the previous buyer with the opponent's previous price
        if self.last_buyer is not None and self.last_opponent_stocked:
            try:
                opponent_price = float(last_sale[1][self.opponent_number])

                if np.isfinite(opponent_price):
                    self.opponent_history.append(
                        (self.last_buyer.copy(), opponent_price)
                    )
                    self.opponent_history = self.opponent_history[-200:]

            except (TypeError, ValueError, IndexError):
                pass

        #remember this customer for the next round
        self.last_buyer = np.asarray(buyer, dtype=float).copy()
        self.last_opponent_stocked = (
            inventories[self.opponent_number] > 0
)



        s = int(inventories[self.this_agent_number])
        t = max(1, min(20, int(time_until_replenish)))

        if s <= 0:
            return 500.0

        features = pd.DataFrame(np.tile(buyer, (len(self.price_grid), 1)), columns=["Covariate1", "Covariate2", "Covariate3"])

        features["price_item"] = self.price_grid

        #purchase probabilities from XGBoost
        probability = self.trained_model.predict_proba(features)[:, 1]

        #Bellman inventory opportunity cost
        #original Bellman opportunity cost
        base_opportunity_cost = self.V[t-1, s] - self.V[t-1, s-1]

        # Estimate how much future competition we face
        opponent_inventory = int(inventories[self.opponent_number])

        competition_pressure = min(
            1.0,
            opponent_inventory / t
        )

        # Reduce the value of saving inventory when competition is strong
        opportunity_cost = base_opportunity_cost * (
            1.0 - 0.25 * competition_pressure
        )



        #probability of offering a lower price than the opponent
        win_probability = np.ones(len(self.price_grid))

        if (
            inventories[self.opponent_number] > 0
            and len(self.opponent_history) >= 20
        ):

            historical_buyers = np.array([
                x for x, p in self.opponent_history
            ])

            historical_prices = np.array([
                p for x, p in self.opponent_history
            ])

            # Standardize covariates for distance comparison
            scale = np.maximum(
                historical_buyers.std(axis=0),
                0.25
            )

            distances = np.sum(
                ((historical_buyers - buyer) / scale) ** 2,
                axis=1
            )

            # Find the 30 most similar previous customers
            k = min(30, len(historical_prices))
            nearest = np.argsort(distances)[:k]

            similar_prices = historical_prices[nearest]

            # Probability of beating prices for similar customers
            local_win = (
                (self.price_grid[:, None] < similar_prices).mean(axis=1)
                + 0.5 * (self.price_grid[:, None] == similar_prices).mean(axis=1)
            )

            # Keep some information from all recent customers
            global_win = (
                (self.price_grid[:, None] < historical_prices).mean(axis=1)
                + 0.5 * (self.price_grid[:, None] == historical_prices).mean(axis=1)
            )

            win_probability = 0.7 * local_win + 0.3 * global_win




        #expected revenue adjusted for competition and inventory
        #original Bellman price without competition
        bellman_score = probability * (self.price_grid - opportunity_cost)
        bellman_price = self.price_grid[np.argmax(bellman_score)]

        #don't discount more than 15% below the Bellman price
        minimum_price = max(
            opportunity_cost,
            0.85 * bellman_price
        )

        #competition-adjusted expected revenue
        score = (
            probability
            * win_probability
            * (self.price_grid - opportunity_cost)
        )

        #exclude prices below our minimum acceptable price
        score = np.where(
            self.price_grid >= minimum_price,
            score,
            -np.inf
        )

        return float(self.price_grid[np.argmax(score)])
