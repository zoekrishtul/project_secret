import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import ndtr   # standard normal CDF

"""
Part 2 agent: capacity-aware pricing against one competitor.

Every customer buys from the cheapest seller that has stock (and whose price is <= their value).
So to sell, my price must be (a) acceptable to the customer AND (b) lower than the rival's price:

    P(I win | p) = q(p | covariates)  *  P(rival price > p)

 q(p | covariates)  : purchase model (gradient boosting, monotone in price) - same as Part 1
 P(rival price > p) : learned online from the rival's past posted prices (recency-weighted, smoothed)
                      = 1 if the rival is out of stock (then I am a monopolist for this customer)
 opportunity cost   : Bellman value of capacity (helper_files_here/bellman_values_xgb.npy):
                      delta = V[t-1, s] - V[t-1, s-1] = future value of the unit I would give up

Chosen price = argmax_p  P(I win | p) * (p - delta).
"""

# knobs (tuned on simulated opponents)
HALF_LIFE = 12.0     # how fast the rival's old prices are forgotten (in customers)
WINDOW = 60          # max number of rival prices remembered
MIN_BW = 2.0         # smoothing (std, in $) of the rival price distribution
REL_BW = 0.05        # ... or this fraction of the rival's typical price, whichever is bigger
DELTA_SCALE = 1.0    # scales the opportunity cost (1 = trust the single-agent Bellman values)


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number
        self.opponent_number = 1 - agent_number
        self.project_part = params['project_part']
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        base = os.path.join(os.path.dirname(__file__), 'helper_files_here')
        with open(os.path.join(base, 'xgb_model.pkl'), 'rb') as f:
            self.trained_model = pickle.load(f)
        self.V = np.load(os.path.join(base, 'bellman_values_xgb.npy'))
        self.price_grid = np.linspace(0, 500, 101)

        self.opp_prices = []          # rival prices observed while the rival had stock
        self.opp_offsets = []         # rival price minus MY price one step earlier (for "follower" rivals)
        self.my_prev = None           # my price in the round before the last one
        self.my_last = None           # my price in the last round
        self.prev_opp_inventory = None
        self.cols = ['Covariate1', 'Covariate2', 'Covariate3']

    # ---- learning the rival from what we observe ------------------------------------
    def _process_last_sale(self, last_sale, inventories):
        opp_price = last_sale[1][self.opponent_number]
        my_price = last_sale[1][self.this_agent_number]
        # the rival posted opp_price after seeing MY price from the round before (self.my_last)
        # a rival with no stock may post junk (e.g. 0) that cannot win -> ignore those
        if (not np.isnan(opp_price)) and self.prev_opp_inventory is not None and self.prev_opp_inventory > 0:
            self.opp_prices.append(float(opp_price))
            if self.my_last is not None:
                self.opp_offsets.append(float(opp_price) - self.my_last)
            else:
                self.opp_offsets.append(np.nan)
            if len(self.opp_prices) > WINDOW:
                self.opp_prices.pop(0)
                self.opp_offsets.pop(0)
        if not np.isnan(my_price):
            self.my_last = float(my_price)
        self.prev_opp_inventory = int(inventories[self.opponent_number])

    def _rival_samples(self):
        """Candidate values for the rival's next price.
        Model A ("fixed-type" rival): its own past prices.
        Model B ("follower" rival, e.g. undercuts me): my latest price + its past offsets from my price.
        Use whichever model has been predicting better over the last 15 rounds."""
        x = np.asarray(self.opp_prices)
        off = np.asarray(self.opp_offsets)
        k = 15
        ok = ~np.isnan(off)
        if self.my_last is None or ok.sum() < 6:
            return x
        xa, xo = x[-k:], off[-k:]
        okk = ~np.isnan(xo)
        # leave-one-out style error: deviation from the mean of the other recent values
        err_a = np.std(xa[okk])
        err_b = np.std(xo[okk])
        if err_b < 0.7 * err_a:
            return np.clip(self.my_last + off[ok], 0, None)
        return x

    def _prob_rival_above(self):
        """P(rival price > p) on the price grid, from recency-weighted, Gaussian-smoothed past prices."""
        x = self._rival_samples()
        n = len(x)
        w = 0.5 ** ((n - 1 - np.arange(n)) / HALF_LIFE)
        w /= w.sum()
        bw = max(MIN_BW, REL_BW * float(np.dot(w, x)))
        z = (self.price_grid[:, None] - x[None, :]) / bw
        # P(rival > p) = sum_i w_i * (1 - Phi(z_i)),  Phi via erf
        cdf = ndtr(z)
        return (w[None, :] * (1.0 - cdf)).sum(axis=1)

    # ---- the decision ----------------------------------------------------------------
    def action(self, obs):
        buyer, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, inventories)

        s = int(inventories[self.this_agent_number])
        t = max(1, min(20, int(time_until_replenish)))
        if s <= 0:
            return 500.0   # cannot sell anyway

        feats = pd.DataFrame(np.tile(buyer, (len(self.price_grid), 1)), columns=self.cols)
        feats['price_item'] = self.price_grid
        q = self.trained_model.predict_proba(feats)[:, 1]

        if int(inventories[self.opponent_number]) <= 0:
            p_beat = np.ones(len(self.price_grid))        # rival has nothing to sell
        elif len(self.opp_prices) == 0:
            p_beat = np.ones(len(self.price_grid))        # no information yet
        else:
            p_beat = self._prob_rival_above()

        delta = DELTA_SCALE * (self.V[t - 1, s] - self.V[t - 1, s - 1])
        score = q * p_beat * (self.price_grid - delta)
        return float(self.price_grid[int(np.argmax(score))])
