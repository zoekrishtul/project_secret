"""Run from the project repository root after training xgb_model.pkl.

Checks the Part 2 agent interface without accessing any hidden valuations.
Run: python smoke_test_part2.py
"""
from agents.agent_part2 import Agent
import numpy as np

for agent_number in (0, 1):
    agent = Agent(agent_number, {'project_part': 2,
                                 'inventory_limit': {'min': 7, 'max': 20},
                                 'inventory_replenish': 20})
    for step in range(40):
        buyer = np.array([float(step % 2), 5.0 + step % 8, 30.0 + step % 21])
        # Price in `last_sale` is the previous round's price, not the current price.
        last_sale = (np.nan, [np.nan, np.nan]) if step == 0 else (0, [60., 40.])
        stocks = [15, 15]
        remaining = 20 - step % 20
        obs = (buyer, last_sale, [0., 0.], stocks, remaining)
        price = agent.action(obs)
        assert isinstance(price, (float, int)), type(price)
        assert np.isfinite(price) and price >= 0, price
    print(f'Agent position {agent_number}: PASS')
print('Basic interface checks passed (not a revenue benchmark).')
