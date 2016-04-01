# -------------------------------------------------- #
# --------------------_F9_Lander_------------------- #
# ----------------------CLIENT---------------------- #
# -------------------------------------------------- #

import numpy as np
from qMLPAgent import QMLPAgent
from F9utils import F9GameClient

# for delay in debug launch
import time

# -------------------------------------------------- #

FEATURES_NUM = 14

def featureExtractor(state, action):
    agent, platform, _ = state
    e1, e2, e3, _ = action
    features = np.array([agent['dist'],
                         agent['angle'],
                         agent['vx'],
                         agent['vy'],
                         agent['px'],
                         agent['py'],
                         agent['contact'],
                         agent['wind'],
                         agent['fuel'],
                         platform['px'],
                         platform['py'],
                         e1,
                         e2,
                         e3],
                         dtype=np.float64)
    assert(len(features) == FEATURES_NUM)
    return features

def solve():
    # Setup agent and experience replay
    client = F9GameClient()
    ai = QMLPAgent(client.actions, featureExtractor, FEATURES_NUM)
    state = client.curState

    while True:
        action = ai.getAction(state)
        client.doAction(action)
        new_state = client.curState
        reward = client.getReward(new_state)
        ai.provideFeedback(state, action, reward, new_state)

        # system_state["flight_status"] | "none", "landed", "destroyed"
        # "none" means that we don't know, whether we landed or destroyed
        if client.isTerminalState(new_state):
            client.reset_game()
            state = client.curState
        else:
            state = new_state

if __name__ == "__main__":
    solve()

# -------------------------------------------------- #
# --------------- you have landed ------------------ #
# -------------------------------------------------- #
