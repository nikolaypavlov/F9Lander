# -------------------------------------------------- #
# --------------------_F9_Lander_------------------- #
# ----------------------CLIENT---------------------- #
# -------------------------------------------------- #

import numpy as np
from qMLPMultiAction import QMLPMultiActionAgent
# from qAgent import QAgent
from F9utils import F9GameClient

# for delay in debug launch
import time

# -------------------------------------------------- #

FEATURES_NUM = 7
MAX_ITERS = 25000

def featureExtractor(state, action=None):
    agent, platform, _ = state
    # e1, e2, e3, _ = action
    features = np.array([agent['dist'],
                         agent['angle'],
                         agent['vx'],
                         agent['vy'],
                         agent['contact'],
                         agent['wind'],
                         agent['fuel']],
                         # e1,
                         # e2,
                         # e3],
                         dtype=np.float64)
    assert(len(features) == FEATURES_NUM)
    return features

def solve():
    # Setup agent and experience replay
    client = F9GameClient()
    ai = QMLPMultiActionAgent(client.actions, featureExtractor, client.isTerminalState, FEATURES_NUM)
    # ai = QAgent(client.actions, featureExtractor, client.isTerminalState, FEATURES_NUM)
    state = client.curState
    scores = []

    while ai.numIters <= MAX_ITERS:
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
            scores.append(ai.total_reward)
            ai.total_reward = 0
        else:
            state = new_state

    print "Average game score: ", np.mean(scores)

if __name__ == "__main__":
    solve()

# -------------------------------------------------- #
# --------------- you have landed ------------------ #
# -------------------------------------------------- #
