# -------------------------------------------------- #
# --------------------_F9_Lander_------------------- #
# ----------------------CLIENT---------------------- #
# -------------------------------------------------- #

import numpy as np
import cPickle as pickle
import glob
import os
from experienceReplay import *
from qSGDAgent import QSGDAgent
from F9utils import F9GameClient

# for delay in debug launch
import time

# -------------------------------------------------- #

FEATURES_NUM = 7
BATCH_SIZE = 32
MIN_REPLAY_SIZE = 2048
STEP_SIZE = 1.0e-7
GAMMA = 0.99
EPS = 0.99
SNAPSHOT_EVERY = 5000
SNAPSHOT_PREFIX = 'snapshots/qsgd'
SYNC_FIXED_MODEL = 2048

def featureExtractor(state, action):
    agent, platform, _ = state
    e1, e2, e3, _ = action
    features = np.array([agent['dist'],
                         agent['angle'],
                         agent['contact'],
                         agent['wind'],
                         e1,
                         e2,
                         e3],
                         dtype=np.float64)
    assert(len(features) == FEATURES_NUM)
    return features

def save_snapshot(state, prefix):
    file_path = '_'.join([prefix, str(state['epoch'])])
    f = file(''.join([file_path, '.pkl']), 'wb')
    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def load_snapshot(prefix):
    file_path = glob.glob(''.join([prefix, "*_[0-9]*.pkl"]))
    file_path.sort(key=os.path.getctime)
    state = None
    if len(file_path):
        f = file(file_path[-1], 'rb')
        print "Loading snapshot", file_path[-1]
        state = pickle.load(f)
        f.close()
    return state

def solve():
    # Setup agent and experience replay
    replay = ExperienceReplay()
    client = F9GameClient()
    ai = QSGDAgent(STEP_SIZE, GAMMA, client.actions, featureExtractor, FEATURES_NUM, EPS, SYNC_FIXED_MODEL)
    state = client.curState
    log = file("output.log", 'a')
    learn = False

    # Load previous state from snapshot if any
    snapshot = load_snapshot(SNAPSHOT_PREFIX)
    if snapshot is not None:
        ai.numIters = snapshot["epoch"]
        ai.learning_rate = snapshot["learning_rate"]
        ai.gamma = snapshot["gamma"]
        ai.w = snapshot["weights"]
        ai.explorationProb = snapshot["explorationProb"]
        client.totalScore = snapshot["totalScore"]

    if not learn:
        ai.explorationProb = 0
        ai.numIters = 0
        client.totalScore = 0

    while True:
        action = ai.getAction(state)
        client.doAction(action)
        new_state = client.curState
        reward = client.getReward(new_state)
        replay.append(state, action, reward, new_state)

        if reward >= 0.0:
            agent, _, system = new_state
            print "Agent state %s\n System state %s\n Reward %s\n Experience replay size %s\n" %\
                    (agent, system, reward, len(replay.experience))

        if learn and replay.getSize() >= MIN_REPLAY_SIZE:
            batch = replay.mini_batch(BATCH_SIZE)
            loss = ai.updateWeights(batch)
            log.write("Iteration: %s Score: %s Loss: %s Reward: %s Action %s Epsilon: %s Learning rate %s\n" %\
                      (ai.numIters, int(client.totalScore), loss, reward, action, ai.explorationProb, ai.learning_rate))
        elif not learn:
            log.write("Iteration: %s Score: %s Reward: %s Action %s\n" % (ai.numIters, int(client.totalScore), reward, action))

        # system_state["flight_status"] | "none", "landed", "destroyed"
        # "none" means that we don't know, whether we landed or destroyed
        if client.isTerminalState(new_state):
            client.reset_game()
            state = client.curState
        else:
            state = new_state

        # Create snapshot
        if ai.numIters % SNAPSHOT_EVERY == 0:
            save_snapshot({"epoch": ai.numIters,
                           "learning_rate": ai.learning_rate,
                           "gamma": ai.gamma,
                           "weights": ai.w,
                           "explorationProb": ai.explorationProb,
                           "totalScore": client.totalScore}, SNAPSHOT_PREFIX)

if __name__ == "__main__":
    solve()

# -------------------------------------------------- #
# --------------- you have landed ------------------ #
# -------------------------------------------------- #
