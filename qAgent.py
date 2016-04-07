import numpy as np
import random
from F9utils import RLAgent
from experienceReplay import ExperienceReplay

# Mini-batch size
BATCH_SIZE = 32
# Discount factor gamma
GAMMA = 0.99
# Epsilon-greedy policy parameters
EPS = 0.99
POWER = 0.1
# Log file path
LOG_FILE = "output.log"
# Learnin rate
STEP_SIZE = 1e-4
# Wait till replay accumulate some experience than start learning
MIN_REPLAY_SIZE = 1024
SYNC_FIXED_MODEL = 2048
# Take snapshots
SNAPSHOT_EVERY = 5000
SNAPSHOT_PREFIX = 'snapshots/qmlp'

class QAgent(RLAgent):
    """Least squares time difference Q-Learning"""
    def __init__(self, actions, featureExtractor, isTerminalState, featuresNum):
        self.alpha = STEP_SIZE
        self.gamma = GAMMA
        self.featuresNum = featuresNum
        self.w = np.random.randn(self.featuresNum)
        self.actions = actions
        self.featureExtractor = featureExtractor
        self.explorationProb0_ = EPS
        self.explorationProb = EPS
        self.numIters = 0
        self.total_reward = 0
        self.log = file(LOG_FILE, 'a')
        self.replay = ExperienceReplay()

    def _getQ(self, state, action):
        return np.dot(self.featureExtractor(state, action), self.w)

    def _getQOpt(self, state):
        actions = self.actions(state)
        q_vals = [(self._getQ(state, action), action) for action in actions]
        max_q = max(q_vals)[0]
        best_actions = [(q, act) for q, act in q_vals if q == max_q]
        return random.choice(best_actions)

    def getOptAction(self, state):
        return self._getQOpt(state)[1]

    # Epsilon-greedy policy
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return self.getOptAction(state)

    def provideFeedback(self, state, action, reward, new_state):
        self.replay.append(state, action, reward, new_state) # Put new data to experience replay
        self.total_reward += reward

        if self.replay.getSize() >= MIN_REPLAY_SIZE:
            batch = self.replay.mini_batch(BATCH_SIZE)

            loss = 0.0
            deltaW = np.zeros(self.featuresNum, dtype=np.float64)
            for sars in batch:
                b_state, b_action, b_reward, b_new_state = sars
                maxQ, targetAction = self._getQOpt(b_new_state)
                target = b_reward + self.gamma * maxQ
                prediction = self._getQ(b_state, b_action)
                err = target - prediction
                deltaW += np.array([err * self.featureExtractor(b_state, b_action)[i] for i in range(self.featuresNum)])
                loss += err ** 2

            self.w += self.alpha * deltaW / BATCH_SIZE
            loss = loss / BATCH_SIZE
            self.explorationProb = self.explorationProb0_ / pow(self.numIters + 1, POWER)

            self.log.write("Iteration: %s Loss: %s Reward: %s Action %s Epsilon: %s\n" %\
                            (self.numIters, loss, reward, action, self.explorationProb))
