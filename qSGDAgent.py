import numpy as np
import random
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class QSGDAgent:
    """Q-Learning agent with SGD linear function approximation"""
    def __init__(self, learning_rate, gamma, actions, featureExtractor, featuresNum, explorationProb=0.2, syncInterval=None):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.featuresNum = featuresNum
        self.w = np.random.randn(self.featuresNum).astype(np.float64)
        self.actions = actions
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.explorationProb0_ = explorationProb
        self.numIters = 0
        self.model = SGDRegressor(loss='squared_loss', eta0=self.learning_rate, random_state=222, penalty='l2', n_iter=1)
        self.model.coef_ = self.w
        self.model.intercept_ = np.zeros(1, dtype=np.float64)
        self.syncInterval = syncInterval

        self._syncModel()

    def _syncModel(self):
        if self.syncInterval is not None:
            self.fixedModel = SGDRegressor(**self.model.get_params())
            self.fixedModel.coef_ = np.copy(self.model.coef_)
            self.fixedModel.intercept_ = np.copy(self.model.intercept_)

    def getQ(self, state, action, fixed=False):
        if fixed:
            return self.fixedModel.predict(self.featureExtractor(state, action).reshape(1, -1))
        else:
            return self.model.predict(self.featureExtractor(state, action).reshape(1, -1))

    def getQOpt(self, state, fixed=False):
        actions = self.actions(state)
        q_vals = [(self.getQ(state, action, fixed), action) for action in actions]
        max_q = max(q_vals)[0]
        best_actions = [(q, act) for q, act in q_vals if q == max_q]
        return random.choice(best_actions)

    def getOptAction(self, state):
        return self.getQOpt(state)[1]

    # Epsilon-greedy policy
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return self.getOptAction(state)

    def updateWeights(self, batch):
        target = np.zeros(len(batch), dtype=np.float64)
        features = np.zeros((len(batch), self.featuresNum), dtype=np.float64)
        for i, sars in enumerate(batch):
            state, action, reward, new_state = sars
            if self.syncInterval is not None:
                if self.numIters % self.syncInterval == 0:
                    self._syncModel()
                maxQ, _ = self.getQOpt(new_state, fixed=True)
            else:
                maxQ, _ = self.getQOpt(new_state)

            target[i] = reward + self.gamma * maxQ
            features[i] = self.featureExtractor(state, action)

        pred = self.model.predict(features)
        self.model.partial_fit(features, target)
        self.w = self.model.coef_
        self.learning_rate = self.model.eta0 / pow(self.model.t_, self.model.power_t)
        self.explorationProb = self.explorationProb0_ / pow(self.numIters, 0.05)

        return mean_squared_error(target, pred)
