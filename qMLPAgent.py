import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import squared_error

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 512
# Number of training sequences in each batch
N_BATCH = 128
# Weight decay
DECAY = 0.0

class QMLPAgent:
    """Q-Learning agent with MLP function approximation"""
    def __init__(self, learning_rate, gamma, actions, featureExtractor, featuresNum, explorationProb=0.2, syncInterval=None):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.featuresNum = featuresNum
        self.actions = actions
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.explorationProb0_ = explorationProb
        self.numIters = 0
        self.syncInterval = syncInterval
        self.net, self.train, self.predict = self.create_network()

        # self._syncModel()

    def build_network(self):
        l_in = InputLayer(shape=(None, self.featuresNum), name="input")
        l_forward_1 = batch_norm(DenseLayer(l_in, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, name="fc1"))
        l_out = batch_norm(DenseLayer(l_forward_1, num_units=1, nonlinearity=lasagne.nonlinearities.identity, name="out"))

        return l_out, l_in.input_var

    def create_network(self):
        print("Building network ...")
        net, input_var = self.build_network()
        target_values = T.matrix('target_output')

        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(net)
        cost = squared_error(network_output, target_values).mean()
        # cost += regularize_network_params(net, l2) * DECAY

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(net, trainable=True)

        # Compute SGD updates for training
        updates = lasagne.updates.adadelta(cost, all_params)

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function([input_var, target_values], cost, updates=updates)
        predict = theano.function([input_var], network_output)

        return net, train, predict

    def _syncModel(self):
        pass
        # if self.syncInterval is not None:
        #     self.fixedModel = SGDRegressor(**self.model.get_params())
        #     self.fixedModel.coef_ = np.copy(self.model.coef_)
        #     self.fixedModel.intercept_ = np.copy(self.model.intercept_)

    def getQOpt(self, state):
        features = np.zeros((len(self.actions(state)), self.featuresNum))
        for i, action in enumerate(self.actions(state)):
            features[i] = self.featureExtractor(state, action)
        pred = self.predict(features.astype(theano.config.floatX))
        return (pred.max(), self.actions(state)[pred.argmax()])

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
        target = np.zeros((len(batch), 1), dtype=np.float64)
        features = np.zeros((len(batch), self.featuresNum), dtype=np.float64)
        for i, sars in enumerate(batch):
            state, action, reward, new_state = sars
            maxQ, _ = self.getQOpt(new_state)

            target[i] = reward + self.gamma * maxQ
            features[i] = self.featureExtractor(state, action)

        loss = self.train(features.astype(theano.config.floatX), target.astype(theano.config.floatX))
        self.explorationProb = self.explorationProb0_ / pow(self.numIters, 0.05)

        return loss
