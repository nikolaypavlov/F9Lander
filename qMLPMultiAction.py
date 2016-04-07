from F9utils import RLAgent
from experienceReplay import ExperienceReplay
import cPickle as pickle
import glob
import os
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
N_BATCH = 1024
# Weight decay
DECAY = 1e-5
# Mini-batch size
BATCH_SIZE = 32
# Discount factor gamma
GAMMA = 0.99
# Epsilon-greedy policy parameters
EPS = 0.99
POWER = 0.1
# Log file path
LOG_FILE = "output.log"

# Wait till replay accumulate some experience than start learning
MIN_REPLAY_SIZE = 2048
SYNC_FIXED_MODEL = 2048

# Take snapshots
SNAPSHOT_EVERY = 5000
SNAPSHOT_PREFIX = 'snapshots/qmlp'

class QMLPMultiActionAgent(RLAgent):
    """Q-Learning agent with MLP function approximation"""
    def __init__(self, actions, featureExtractor, isTerminalState, featuresNum):
        self.actions = actions
        self.actionsNum = len(actions())
        self.featuresNum = featuresNum
        self.featureExtractor = featureExtractor
        self.isTerminalState = isTerminalState
        self.explorationProb = EPS
        self.explorationProb0_ = EPS
        self.numIters = 0
        self.total_reward = 0
        self.gamma = GAMMA
        self.log = file(LOG_FILE, 'a')
        self.replay = ExperienceReplay()
        self.model, self.train, self.predict = self._create_network()
        self.fixedModel, _, self.predictFixed = self._create_network()
        self.syncInterval = SYNC_FIXED_MODEL

        self._syncModel()

    def _build_network(self):
        l_in = InputLayer(shape=(None, self.featuresNum), name="input")
        l_forward_1 = batch_norm(DenseLayer(l_in, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, name="fc1"))
        l_out = batch_norm(DenseLayer(l_forward_1, num_units=self.actionsNum, nonlinearity=lasagne.nonlinearities.identity, name="out"))

        return l_out, l_in.input_var

    def _create_network(self):
        print("Building network ...")
        net, input_var = self._build_network()
        target_values = T.matrix('target_output')
        # mask = theano.shared(np.zeros((BATCH_SIZE, self.actionsNum)).astype(np.int32))
        # mask = T.set_subtensor(mask[np.arange(BATCH_SIZE), target_values.argmax(1)], 1)

        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(net)
        # new_network_output = network_output * mask
        # new_target_values = target_values * mask
        cost = squared_error(network_output, target_values).mean()

        # Add regularization penalty
        cost += regularize_network_params(net, l2) * DECAY

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(net, trainable=True)

        # Compute SGD updates for training
        updates = lasagne.updates.adadelta(cost, all_params)

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function([input_var, target_values], cost, updates=updates)
        predict = theano.function([input_var], lasagne.layers.get_output(net))

        return net, train, predict

    def _syncModel(self):
        if self.syncInterval is not None:
            all_param_values = lasagne.layers.get_all_param_values(self.model)
            lasagne.layers.set_all_param_values(self.fixedModel, all_param_values)

    def _getQOpt(self, state):
        pred = self.predict(self.featureExtractor(state).reshape(-1, self.featuresNum).astype(theano.config.floatX))
        self.log.write("Prediction %s\n" % pred)
        return (pred.max(), self.actions(state)[pred.argmax()], pred.argmax())

    def _getOptAction(self, state):
        return self._getQOpt(state)[1]

    # Epsilon-greedy policy
    def getAction(self, state):
        self.numIters += 1
        if self.replay.getSize() < MIN_REPLAY_SIZE or random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return self._getOptAction(state)

    def provideFeedback(self, state, action, reward, new_state):
        self.replay.append(state, action, reward, new_state) # Put new data to experience replay
        self.total_reward += reward

        if self.replay.getSize() >= MIN_REPLAY_SIZE:
            batch = self.replay.mini_batch(BATCH_SIZE)
            target = np.zeros((len(batch), self.actionsNum), dtype=np.float64)
            features = np.zeros((len(batch), self.featuresNum), dtype=np.float64)

            for i, sars in enumerate(batch):
                b_state, b_action, b_reward, b_new_state = sars
                features[i] = self.featureExtractor(b_state)

                if self.isTerminalState(b_new_state):
                    target[i].fill(b_reward)
                else:
                    pred = self.predictFixed(self.featureExtractor(b_new_state).reshape(-1, self.featuresNum).astype(theano.config.floatX))
                    target[i] = b_reward + self.gamma * pred

            _ = self.train(features.astype(theano.config.floatX), target.astype(theano.config.floatX))
            self.explorationProb = self.explorationProb0_ / pow(self.numIters - MIN_REPLAY_SIZE + 1, POWER)

            # self.log.write("Iteration: %s Score: %s Loss: %s Reward: %s Action %s Epsilon: %s\n" %\
            #                 (self.numIters, int(self.total_reward), loss, reward, action, self.explorationProb))

        if self.syncInterval is not None and self.numIters % self.syncInterval == 0:
            self.log.write("Syncing models\n")
            self._syncModel()

    def _save_snapshot(state, prefix):
        file_path = '_'.join([prefix, str(state['epoch'])])
        f = file(''.join([file_path, '.pkl']), 'wb')
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def _load_snapshot(prefix):
        file_path = glob.glob(''.join([prefix, "*_[0-9]*.pkl"]))
        file_path.sort(key=os.path.getctime)
        state = None
        if len(file_path):
            f = file(file_path[-1], 'rb')
            print "Loading snapshot", file_path[-1]
            state = pickle.load(f)
            f.close()

        return state
