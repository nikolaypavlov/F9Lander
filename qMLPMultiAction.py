from F9utils import RLAgent
from F9utils import Snapshot
from experienceReplay import ExperienceReplay
import numpy as np
import random
import logging

from sklearn.preprocessing import StandardScaler
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm
from lasagne.regularization import regularize_network_params, l2
from lasagne.objectives import squared_error

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 512
# Weight decay
DECAY = 1.0e-6
# Mini-batch size
BATCH_SIZE = 16
# Discount factor gamma
GAMMA = 0.99
# Epsilon-greedy policy parameters
EPS0 = 1.0
EPS = 0.1
# Synchronise target model afterwhile
SYNC_TARGET_MODEL = 5000
# Wait till replay accumulate some experience than start learning
MIN_REPLAY_SIZE = 1000
# Log file path
LOG_FILE = "output.log"
# Take snapshots
SNAPSHOT_EVERY = 10000
SNAPSHOT_PREFIX = 'snapshots/qmlp'

class QMLPMultiActionAgent(RLAgent):
    """Q-Learning agent with MLP function approximation"""
    def __init__(self, actions, featureExtractor, isTerminalState, featuresNum, max_iters):
        self.actions = actions
        self.actionsNum = len(actions())
        self.featuresNum = featuresNum
        self.featureExtractor = featureExtractor
        self.isTerminalState = isTerminalState
        self.explorationProb = EPS0
        self.numIters = 0
        self.max_iters = max_iters
        self.start_learning = False
        self.total_reward = 0
        self.gamma = GAMMA
        self.replay = ExperienceReplay(clean_at=MIN_REPLAY_SIZE)

        logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)
        self.snapshot = Snapshot(SNAPSHOT_PREFIX)

        self.model, self.train, self.predict = self._create_network()
        self.targetModel, self.predictTarget = self._create_fixed_network()
        self._syncModel()
        self._load_snapshot()

    def _build_network(self):
        l_in = InputLayer(shape=(None, self.featuresNum), name="input")
        l_forward_1 = batch_norm(DenseLayer(l_in, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, name="fc1"))
        l_forward_2 = batch_norm(DenseLayer(l_forward_1, num_units=N_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify, name="fc2"))
        l_out = DenseLayer(l_forward_2, num_units=self.actionsNum, nonlinearity=lasagne.nonlinearities.identity, name="out")

        return l_out, l_in.input_var

    def _create_fixed_network(self):
        print("Building network with fixed weights...")
        net, input_var = self._build_network()

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        predict = theano.function([input_var], lasagne.layers.get_output(net))

        return net, predict

    def _create_network(self):
        print("Building network ...")
        net, input_var = self._build_network()
        target_values = T.matrix('target_output')
        maxQ_idx = target_values.argmax(1)

        # Create masks
        mask = theano.shared(np.ones((BATCH_SIZE, self.actionsNum)).astype(np.int32))
        maxQ_mask = theano.shared(np.zeros((BATCH_SIZE, self.actionsNum)).astype(np.int32))
        mask = T.set_subtensor(mask[np.arange(BATCH_SIZE), maxQ_idx], 0)
        maxQ_mask = T.set_subtensor(maxQ_mask[np.arange(BATCH_SIZE), maxQ_idx], 1)

        # lasagne.layers.get_output produces a variable for the output of the net
        network_output = lasagne.layers.get_output(net)
        new_target_values = target_values * maxQ_mask + network_output * mask

        cost = squared_error(network_output, new_target_values).mean()

        # Add regularization penalty
        cost += regularize_network_params(net, l2) * DECAY

        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(net, trainable=True)

        # Compute SGD updates for training
        updates = lasagne.updates.adadelta(cost, all_params)

        # Theano functions for training and computing cost
        print("Compiling functions ...")
        train = theano.function([input_var, target_values], [cost, new_target_values, network_output, maxQ_idx], updates=updates)
        predict = theano.function([input_var], lasagne.layers.get_output(net))

        return net, train, predict

    def _syncModel(self):
        net_params = lasagne.layers.get_all_param_values(self.model)
        fixed_net_param = lasagne.layers.get_all_param_values(self.targetModel)
        diff = np.mean([np.mean(layer - fixed_net_param[i]) for i, layer in enumerate(net_params)])
        logging.debug("Syncing models, average weight diff %s" % diff)
        lasagne.layers.set_all_param_values(self.targetModel, net_params)

    def _getOptAction(self, state):
        pred = self.predict(self.featureExtractor(state).reshape(-1, self.featuresNum).astype(theano.config.floatX))
        return self.actions(state)[pred.argmax()]

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
        if self.replay.getSize() > MIN_REPLAY_SIZE:
            self.start_learning = True

        if self.start_learning:
            batch = self.replay.mini_batch(BATCH_SIZE)
            target = np.zeros((BATCH_SIZE, self.actionsNum), dtype=np.float64)
            features = np.zeros((BATCH_SIZE, self.featuresNum), dtype=np.float64)

            for i, sars in enumerate(batch):
                b_state, b_action, b_reward, b_new_state = sars
                features[i] = self.featureExtractor(b_state)

                if self.isTerminalState(b_new_state):
                    target[i].fill(b_reward - 1)
                    target[i][np.random.randint(0, self.actionsNum)] = b_reward
                else:
                    # Double Q-learning target
                    act = np.argmax(self.predict(self.featureExtractor(b_new_state).reshape(-1, self.featuresNum).astype(theano.config.floatX)), 1)
                    q_vals = self.predictTarget(self.featureExtractor(b_new_state).reshape(-1, self.featuresNum).astype(theano.config.floatX))
                    t = b_reward + self.gamma * np.ravel(q_vals)
                    target[i] = np.tile(t[act] - 1, self.actionsNum)
                    target[i][act] = t[act]

            assert(target.shape == (BATCH_SIZE, self.actionsNum))
            loss, target_val, net_out, maxQ_idx = self.train(features.astype(theano.config.floatX), target.astype(theano.config.floatX))
            self.explorationProb -= (EPS0 - EPS) / (self.max_iters - MIN_REPLAY_SIZE)
            assert(np.sum(np.invert(np.isclose(target_val, net_out))) <= BATCH_SIZE)

            logging.info("Iteration: %s Replay: %s TD-err: %s Reward: %s Action %s Epsilon: %s" %\
                            (self.numIters, self.replay.getSize(), loss, reward, action, self.explorationProb))
            logging.debug("maxQ_idx %s" % maxQ_idx)

            if self.numIters % SYNC_TARGET_MODEL == 0:
                self._syncModel()

            if self.numIters % SNAPSHOT_EVERY == 0:
                self._save_snapshot()

    def _save_snapshot(self):
        snap = {"iter": self.numIters,
                "epsilon": self.explorationProb,
                "target_params": lasagne.layers.get_all_param_values(self.targetModel),
                "params": lasagne.layers.get_all_param_values(self.model),
                "replay": self.replay,
                "start_learning": self.start_learning}
        self.snapshot.save(snap, self.numIters)

    def _load_snapshot(self):
        snap = self.snapshot.load()
        if snap is not None:
            lasagne.layers.set_all_param_values(self.model, snap['params'])
            lasagne.layers.set_all_param_values(self.targetModel, snap['target_params'])
            self.numIters = snap["iter"]
            self.explorationProb = snap["epsilon"]
            self.replay = snap["replay"]
            self.start_learning = snap["start_learning"]
