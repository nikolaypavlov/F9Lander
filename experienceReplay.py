import numpy as np

class ExperienceReplay:
    """Stores all trajectories during the life of the agent"""

    def __init__(self):
        self.experience = []

    def append(self, state, action, reward, next_state):
        self.experience.append((state, action, reward, next_state))

    def mini_batch(self, size):
        idx = np.random.choice(range(len(self.experience)), size)
        return [self.experience[n] for n in idx]

    def getSize(self):
        return len(self.experience)
