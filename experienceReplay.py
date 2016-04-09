import numpy as np

class ExperienceReplay:
    """Stores all trajectories during the life of the agent"""

    def __init__(self, min_usages=8, clean_at=100):
        self.experience = []
        self.min_usages = min_usages
        self.clean_at = clean_at

    def append(self, state, action, reward, next_state):
        self.experience.append((state, action, reward, next_state, 0))

    def mini_batch(self, size):
        idx = np.random.choice(range(len(self.experience)), size, replace=False)
        batch = [None] * size
        removelist = []
        for i, n in enumerate(idx):
            state, action, reward, new_action, num = self.experience[n]
            batch[i] = (state, action, reward, new_action)
            num += 1
            if num >= self.min_usages:
                removelist.append(n)

            self.experience[n] = (state, action, reward, new_action, num)

        if self.getSize() >= self.clean_at:
            for i in sorted(removelist, reverse=True):
                del self.experience[i]

        return batch

    def getSize(self):
        return len(self.experience)
