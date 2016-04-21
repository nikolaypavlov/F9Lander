import numpy as np
from blist import sortedset

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

class PrioritizedExperienceReplay():
    def __init__(self, alpha=0, min_usages=8, clean_at=1000):
        self.sum = 0
        self._alpha = alpha
        self.min_usages = min_usages
        self.clean_at = clean_at
        self.experience = sortedset(key=lambda x: x[0])

    def append(self, state, action, reward, next_state, count=0, td_err=None):
        if count < self.min_usages or self.getSize() < self.clean_at:
            priority = (abs(td_err) + np.finfo(np.float32).eps) ** self._alpha
            transition = (priority, (state, action, reward, next_state), count)
            self.experience.add(transition)
            self.sum += priority

    def mini_batch(self, size):
        space = np.linspace(0, self.sum, size + 1)
        ranges = zip(space, space[1:])
        batch = [None] * size
        for i, (left, right) in enumerate(ranges):
            start = self.experience.bisect_left((left, None, None))
            end = self.experience.bisect_right((right, None, None))
            sample_idx = (np.random.choice(np.arange(start, end), 1, replace=False))
            priority, (state, action, reward, next_state), count = self.experience.pop(sample_idx)
            self.sum -= priority
            batch[i] = ((state, action, reward, next_state), count + 1)

        return batch

    def getSize(self):
        return len(self.replay)
