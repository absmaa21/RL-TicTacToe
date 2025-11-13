import random, pickle
from collections import defaultdict
from typing import Tuple, List

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q[state_key][action] = value
        self.Q = defaultdict(lambda: defaultdict(float))

    def get_action(self, state_key: Tuple, legal: List[int], training=True) -> int:
        # legal should be list of ints
        if training and (not legal):
            raise ValueError("No legal actions provided")
        if training and random.random() < self.epsilon:
            return random.choice(legal)
        # pick best action among legal ones
        qvals = [(self.Q[state_key][a], a) for a in legal]
        max_q = max(qvals, key=lambda x: x[0])[0]
        best = [a for q,a in qvals if q == max_q]
        return random.choice(best)

    def update(self, state_key, action, reward, next_state_key, next_legal, done):
        q_old = self.Q[state_key][action]
        target = reward
        if not done:
            # best next-state q
            best_next = 0 if not next_legal else max(self.Q[next_state_key][a] for a in next_legal)
            target += self.gamma * best_next
        self.Q[state_key][action] += self.alpha * (target - q_old)

    def save(self, path):
        # convert nested defaultdicts to normal dicts for pickle
        raw = {s: dict(a) for s,a in self.Q.items()}
        with open(path, "wb") as f:
            pickle.dump(raw, f)

    def load(self, path):
        with open(path, "rb") as f:
            raw = pickle.load(f)
        # restore to defaultdict structure
        dd = defaultdict(lambda: defaultdict(float))
        for s, amap in raw.items():
            inner = defaultdict(float)
            inner.update(amap)
            dd[s] = inner
        self.Q = dd
