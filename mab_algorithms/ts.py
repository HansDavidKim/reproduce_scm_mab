import numpy as np

from mab_algorithms.bandit import Bandit
from mab_algorithms.bandit import ArmStats

class TSBandit(Bandit):
    def __init__(self):
        super().__init__()

    def priority_fn(self, arm_stats: ArmStats) -> float:
        """
        Sample each arm's reward from Beta distribution estimated per each arm.
        Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
        """
        alpha = arm_stats.total_reward + 1  # successes + prior
        beta = (arm_stats.n_pulls - arm_stats.total_reward) + 1  # failures + prior
        return np.random.beta(alpha, beta)
        