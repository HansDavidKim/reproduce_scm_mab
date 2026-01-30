import numpy as np

from mab_algorithms.bandit import Bandit
from mab_algorithms.bandit import ArmStats

class UCBBandit(Bandit):
    def __init__(self):
        super().__init__()
    
    def priority_fn(self, arm_stats: ArmStats) -> float:
        """
        Calculating upper confidence bound of reward per each arm.
        Theoretical Background - Hoeffding's Inequality
        """
        try:
            upper_bound = arm_stats.mean_reward + np.sqrt(np.log(self.t) / (2 * arm_stats.n_pulls))
        except:
            upper_bound = np.inf

        return upper_bound
        