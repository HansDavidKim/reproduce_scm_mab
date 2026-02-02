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
        if arm_stats.n_pulls == 0:
            return np.inf
            
        # Use max(self.t, 1) to avoid log(0) at the start
        upper_bound = arm_stats.mean_reward + np.sqrt(np.log(max(self.t, 1)) / (2 * arm_stats.n_pulls))

        return upper_bound
        