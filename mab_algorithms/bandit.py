from abc import ABC, abstractmethod
import numpy as np

# Super-Class for MAB algorithms : UCB, KL-UCB, TS

class ArmStats:
    """Stores statistics/parameters for each arm."""
    def __init__(self, arm_idx: int):
        self.arm_idx = arm_idx
        self.n_pulls = 0          # Number of times this arm was pulled
        self.total_reward = 0.0   # Sum of rewards
        self.mean_reward = 0.0    # Empirical mean reward
        
    def update(self, reward: float):
        """Update statistics after pulling this arm."""
        self.n_pulls += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.n_pulls


# This bandit must be compatible with Structural Causal Bandits Paper
class Bandit(ABC):
    def __init__(self):
        super().__init__()
        self.arms = []
        self.arm_stats = {}  # Maps arm_idx -> ArmStats
        self.t = 0  # Total number of rounds

    '''
    @Argument
    arms : 
    - list of indexes of each arm
    - Global Arms are mapped to an integer
    '''
    def set_arms(self, arms: list):
        self.arms = arms
        # Initialize ArmStats for each arm
        self.arm_stats = {arm_idx: ArmStats(arm_idx) for arm_idx in arms}

    @abstractmethod
    def priority_fn(self, arm_stats: ArmStats) -> float:
        """Calculate priority score for an arm (higher = more likely to be selected).
        
        This should be implemented by subclasses (UCB, TS, etc.)
        """
        pass

    def select_arm(self) -> int:
        """Select the arm with highest priority using argmax."""
        if not self.arms:
            raise ValueError("No arms set. Call set_arms() first.")
        
        # Compute priorities for all arms and select the one with max priority
        priorities = [self.priority_fn(self.arm_stats[arm_idx]) for arm_idx in self.arms]
        best_idx = np.argmax(priorities)
        return self.arms[best_idx]

    def update(self, arm_idx: int, reward: float):
        """Update arm statistics after observing reward."""
        self.t += 1
        self.arm_stats[arm_idx].update(reward)