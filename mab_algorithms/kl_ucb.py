import numpy as np
from mab_algorithms.bandit import Bandit, ArmStats

def kl_bernoulli(p, q):
    """
    Binary KL divergence for Bernoulli distributions.
    d(p, q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    """
    # Numerical stability
    p = min(max(p, 1e-12), 1 - 1e-12)
    q = min(max(q, 1e-12), 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_ucb_solve(p, rhs, precision=1e-7):
    """
    Solve d(p, q) = rhs for q in (p, 1) using binary search.
    Since d(p, q) is convex and increasing for q > p, we find the largest q.
    """
    if rhs <= 0:
        return p
    
    low = p
    high = 1.0
    
    # Binary search for q
    for _ in range(32): # Sufficient for high precision
        mid = (low + high) / 2
        if kl_bernoulli(p, mid) <= rhs:
            low = mid
        else:
            high = mid
            
    return low

class KLUCBBandit(Bandit):
    def __init__(self, c=0):
        super().__init__()
        self.c = c
    
    def priority_fn(self, arm_stats: ArmStats) -> float:
        """
        KL-UCB Priority Function.
        Theoretical Background - Kullback-Leibler Upper Confidence Bounds
        """
        if arm_stats.n_pulls == 0:
            return np.inf
        
        # log(t) + c*log(log(t))
        # Usually c=0 or c=3 is used in practice for exploration tuning
        num = np.log(max(self.t, 1))
        if self.c > 0 and self.t > 1:
            num += self.c * np.log(np.log(self.t))
            
        rhs = num / arm_stats.n_pulls
        
        return kl_ucb_solve(arm_stats.mean_reward, rhs)
