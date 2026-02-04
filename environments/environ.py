from abc import ABC, abstractmethod
from pathlib import Path
from itertools import combinations, product

import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent.parent

class Environ(ABC):
    def __init__(self, seed: int=42):
        super().__init__()
        self.seed: int = seed

        # Causal Diagram : will load structure from json file
        # Common Scenario : Mainpulable Variables can only have 0 or 1
        
        # Manipulable Variables can only have 0 or 1 (Binary Variables)

        np.random.seed(seed)

        # Intervention Set
        self.IS = None
        self.G = None

    def load_graph(self, file_name: str):
        """Load causal graph structure from a JSON file.
        
        Args:
            file_name: JSON file name in the benchmark directory
        """
        json_path = base_dir / "benchmark" / file_name
        
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.G = nx.DiGraph()
        self.latent_nodes = []  # Track latent confounder nodes

        # Add nodes from vars
        for var in data.get("vars", []):
            self.G.add_node(var, latent=False)
        
        # Add directed edges
        for edge in data.get("direct", []):
            self.G.add_edge(edge[0], edge[1])
        
        # Add bidirected edges as explicit latent confounders
        # X ↔ Z becomes: U_XZ → X, U_XZ → Z
        for i, edge in enumerate(data.get("bidirect", [])):
            latent_name = f"U_{edge[0]}_{edge[1]}"
            self.G.add_node(latent_name, latent=True)
            self.G.add_edge(latent_name, edge[0])
            self.G.add_edge(latent_name, edge[1])
            self.latent_nodes.append(latent_name)
        
        # Generate all possible intervention sets (excluding Y and latent nodes)
        # Power set of non-Y, non-latent variables: {}, {X}, {Z}, {X,Z}, ...
        non_y_vars = [var for var in data.get("vars", []) if var != "Y"]
        self.IS = []
        for r in range(len(non_y_vars) + 1):
            for subset in combinations(non_y_vars, r):
                self.IS.append(frozenset(subset))
        
        # Generate concrete interventions for each intervention set
        # Maps each set to list of possible interventions: {X} -> [{X:0}, {X:1}]
        self.arms = []
        self.arm_to_set = {}  # Maps arm index to its intervention set
        for intervention_set in self.IS:
            if len(intervention_set) == 0:
                # Empty set: observational (no intervention)
                self.arm_to_set[len(self.arms)] = intervention_set
                self.arms.append({})
            else:
                # Generate all 0/1 combinations for variables in the set
                vars_list = sorted(intervention_set)
                for values in product([0, 1], repeat=len(vars_list)):
                    intervention = dict(zip(vars_list, values))
                    self.arm_to_set[len(self.arms)] = intervention_set
                    self.arms.append(intervention)

    def get_arm_indices(self, intervention_sets: any = None) -> list:
        """Get arm indices for specified intervention sets.
        
        Args:
            intervention_sets: Can be:
                - None: returns all arm indices.
                - set/frozenset of strings: e.g., {'X', 'Z'} (single set)
                - iterable of sets: e.g., [{X}, {Z}] or {frozenset({X}), frozenset({Z})} (POMIS result)
        
        Returns:
            List of arm indices that belong to the specified intervention sets.
        """
        if intervention_sets is None:
            return list(range(len(self.arms)))
        
        # Determine if we are handling a single set of variables or a collection of sets
        if isinstance(intervention_sets, (set, frozenset)):
            if not intervention_sets:
                # Empty set (observational)
                target_sets = {frozenset()}
            else:
                first_elem = next(iter(intervention_sets))
                if isinstance(first_elem, str):
                    # Single set of variables: {'X', 'Z'} -> {frozenset({'X', 'Z'})}
                    target_sets = {frozenset(intervention_sets)}
                else:
                    # Collection of sets: {frozenset({'X'}), frozenset({'Z'})}
                    target_sets = {frozenset(s) for s in intervention_sets}
        else:
            # Assume it's an iterable of sets (list, tuple, etc.)
            target_sets = {frozenset(s) for s in intervention_sets}
        
        return [
            idx for idx, arm_set in self.arm_to_set.items()
            if arm_set in target_sets
        ]

    def show_graph(self):
        assert self.G != None, "Graph is None"
        pos = nx.spring_layout(self.G, seed=42)

        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color='lightblue',
            arrows=True
        )

        plt.show()

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _prepare_sampling(self):
        """Precompute and cache graph structure for fast sampling."""
        self.topo_nodes = list(nx.topological_sort(self.G))
        self.node_to_idx = {node: i for i, node in enumerate(self.topo_nodes)}
        self.num_nodes = len(self.topo_nodes)
        
        # Cache weights and parent indices
        self.node_metadata = []
        for node in self.topo_nodes:
            is_latent = self.G.nodes[node].get('latent', False)
            parents = list(self.G.predecessors(node))
            parent_indices = [self.node_to_idx[p] for p in parents]
            weights = np.array([self.G[p][node]['weight'] for p in parents])
            
            self.node_metadata.append({
                'is_latent': is_latent,
                'parent_indices': parent_indices,
                'weights': weights,
                'name': node
            })

    def allocate_weight(self):
        """Allocate random weights to each edge using sigmoid transformation."""
        assert self.G is not None, "Graph is None"
        
        for u, v in self.G.edges():
            raw_weight = np.random.randn()
            weight = self.sigmoid(raw_weight)
            self.G[u][v]['weight'] = weight
        
        # Prepare sampling cache after weights are allocated
        self._prepare_sampling()

    def sample_node_values_vectorized(self, interventions: dict = None, n_samples: int = 1, noise_scale: float = 0.1, noise_matrix: np.ndarray = None):
        """Sample node values using NumPy vectorization for massive speedup.
        
        Args:
            interventions: dict of {node_name: fixed_value}
            n_samples: number of samples
            noise_scale: scale of noise
            noise_matrix: Pre-generated noise of shape (n_samples, num_nodes). 
                         If provided, greatly speeds up multi-arm evaluations.
        
        Returns:
            NumPy array of shape (n_samples, num_nodes)
        """
        interventions = interventions or {}
        # Shape: (n_samples, num_nodes)
        values = np.zeros((n_samples, self.num_nodes))
        
        # Use provided noise or generate new
        if noise_matrix is None:
            noise_matrix = np.random.randn(n_samples, self.num_nodes)
        
        for i, meta in enumerate(self.node_metadata):
            node_name = meta['name']
            
            if meta['is_latent']:
                # Latent node (U): continuous noise
                values[:, i] = noise_matrix[:, i] * noise_scale
            elif node_name in interventions:
                # do-intervention: fix value
                values[:, i] = interventions[node_name]
            else:
                # Observed node: sigmoid(sum(W * parent) + noise)
                node_noise = noise_matrix[:, i] * noise_scale
                if not meta['parent_indices']:
                    prob = self.sigmoid(node_noise)
                else:
                    # Select parent columns and multiply by weights
                    parent_values = values[:, meta['parent_indices']]
                    parent_sum = np.dot(parent_values, meta['weights'])
                    prob = self.sigmoid(parent_sum + node_noise)
                
                values[:, i] = np.random.binomial(1, prob)
        
        return values

    def sample_node_values(self, interventions: dict = None, noise_scale: float = 0.1):
        """Legacy single-sample wrapper for compatibility."""
        res = self.sample_node_values_vectorized(interventions, 1, noise_scale)
        return {self.topo_nodes[i]: res[0, i] for i in range(self.num_nodes)}

    def get_optimal_expected_reward(self, n_samples: int = 1000):
        """Estimate optimal reward using vectorized Monte Carlo sampling and CRN optimization.
        
        Args:
            n_samples: Number of samples per arm
        """
        from tqdm import tqdm
        self.expected_rewards = {}
        y_idx = self.node_to_idx['Y']
        
        # CRN (Common Random Numbers) Optimization:
        # Pre-generate noise once and reuse across all arms to save billions of random calls
        # and reduce variance in comparison between arms.
        noise_matrix = np.random.randn(n_samples, self.num_nodes)
        
        for arm_idx, intervention in enumerate(tqdm(self.arms, desc="Calculating Oracle Rewards", leave=False)):
            samples = self.sample_node_values_vectorized(
                interventions=intervention, 
                n_samples=n_samples,
                noise_matrix=noise_matrix
            )
            self.expected_rewards[arm_idx] = np.mean(samples[:, y_idx])
        
        self.optimal_arm_idx = max(self.expected_rewards, key=self.expected_rewards.get)
        self.max_expected_reward = self.expected_rewards[self.optimal_arm_idx]
        
        return self.max_expected_reward

if __name__ == '__main__':
    env = Environ()
    env.load_graph('confounder.json')
    env.allocate_weight()
    env.show_graph()
