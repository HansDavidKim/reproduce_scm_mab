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

    def get_arm_indices(self, intervention_sets: list = None) -> list:
        """Get arm indices for specified intervention sets.
        
        Args:
            intervention_sets: List of intervention sets (as set or frozenset).
                               If None, returns all arm indices.
        
        Returns:
            List of arm indices that belong to the specified intervention sets.
        """
        if intervention_sets is None:
            return list(range(len(self.arms)))
        
        # Convert to frozensets for comparison
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

    def allocate_weight(self):
        """Allocate random weights to each edge using sigmoid transformation.
        
        Each edge weight is: sigmoid(randn()) -> value in (0, 1)
        """
        assert self.G is not None, "Graph is None"
        
        for u, v in self.G.edges():
            raw_weight = np.random.randn()
            weight = self.sigmoid(raw_weight)
            self.G[u][v]['weight'] = weight

    def sample_node_values(self, interventions: dict = None, noise_scale: float = 0.1):
        """Sample node values following SCM with Bernoulli sampling.
        
        V ~ Bernoulli(sigmoid(sum(W_parent * parent_value) + U_v))
        Latent nodes generate continuous values, observed nodes are binary.
        
        Args:
            interventions: dict of {node_name: fixed_value} for do-interventions
            noise_scale: scale of the noise term U
            
        Returns:
            dict of {node_name: sampled_value (0 or 1 for observed, float for latent)}
        """
        assert self.G is not None, "Graph is None"
        
        interventions = interventions or {}
        values = {}
        
        # Process nodes in topological order (parents before children)
        for node in nx.topological_sort(self.G):
            is_latent = self.G.nodes[node].get('latent', False)
            
            if is_latent:
                # Latent node (U): generate continuous value from noise
                # Cannot be intervened on
                values[node] = np.random.randn() * noise_scale
            elif node in interventions:
                # do(X=x) intervention: fix the value
                values[node] = interventions[node]
            else:
                # Observed node: compute based on parents
                parents = list(self.G.predecessors(node))
                
                if not parents:
                    # Root node: sample from prior
                    noise = np.random.randn() * noise_scale
                    prob = self.sigmoid(noise)
                else:
                    # Child node: prob = sigmoid(sum(W * parent) + U)
                    parent_sum = sum(
                        self.G[parent][node]['weight'] * values[parent]
                        for parent in parents
                    )
                    noise = np.random.randn() * noise_scale
                    prob = self.sigmoid(parent_sum + noise)
                
                # Bernoulli sampling: sample 0 or 1 based on probability
                values[node] = np.random.binomial(1, prob)
        
        return values

if __name__ == '__main__':
    env = Environ()
    env.load_graph('chain.json')
    env.allocate_weight()
    env.show_graph()
