import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import networkx as nx
from causal_engine.muct import MUCT, IB
from copy import deepcopy

from environments.environ import Environ

def intervene(graph: nx.DiGraph, treatment: str):
    incoming_edges = list(graph.in_edges(treatment))
    graph.remove_edges_from(incoming_edges)

    return graph

def POMIS(graph: nx.DiGraph):
    muct = MUCT(graph)
    ib = IB(graph, muct)

    pomis_set = set()

    H = deepcopy(graph)
    # Prune all incoming edges of IB Nodes
    for node in ib:
        intervene(H, node)

    order = list(reversed(list(nx.topological_sort(H))))

    # Filtering Order : It should have element in MUCT \ Y
    order = [node for node in order if node in (muct - {'Y'})]
    pomis_set.add(frozenset(ib))

    return pomis_set | subpomis(H, order, set())

def subpomis(graph: nx.DiGraph, order: list, memo: set):
    pomis_set = set()
    
    for i in range(len(order)):
        var = order[i]
        next_graph = deepcopy(graph)
        next_graph = intervene(next_graph, var)

        muct = MUCT(next_graph)
        ib = IB(next_graph, muct)

        next_order = order[i + 1:]
        next_memo = memo | set(order[:i])

        if not (ib & next_memo):
            # Since ib is a set, and we want to collect sets of nodes (POMISs), 
            # we might need to adjust how pomis_set stores results.
            # But according to the context, it seems it wants to return a set of nodes or similar.
            # Assuming it wants to collect nodes that form POMIS.
            if ib:
                pomis_set.add(frozenset(ib))
                pomis_set |= subpomis(next_graph, next_order, next_memo)

    return pomis_set

if __name__ == '__main__':
    env = Environ()
    env.load_graph('chain_4.json')

    pomis_res = POMIS(env.G)
    print(f"POMIS Result: {pomis_res}")