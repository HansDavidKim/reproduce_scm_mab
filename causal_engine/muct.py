# Get POMISs using Algorithm 1
import networkx as nx
from collections import deque

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from environments.environ import Environ

# Minimal Unobserved Confounders' Territory
def get_candidates(graph: nx.DiGraph, node: str, ancestor: set, memo: set):
    children = set(graph.successors(node)) & ancestor
    
    cc = set() # connected component
    for parent in list(graph.predecessors(node)):
        if not ('U' in parent):
            continue
        
        for child in list(graph.successors(parent)):
            cc.add(child)
    cc &= ancestor
    return (children | cc) - memo

def MUCT(graph: nx.DiGraph):
    muct = set('Y')
    ancestor = set(nx.ancestors(graph, 'Y'))

    queue = deque()
    # BFS-like expansion
    queue.append('Y')

    while len(queue) > 0:
        node = queue.popleft()
        candidate = get_candidates(graph, node, ancestor, muct)

        muct |= candidate
        for next in candidate:
            queue.append(next)

    return muct

def IB(graph: nx.DiGraph, muct: set):
    ib_candidate = set()
    ib = set()
    for node in muct:
        parents = set(graph.predecessors(node))
        if not (parents & muct):
            ib_candidate |= parents

    for node in ib_candidate:
        if not 'U' in node:
            ib.add(node)
    return ib

if __name__ == '__main__':
    env = Environ()
    env.load_graph('chain_3.json')
    env.allocate_weight()
    # env.show_graph()

    muct = MUCT(env.G)
    print(IB(env.G, muct))