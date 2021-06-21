import graphs
import time
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import itertools
import random
from decimal import *

def message_up(T, nodes, node):
    neighbors = list(nx.neighbors(T, node))
    if len(neighbors) == 0:
        nodes[node] = (Decimal(1.0), Decimal(1.0), Decimal(0.0))
    else:
        for n in neighbors:
            message_up(T, nodes, n)
        t = Decimal(1) + sum([nodes[j][0] for j in neighbors])
        nodes[node] = (t, t*np.product([nodes[j][1] for j in neighbors]), Decimal(0.0))

def message_down(T, nodes, node, N):
    neighbors = list(nx.neighbors(T, node))
    for n in neighbors:
        nodes[n] = (nodes[n][0], nodes[n][1], nodes[node][2]*Decimal(nodes[n][0])/Decimal(N - nodes[n][0]))
        message_down(T, nodes, n, N)

def rumor_center(G, x):
    GN = G.subgraph(list(x))
    nodes = dict.fromkeys(GN.nodes(), (0.0,0.0,0.0))
    root = list(GN.nodes)[0]
    T = nx.bfs_tree(GN, root)
    getcontext().prec = 28

    message_up(T, nodes, root)
    neighbors = list(nx.neighbors(T, root))
    N = len(nodes.keys())
    r_root = np.product([Decimal(i) for i in range(1, N+1)])
    r_root = r_root/Decimal(N*np.product([nodes[j][1] for j in neighbors]))
    nodes[root] = (nodes[root][0], nodes[root][1], r_root)
    message_down(T, nodes, root, N)
    getcontext().prec = 20
    rnd = Decimal(1.0)
    
    ml = -1
    rs = Decimal(-1)
    center = -1
    rcenter = -1
    for n in nodes.keys():
        edges = nx.bfs_edges(GN, n)
        sigma = [n] + [v for u, v in edges]
        p = np.product([Decimal(1.0/(G.degree[n]+sum([G.degree[sigma[i]]-2 for i in range(1, k-1)]))) for k in range(1,N)])
        if rnd*nodes[n][2] > rs:
            rs = rnd*nodes[n][2]
            rcenter = n
        elif rnd*nodes[n][2] == rs:
            if random.random() < 0.5:
                rs = rnd*nodes[n][2]
                rcenter = n
        if nodes[n][2]*p > ml:
            center = n
            ml = nodes[n][2]*p
        nodes[n] = nodes[n][2]*p
    return center, ml, rcenter, rs, nodes

def rumor_center_topk(G, x, k):
    GN = G.subgraph(list(x))
    nodes = dict.fromkeys(GN.nodes(), (0.0,0.0,0.0))
    root = list(GN.nodes)[0]
    T = nx.bfs_tree(GN, root)
    getcontext().prec = 28

    message_up(T, nodes, root)
    neighbors = list(nx.neighbors(T, root))
    N = len(nodes.keys())
    r_root = np.product([Decimal(i) for i in range(1, N+1)])
    r_root = r_root/Decimal(N*np.product([nodes[j][1] for j in neighbors]))
    nodes[root] = (nodes[root][0], nodes[root][1], r_root)
    message_down(T, nodes, root, N)
    getcontext().prec = 20
    rnd = Decimal(1.0)
    
    ml_dict = {}
    for n in nodes.keys():
        ml_dict[n] = rnd*nodes[n][2]

    ml_sort = sorted(ml_dict.items(), key=lambda x: x[1], reverse=True)
    topk = []
    for n in ml_sort:
        if len(topk) < k:
            topk += [n[0]]
            continue
        break

    return list(topk)
