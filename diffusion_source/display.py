import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def display_infected(G, x, source):
    nodes = []
    colors = []
    for n in G.graph.nodes():
        p = False
        c = [0.0, 0.0, 0.0]
        if n in x:
            p = True
            c[0] += 1.0
        if (not p) and any([(neighbor in x) for neighbor in G.graph.neighbors(n)]):
            p = True
            c = (0.5, 0.5, 0.5)
        if n == source:
            p = True
            c = [0.0, 1.0, 0.0]
        if p:
            nodes += [n]
            colors += [c]
    nx.draw_networkx(G.graph.subgraph(nodes), nodelist=nodes, with_labels=True, node_color=colors)
    plt.show()

def display_stationary_dist(G):
    A = nx.adjacency_matrix(G.graph, weight=None)
    A = A.asfptype()
    print(A.shape)
    A = (A.T/A.sum(axis=1)).T

    evals, evecs = np.linalg.eig(A.T)
    U = evecs[:, np.isclose(evals, 1)]

    U = U[:,0]
    U = U/U.sum()
    print(U)

    labels = {}

    nodes = []
    colors = []
    for n in G.graph.nodes():
        p = False
        c = [0.0, 0.0, 0.0]
        labels[n] = "{:.4f}".format(U[n])
        if p:
            nodes += [n]
            colors += [c]
    nx.draw_networkx(G.graph.subgraph(nodes), nodelist=nodes, with_labels=True, node_color=colors, labels=labels)
    plt.show()
