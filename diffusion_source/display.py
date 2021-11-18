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

def alpha_v_coverage(results, steps=1000):
    alpha = np.linspace(0, 1, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    lranges = [np.zeros(steps) for _ in range(nl)]

    for t, result in results.items():
        s = result["meta"][2]
        for i in range(nl):
            p = result["p_vals"][s][i]
            lranges[i] += (alpha > 1-p)

    K = len(results)
    for i in range(nl):
        lranges[i] /= K

    diff = 0
    better = 0
    for i in range(nl):
        plt.plot(alpha, lranges[i])
        diff += np.mean(lranges[i] - alpha)
        better += np.mean(lranges[i] >= alpha)
    plt.plot(alpha, alpha)
    print("average deviation {}".format(diff/nl))
    print("better rate {}".format(better/nl))
    plt.show()

    return alpha, lranges

def alpha_v_size(results, steps=1000):
    alpha = np.linspace(0, 1, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    lranges = [np.zeros(steps) for _ in range(nl)]

    for t, result in results.items():
        T = len(result["meta"][1])
        for i in range(nl):
            for s, p in result["p_vals"].items():
                lranges[i] += (alpha >= 1-p[i])/T

    K = len(results)
    for i in range(nl):
        lranges[i] /= K

    for i in range(nl):
        plt.plot(alpha, lranges[i])
    plt.plot(alpha, alpha)
    plt.show()

    return alpha, lranges

def sample_size_cdf(I, steps=1000, K=1000):
    x = np.linspace(0, len(I.G.graph), num=steps)

    cdf = np.zeros(steps)
    for k in range(K):
        s = I.select_uniform_source()
        cdf += len(I.data_gen(s)) < x

    plt.plot(x, cdf)
    plt.show()
