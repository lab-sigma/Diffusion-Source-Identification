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

def alpha_v_coverage(results, opacity=0.7, steps=1000, l_indices=None, l_names=None, x_label="Confidence Level", y_label="Coverage", title="Observed Coverage"):

    alpha = np.linspace(0, 1+1/steps, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    if l_indices is None:
        l_indices = list(range(nl))

    lranges = [np.zeros(steps) for _ in range(len(l_indices))]

    for t, result in results.items():
        s = result["meta"][2]
        for i, li in enumerate(l_indices):
            p = result["p_vals"][s][li]
            lranges[i] += (1-alpha < p)

    K = len(results)
    for i, li in enumerate(l_indices):
        lranges[i] /= K

    if l_names is None:
        l_names = list(range(nl))

    diff = 0
    better = 0
    for i, (li, ln) in enumerate(zip(l_indices, l_names)):
        plt.plot(alpha, lranges[i], label=ln, alpha=opacity)
        diff += np.mean(lranges[i] - alpha)
        better += np.mean(lranges[i] > alpha)
    plt.plot(alpha, alpha)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    print("average deviation {}".format(diff/nl))
    print("better rate {}".format(better/nl))
    plt.savefig("figures/{}.png".format(title))
    plt.show()

    return alpha, lranges

def alpha_v_size(results, opacity=0.7, steps=1000, l_indices=None, l_names=None, x_label="Confidence Level", y_label="Avg. Size", title="Average Confidence Set Size"):
    alpha = np.linspace(0, 1+1/steps, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    if l_indices is None:
        l_indices = list(range(nl))

    lranges = [np.zeros(steps) for _ in range(len(l_indices))]

    for t, result in results.items():
        T = len(result["meta"][1])
        for i, li in enumerate(l_indices):
            for s, p in result["p_vals"].items():
                lranges[i] += (1-alpha < p[li])/T

    K = len(results)
    for i, li in enumerate(l_indices):
        lranges[i] /= K

    if l_names is None:
        l_names = list(range(nl))

    for i, (li, ln) in enumerate(zip(l_indices, l_names)):
        plt.plot(alpha, lranges[i], label=ln, alpha=opacity)
    plt.plot(alpha, alpha)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("figures/{}.png".format(title))
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
