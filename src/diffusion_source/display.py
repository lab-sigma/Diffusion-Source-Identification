import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

axis_fontsize = 20
tick_fontsize = 16
legend_fontsize = 16
opacity_setting = 0.7
linewidth = 3.0

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

def alpha_v_coverage(I, opacity=opacity_setting, steps=1000, l_indices=None, l_names=None, x_label="Confidence Level", y_label="Coverage", title="Observed Coverage", trunc=-1, save=False, legend=True, show_x_label=True, show_y_label=True, colors=None, filename=None, randomized=True):
    results = I.results

    alpha = np.linspace(0, 1+1/steps, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    if l_indices is None:
        l_indices = list(range(nl))

    lranges = [np.zeros(steps) for _ in range(len(l_indices))]

    if colors is None:
        colors = [None for _ in range(nl)]

    K = 0
    rand = {}
    for t, result in results.items():
        s = result["meta"][2]
        if len(result["meta"][1]) <= trunc:
            continue
        K += 1
        for i, li in enumerate(l_indices):
            p = result["p_vals"][s][li]
            u = 1.0
            if randomized:
                u = random.random()
            lranges[i] += (1-alpha < p[0] + u*p[1])

    for i, li in enumerate(l_indices):
        lranges[i] /= K

    if l_names is None:
        l_names = list(range(nl))

    diff = 0
    better = 0
    rcParams.update({'figure.autolayout': True})
    for i, (li, ln, ci) in enumerate(zip(l_indices, l_names, colors)):
        plt.plot(alpha, lranges[i], label=ln, alpha=opacity, color=ci, linewidth=linewidth)
        diff += np.mean(lranges[i] - alpha)
        better += np.mean(lranges[i] > alpha)
    plt.plot(alpha, alpha, color="black", alpha=0.5)
    if legend:
        plt.legend(fontsize=legend_fontsize)
    if show_x_label:
        plt.xlabel(x_label, fontsize=axis_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    if show_y_label:
        plt.ylabel(y_label, fontsize=axis_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(title, fontsize=axis_fontsize)
    plt.grid()
    print("average deviation {}".format(diff/nl))
    print("better rate {}".format(better/nl))
    plt.autoscale()
    if save:
        if filename is None:
            filename = title
        plt.savefig("figures/{}.pdf".format(filename))
    plt.show()

    return alpha, lranges

def alpha_v_size(I, opacity=opacity_setting, steps=1000, l_indices=None, l_names=None, x_label="Confidence Level", y_label="Avg. Size", title="Average Confidence Set Size", trunc=-1, save=False, ratio=False, legend=True, show_x_label=True, show_y_label=True, colors=None, filename=None, randomized=True):
    results = I.results
    alpha = np.linspace(-1/steps, 1+1/steps, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    if l_indices is None:
        l_indices = list(range(nl))

    lranges = [np.zeros(steps) for _ in range(len(l_indices))]

    if colors is None:
        colors = [None for _ in range(nl)]

    K = 0
    mean_T = 0
    for t, result in results.items():
        T = len(I.source_candidates(result["meta"][1]))
        if T <= trunc:
            continue
        mean_T += T
        K += 1
        for i, li in enumerate(l_indices):
            for s, p in result["p_vals"].items():
                u = 1.0
                if randomized:
                    u = random.random()
                if ratio:
                    lranges[i] += (1-alpha < p[li][0] + u*p[li][1])/T
                else:
                    lranges[i] += (1-alpha < p[li][0] + u*p[li][1])

    for i, li in enumerate(l_indices):
        lranges[i] /= K

    if l_names is None:
        l_names = list(range(nl))

    mean_T /= K
    print(mean_T)

    rcParams.update({'figure.autolayout': True})
    for i, (li, ln, ci) in enumerate(zip(l_indices, l_names, colors)):
        plt.plot(alpha, lranges[i], label=ln, alpha=opacity, color=ci, linewidth=linewidth)
    if ratio:
        plt.plot(alpha, alpha, color="black", alpha=0.5)
    else:
        plt.plot(alpha, T*alpha, color="black", alpha=0.5)
    if legend:
        plt.legend(fontsize=legend_fontsize)
    if show_x_label:
        plt.xlabel(x_label, fontsize=axis_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    if show_y_label:
        plt.ylabel(y_label, fontsize=axis_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(title, fontsize=axis_fontsize)
    plt.grid()
    plt.autoscale()
    if save:
        if filename is None:
            filename = title
        plt.savefig("figures/{}.pdf".format(filename))
    plt.show()

    return alpha, lranges

def coverage_v_size(I, opacity=opacity_setting, steps=1000, l_indices=None, l_names=None, x_label="True Confidence", y_label="Avg. Size", title="Confidence Set Size vs True Confidence", trunc=-1, save=False, ratio=False, legend=True, show_x_label=True, show_y_label=True, colors=None, filename=None, randomized=True):
    results = I.results
    alpha = np.linspace(0, 1+1/steps, num=steps)

    ei = next(iter(results))
    si = next(iter(results[ei]["p_vals"]))
    nl = len(results[ei]["p_vals"][si])

    if l_indices is None:
        l_indices = list(range(nl))

    cranges = [np.zeros(steps) for _ in range(len(l_indices))]
    sranges = [np.zeros(steps) for _ in range(len(l_indices))]

    if colors is None:
        colors = [None for _ in range(nl)]

    K = 0
    for t, result in results.items():
        T = len(I.source_candidates(result["meta"][1]))
        s = result["meta"][2]
        if T <= trunc:
            continue
        K += 1
        for i, li in enumerate(l_indices):
            s = result["meta"][2]
            p = result["p_vals"][s][li]
            u = 1.0
            if randomized:
                u = random.random()
            cranges[i] += (1-alpha < p[0] + u*p[1])
            for s, p in result["p_vals"].items():
                if randomized:
                    u = random.random()
                if ratio:
                    sranges[i] += (1-alpha < p[li][0] + u*p[li][1])/T
                else:
                    sranges[i] += (1-alpha < p[li][0] + u*p[li][1])

    for i, li in enumerate(l_indices):
        sranges[i] /= K
        cranges[i] /= K

    if l_names is None:
        l_names = list(range(nl))

    rcParams.update({'figure.autolayout': True})
    for i, (li, ln, ci) in enumerate(zip(l_indices, l_names, colors)):
        plt.plot(cranges[i], sranges[i], label=ln, alpha=opacity, color=ci, linewidth=linewidth)
    if ratio:
        plt.plot(alpha, alpha, color="black", alpha=0.5)
    else:
        plt.plot(alpha, T*alpha, color="black", alpha=0.5)
    if legend:
        plt.legend(fontsize=legend_fontsize)
    if show_x_label:
        plt.xlabel(x_label, fontsize=axis_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    if show_y_label:
        plt.ylabel(y_label, fontsize=axis_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(title, fontsize=axis_fontsize)
    plt.grid()
    plt.autoscale()
    if save:
        if filename is None:
            filename = title
        plt.savefig("figures/{}.pdf".format(filename))
    plt.show()

    return alpha, cranges, sranges

def sample_size_cdf(I, steps=1000, K=1000):
    x = np.linspace(0, len(I.G.graph), num=steps)

    cdf = np.zeros(steps)
    sizes = []
    for k in range(K):
        s = I.select_uniform_source()
        xi = I.data_gen(s)
        cdf += len(xi) < x
        sizes += [len(xi)]

    #plt.plot(x, cdf)
    plt.histogram(sizes)
    plt.show()
