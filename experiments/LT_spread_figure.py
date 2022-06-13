import matplotlib.pyplot as plt
import networkx as nx


def display_AxI(G, active, inactive, node_labels, layout, edge_labels, i=0):
    node_colors = []
    nodes = list(G.nodes)
    for n in nodes:
        c = [0.35, 0.35, 0.35]
        if n in active:
            c = [1.0, 0.0, 0.0]
        elif n in inactive:
            c = [0.5, 0.3, 0.3]
        node_colors += [c]

    edge_colors = []
    for e in list(G.edges):
        c = [0.0, 0.0, 0.0]
        if (e[0] in active or e[1] in active) and (e[0] not in active.union(inactive) or e[1] not in active.union(inactive)):
            c = [1.0, 0.5, 0.0]
        edge_colors += [c]
    nx.draw_networkx(
            G,
            pos=layout,
            nodelist=nodes,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            labels=node_labels,
            width=1.5,
            font_size=10
    )
    nx.draw_networkx_nodes(
            G,
            pos=layout,
            nodelist=nodes,
            node_color=node_colors,
            edgecolors=[0.0, 0.0, 0.0],
            linewidths=1.5
    )
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels)
    plt.savefig("spread_figures/LTspread_{}.pdf".format(i))
    plt.show()


G = nx.Graph()

G.add_nodes_from([1, 2, 3, 4, 5])

G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(2, 4)
G.add_edge(2, 5)
G.add_edge(3, 4)
G.add_edge(4, 5)

layout = {
    1: (0.5, 0.1),
    2: (0.7, 0.5),
    3: (0.25, 0.4),
    4: (0.35, 0.8),
    5: (0.65, 1)
}

labels = {
    1: "",
    2: "0.6",
    3: "0.2",
    4: "?",
    5: "?"
}

edge_labels = {
    (1, 2): 0.3,
    (1, 3): 0.3,
    (3, 4): 0.2,
    (2, 4): 0.3,
    (4, 5): 0.1,
    (2, 5): 0.4,
}

display_AxI(G, {1}, {}, labels, layout, edge_labels, 1)

labels = {
    1: "",
    2: "0.6",
    3: "",
    4: "0.1",
    5: "?"
}
edge_labels[(1, 3)] = ""

display_AxI(G, {1, 3}, {}, labels, layout, edge_labels, 2)

labels = {
    1: "",
    2: "0.6",
    3: "",
    4: "",
    5: "0.8"
}
edge_labels[(3, 4)] = ""

display_AxI(G, {1, 3, 4}, {}, labels, layout, edge_labels, 3)

labels = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "0.8"
}
edge_labels[(2, 4)] = ""
edge_labels[(1, 2)] = ""

display_AxI(G, {1, 2, 3, 4}, {}, labels, layout, edge_labels, 4)
