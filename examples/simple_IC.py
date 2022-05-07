import diffusion_source.graphs as graphs
import networkx as nx
from diffusion_source.graphs import GraphWrapper
from diffusion_source.infection_model import save_model, FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.display import alpha_v_coverage, alpha_v_size, coverage_v_size

losses = [L2_h, ADiT_h, ADT_h]

graph = nx.DiGraph()

graph.add_node(1)
graph.add_node(2)
graph.add_edge(1, 2)
graph.add_edge(2, 1)

graph[1][2]['weight'] = 0.1
graph[2][1]['weight'] = 0.9

G = GraphWrapper(graph)

I = ICM(G, losses, m=2000, T=-1)
K = 1000
C = 0
S = 0
both = 0
bothC = 0
bothS = 0
for k in range(K):
    source = I.select_uniform_source()
    x = I.data_gen(source)

    c_set = I.confidence_set(x, 0.5, meta=("IC", x, source))
    print(x)
    print(I.results)
    quit()
    if source in c_set["L2_h"]:
        C += 1
        if len(x) == 2:
            bothC += 1
    S += len(c_set["L2_h"])
    if len(x) == 2:
        both += 1
        bothS += len(c_set["L2_h"])
    print("{:.2f} % : {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(k/K, C/(k+1), S/(k+1), bothC/both, bothS/both), end="\r")

l_indices = [0, 1, 2]
l_names = ["L2", "ADiT", "ADT"]
colors = ["green", "orange", "blue"]

alpha_v_coverage(I, l_indices=l_indices, l_names=l_names, legend=True, show_y_label=True, title="avc", colors=colors)
alpha_v_size(I, l_indices=l_indices, l_names=l_names, ratio=True, legend=True, show_y_label=True, title="avs", colors=colors)
coverage_v_size(I, l_indices=l_indices, l_names=l_names, ratio=True, legend=True, show_y_label=True, title="cvs", colors=colors)
print()
print()

print(C/K)
