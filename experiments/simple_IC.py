import networkx as nx
from diffusion_source.graphs import GraphWrapper
from diffusion_source.infection_model import FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, ADiT_h, ADT_h, L2, loss_creator
from diffusion_source.display import alpha_v_coverage, alpha_v_size, coverage_v_size

losses = [L2_h, ADiT_h, ADT_h]
#losses = [loss_creator(L2)]

graph = nx.DiGraph()

graph.add_node(1)
graph.add_node(2)
graph.add_edge(1, 2)
graph.add_edge(2, 1)

graph[1][2]['weight'] = 0.9
graph[2][1]['weight'] = 0.1

G = GraphWrapper(graph)

I = ICM(G, losses, canonical=True, m_l=2000, m_p=2000, T=-1)
K = 100
A = [0, 0]
B = [0, 0]
AB = [0, 0, 0, 0]
for k in range(K):
    source = I.select_uniform_source()
    x = I.data_gen(source)

    c_set = I.confidence_set(x, 0.9, meta=("IC", x, source))
    C = c_set["L2_h"]
    if len(x) == 1:
        if 1 in x:
            if len(C) == 1:
                A[1] += 1
            else:
                A[0] += 1
        else:
            if len(C) == 1:
                B[1] += 1
            else:
                B[0] += 1
    else:
        if len(C) == 2:
            AB[0] += 1
        elif 1 in C:
            AB[1] += 1
        elif 2 in C:
            AB[2] += 1
        else:
            AB[3] += 1

print(A[0]/sum(A))
print(A[1]/sum(A))
print(B[0]/sum(B))
print(B[1]/sum(B))
print(AB[0]/sum(AB))
print(AB[1]/sum(AB))
print(AB[2]/sum(AB))
print(AB[3]/sum(AB))

l_indices = [0, 1, 2]
l_names = ["L2", "ADiT", "ADT"]
colors = ["green", "orange", "blue"]

#print()
#print()
alpha_v_coverage(I, l_indices=l_indices, l_names=l_names, legend=True, show_y_label=True, title="avc", colors=colors)
alpha_v_size(I, l_indices=l_indices, l_names=l_names, ratio=True, legend=True, show_y_label=True, title="avs", colors=colors)
coverage_v_size(I, l_indices=l_indices, l_names=l_names, ratio=True, legend=True, show_y_label=True, title="cvs", colors=colors)
#print()
#print()
#print()
