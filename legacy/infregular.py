import graphs
from main import run_rs, rumor_center, run_graph, run_graph_source
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from scipy.special import factorial

degrees = [3, 4]
Ts = list(range(100, 1600, 100))
#Ts.reverse()

for d in range(len(degrees)):
    degree = degrees[d]
    G = graphs.InfRegular(degree)
    for t in range(len(Ts)):
        T = Ts[t]
        print("Degree: {}, T: {}".format(degree, T))
        run_graph_source(G, T, 0, "Infinite Regular")
        print()
    print()
#Ts = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#Tb = [[[] for i in range(len(Ts))] for j in range(len(degrees))]
#Tbhops = [[[] for i in range(len(Ts))] for j in range(len(degrees))]
#Tr = [[[] for i in range(len(Ts))] for j in range(len(degrees))]
#Trhops = [[[] for i in range(len(Ts))] for j in range(len(degrees))]
#for d in range(len(degrees)):
#    degree = degrees[d]
#    G = graphs.InfRegular(degree)
#    for t in range(len(Ts)):
#        T = Ts[t]
#        print("Degree: {}, T: {}".format(degree, T))
#        b, bhops, r, rhops = run_rs(G, T, "Infinite Regular")
#        Tb[d][t] = b
#        Tbhops[d][t] = bhops
#        Tr[d][t] = r
#        Trhops[d][t] = rhops
#    print()
#
#print("Tb")
#print(Ts)
#for d in range(len(degrees)):
#    print(degrees[d], end=" & ")
#    for t in range(len(Ts)):
#        print(Tb[d][t], end=" & ")
#    print()
#print("Tbhops")
#print(Ts)
#for d in range(len(degrees)):
#    print(degrees[d], end=" & ")
#    for t in range(len(Ts)):
#        print(Tbhops[d][t], end=" & ")
#    print()
#print("Tr")
#print(Ts)
#for d in range(len(degrees)):
#    print(degrees[d], end=" & ")
#    for t in range(len(Ts)):
#        print(Tr[d][t], end=" & ")
#    print()
#print("Trhops")
#print(Ts)
#for d in range(len(degrees)):
#    print(degrees[d], end=" & ")
#    for t in range(len(Ts)):
#        print(Trhops[d][t], end=" & ")
#    print()

#G = graphs.InfRegular(2)
#for i in range(50):
#    x = G.sample(0, 50)
#    print()
#    center, centrality, rc, rcentrality, nodes = rumor_center(G.graph, x)
#    print(rc)
#    print(rcentrality)
#    print(G.dist(0, rc))
#x = G.sample(0, 50)
#center, centrality, rc, rcentrality, nodes = rumor_center(G.graph, x)
#color_map = []
#for node in G.graph.nodes:
#    if node == rc:
#        color_map.append('blue')
#    else:
#        color_map.append('green')
#nx.draw(G.graph, node_color=color_map, with_labels=True)
#plt.show()
