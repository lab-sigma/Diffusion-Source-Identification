import graphs
from isomorphisms import get_iso, first_order_iso, get_iso_full, first_order_iso_full
from main import N, degree, K, T, print_iso, display_infected, run_graph_iso
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

#G = graphs.RegularTree(N, degree, 0)
#print(len(first_order_iso_full(G)))
#G = graphs.ErdosRenyi(N, degree)
#f_iso , f_iso_f = print_iso(G, "Erdos Renyi")
#x = set()
#for p in f_iso:
#    x.add(p[0])
#    x.add(p[1])
#display_infected(G, x)
#nx.draw_networkx(G.graph, with_labels=True)
#plt.show()

#print_iso(graphs.RegularGraph(N, degree), "Regular Graph")
#print_iso(graphs.WattsStrogatz(N, degree), "Small World")

RT = graphs.RegularTree(N, degree, 0)
PA = graphs.PreferentialAttachment(N, 1)

print_iso(RT, "Regular Tree")
print_iso(PA, "Preferential Attachment")

#run_graph_iso(RT, T, "Regular Tree")
#run_graph_iso(PA, T, "Preferential Attachment")
