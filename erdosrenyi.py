import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, first_order_iso, save_graph, run_graph_multi_source, run_graph_compare, final_results, load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

#G = graphs.ErdosRenyi(300, degree)

#s = G.select_source()
#L = 0
#for i in range(100):
#    x, _ = G.sample(s, 100)
#    L += len(first_order_iso(G, x))

#print(L)
#large_run_graph(G, "Erdos Renyi")

G = graphs.ErdosRenyi(N, degree)
#G, s = load_graph("ERFinal")
s = final_results(G, "Erdos Renyi")
#save_graph(G, s, "ERFinal")
