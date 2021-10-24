import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, display_only_sample, save_graph, run_graph_multi_source, run_graph_compare, final_results, load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random
import sys

#index = (int(sys.argv[1]) % 5)

#random.seed(int(sys.argv[1])//10)

#G = graphs.WattsStrogatz(300, degree)
#large_run_graph(G, "Small World")
#
#G = graphs.WattsStrogatz(N, degree)
#large_run_graph(G, "Small World")

#G = graphs.WattsStrogatz(25, 4)
#degrees = [G.graph.degree[v] for v in list(G.graph.nodes)]
#print(np.mean(degrees))
#s = G.select_source()
#x, x_l = G.sample(s, 25)
#display_only_sample(G, x, s)

#G = graphs.WattsStrogatz(N, degree)
G, s = load_graph("SWRandom")
final_results(G, "Small World", t = 200, source=s[0])
#save_graph(G, s, "SWFinal")
