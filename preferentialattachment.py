import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, save_graph, load_graph, small_run_graph, simple_run_large_source, display_only_sample, run_graph_multi_source, run_graph_compare, final_results
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random
import sys
import os, psutil

#index = (int(sys.argv[1]) % 5)

#random.seed(int(sys.argv[1])//10)

#G = graphs.PreferentialAttachment(N, 1)
#degrees = [G.graph.degree[v] for v in list(G.graph.nodes)]
#print(np.mean(degrees))
#s = G.select_source()
#x, x_l = G.sample(s, 25)
#display_only_sample(G, x, s)

#G = graphs.PreferentialAttachment(300, degree)
G, s = load_graph("PARandom")
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

s = final_results(G, "Preferential Attachment", t = 200, source=s[0])
#s = final_results(G, "Preferential Attachment", t = 150)
#save_graph(G, s, "PAFinal")
