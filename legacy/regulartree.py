import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, first_order_iso, save_graph, run_graph_multi_source, run_graph_compare, final_results, load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random
import sys

#index = (int(sys.argv[1]) % 5)

#G = graphs.RegularTree(300, degree, height)
#
#s = G.select_source()
#L = 0
#for i in range(100):
#    x, _ = G.sample(s, 100)
#    L += len(first_order_iso(G, x))
#
#print(L)

#run_gc_topk(G, T, "Regular Tree")
#large_run_graph(G, "Regular Tree")
#display_sample(G, T)

#G = graphs.RegularTree(N, degree, height)

#large_run_graph(G, "Regular Tree")

#random.seed(int(sys.argv[1])//10)

#G = graphs.RegularTree(N, degree, height)
G, s = load_graph("RTRandom")
final_results(G, "Regular Tree", t = 200, source=s[0])
#save_graph(G, s, "RTFinal")
