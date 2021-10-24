import graphs
from main import display_sample, run_graph, run_graph2, run_rs, run_gc_topk, N, degree, T, height, save_graph, run_graph_multi_source, run_graph_compare, final_results, load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

#G = graphs.RegularGraph(N, degree)

#run_gc_topk(G, T, "Regular Graph")
#run_graph2(G, T, "Regular Graph")
#run_graph(G, T, "Regular Graph")
#display_sample(G, T)

G = graphs.RegularGraph(N, degree)
#G, s = load_graph("RGFinal")
s = final_results(G, "Regular Graph")
#save_graph(G, s, "RGFinal")
