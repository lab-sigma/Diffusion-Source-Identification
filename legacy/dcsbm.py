import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, first_order_iso, save_graph, run_graph_multi_source, run_graph_compare, final_results, load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

G = graphs.StochasticBlock(N, degree)
#G, s = load_graph("SBMFinal")
s = final_results(G, "DCSBM")
#save_graph(G, s, "SBMFinal")
