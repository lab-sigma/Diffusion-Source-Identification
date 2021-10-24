import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, save_graph, load_graph, small_run_graph, run_graph_source, run_large_source, simple_run_large_source, display_only_sample, print_iso
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

G, s = load_graph("ER3280")
print_iso(G, "Erdos Renyi")

#x, x_l = G.sample(s, 1)
#display_only_sample(G, x, s)
#source, m_mat = simple_run_large_source(G, s, "Preferential Attachment")

#np.savetxt("saved/p_values3.csv", m_mat)

#G = graphs.PreferentialAttachment(300, degree)
#large_run_graph(G, "Preferential Attachment")
