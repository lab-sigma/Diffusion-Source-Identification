import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, save_graph, load_graph, small_run_graph, run_graph_source, run_large_source, simple_run_large_source
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

G, s = load_graph("PA300")
source, m_mat = simple_run_large_source(G, s, "Preferential Attachment")

np.savetxt("saved/p_values_small.csv", m_mat)

#G = graphs.PreferentialAttachment(300, degree)
#large_run_graph(G, "Preferential Attachment")
