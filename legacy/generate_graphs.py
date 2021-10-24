import graphs
from main import display_sample, run_graph, run_graph2, large_run_graph, run_rs, run_gc_topk, N, degree, T, height, first_order_iso, save_graph, run_graph_multi_source, run_graph_compare, final_results, load_graph
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import random
import sys

G = graphs.RegularTree(N, degree, height)
sources = [G.select_source() for _ in range(5)]
save_graph(G, sources, "RTRandom")

G = graphs.PreferentialAttachment(N, 1)
sources = [G.select_source() for _ in range(5)]
save_graph(G, sources, "PARandom")

G = graphs.WattsStrogatz(N, degree)
sources = [G.select_source() for _ in range(5)]
save_graph(G, sources, "SWRandom")
