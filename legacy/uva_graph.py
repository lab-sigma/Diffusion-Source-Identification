import graphs
from main import display_sample, run_graph, run_graph2, run_rs, run_gc_topk, N, degree, T, height
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

G = graphs.EdgeList("data/UVA-EdgeList.csv")

run_graph(G, 50, "UVA Graph")
