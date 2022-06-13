import sys
from os.path import exists
import pickle

import diffusion_source.graphs as graphs
from diffusion_source.infection_model import FixedTSI
from diffusion_source.discrepancies import L2_h, ADiT_h, ADT_h

networks = pickle.load(open("data/CommunityFitNet/Benchmark/CommunityFitNet.pickle", "rb"))

edgelists = networks['edges_id']
gp = networks['graphProperties']
nn = networks['number_nodes']
network_titles = networks['title']
network_domains = networks['networkDomain']
url = networks['sourceUrl']

K = 5
arg = (int(sys.argv[1]) - 1)
global_m = int(sys.argv[2])
index = arg % networks.shape[0]

edges = edgelists.iloc[index]
title = network_titles.iloc[index]
G = graphs.PyEdgeList(edges)

losses = [L2_h, ADiT_h, ADT_h]
expectation_after = [False, False, False]
canonical = [True, True, True]

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))

def run_graph(I, name, k, index):
    if exists("results/community_timing/{}_{}_{}_{}_{}.p".format(name, global_m, index, arg+1, k)):
        return
    s = I.select_uniform_source()
    x = I.data_gen(s)

    print(I.p_values(x, meta=(name, x, s)))
    I.store_results("results/community_timing/{}_{}_{}_{}_{}.p".format(name, global_m, index, arg+1, k))


for k in range(K):
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=False)
    run_graph(I, "neither_{}".format(title), k, index)
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=True, d1=True)
    run_graph(I, "both_{}".format(title), k, index)
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=True)
    run_graph(I, "d1_{}".format(title), k, index)
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=True, d1=False)
    run_graph(I, "iso_{}".format(title), k, index)
