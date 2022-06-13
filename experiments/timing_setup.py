import sys
import diffusion_source.graphs as graphs
import networkx as nx
from diffusion_source.infection_model import save_model, FixedTSI, FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, ADiT_h, ADT_h

files = [
    "data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    "data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    "data/NorthAmericaHiring/ComputerScienceHiring.txt",
    "data/NorthAmericaHiring/HistoryHiring.txt"
]

names = [
    "AirportFlightTraffic",
    "StatisticianCitations",
    "BSchoolHiring",
    "ComputerScienceHiring",
    "HistoryHiring"
]

exc = [
    "AirportFlightTraffic",
    "StatisticianCitations"
]

IC_scale = 10
LT_scale = 0.01
N = 1365

losses = [L2_h, ADiT_h, ADT_h]
expectation_after = [False, False, False]
canonical = [True, True, True]

global_m = int(sys.argv[1])

#######################################

G = graphs.RegularTree(N, 4)

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
s = I.select_uniform_source()
x = I.data_gen(s)

save_model(I, s, x, "regular_tree_both_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False)
save_model(I, s, x, "regular_tree_d1_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), d1=False)
save_model(I, s, x, "regular_tree_iso_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=False)
save_model(I, s, x, "regular_tree_neither_{}".format(global_m))

G = graphs.WattsStrogatz(N, 4)

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
s = I.select_uniform_source()
x = I.data_gen(s)

save_model(I, s, x, "small_world_both_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False)
save_model(I, s, x, "small_world_d1_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), d1=False)
save_model(I, s, x, "small_world_iso_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=False)
save_model(I, s, x, "small_world_neither_{}".format(global_m))

G = graphs.PreferentialAttachment(N, 1)

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
s = I.select_uniform_source()
x = I.data_gen(s)

save_model(I, s, x, "preferential_attachment_both_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False)
save_model(I, s, x, "preferential_attachment_d1_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), d1=False)
save_model(I, s, x, "preferential_attachment_iso_{}".format(global_m))
I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=False)
save_model(I, s, x, "preferential_attachment_neither_{}".format(global_m))
#######################################

for index in range(len(names)):
    f = files[index]
    name = names[index]
    G = graphs.GeneralAdjacency(f)

    I = FixedTSI(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "directed_UW_{}_both_{}".format(name, global_m))
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False)
    save_model(I, s, x, "directed_UW_{}_d1_{}".format(name, global_m))
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), d1=False)
    save_model(I, s, x, "directed_UW_{}_iso_{}".format(name, global_m))
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=False)
    save_model(I, s, x, "directed_UW_{}_neither_{}".format(name, global_m))

    I = FixedTSI_IW(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "directed_IW_{}_{}".format(global_m, name))

    if name in exc:
        continue

    max_w = max([w['weight'] for (u, v, w) in G.graph.edges(data=True)])
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(IC_scale+max_w)
    I = ICM(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=-1)
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "directed_IC_{}_{}".format(global_m, name))

    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']*(IC_scale+max_w)
    max_w = 0
    influence = {}
    for v in G.graph.nodes():
        vtw = 0
        if nx.is_directed(G.graph):
            for (u, vp, w) in G.graph.in_edges(v, data=True):
                vtw += w['weight']
        else:
            for (u, vp, w) in G.graph.edges(v, data=True):
                vtw += w['weight']
        if vtw > max_w:
            max_w = vtw
        influence[v] = vtw
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(max(LT_scale*max_w, influence[v]))
    I = LTM(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=-1)
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "directed_LT_{}_{}".format(global_m, name))

for index in range(len(names)):
    f = files[index]
    name = names[index]
    G = graphs.WeightedAdjacency(f)

    I = FixedTSI(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "undirected_UW_{}_both_{}".format(name, global_m))
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False)
    save_model(I, s, x, "dunirected_UW_{}_d1_{}".format(name, global_m))
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), d1=False)
    save_model(I, s, x, "undirected_UW_{}_iso_{}".format(name, global_m))
    I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5), iso=False, d1=False)
    save_model(I, s, x, "undirected_UW_{}_neither_{}".format(name, global_m))

    I = FixedTSI_IW(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=min(150, len(G.graph)//5))
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "undirected_IW_{}_{}".format(global_m, name))

    if name in exc:
        continue

    max_w = max([w['weight'] for (u, v, w) in G.graph.edges(data=True)])
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(IC_scale+max_w)
    I = ICM(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=-1)
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "undirected_IC_{}_{}".format(global_m, name))

    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']*(IC_scale+max_w)
    max_w = 0
    influence = {}
    for v in G.graph.nodes():
        vtw = 0
        if nx.is_directed(G.graph):
            for (u, vp, w) in G.graph.in_edges(v, data=True):
                vtw += w['weight']
        else:
            for (u, vp, w) in G.graph.edges(v, data=True):
                vtw += w['weight']
        if vtw > max_w:
            max_w = vtw
        influence[v] = vtw
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(max(LT_scale*max_w, influence[v]))
    I = LTM(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=global_m, m_p=global_m, T=-1)
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "undirected_LT_{}_{}".format(global_m, name))
