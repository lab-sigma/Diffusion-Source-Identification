import diffusion_source.graphs as graphs
import networkx as nx
from diffusion_source.infection_model import save_model, FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus

files = [
    #"data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    #"data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    "data/NorthAmericaHiring/ComputerScienceHiring.txt",
    "data/NorthAmericaHiring/HistoryHiring.txt"
]

names = [
    #"AirportFlightTraffic",
    #"StatisticianCitations",
    "BSchoolHiring",
    "ComputerScienceHiring",
    "HistoryHiring"
]

IC_scale = 10
LT_scale = 0.01

losses = [L2_h, ADiT_h, ADT_h]
expectation_after = [False, False, False]
canonical = [True, True, True]

for index in range(len(names)):
    f = files[index]
    name = names[index]
    G = graphs.GeneralAdjacency(f)

    I = FixedTSI_IW(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=2000, m_p=2000, T=min(150, len(G.graph)//5))
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "SI_{}".format(name))

    max_w = max([w['weight'] for (u, v, w) in G.graph.edges(data=True)])
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(IC_scale+max_w)

    I = ICM(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=2000, m_p=2000, T=-1)
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "IC_{}".format(name))

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
    I = LTM(G, losses, canonical=canonical, expectation_after=expectation_after, m_l=2000, m_p=2000, T=-1)
    s = I.select_uniform_source()
    x = I.data_gen(s)
    save_model(I, s, x, "LT_{}".format(name))
