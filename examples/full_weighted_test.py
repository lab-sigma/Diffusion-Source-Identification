import diffusion_source.graphs as graphs
import networkx as nx
from diffusion_source.infection_model import save_model, FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h

files = [
    "data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    "data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    "data/NorthAmericaHiring/ComputerScienceHiring.txt",
    "data/NorthAmericaHiring/HistoryHiring.txt",
    "data/NorthAmericaHiring/StatisticsHiring.txt"
]

names = [
    "AirportFlightTraffic",
    "StatisticianCitations",
    "BSchoolHiring",
    "ComputerScienceHiring",
    "HistoryHiring",
    "StatisticsHiring"
]

K = 100
scale = 0.1

losses = [L2_h, L2_after, ADiT_h, ADT_h]
losses = [L2_h, ADiT_h, ADT_h]
expectation_after = [False, False, False]

for index in range(len(names)):
    f = files[index]
    name = names[index]
    G = graphs.WeightedAdjacency(f)

    def run_K(I, model_name, name):
        C = [0 for _ in range(len(losses))]
        S = [0 for _ in range(len(losses))]
        print(model_name)
        print(name)
        for k in range(K):
            source = I.select_uniform_source()
            x = I.data_gen(source)

            c_set = I.confidence_set(x, 0.1)
            i = 0
            print("{:.2f} % : ".format(100*k/K), end="")
            for l in losses:
                if source in c_set[l.__name__]:
                    C[i] += 1
                S[i] += len(c_set[l.__name__])
                print("{} {:.2f}/{:.2f}, ".format(l.__name__, C[i]/(k+1), S[i]/(k+1)), end="")
                i += 1
            print("", end="\r")
        print()
        print()

    I = FixedTSI_IW(G, losses, expectation_after=expectation_after, m=10, T=1)
    #run_K(I, "SI", name)

    max_w = max([w['weight'] for (u, v, w) in G.graph.edges(data=True)])
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(scale*max_w)
    I = ICM(G, losses, expectation_after=expectation_after, m=10, T=10)
    #run_K(I, "IC", name)

    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']*(scale*max_w)
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
        G.graph[u][v]['weight'] = w['weight']/(max(scale*max_w, influence[v]))
    I = LTM(G, losses, expectation_after=expectation_after, m=10, T=10)
    run_K(I, "LT", name)
