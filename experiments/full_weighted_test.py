import diffusion_source.graphs as graphs
import networkx as nx
from diffusion_source.infection_model import FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.display import sample_size_cdf, alpha_v_coverage, alpha_v_size

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

K = 100
IC_scale = 10
LT_scale = 0.01

si_losses = [L2_h, L2_after, ADiT_h, ADT_h]
si_expectation_after = [False, True, False, False]
si_canonical = [True, False, True, True]

o_losses = [Z_minus]
o_expectation_after = [True]
o_canonical = [False]

for index in range(len(names)):
    f = files[index]
    name = names[index]
    G = graphs.WeightedAdjacency(f)

    def run_K(I, model_name, name, losses):
        C = [0 for _ in range(len(losses))]
        A = [0 for _ in range(len(losses))]
        S = [0 for _ in range(len(losses))]
        T = 0
        print(model_name)
        print(name)
        for k in range(K):
            source = I.select_uniform_source()
            x = I.data_gen(source)

            c_set = I.confidence_set(x, 0.1, meta=(name, x, source))
            i = 0
            T += len(x)
            print("{:.2f} %, {}/{:.2f} : ".format(100*k/K, len(I.G.graph), T/(k+1)), end="")
            for l in losses:
                if source in c_set[l.__name__]:
                    C[i] += 1
                A[i] += len(c_set[l.__name__])/len(x)
                S[i] += len(c_set[l.__name__])
                print("{} {:.2f}/{:.2f}/{:.2f}, ".format(l.__name__, C[i]/(k+1), A[i]/(k+1), S[i]/(k+1)), end="")
                i += 1
            print("", end="\r")
        print()
        print()
        alpha_v_coverage(I)
        alpha_v_size(I)

    I = FixedTSI_IW(G, si_losses, canonical=si_canonical, expectation_after=si_expectation_after, m_l=10, m_p=10, T=min(150, len(G.graph)//5))
    run_K(I, "SI", name, si_losses)

    max_w = max([w['weight'] for (u, v, w) in G.graph.edges(data=True)])
    for (u, v, w) in G.graph.edges(data=True):
        G.graph[u][v]['weight'] = w['weight']/(IC_scale+max_w)
    I = ICM(G, o_losses, canonical=o_canonical, expectation_after=o_expectation_after, m_l=10, m_p=10, T=-1)
    run_K(I, "IC", name, o_losses)
    sample_size_cdf(I)

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
    I = LTM(G, o_losses, canonical=o_canonical, expectation_after=o_expectation_after, m_l=10, m_p=10, T=-1)
    sample_size_cdf(I)
    run_K(I, "LT", name, o_losses)
