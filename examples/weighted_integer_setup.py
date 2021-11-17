import diffusion_source.graphs as graphs
from diffusion_source.infection_model import save_model, FixedTSI_IW
from diffusion_source.discrepancies import L2_after, L2_h, ADiT_h

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

for index in range(6):
    f = files[index]
    name = names[index]
    G = graphs.WeightedAdjacency(f)

    I = FixedTSI_IW(G, [L2_after, L2_h, ADiT_h], expectation_after=[True, True, True], canonical=[False, True, True], m=1, T=min(150, len(G.graph)//5))
    #I = FixedTSI_IW(G, [L2_after, L2_h_after, ADiT_h_after], expectation_after=[True, True, True], m=2000, T=min(150, len(G.graph)//5))
    #I = FixedTSI_IW(G, [L2_after, L2_h_after, ADiT_h_after], expectation_after=[True, True, True], m=10, T=10)
    s = I.select_uniform_source()
    x = I.data_gen(s)

    save_model(I, s, x, name)
