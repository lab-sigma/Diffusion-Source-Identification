import diffusion_source.graphs as graphs
from diffusion_source.infection_model import save_model, FixedTSI_IW
from diffusion_source.discrepancies import L2_after, L2_h, ADiT_h, ADT_h, Z_minus


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

losses = [L2_h, ADiT_h, ADT_h]
expectation_after = [False, False, False]
canonical = [True, True, True]

for index in range(6):
    f = files[index]
    name = names[index]
    G = graphs.GeneralAdjacency(f)

    I = FixedTSI_IW(G, losses, expectation_after=expectation_after, canonical=canonical, m_l=2000, m_p=2000, T=min(150, len(G.graph)//5))
    s = I.select_uniform_source()
    x = I.data_gen(s)

    save_model(I, s, x, name)
