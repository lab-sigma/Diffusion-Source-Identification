import diffusion_source.graphs as graphs
from diffusion_source.infection_model import FixedTSI_IW
from diffusion_source.display import alpha_v_size

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

for index in range(6):
    f = files[index]
    name = names[index]
    G = graphs.WeightedAdjacency(f)

    I = FixedTSI_IW(G, [L2_after, L2_h, ADiT_h], expectation_after=[True, True, True], canonical=[False, True, True], m_l=10, m_p=10, T=10)

    for k in range(K):
        s = I.select_uniform_source()
        x = I.data_gen(s)
        I.p_values(x, meta=("", x, s))

    alpha_v_size(I)
