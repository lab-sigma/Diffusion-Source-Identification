import sys

import diffusion_source.graphs as graphs
import networkx as nx
from diffusion_source.infection_model import load_model, FixedTSI_IW, ICM, LTM
from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.display import sample_size_cdf, alpha_v_coverage, alpha_v_size

files = [
    #"data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    #"data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    "data/NorthAmericaHiring/ComputerScienceHiring.txt",
    "data/NorthAmericaHiring/HistoryHiring.txt",
    "data/NorthAmericaHiring/StatisticsHiring.txt"
]

names = [
    #"AirportFlightTraffic",
    #"StatisticianCitations",
    "BSchoolHiring",
    "ComputerScienceHiring",
    "HistoryHiring",
    "StatisticsHiring"
]

arg = (int(sys.argv[1]) - 1)

for index in range(len(names)):
    f = files[index]
    name = names[index]
    G = graphs.WeightedAdjacency(f)


    I, s, x = load_model("IC_{}".format(name))
    I.load_probabilities("probs/LT_{}.p".format(name))
    s = I.select_uniform_source()
    x = I.data_gen(s)

    I.p_values(x, meta=(name, x, s))
    I.store_results("results/IC_results/{}_{}.p".format(name, arg+1))

    I, s, x = load_model("LT_{}".format(name))
    I.load_probabilities("probs/LT_{}.p".format(name))
    s = I.select_uniform_source()
    x = I.data_gen(s)

    I.p_values(x, meta=(name, x, s))
    I.store_results("results/LT_results/{}_{}.p".format(name, arg+1))
