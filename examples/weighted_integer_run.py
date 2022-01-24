import sys

import diffusion_source.graphs as graphs
from diffusion_source.infection_model import load_model

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

arg = (int(sys.argv[1]) - 1)

K = 1
for k in range(K):
    for index in range(len(names)):
        f = files[index]
        name = names[index]

        I, s, x = load_model(name)

        #I.load_probabilities("probs/SI_{}.p".format(name))

        s = I.select_uniform_source()
        x = I.data_gen(s)

        I.p_values(x, meta=(name, x, s))

        I.store_results("results/final_weighted/{}_{}_{}.p".format(name, arg+1, k))
