import sys

import diffusion_source.graphs as graphs
from diffusion_source.infection_model import save_model, load_model

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

arg = int(sys.argv[1])
index = arg % len(names)

f = files[index]
name = names[index]

model_id = int(sys.argv[2])

I, s, x = load_model(name)

m_p = 1000

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, m_p)

save_model(I, s, x, "{}_{}".format(name, model_id))
