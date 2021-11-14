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
mname = "SI_{}".format(name)

model_id = int(sys.argv[2])

I, s, x = load_model(mname)

m_p = 1

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, m_p)
    #print(s)
    #print(I.probabilities[1].getrow(s))

I.store_probabilities("probs/{}_{}.p".format(mname, model_id))
