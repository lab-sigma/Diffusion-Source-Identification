import sys
from diffusion_source.infection_model import load_model

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

arg = int(sys.argv[1])
index = arg % len(names)

f = files[index]
name = names[index]
mname = "IC_{}".format(name)

model_id = int(sys.argv[2])

I, s, x = load_model(mname)

m_p = 1000

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, m_p, convert=False)

I.store_probabilities("probs/{}_{}.p".format(mname, model_id))
