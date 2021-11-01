import sys
from os import listdir
from os.path import isfile, join

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

I, s, x = load_model(name)

probs_dir = "probs1"
rfiles1 = [join(probs_dir, f) for f in listdir(probs_dir) if isfile(join(probs_dir, f)) and name in f]
probs_dir = "probs2"
rfiles = rfiles1 + [join(probs_dir, f) for f in listdir(probs_dir) if isfile(join(probs_dir, f)) and name in f]

for f in rfiles:
    I.load_probabilities(f)

I.store_probabilities("probs/{}.p".format(name))
