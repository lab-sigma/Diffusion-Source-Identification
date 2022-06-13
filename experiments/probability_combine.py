from os import listdir
from os.path import isfile, join

import diffusion_source.graphs as graphs
from diffusion_source.infection_model import load_model

files = [
    "data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    #"data/StatisticianCitation/TotalCite.txt",
    #"data/NorthAmericaHiring/BSchoolHiring.txt",
    #"data/NorthAmericaHiring/ComputerScienceHiring.txt",
    #"data/NorthAmericaHiring/HistoryHiring.txt"
]

names = [
    "AirportFlightTraffic",
    #"StatisticianCitations",
    #"BSchoolHiring",
    #"ComputerScienceHiring",
    #"HistoryHiring"
]

probs_dir = "AFT_probs"

for index in range(len(names)):
    f = files[index]
    name = names[index]
    mname = "SI_{}".format(name)

    I, s, x = load_model(mname)

    rfiles = [join(probs_dir, f) for f in listdir(probs_dir) if isfile(join(probs_dir, f)) and mname in f]

    for f in rfiles:
        I.load_probabilities(f)

    if not I.probabilities[0] is None:
        I.store_probabilities("probs/{}.p".format(mname))
