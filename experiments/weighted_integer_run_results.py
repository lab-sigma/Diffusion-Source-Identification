from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, L2_after
from diffusion_source.infection_model import load_model



results_dir = "results/l2_test"

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

results = pd.DataFrame([], columns=["Name", "N", "T", "K"])

alphas = [i/10 for i in range(1,10)]

losses = [L2_h, L2_after]
lnames = [l.__name__ for l in losses]

for a in alphas:
    for l in lnames:
        results["{}_coverage_{}".format(l, a)] = None
        results["{}_size_{}".format(l, a)] = None

#for index in range(len(names)):
for index in [1]:
    name = names[index]
    I, s, x = load_model(name)

    new_row = {
            "Name": name,
            "N": len(I.G.graph),
            "T": min(150, len(I.G.graph)//5),
            "K": 0
    }
    for a in alphas:
        for l in lnames:
            new_row["{}_coverage_{}".format(l, a)] = []
            new_row["{}_size_{}".format(l, a)] = []

    rfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and name in f]
    for f in rfiles:
        new_row["K"] += 1
        I.load_results(f)

    for a in alphas:
        C_sets = I.confidence_set(x, a, new_run=False, full=True)
        for t, Cs in C_sets.items():
            used_s = I.results[t]["meta"][2]
            for l, C in Cs.items():
                if used_s in C:
                    #new_row["{}_coverage_{}".format(l, a)] += 1/new_row["K"]
                    new_row["{}_coverage_{}".format(l, a)]
                new_row["{}_size_{}".format(l, a)] += len(C)/new_row["K"]

    results = results.append(new_row, ignore_index=True)

#results.transpose().to_csv("results/weighted_L2.csv")
print(results)
