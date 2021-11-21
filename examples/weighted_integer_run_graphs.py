from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h
from diffusion_source.infection_model import load_model
from diffusion_source.display import alpha_v_coverage, alpha_v_size


results_dir = "results/final_weighted"

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

losses = [L2_after, L2_h, ADiT_h]
lnames = [l.__name__ for l in losses]

for index in range(len(names)):
#for index in [1]:
    name = names[index]
    I, s, x = load_model(name)

    rfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and name in f]
    if len(rfiles) == 0:
        continue

    for f in rfiles:
        I.load_results(f)

    print(name)
    print("m = 2000")
    alpha_v_coverage(I.results, l_names=I.loss_names, title="Weighted SI Mean Coverage; {}".format(name))
    alpha_v_size(I.results, l_names=I.loss_names, title="Weighted SI Mean Size; {}".format(name))
