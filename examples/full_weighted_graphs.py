from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.infection_model import load_model
from diffusion_source.display import alpha_v_coverage, alpha_v_size, coverage_v_size


files = [
    #"data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    #"data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    #"data/NorthAmericaHiring/ComputerScienceHiring.txt",
    #"data/NorthAmericaHiring/HistoryHiring.txt",
    #"data/NorthAmericaHiring/StatisticsHiring.txt"
]

names = [
    #"AirportFlightTraffic",
    #"StatisticianCitations",
    "BSchoolHiring",
    #"ComputerScienceHiring",
    #"HistoryHiring",
    #"StatisticsHiring"
]

losses = [L2_h, L2_after, ADiT_h, ADT_h, Z_minus]
lnames = [l.__name__ for l in losses]

l_indices = [0, 2, 3, 4]
l_names = ["Z+", "ADiT", "ADT", "Z-"]
colors = ["green", "orange", "blue", "purple"]

IC_results = "results/IC_results"
LT_results = "results/LT_results"

for index in range(len(names)):
#for index in [1]:
    name = names[index]
    I, s, x = load_model("IC_{}".format(name))

    rfiles = [join(IC_results, f) for f in listdir(IC_results) if isfile(join(IC_results, f)) and name in f]
    if len(rfiles) == 0:
        continue

    for f in rfiles:
        I.load_results(f)

    print(name)
    print("IC")
    alpha_v_coverage(I.results, l_indices=l_indices, l_names=l_names, filename="IC Mean Coverage; {}".format(name), save=True, legend=True, show_y_label=True, title=name, colors=colors)
    alpha_v_size(I.results, l_indices=l_indices, l_names=l_names, filename="IC Mean Size; {}".format(name), ratio=True, save=True, legend=True, show_y_label=True, title=name, colors=colors)
    coverage_v_size(I.results, l_indices=l_indices, l_names=l_names, filename="IC Coverage v Size; {}".format(name), ratio=True, save=True, legend=True, show_y_label=True, title=name, colors=colors)

    I, s, x = load_model("LT_{}".format(name))

    rfiles = [join(LT_results, f) for f in listdir(LT_results) if isfile(join(LT_results, f)) and name in f]
    if len(rfiles) == 0:
        continue

    for f in rfiles:
        I.load_results(f)

    print("LT")
    alpha_v_coverage(I.results, l_indices=l_indices, l_names=l_names, filename="LT Mean Coverage; {}".format(name), save=True, legend=False, show_y_label=False, title=name, colors=colors)
    alpha_v_size(I.results, l_indices=l_indices, l_names=l_names, filename="LT Mean Size; {}".format(name), ratio=True, save=True, legend=False, show_y_label=False, title=name, colors=colors)
    coverage_v_size(I.results, l_indices=l_indices, l_names=l_names, filename="LT Coverage v Size; {}".format(name), ratio=True, save=True, legend=False, show_y_label=False, title=name, colors=colors)