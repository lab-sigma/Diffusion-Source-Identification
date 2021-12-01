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
    #"data/NorthAmericaHiring/BSchoolHiring.txt",
    #"data/NorthAmericaHiring/ComputerScienceHiring.txt",
    #"data/NorthAmericaHiring/HistoryHiring.txt",
    #"data/NorthAmericaHiring/StatisticsHiring.txt"
]

names = [
    "AirportFlightTraffic",
    "StatisticianCitations",
    #"BSchoolHiring",
    #"ComputerScienceHiring",
    #"HistoryHiring",
    #"StatisticsHiring"
]

#losses = [L2_after, L2_h, ADiT_h]
#lnames = [l.__name__ for l in losses]

l_indices = [0, 2, 3]
l_names = ["L2", "ADiT", "ADT"]
colors = ["green", "orange", "blue"]

#l_indices = [1, 2]
#l_names = ["L2", "ADiT"]
#colors = ["green", "orange"]

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
    legend=True
    show_y_label=True
    if (index > 0):
        legend = False
        show_y_label=False
    alpha_v_coverage(I.results, l_indices=l_indices, l_names=l_names, filename="Weighted SI Mean Coverage; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=name, colors=colors)
    alpha_v_size(I.results, l_indices=l_indices, l_names=l_names, filename="Weighted SI Mean Size; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=name, colors=colors)
