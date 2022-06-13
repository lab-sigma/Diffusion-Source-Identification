from os import listdir
from os.path import isfile, join

from diffusion_source.infection_model import load_model
from diffusion_source.display import alpha_v_coverage, alpha_v_size, coverage_v_size


results_dir = "directed_results/final_weighted"

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

l_indices = [0, 2, 3]
l_names = ["L2", "ADiT", "ADT"]
colors = ["green", "orange", "blue"]

for index in range(len(names)):
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

    alpha_v_coverage(I, l_indices=l_indices, l_names=l_names, filename="Directed Weighted SI Mean Coverage; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=name, colors=colors)
    alpha_v_size(I, l_indices=l_indices, l_names=l_names, filename="Directed Weighted SI Mean Size; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=name, colors=colors, ratio=True)
    coverage_v_size(I, l_indices=l_indices, l_names=l_names, filename="Directed Weighted SI Coverage v Size; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=name, colors=colors, ratio=True)
