from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, L2_after
from diffusion_source.infection_model import load_model
from diffusion_source.display import alpha_v_coverage


results_dir = "broken_results/final_vary_m"

files = [
    #"data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    "data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    "data/NorthAmericaHiring/ComputerScienceHiring.txt",
    "data/NorthAmericaHiring/HistoryHiring.txt"
]

names = [
    #"AirportFlightTraffic",
    "StatisticianCitations",
    "BSchoolHiring",
    "ComputerScienceHiring",
    "HistoryHiring"
]

losses = [L2_h, L2_after]
lnames = [l.__name__ for l in losses]

ms = [200, 800, 2000, 10000]

for m in ms:
    for index in range(len(names)):
    #for index in [1]:
        name = names[index]
        I, s, x = load_model(name)
        mname = "{}_{}".format(m, name)

        rfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and mname in f]
        if len(rfiles) == 0:
            continue

        print(mname)

        for f in rfiles:
            I.load_results(f)

        alpha_v_coverage(I, l_names=I.loss_names, title="LT Mean Coverage; {}; m = {}".format(name, m))
