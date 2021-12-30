from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, ADiT_h, ADT_h
from diffusion_source.infection_model import load_model


names = ["regular_tree", "preferential_attachment", "small_world"]

results_dir = "results/unweighted_results_10000"

results = pd.DataFrame([], columns=["Name", "N", "T", "K"])

alphas = [i/10 for i in range(1,10)]

losses = [L2_h, ADiT_h, ADT_h]
lnames = [l.__name__ for l in losses]

for a in alphas:
    for l in lnames:
        results["{}_coverage_{}".format(l, a)] = None
        results["{}_size_{}".format(l, a)] = None

#for index in range(len(names)):
for index in range(len(names)):
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
            new_row["{}_coverage_{}".format(l, a)] = 0
            new_row["{}_size_{}".format(l, a)] = 0

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
                    new_row["{}_coverage_{}".format(l, a)] += 1/new_row["K"]
                new_row["{}_size_{}".format(l, a)] += len(C)/new_row["K"]

    results = results.append(new_row, ignore_index=True)

results.transpose().to_csv("results/unweighted_10000.csv")
print(results)
