import pandas as pd
import numpy as np

import diffusion_source.graphs as graphs
from diffusion_source.discrepancies import L2_h, ADiT_h
from diffusion_source.infection_model import load_model
import pickle


results_dir = "results/l2_test"

results = pd.DataFrame([], columns=["Model", "Date", "Threshold", "T", "p", "0.1_size", "0.2_size"])

losses = [L2_h, ADiT_h]
lnames = [l.__name__ for l in losses]

df = pd.read_csv("data/BaiduMobility/TrafficVol.csv", sep=',', index_col=0)
mat = df.to_numpy()
mat = np.nan_to_num(df)
trunc = 1
mat[abs(mat) < trunc] = 0.0

GD = graphs.DirectedFromAdjacency(mat)
G = graphs.FromAdjacency(mat)

cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
#cc = pd.read_csv("data/BaiduMobility/InfectionStatus.csv", index_col=0)
names = np.array(list(df.columns))
dates = list(range(5, 20, 5))
thresh = list(range(1, 10, 2))
#thresh = [10]
s = np.where(names == 'Wuhan')[0][0]
source = s

unweighted = pickle.load(open("broken_results/baidu/unweighted_2000.p", "rb"))
weighted = pickle.load(open("broken_results/baidu/weighted_2000.p", "rb"))

for t in thresh:
    for d in dates:
        x = set()
        for i in range(len(G.graph)):
            if cc.iloc[i, d] >= t:
                x.add(i)
        if len(x) <= 1:
            continue

        for l in range(len(losses)):
            lname = lnames[l]
            p_vals = weighted[(t, d)]["p_vals"]
            size1 = 0
            size2 = 0
            for s, ps in p_vals.items():
                if ps[l] >= 0.1:
                    size1 += 1
                if ps[l] >= 0.2:
                    size2 += 1

            new_row = {
                "Model": "Weighted",
                "Date": d,
                "Threshold": t,
                "T": len(x)-1,
                "loss": lname,
                "p": p_vals[s][l],
                "0.1_size": size1,
                "0.2_size": size1
            }

            results = results.append(new_row, ignore_index=True)

        for l in range(len(losses)):
            lname = lnames[l]
            p_vals = unweighted[(t, d)]["p_vals"]
            size1 = 0
            size2 = 0
            for s, ps in p_vals.items():
                if ps[l] >= 0.1:
                    size1 += 1
                if ps[l] >= 0.2:
                    size2 += 1

            new_row = {
                "Model": "Unweighted",
                "Date": d,
                "Threshold": t,
                "T": len(x)-1,
                "loss": lname,
                "p": p_vals[s][l],
                "0.1_size": size1,
                "0.2_size": size1
            }

            results = results.append(new_row, ignore_index=True)

results.transpose().to_csv("broken_results/baidu_2000.csv")
print(results)
