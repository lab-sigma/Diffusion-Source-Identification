from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle

networks = pickle.load(open("data/CommunityFitNet/Benchmark/CommunityFitNet.pickle", "rb"))

edgelists = networks['edges_id']
gp = networks['graphProperties']
nn = networks['number_nodes']
network_titles = networks['title']
network_domains = networks['networkDomain']
url = networks['sourceUrl']

results_dir = "results/community_timing"

mnames = [f[:-4] for f in listdir(results_dir) if isfile(join(results_dir, f))]
fnames = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f))]

results_frame = pd.DataFrame([], columns=["Name", "T", "K", "m", "domain", "d1", "iso", "setup_timing", "sample_timing", "loss_timing", "total_timing"])

for m, f in zip(mnames, fnames):
    results_dict = pickle.load(open(f, "rb"))
    for t, r in results_dict.items():
        results = r

    name = m[:len(m) - 1 - m[::-1].index("_")]

    index = int(name[len(name) - name[::-1].index("_"):])

    new_row = {
            "Name": name,
            "network_title": network_titles.iloc[index],
            "T": len(results["meta"][1]),
            "K": 1,
            "m": 0,
            "domain": network_domains.iloc[index],
            "d1": "No",
            "iso": "No",
            "setup_timing": results["grouping_time"],
            "sample_timing": results["sampling_time"],
            "loss_timing": results["loss_time"],
            "total_timing": results["grouping_time"] + results["sampling_time"] + results["loss_time"]
    }

    if name in results_frame.Name.unique():
        results_frame.loc[results_frame["Name"] == name, ["K"]] += 1
        results_frame.loc[results_frame["Name"] == name, ["setup_timing"]] += new_row["setup_timing"]
        results_frame.loc[results_frame["Name"] == name, ["sample_timing"]] += new_row["sample_timing"]
        results_frame.loc[results_frame["Name"] == name, ["loss_timing"]] += new_row["loss_timing"]
        results_frame.loc[results_frame["Name"] == name, ["total_timing"]] += new_row["total_timing"]
        continue

    if "_2000_" in m:
        new_row["m"] = 2000
    else:
        new_row["m"] = 10000

    if "d1" in m or "both" in m:
        new_row["d1"] = "Yes"

    if "iso" in m or "both" in m:
        new_row["iso"] = "Yes"

    results_frame = results_frame.append(new_row, ignore_index=True)

results_frame["setup_timing"] = (1/results_frame["K"])*results_frame["setup_timing"]
results_frame["sample_timing"] = (1/results_frame["K"])*results_frame["sample_timing"]
results_frame["loss_timing"] = (1/results_frame["K"])*results_frame["loss_timing"]
results_frame["total_timing"] = (1/results_frame["K"])*results_frame["total_timing"]

results_frame["Name"] = results_frame["Name"].str.replace(',', '')
results_frame["Name"] = results_frame["Name"].str.replace(';', '')
results_frame.to_csv("results/community_timing_results.csv")
print(results_frame)
