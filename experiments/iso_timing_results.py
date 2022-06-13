from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle


results_dir = "results/iso_timing_results"

mnames = [f[:-2] for f in listdir(results_dir) if isfile(join(results_dir, f))]
fnames = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f))]

results_frame = pd.DataFrame([], columns=["m", "iso_k", "setup_timing", "sample_timing", "loss_timing", "total_timing"])

for m, f in zip(mnames, fnames):
    results_dict = pickle.load(open(f, "rb"))
    for t, r in results_dict.items():
        results = r

    new_row = {
            "m": 0,
            "iso_k": 0,
            "setup_timing": results["grouping_time"],
            "sample_timing": results["sampling_time"],
            "loss_timing": results["loss_time"],
            "total_timing": results["grouping_time"] + results["sampling_time"] + results["loss_time"]
    }

    if "2000" in m:
        new_row["m"] = 2000
    elif "10000" in m:
        new_row["m"] = 10000
    elif "500" in m:
        new_row["m"] = 500
    else:
        new_row["m"] = 1000

    new_row["iso_k"] = int(m[-1])
    if new_row["iso_k"] == 0:
        new_row["iso_k"] = 10

    results_frame = results_frame.append(new_row, ignore_index=True)

results_frame.transpose().to_csv("results/iso_timing_results.csv")
print(results_frame)
