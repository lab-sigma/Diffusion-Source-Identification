from os import listdir
from os.path import isfile, join
from diffusion_source.infection_model import load_model


l_indices = [0, 1, 2]
l_names = ["L2", "ADiT", "ADT"]
colors = ["green", "orange", "blue"]

results_dir = "results/unweighted_timing"
#results_dir = "results/preferential_attachment_results_10000_2"

titles = ["regular tree", "small world", "preferential attachment"]
names = ["regular_tree", "small_world", "preferential_attachment"]

#titles = ["preferential attachment"]
#names = ["preferential_attachment"]

def gen_graph(mname, title):
    I, s, x = load_model(mname)
    rfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and mname in f]

    for f in rfiles:
        I.load_results(f)

    for i in I.results:
        print(I.results[i]["runtime"])


for name, title in zip(names, titles):
    gen_graph(name, title)
