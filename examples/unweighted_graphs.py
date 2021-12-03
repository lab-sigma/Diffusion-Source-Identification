from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.infection_model import load_model
from diffusion_source.display import alpha_v_coverage, alpha_v_size


l_indices = [0, 2, 3]
l_names = ["L2", "ADiT", "ADT"]
colors = ["green", "orange", "blue"]

results_dir = "results/unweighted_results"

titles = ["regular tree", "small world", "preferential attachment"]
names = ["regular_tree", "small_world", "preferential_attachment"]

def gen_graph(mname, title):
    I, s, x = load_model(mname)
    rfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and mname in f]

    for f in rfiles:
        I.load_results(f)


    legend=True
    show_y_label=True
    #if not mname == "regular_tree":
    #    legend=False
    #    show_y_label=False

    alpha_v_coverage(I, l_indices=l_indices, l_names=l_names, filename="Unweighted SI Mean Coverage; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=title, colors=colors)
    alpha_v_size(I, l_indices=l_indices, l_names=l_names, filename="Unweighted SI Mean Size; {}".format(name), save=True, legend=legend, show_y_label=show_y_label, title=title, colors=colors)
    #alpha_v_coverage(I, l_names=I.loss_names, title="Unweighted SI Mean Coverage; {}".format(mname))
    #alpha_v_size(I, l_names=I.loss_names, title="Unweighted SI Mean Size; {}".format(mname))

for name, title in zip(names, titles):
    gen_graph(name, title)
