from os import listdir
from os.path import isfile, join
import pandas as pd

from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.infection_model import load_model
from diffusion_source.display import alpha_v_coverage, alpha_v_size


#l_indices = [0, 2, 3]
#l_names = ["L2", "ADiT", "ADT"]

results_dir = "results/unweighted_results"

names = ["regular_tree", "small_world", "preferential_attachment"]

def gen_graph(mname):
    I, s, x = load_model(mname)
    rfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f)) and name in f]

    for f in rfiles:
        I.load_results(f)


    #alpha_v_coverage(I.results, l_indices=l_indices, l_names=l_names, title="Unweighted SI Mean Coverage; {}".format(name))
    #alpha_v_size(I.results, l_indices=l_indices, l_names=l_names, title="Unweighted SI Mean Size; {}".format(name))
    alpha_v_coverage(I.results, l_names=I.loss_names, title="Unweighted SI Mean Coverage; {}".format(name))
    alpha_v_size(I.results, l_names=I.loss_names, title="Unweighted SI Mean Size; {}".format(name))

for name in names:
    gen_graph(name)
