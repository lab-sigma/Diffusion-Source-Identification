from os import listdir
from os.path import isfile, join

from diffusion_source.infection_model import load_model

names = ["regular_tree", "small_world", "preferential_attachment"]
probs_dir = "probs"

def combine_name(mname):
    I, s, x = load_model(mname)

    rfiles = [join(probs_dir, f) for f in listdir(probs_dir) if isfile(join(probs_dir, f)) and mname in f]

    for f in rfiles:
        I.load_probabilities(f)

    if not I.probabilities[0] is None:
        I.store_probabilities("probs/{}.p".format(mname))

for name in names:
    combine_name(name)
