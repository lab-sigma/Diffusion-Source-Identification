import sys
from os.path import exists
from diffusion_source.infection_model import load_model

names = ["regular_tree", "small_world", "preferential_attachment"]
K = 1
arg = (int(sys.argv[1]) - 1)

def run_name(mname, k):
    if exists("results/unweighted_results/{}_{}_{}.p".format(mname, arg+1, k)):
        return
    I, s, x = load_model(mname)
    #I.load_probabilities("probs/{}.p".format(mname))
    s = I.select_uniform_source()
    x = I.data_gen(s)

    I.p_values(x, meta=(mname, x, s))
    I.store_results("results/unweighted_results/{}_{}_{}.p".format(mname, arg+1, k))

for k in range(K):
    for name in names:
        run_name(name, k)
