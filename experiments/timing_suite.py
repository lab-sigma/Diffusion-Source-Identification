import sys
from os import listdir
from os.path import exists, isfile, join
from diffusion_source.infection_model import load_model

saved_dir = "saved"

mnames = [f[:-2] for f in listdir(saved_dir) if isfile(join(saved_dir, f))]

K = 1
arg = (int(sys.argv[1]) - 1)
index = arg % len(mnames)

def run_name(mname, k):
    print(mname)
    if exists("results/unweighted_results/{}_{}_{}.p".format(mname, arg+1, k)):
        return
    I, s, x = load_model(mname)
    #I.load_probabilities("probs/{}.p".format(mname))
    s = I.select_uniform_source()
    x = I.data_gen(s)

    I.p_values(x, meta=(mname, x, s))
    I.store_results("results/timing_results/{}_{}_{}.p".format(mname, arg+1, k))


for k in range(K):
    run_name(mnames[index], k)
