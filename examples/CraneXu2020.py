import sys
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random

import networkx as nx
import numpy as np
from diffusion_source.graphs import InfRegular, Graph, RegularTree, WattsStrogatz, PreferentialAttachment
from diffusion_source.infection_model import load_model, FixedTSI, ICM, LTM
from diffusion_source.discrepancies import L2_h, L2_after, ADiT_h, ADT_h, Z_minus
from diffusion_source.display import sample_size_cdf, alpha_v_coverage, alpha_v_size
from diffusion_source.algos import CraneXuConfidence, CraneXuVals

axis_fontsize = 20
tick_fontsize = 16
legend_fontsize = 16
opacity_setting = 0.7
linewidth = 3.0

K = 100
T = 150
steps = 1000
N = 3000

alpha = np.linspace(0, 1+1/steps, num=steps)
crange = np.zeros(steps)
srange = np.zeros(steps)
Ps = []
for k in range(K):
    #G = InfRegular(4)
    G = FixedTSI(RegularTree(N, 4), [])
    #G = FixedTSI(PreferentialAttachment(N, 1), [])
    #G = FixedTSI(WattsStrogatz(N, 4), [])
    s = random.sample(set(G.G.graph.nodes()), 1)[0]
    x = G.data_gen(s)

    p = CraneXuVals(G.G, x)
    Ps += [p[s]]
    crange += (alpha >= p[s])/K

    for s, pi in p.items():
        srange += (alpha >= pi)/(T*K)

alphas = [i/10.0 for i in range(1, 10)]

#for alpha in alphas:
#    print("{}: {}".format(alpha, sum([alpha >= p for p in Ps])/K))

plt.plot(alpha, alpha, color="black", alpha=0.5)
plt.plot(alpha, crange, label="alpha v coverage", alpha=opacity_setting, color="green", linewidth=linewidth)
plt.show()

plt.plot(alpha, alpha, color="black", alpha=0.5)
plt.plot(alpha, srange, label="alpha v size", alpha=opacity_setting, color="green", linewidth=linewidth)
plt.show()

plt.plot(alpha, alpha, color="black", alpha=0.5)
plt.plot(crange, srange, label="coverage v size", alpha=opacity_setting, color="green", linewidth=linewidth)
plt.show()

alphas = [i/10.0 for i in range(1, 10)]

for alpha in alphas:
    print("{}: {}".format(alpha, sum([alpha >= p for p in Ps])/K))
