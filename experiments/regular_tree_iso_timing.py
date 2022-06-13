from diffusion_source.infection_model import FixedTSI
from diffusion_source.graphs import RegularTree
from diffusion_source.discrepancies import L2_h
import sys

k_iso = int(sys.argv[1])
m = int(sys.argv[2])

G = RegularTree(1365, 4)

I = FixedTSI(G, L2_h, T=150, m_l=m, m_p=m, d1=False, iso=True, k_iso=k_iso)

source = I.select_uniform_source()
x = I.data_gen(source)

I.p_values(x, meta=("k: {}".format(k_iso), x, source))
I.store_results("results/iso_timing_results/{}_{}_{}.p".format("regular_tree", m, k_iso))
