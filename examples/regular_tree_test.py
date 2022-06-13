from diffusion_source.infection_model import FixedTSI
from diffusion_source.graphs import RegularTree
from diffusion_source.discrepancies import L2_h

G = RegularTree(300, 4)

I = FixedTSI(G, L2_h, T=10, m_l=10, m_p=10, d1=False, iso=False)

K = 1000
C = 0
S = 0
for k in range(K):
    source = I.select_uniform_source()
    x = I.data_gen(source)

    c_set = I.confidence_set(x, 0.1)
    if source in c_set["L2_h"]:
        C += 1
    S += len(c_set["L2_h"])
    print("{:.2f} % : {:.2f}, {:.2f}".format(k/K, C/(k+1), S/(k+1)), end="\r")

print()
print()

print(C/K)
