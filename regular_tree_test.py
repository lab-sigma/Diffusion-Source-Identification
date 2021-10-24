from infection_model import FixedTSI
from graphs import RegularTree
from discrepancies import L2_h

G = RegularTree(300, 4, 4)

I = FixedTSI(G, L2_h, T=10, m=100)

K = 10000
C = 0
for k in range(K):
    source = I.select_uniform_source()
    x = I.data_gen(source)

    c_set = I.confidence_set(x, 0.1)
    if source in c_set["L2_h"]:
        C += 1
    print("{:.2f} % : {:.2f}".format(k/K, C/(k+1)), end="\r")

print()
print()

print(C/10000)
