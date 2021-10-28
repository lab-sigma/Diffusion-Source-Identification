from diffusion_source.infection_model import FixedTSI
from diffusion_source.graphs import RegularTree
from diffusion_source.discrepancies import L2_after

G = RegularTree(300, 4)

I = FixedTSI(G, L2_after, expectation_after=True, T=50)

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, 1000)

I2 = FixedTSI(G, L2_after, expectation_after=True, T=50)

for s in list(I2.G.graph.nodes):
    I2.precompute_probabilities(s, 1000)

I.include_probabilities(I2)

s = I.select_uniform_source()
x = I.data_gen(s)

print(I.confidence_set(x, 0.1))
