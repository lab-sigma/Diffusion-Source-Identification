import diffusion_source.graphs as graphs
from diffusion_source.infection_model import save_model, FixedTSI
from diffusion_source.discrepancies import L2_after, L2_h, ADiT_h, ADT_h, Z_minus

N = 1365

losses = [L2_h, L2_after, ADiT_h, ADT_h, Z_minus]
expectation_after = [True, True, True, True, True]
canonical = [True, False, True, True, False]

G = graphs.RegularTree(N, 4)

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m=2000, T=min(150, len(G.graph)//5))
s = I.select_uniform_source()
x = I.data_gen(s)

save_model(I, s, x, "regular_tree")



G = graphs.WattsStrogatz(N, 4)

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m=2000, T=min(150, len(G.graph)//5))
s = I.select_uniform_source()
x = I.data_gen(s)

save_model(I, s, x, "small_world")



G = graphs.PreferentialAttachment(N, 1)

I = FixedTSI(G, losses, expectation_after=expectation_after, canonical=canonical, m=2000, T=min(150, len(G.graph)//5))
s = I.select_uniform_source()
x = I.data_gen(s)

save_model(I, s, x, "preferential_attachment")
