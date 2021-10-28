from diffusion_source.infection_model import FixedTSI
from diffusion_source.graphs import RegularTree
from diffusion_source.discrepancies import L2_h

G = RegularTree(100, 4, 4)

I = FixedTSI(G, L2_h)
