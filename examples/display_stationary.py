from diffusion_source.graphs import RegularTree
from diffusion_source.display import display_stationary_dist

G = RegularTree(20, 3)

display_stationary_dist(G)
