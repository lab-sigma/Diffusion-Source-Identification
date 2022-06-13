from diffusion_source.infection_model import load_model
from diffusion_source.display import display_infected

I, x, s = load_model("iso_fail")

#x = set(I.G.graph.nodes)

#groupings, permutations = I.create_groupings(x)

#print(groupings)
#print(permutations.keys())
print(s)
print(x)

display_infected(I.G, x, s)

