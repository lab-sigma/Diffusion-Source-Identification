import sys

import diffusion_source.graphs as graphs
from diffusion_source.infection_model import save_model, load_model

model_id = int(sys.argv[1])

m_p = 100

I, s, x = load_model("regular_tree")

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, m_p, convert=False)

I.store_probabilities("probs/{}_{}.p".format("regular_tree", model_id))


I, s, x = load_model("small_world")

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, m_p, convert=False)

I.store_probabilities("probs/{}_{}.p".format("small_world", model_id))


I, s, x = load_model("preferential_attachment")

for s in list(I.G.graph.nodes):
    I.precompute_probabilities(s, m_p, convert=False)

I.store_probabilities("probs/{}_{}.p".format("preferential_attachment", model_id))
