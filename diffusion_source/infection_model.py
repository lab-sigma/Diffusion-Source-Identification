import networkx as nx
import numpy as np
from scipy.stats import powerlaw
from scipy.stats import uniform
from collections import Counter
import random
import time
import copy
import pickle
import warnings

from scipy.sparse.linalg import eigsh
from scipy import sparse

from diffusion_source.isomorphisms import general_iso, compile_permutations
from diffusion_source.display import display_infected
from diffusion_source.discrepancies import L2_h

from abc import ABC

def save_model(I, s, x, name):
    pickle.dump((I, s, x), open("saved/" + name + ".p", "wb"))

def load_model(name):
    return pickle.load(open("saved/" + name + ".p", "rb"))

# Conditionally expand element
def cexpand(d, l):
    if not hasattr(d, "__len__"):
        return [d for _ in range(l)]
    return d

"""
    Infection model abstract base class; defines same confidence set computation structure for each type
"""
class InfectionModelBase(ABC):
    def __init__(self, G, discrepancies, discrepancy_names=None):
        self.G = G
        self.losses = discrepancies
        self.results = {}
        self.current = None
        self.losses = cexpand(discrepancies, 1)
        if discrepancy_names is None:
            self.loss_names = [l.__name__ for l in self.losses]
        else:
            self.loss_names = cexpand(discrepancy_names, 1)

    def store_results(self, fname):
        pickle.dump(self.results, open(fname, "wb"))

    def load_results(self, fname, append=True):
        results = pickle.load(open(fname, "rb"))
        if append:
            for t, r in results.items():
                if t in self.results:
                    warnings.warn("Added result timestamp {} already exists and was overwritten".format(t))
                self.results[t] = r
                if self.current is None:
                    self.current = t
                self.current = max(self.current, t)
        else:
            self.results = results

    def add_discrepancy(self, disc, name=None):
        self.losses.append(disc)
        if name is None:
            self.loss_names.append(disc.__name__)

    def create_groupings(self, x):
        pass

    def sample_prep(self, s, leader, samples, permutations, dependencies):
        pass

    def sample(self, s, dependencies):
        pass

    def compute_p_vals(self, x, s, samples_p, ratios):
        pass

    def data_gen(self, s):
        pass

    def select_uniform_source(self):
        self.source = random.sample(set(self.G.graph.nodes()), 1)[0]
        return self.source

    def p_values(self, x, meta=None, s=None):
        results = {
            "p_vals": {},
            "mu_x": {}
        }

        if not meta is None:
            results["meta"] = meta

        groupings, permutations = self.create_groupings(x)

        for leader, group, dependencies in groupings:

            if not s is None and not s in group:
                continue

            samples = self.sample(leader, dependencies)

            for sg in group:
                s = sg
                if hasattr(sg, "__len__"):
                    s = next(iter(sg))
                else:
                    sg = [sg]
                samples_p, ratios = self.sample_prep(s, leader, samples, permutations, dependencies)
                losses = self.compute_p_vals(x, s, samples_p, ratios)
                for si in sg:
                    results["p_vals"][si] = []
                    results["mu_x"][si] = []
                    for mu, psi in losses:
                        results["p_vals"][si] += [psi]
                        results["mu_x"][si] += [mu]

        self.current = time.time()
        self.results[self.current] = results

        return results

    def confidence_set(self, x, alpha, new_run=True, meta=None, full=False):
        if new_run:
            new_results = self.p_values(x, meta)
        else:
            if self.current is None:
                raise RuntimeError('Requested confidence set based on previous run with empty history.')
            new_results = self.results[self.current]

        def csets(results):
            p_vals = results["p_vals"]

            C_sets = {}
            for l_name in self.loss_names:
                C_sets[l_name] = set()

            for si, p in p_vals.items():
                for i, l_name in enumerate(self.loss_names):
                    if p[i] >= alpha:
                        C_sets[l_name].add(si)

            return C_sets

        if full:
            full_C = {}
            for t, r in self.results.items():
                full_C[t] = csets(r)
            return full_C

        return csets(new_results)


##########################################################################################################
##########################################################################################################
##########################################################################################################

"""
    Fixed infection size (T) Susceptible-Infected (SI) model; Model used in work presented at ICML 2021
"""
class FixedTSI(InfectionModelBase):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m=1000, T=150, iso=True, k_iso=10, d1=True):

        self.iso = iso
        self.k_iso = k_iso
        self.d1 = d1
        self.T = T
        self.m = m

        self.probabilities = [None for _ in range(T+1)]
        self.m_p = 0

        super().__init__(G, discrepancies, discrepancy_names)

        self.canonical = cexpand(canonical, len(self.losses))
        self.expectation_after = cexpand(expectation_after, len(self.losses))

    def add_discrepancy(self, disc, name=None, canonical=True, expectation_after=False):
        self.canonical.append(canonical)
        self.expectation_after.append(expectation_after)
        super().add_discrepancy(disc, name)

    def create_groupings(self, x):
        groupings = list()
        permutations = {}
        if not self.iso:
            leaders = {}
            for xi in x:
                groupings.append((xi, set([xi]), {}))
        else:
            f_iso = general_iso(self.G, x, self.k_iso)
            leaders, permutations = compile_permutations(f_iso)
            for v in x:
                if not any([v == l or v in leaders[l] for l in leaders.keys()]):
                    leaders[v] = set()
            for l, mem in leaders.items():
                mem.add(l)
                groupings.append((l, mem, {}))

        if self.d1:
            d1groupings = list()
            regroupings = list()
            for grouping in groupings:
                if self.G.graph.degree[grouping[0]] == 1:
                    d1groupings.append(grouping)
                else:
                    regroupings.append(grouping)
            for d1 in d1groupings:
                parent = next(iter(self.G.neighbors[d1[0]]))
                for i in range(len(regroupings)):
                    if parent in regroupings[i][1]:
                        regroupings[i][1].update(d1[1])
                        for v in d1[1]:
                            v_parent = next(iter(self.G.neighbors[v]))
                            regroupings[i][2][v] = v_parent
                            if (v_parent, regroupings[i][0]) in permutations:
                                permutations[(v, regroupings[i][0])] = permutations[(v_parent, regroupings[i][0])]

            groupings = regroupings

        return groupings, permutations

    def sample_prep(self, s, leader, samples, permutations, dependencies):
        if (s, leader) in permutations:
            permutation = permutations[(s, leader)]
            def permf(x):
                if x in permutation:
                    return permutation[x]
                return x
            permuted_samples = list()
            for sample_i in samples:
                permuted = {}
                for v in sample_i:
                    permuted[permf(v)] = sample_i[v].copy()
                permuted_samples.append(permuted)

            ratios = [1 for _ in range(len(samples))]

            if s in dependencies:
                return self.compute_ratios(permuted_samples, s, permutations)

            return permuted_samples, ratios

        if s in dependencies:
            return self.compute_ratios(copy.deepcopy(samples), s, permutations)

        return samples, [1 for _ in range(len(samples))]

    def sample(self, s, dependents):
        m = 2*self.m
        if s in self.probabilities and all(self.expectation_after):
            m = self.m
        return [self.single_sample(s) for i in range(m)]

    def precompute_probabilities(self, s, m_p=None, time=True, samples=None):
        if m_p is None:
            m_p = self.m
        self.m_p = m_p
        if time:
            for i in range(1, self.T+1):
                self.probabilities[i] = sparse.dok_matrix((len(G.graph), len(G.graph)), dtype=np.float32)
        else:
            self.probabilities[0] = sparse.dok_matrix((len(G.graph), len(G.graph)), dtype=np.float32)
        if samples is None:
            samples = [self.single_sample(s) for i in range(m_p)]
        for sample in samples:
            for v, (t, d) in sample.items():
                self.probabilities[0][s, v] += 1/m_p
                if time:
                    self.probabilities[t][s, v] += 1/m_p
        for i in range(self.T+1):
            self.probabilities[i] = self.probabilities[i].tocsr()

    def include_probabilities(self, probabilities, m_p):
        for t in range(len(probabilities)):
            if self.probabilities[t] is None and probabilities[t] is not None:
                self.probabilities[t] = probabilities[t]*m_p/(self.m_p+m_p)
            elif probabilities[t] is None and self.probabilities[t] is not None:
                self.probabilities[t] = self.probabilities[t]*self.m_p/(self.m_p+m_p)
            elif probabilities[t] is not None and self.probabilities[t] is not None:
                self.probabilities[t] = (probabilities[t]*m_p + self.probabilities[t]*self.m_p)/(self.m_p+m_p)
        self.m_p += m_p

    def load_probabilities(self, fname):
        m_p, probs = pickle.load(open(fname, "rb"))
        for i in range(len(probs)):
            probs[i] = probs[i].tocsr()
        self.include_probabilities(m_p, probs)

    def store_probabilities(self, fname):
        pickle.dump((self.m_p, self.probabilities), open(fname, "wb"))

    def node_vals(self, h_t, samples, ratios):
        vals = {}
        for sample, ratio in zip(samples, ratios):
            for v in sample.keys():
                if not v in vals.keys():
                    vals[v] = 0
                vals[v] += ratio * h_t(sample[v][0], self.T)
        return vals

    def temporal_loss(self, x, vals):
        Tx = 0
        for x_i in x:
            if x_i in vals.keys():
                Tx += vals[x_i]
        return -Tx

    def compute_p_vals(self, x, s, samples, ratios=None):
        if ratios is None:
            ratios = [1 for _ in range(len(samples))]

        losses = []
        for loss, canonical, expectation_after in zip(self.losses, self.canonical, self.expectation_after):
            if canonical or expectation_after:
                if expectation_after:
                    lf = loss
                    if not s in self.probabilities:
                        self.precompute_probabilities(s, samples=samples[self.m:])
                    samples_vals = list()
                    for i in range(len(self.probabilities)):
                        p_s = None
                        if self.probabilities[i] is not None:
                            p_s = self.probabilities[i].getrow(s)
                        samples_vals.append(p_s)
                else:
                    lf = self.temporal_loss
                    samples_vals = self.node_vals(loss, samples[self.m:], ratios[self.m:])

                mu_x = lf(x, samples_vals)
                psi = sum([ratio*(lf(yi, samples_vals) >= mu_x) for yi, ratio in zip(samples[:self.m], ratios[:self.m])])/(self.m)
            else:
                mu_x = loss(G, x, samples[:self.m], ratios[:self.m], s)
                psi = sum([loss(G, yi, samples[:self.m], s) >= mu_x for yi in samples[self.m:]])/self.m
            losses += [(mu_x, psi)]
        return losses

    def compute_ratios(self, samples, sp, permutations):
        sample_ratios = list()
        for i in range(len(samples)):
            if not sp in samples[i]:
                sample_ratios.append(0)
                continue
            ratio = 1/self.T
            sp_index = samples[i][sp][0]
            for v in samples[i]:
                if samples[i][v][0] < sp_index-1:
                    ratio *= 1/(1 - 1/samples[i][v][1])
                    samples[i][v][0] += 1
                elif samples[i][v][0] == sp_index-1:
                    ratio *= samples[i][v][1]
            samples[i][sp][0] = 1
            sample_ratios.append(ratio)
        return samples, sample_ratios

    def single_sample(self, s):
        edges = set()
        for n in self.G.neighbors[s]:
            edges.add((s, n))
        infected = {}
        infected[s] = [1, len(edges)]
        for i in range(1, self.T):
            jump = random.sample(edges, 1)[0]
            infected[jump[1]] = [i+1, 0]
            for n in self.G.neighbors[jump[1]]:
                if n in infected:
                    edges.discard((n, jump[1]))
                else:
                    edges.add((jump[1], n))
            infected[jump[1]][1] = len(edges)
        return infected

    def data_gen(self, s):
        return set(iter(self.single_sample(s).keys()))

    def select_source(self, seed=None):
        if seed is not None:
            random.seed(seed)
        u = nx.eigenvector_centrality_numpy(self.G.graph)
        median = np.median(np.absolute(list(u.values())))
        u = [ui for ui in u.keys() if abs(u[ui]) >= median]
        if len(u) == 0:
            u = set(self.G.graph.nodes)
        else:
            u = set(u)
        self.source = random.sample(u, 1)[0]
        return self.source


"""
    FixedTSI with weighted sampling
"""
class FixedTSI_Weighted(FixedTSI):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m=1000, T=150):
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m, T, iso=False, k_iso=10, d1=False)

    def single_sample(self, s):
        edges = list()
        weights = list()
        for e in self.G.graph.edges(s, data=True):
            edges += [e]
        infected = {}
        infected[s] = [1]
        for i in range(1, self.T):
            if len(edges) == 0:
                break
            weights = np.array([e[2]["weight"] for e in edges])
            jump_index = np.random.choice(list(range(len(edges))), 1, p=weights/sum(weights))[0]
            jump = edges[jump_index]
            infected[jump[1]] = [i+1]
            for e in self.graph.edges(jump[1], data=True):
                end = e[0]
                if e[0] == jump[1]:
                    end = e[1]
                if not end in infected:
                    edges += [e]
            edges = [e for e in edges if not (e[0] in infected and e[1] in infected)]
        return infected


"""
    FixedTSI with weighted sampling and integer weights
"""
class FixedTSI_IW(FixedTSI):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m=1000, T=150):
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m, T, iso=False, k_iso=10, d1=False)

    def single_sample(self, s):
        edges = set()
        for n in self.G.neighbors[s]:
            for i in range(int(self.G.graph[s][n]["weight"])):
                edges.add((s, n, i))
        infected = {}
        infected[s] = [1]
        for i in range(1, self.T):
            jump = random.sample(edges, 1)[0]
            infected[jump[1]] = [i+1]
            for n in self.G.neighbors[jump[1]]:
                if n in infected:
                    for j in range(int(self.G.graph[jump[1]][n]["weight"])):
                        edges.discard((n, jump[1], j))
                else:
                    for j in range(int(self.G.graph[jump[1]][n]["weight"])):
                        edges.add((jump[1], n, j))
        return infected


"""
    Independent Cascades Model
"""
class ICM(FixedTSI):
    def single_sample(self, s):
        for n in self.G.neighbors[s]:
            for i in range(int(self.G.graph[s][n]["weight"])):
                edges.add((s, n, i))
        infected = {}
        infected[s] = [1]
        for i in range(1, self.T):
            jump = random.sample(edges, 1)[0]
            infected[jump[1]] = [i+1]
            for n in self.G.neighbors[jump[1]]:
                if n in infected:
                    for j in range(int(self.G.graph[jump[1]][n]["weight"])):
                        edges.discard((n, jump[1], j))
                else:
                    for j in range(int(self.G.graph[jump[1]][n]["weight"])):
                        edges.add((jump[1], n, j))
        return infected
