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
from collections import defaultdict

from scipy.sparse.linalg import eigsh
from scipy import sparse

from diffusion_source.isomorphisms import general_iso, compile_permutations
from diffusion_source.display import display_infected
from diffusion_source.discrepancies import L2_h

from abc import ABC

"""
    Functions for saving and loading an infection model as a pickle file
"""

def save_model(I, s, x, name):
    pickle.dump((I, s, x), open("saved/" + name + ".p", "wb"))

def load_model(name):
    return pickle.load(open("saved/" + name + ".p", "rb"))

# Conditionally expand element (helper function)
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

    """
        Store results as a pickle file at the given filename
    """
    def store_results(self, fname):
        pickle.dump(self.results, open(fname, "wb"))

    """
        Load results from
    """
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

    def source_candidates(self, x):
        return x

    def p_values(self, x, ncore=1, meta=None, true_s=None):
        results = {
            "p_vals": {},
            "mu_x": {},
            "runtime": time.time(),
            "grouping_time": 0,
            "sampling_time": 0,
            "loss_time": 0
        }

        if not meta is None:
            results["meta"] = meta

        candidates = self.source_candidates(x)

        results["grouping_time"] = time.time()
        groupings, permutations = self.create_groupings(candidates)
        results["grouping_time"] = time.time() - results["grouping_time"]

        def check_group(pack):
            leader, group, dependencies = pack[0], pack[1], pack[2]
            if not true_s is None and not true_s in group:
                return None, None, None

            sampling_time = time.time()
            samples = self.sample(leader, dependencies)
            sampling_time = time.time() - sampling_time

            mapping = {}

            loss_time = time.time()
            for sg in group:
                s = sg
                if hasattr(sg, "__len__"):
                    s = next(iter(sg))
                else:
                    sg = [sg]
                samples_p, ratios = self.sample_prep(s, leader, samples, permutations, dependencies)
                losses = self.compute_p_vals(x, s, samples_p, ratios)
                for si in sg:
                    mapping[si] = losses
            loss_time = time.time() - loss_time
            return mapping, sampling_time, loss_time

        if ncore <= 1:
            mappings = [(check_group((l, g, d))) for l, g, d in groupings] # sequential
        else:
            with Pool(ncore) as p:
                mappings = p.map(check_group, groupings) # parallel

        bt = float('inf')
        for mapping, st, lt in mappings:
            if mapping is None:
                continue
            if ncore <= 1:
                results["sampling_time"] += st
                results["loss_time"] += lt
            else:
                if st + lt < bt:
                    results["sampling_time"] = st
                    results["loss_time"] = lt
            for si in mapping.keys():
                results["p_vals"][si] = []
                results["mu_x"][si] = []
                for mu, psi_g, psi_e in mapping[si]:
                    results["p_vals"][si] += [(psi_g, psi_e)]
                    results["mu_x"][si] += [mu]

        results["runtime"] = time.time() - results["runtime"]
        self.current = time.time()
        self.results[self.current] = results

        return results

    def confidence_set(self, x, alpha, new_run=True, meta=None, full=False):
        if new_run:
            new_results = self.p_values(x, meta=meta)
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
                    if p[i][0] + random.random()*p[i][1] > alpha:
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
            expectation_after=False, m_l=1000, m_p=1000, T=150, iso=True, k_iso=10, d1=True):

        self.iso = iso
        self.k_iso = k_iso
        self.d1 = d1
        self.T = T
        self.m_l = m_l
        self.m_p = m_p

        self.saved_probs = set()
        self.probabilities = [None for _ in range(T+2)]

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
        m = self.m_l + self.m_p
        if s in self.saved_probs and all(self.expectation_after):
            m = self.m_p
        return [self.single_sample(s) for i in range(m)]

    def precompute_probabilities(self, s, time=True, samples=None, convert=True):
        self.saved_probs.add(s)
        if time:
            for t in range(1, self.T+2):
                if self.probabilities[t] is None:
                    self.probabilities[t] = sparse.dok_matrix((len(self.G.graph), len(self.G.graph)), dtype=np.short)
                elif convert:
                    self.probabilities[t] = self.probabilities[t].todok()
        if self.probabilities[0] is None:
            self.probabilities[0] = sparse.dok_matrix((len(self.G.graph), len(self.G.graph)), dtype=np.short)
        elif convert:
            self.probabilities[0] = self.probabilities[0].todok()
        if samples is None:
            samples = [self.single_sample(s) for i in range(m_l)]
        for sample in samples:
            for v, meta in sample.items():
                self.probabilities[0][s, v] += 1
                if time:
                    if meta[0] >= len(self.probabilities):
                        for t in range(len(self.probabilities), meta[0]+1):
                            self.probabilities.append(sparse.dok_matrix((len(self.G.graph), len(self.G.graph)), dtype=np.short))
                    self.probabilities[meta[0]][s, v] += 1
        if convert:
            for t in range(self.T+2):
                self.probabilities[t] = self.probabilities[t].tocsr()

    def include_probabilities(self, probabilities, m_l):
        for t in range(len(probabilities)):
            if t >= len(self.probabilities):
                self.probabilities.append(probabilities[t])
            elif self.probabilities[t] is None and probabilities[t] is not None:
                self.probabilities[t] = probabilities[t]
            elif probabilities[t] is None and self.probabilities[t] is not None:
                self.probabilities[t] = self.probabilities[t]
            elif probabilities[t] is not None and self.probabilities[t] is not None:
                self.probabilities[t] = probabilities[t] + self.probabilities[t]
        self.m_l += m_l
        for v in self.G.graph.nodes():
            if self.probabilities[0].getrow(v).sum() > 0:
                self.saved_probs.add(v)

    def load_probabilities(self, fname):
        m_l, probs = pickle.load(open(fname, "rb"))
        for i in range(len(probs)):
            probs[i] = probs[i].tocsr()
        self.include_probabilities(probs, m_l)

    def store_probabilities(self, fname):
        for i in range(len(self.probabilities)):
            self.probabilities[i] = self.probabilities[i].tocsr()
        pickle.dump((self.m_l, self.probabilities), open(fname, "wb"))

    def node_vals(self, h_t, samples, ratios):
        vals = defaultdict(int)
        for sample, ratio in zip(samples, ratios):
            for v in sample.keys():
                vals[v] += ratio * h_t(sample[v][0], self.T)
        return vals

    def temporal_loss(self, x, vals):
        Tx = len(x)*self.m_l
        for x_i in x:
            Tx -= 2*vals[x_i]
        return Tx

    def temporal_loss_after(self, x, P):
        Tx = len(x)*self.m_l
        for x_i in x:
            Tx -= 2*P[0, x_i]
        return Tx

    def compute_p_vals(self, x, s, samples, ratios=None):
        if ratios is None:
            ratios = [1 for _ in range(len(samples))]

        losses = []

        for loss, canonical, expectation_after in zip(self.losses, self.canonical, self.expectation_after):
            if canonical or expectation_after:
                if expectation_after:
                    if not s in self.saved_probs:
                        self.precompute_probabilities(s, samples=samples[self.m_p:])
                    if canonical:
                        sample_vals = loss(1, self.T)*self.probabilities[1].getrow(s)
                        for t in range(2, len(self.probabilities)):
                            sample_vals += loss(t, self.T)*self.probabilities[t].getrow(s)
                        lf = self.temporal_loss_after
                    else:
                        sample_vals = list()
                        for i in range(len(self.probabilities)):
                            p_s = None
                            if self.probabilities[i] is not None:
                                p_s = self.probabilities[i].getrow(s)
                            sample_vals.append(p_s)
                        lf = lambda t, T: loss(t, T, self.m_p)
                    mu_x = lf(x, sample_vals)
                    lvals = [lf(yi, sample_vals) for yi in samples[:self.m_p]]
                else:
                    samples_vals = self.node_vals(loss, samples[self.m_p:], ratios[self.m_p:])

                    mu_x = self.temporal_loss(x, samples_vals)
                    lvals = [self.temporal_loss(yi, samples_vals) for yi in samples[:self.m_p]]
                psi_g = sum([ratio*(lv > mu_x) for lv, ratio in zip(lvals, ratios[:self.m_p])])/self.m_p
                psi_e = sum([ratio*(lv == mu_x) for lv, ratio in zip(lvals, ratios[:self.m_p])])/self.m_p
            else:
                mu_x = loss(self.G, x, samples[:self.m_p], ratios[:self.m_p], s)
                lvals = [loss(self.G, yi, samples[:self.m_p], ratios[:self.m_p], s) for yi in samples[self.m_p:]]
                psi_g = sum([lv > mu_x for lv in lvals])/self.m_p
                psi_e = sum([lv == mu_x for lv in lvals])/self.m_p
            losses += [(mu_x, psi_g, psi_e)]
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
        for i in range(1, self.T+1):
            if len(edges) == 0:
                break
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


class FixedTSI_Directed(FixedTSI):
    def source_candidates(self, x):
        gs = self.G.graph.subgraph(x)
        sub_distances = {}
        for d in nx.all_pairs_shortest_path_length(gs):
            sub_distances[d[0]] = d[1]
        candidates = []
        for n in x:
            if n in sub_distances.keys() and len((set(x) - set(sub_distances[n].keys())) - set([n])) == 0:
                candidates += [n]

        return set(candidates)

"""
    FixedTSI with weighted sampling
"""
class FixedTSI_Weighted(FixedTSI_Directed):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m_l=1000, m_p=1000, T=150):
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m_l, m_p, T, iso=False, k_iso=10, d1=False)

    def single_sample(self, s):
        edges = list()
        weights = list()
        for e in self.G.graph.edges(s, data=True):
            edges += [e]
        infected = {}
        infected[s] = [1]
        for i in range(1, self.T+1):
            if len(edges) == 0:
                break
            weights = np.array([e[2]["weight"] for e in edges])
            jump_index = np.random.choice(list(range(len(edges))), 1, p=weights/sum(weights))[0]
            jump = edges[jump_index]
            infected[jump[1]] = [i+1]
            for e in self.G.graph.edges(jump[1], data=True):
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
class FixedTSI_IW(FixedTSI_Directed):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m_l=1000, m_p=1000, T=150):
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m_l, m_p, T, iso=False, k_iso=10, d1=False)

    def single_sample(self, s):
        edges = set()
        for n in self.G.neighbors[s]:
            for i in range(int(self.G.graph[s][n]["weight"])):
                edges.add((s, n, i))
        infected = {}
        infected[s] = [1]
        for i in range(1, self.T+1):
            if len(edges) == 0:
                break
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
    Independent Cascades Model with fixed propagation probability
"""
class ICM_fp(FixedTSI_Directed):
    def __init__(self, G, discrepancies, p, discrepancy_names=None, canonical=True,
            expectation_after=False, m_l=1000, m_p=1000, iso=True, k_iso=10):
        self.p = p
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m_l, m_p, -1, iso=iso, k_iso=k_iso, d1=False)
    def single_sample(self, s):
        active = set([s])
        inactive = {}
        inactive[s] = [1]
        t = 1
        while not len(active) == 0:
            t += 1
            new_active = set()
            for a in active:
                for n in self.G.neighbors[a]:
                    if n in inactive:
                        continue
                    if random.random() < p:
                        new_active.add(n)
                        inactive[n] = [t]
            active = new_active
        return inactive

"""
    Independent Cascades Model
"""
class ICM(FixedTSI_Directed):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m_l=1000, m_p=1000, T=-1):
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m_l, m_p, T, iso=False, k_iso=10, d1=False)
    def single_sample(self, s):
        active = set([s])
        inactive = {}
        inactive[s] = [1]
        t = 1
        while not (len(active) == 0 or self.T - t == -1):
            t += 1
            new_active = set()
            for a in active:
                for n in self.G.neighbors[a]:
                    if n in inactive or n in new_active:
                        continue
                    if random.random() < self.G.graph[a][n]["weight"]:
                        new_active.add(n)
                    #if self.T >= 0 and len(new_active) + len(inactive) > self.T + 1:
                    #    return inactive
            for a in new_active:
                inactive[a] = [t]
            active = new_active
        return inactive

"""
    Linear Threshold Model
"""
class LTM(FixedTSI_Directed):
    def __init__(self, G, discrepancies, discrepancy_names=None, canonical=True,
            expectation_after=False, m_l=1000, m_p=1000, T=-1):
        super().__init__(G, discrepancies, discrepancy_names, canonical, expectation_after, m_l, m_p, T, iso=False, k_iso=10, d1=False)
    def single_sample(self, s):
        added = set([s])
        active = {}
        active[s] = [1]
        thresholds = {}
        influence = {}
        t = 1
        while not (len(added) == 0 or self.T - t == -1):
            t += 1
            newly_added = set()
            for a in added:
                for n in self.G.neighbors[a]:
                    if n in active:
                        continue
                    if n not in influence:
                        influence[n] = 0
                    if not n in thresholds:
                        thresholds[n] = random.random()
                    influence[n] += self.G.graph[a][n]["weight"]
                    if influence[n] >= thresholds[n]:
                        newly_added.add(n)
            for n in newly_added:
                active[n] = [t]
            added = newly_added
        return active
