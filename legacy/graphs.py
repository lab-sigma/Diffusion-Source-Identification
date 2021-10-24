import networkx as nx
import numpy as np
from scipy.stats import powerlaw
import random

class Graph:
    def __init__(self, N, setup_distances=True):
        self.N = N

        #print("Generating neighbor dict")
        self.neighbors = {}
        for n in self.graph.nodes:
            self.neighbors[n] = set(nx.neighbors(self.graph, n))
        
        #print("Generating distance dict")
        self.distances = {}
        if setup_distances:
            for d in nx.all_pairs_shortest_path_length(self.graph):
                self.distances[d[0]] = d[1]
            for n in self.graph.nodes():
                for m in set(self.graph.nodes()) - set(self.distances[n].keys()):
                    self.distances[n][m] = float('inf')
        #print("Finished graph setup")

    def dist(self, i, j):
        if not i in self.distances:
            self.distances[i] = {}
            self.distances[i][j] = nx.shortest_path_length(self.graph, i, j)
        elif not j in self.distances[i]:
            self.distances[i][j] = nx.shortest_path_length(self.graph, i, j)
        return self.distances[i][j]

    def sample(self, s, T):
        edges = set()
        for n in self.neighbors[s]:
            edges.add((s, n))
        infected = set([s])
        infection_order = [s]
        for _ in range(1, T):
            jump = random.sample(edges, 1)[0]
            infected.add(jump[1])
            infection_order.append(jump[1])
            for n in self.neighbors[jump[1]]:
                if n in infected:
                    edges.discard((n, jump[1]))
                else:
                    edges.add((jump[1], n))
        return infected, infection_order

    def sample_d1(self, s, T):
        edges = set()
        for n in self.neighbors[s]:
            edges.add((s, n))
        infected = set([s])
        infection_order = [s]
        degree_list = [len(edges)]
        for _ in range(1, T):
            jump = random.sample(edges, 1)[0]
            infected.add(jump[1])
            infection_order.append(jump[1])
            for n in self.neighbors[jump[1]]:
                if n in infected:
                    edges.discard((n, jump[1]))
                else:
                    edges.add((jump[1], n))
            degree_list.append(len(edges))
        return infected, infection_order, degree_list

    def select_source(self, seed=None):
        if seed is not None:
            random.seed(seed)
        #self.source = random.sample(set(self.graph.nodes), 1)[0]
        #return self.source
        #for n in self.graph.nodes:
        #    if self.graph.degree[n] == 1:
        #        return n
        u = nx.eigenvector_centrality_numpy(self.graph)
        median = np.median(np.absolute(list(u.values())))
        #u_min = min([abs(u[ui]) for ui in u.keys() if abs(u[ui]) >= median])
        #u = [ui for ui in u.keys() if abs(u[ui]) == u_min]
        #return u[0]
        u = [ui for ui in u.keys() if abs(u[ui]) >= median]
        if len(u) == 0:
            u = set(self.graph.nodes)
        else:
            u = set(u)
        self.source = random.sample(u, 1)[0]
        return self.source

    def select_uniform_source(self):
        self.source = random.sample(set(self.graph.nodes()), 1)[0]
        return self.source

class DirectedGraph(Graph):
    def __init__(self, N):
        super().__init__(N)

    def sample(self, s, T):
        edges = set()
        for e in self.graph.out_edges(s):
            edges.add(e)
        infected = set([s])
        infection_order = [s]
        for _ in range(1, T):
            jump = random.sample(edges, 1)[0]
            infected.add(jump[1])
            infection_order.append(jump[1])
            for e in self.graph.out_edges(jump[1]):
                if not e[1] in infected:
                    edges.add(e)
            edges = set([e for e in edges if (not e[1] in infected)])
        return infected, infection_order

    def select_source(self):
        best = -1
        reachable = -1
        for n in self.graph.nodes:
            r = len(self.distances[n].keys())
            if r > reachable:
                best = n
                reachable = r
        return best

    def source_candidates(self, x):
        gs = self.graph.subgraph(x)
        sub_distances = {}
        for d in nx.all_pairs_shortest_path_length(gs):
            sub_distances[d[0]] = d[1]
        candidates = []
        for n in x:
            if n in sub_distances.keys() and len(set(x) - set(sub_distances[n].keys()) - set([n])) == 0:
                candidates += [n]

        return candidates

class WeightedGraph(Graph):
    def __init__(self, N):
        super().__init__(N)

    def sample(self, s, T):
        edges = list()
        weights = list()
        for e in self.graph.edges(s, data=True):
            edges += [e]
        infected = set([s])
        infection_order = [s]
        for _ in range(1, T):
            weights = np.array([e[2]["weight"] for e in edges])
            jump_index = np.random.choice(list(range(len(edges))), 1, p=weights/sum(weights))[0]
            jump = edges[jump_index]
            infected.add(jump[1])
            infection_order.append(jump[1])
            for e in self.graph.edges(jump[1], data=True):
                end = e[0]
                if e[0] == jump[1]:
                    end = e[1]
                if not end in infected:
                    edges += [e]
            edges = [e for e in edges if not (e[0] in infected and e[1] in infected)]
        return infected, infection_order

class RegularTree(Graph):
    def __init__(self, N, degree, height):
        assert(N > degree)
        self.degree = degree
        #self.graph = nx.balanced_tree(degree, height)
        self.graph = nx.Graph()
        neighbors = list(range(1, self.degree+1))
        self.graph.add_node(0)
        self.graph.add_nodes_from(neighbors)
        self.graph.add_edges_from([(0, n) for n in neighbors])
        count = self.degree + 1
        q = neighbors
        while count < N:
            parent = q.pop(0)
            neighbors = list(range(count, count + self.degree - 1))
            for n in neighbors:
                self.graph.add_node(n)
                self.graph.add_edges_from([(parent, n)])
                count = count + 1
                q.append(n)
                if count >= N:
                    break
        self.distances = {}
        for d in nx.all_pairs_shortest_path_length(self.graph):
            self.distances[d[0]] = d[1]
        for n in self.graph.nodes():
            for m in set(self.graph.nodes()) - set(self.distances[n].keys()):
                self.distances[n][m] = float('inf')
        super().__init__(N)

class RegularGraph(Graph):
    def __init__(self, N, degree):
        assert(N > degree)
        self.graph = nx.random_regular_graph(degree, N)
        super().__init__(N)

class ErdosRenyi(Graph):
    def __init__(self, N, degree):
        assert(N > degree)
        self.graph = nx.fast_gnp_random_graph(N, degree/(N-1))
        super().__init__(N)

class WattsStrogatz(Graph):
    def __init__(self, N, degree):
        assert(N > degree)
        self.graph = nx.watts_strogatz_graph(N, degree, 1/degree)
        super().__init__(N)

class PreferentialAttachment(Graph):
    def __init__(self, N, degree):
        assert(N > degree)
        self.graph = nx.barabasi_albert_graph(N, degree)
        super().__init__(N)

class StochasticBlock(Graph):
    def __init__(self, N, degree):
        assert(N > degree)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(list(range(N)))
        beta = 0.2
        k = 6
        randomize = True

        Z = np.zeros((N, k))
        if randomize:
            for n in range(N):
                c = random.randint(0, k-1)
                Z[n][c] = 1
        else:
            for n in range(N):
                c = n % k
                Z[n][c] = 1

        B0 = (1 - beta)*np.eye(k) + beta*(-1*np.eye(k) + 1)
        Theta = np.diag(powerlaw.rvs(1, size=N))

        Z, B0, Theta = np.matrix(Z), np.matrix(B0), np.matrix(Theta)

        M = Theta * Z * B0 * Z.T * Theta
        M = (degree/np.sum(np.mean(M, axis=0)))*M

        for i in range(N):
            for j in range(i+1, N):
                if random.random() <= M[i, j]:
                    self.graph.add_edge(i, j)

        super().__init__(N)

class InfRegular(Graph):
    def __init__(self, degree):
        self.degree = degree
        G = nx.Graph()
        G.add_node(0)
        neighbors = list(range(1, self.degree+1))
        nodes = set(neighbors)
        G.add_nodes_from(neighbors)
        G.add_edges_from([(0, n) for n in neighbors])
        self.graph = G
        self.distances = {}
        for d in nx.all_pairs_shortest_path_length(self.graph):
            self.distances[d[0]] = d[1]

    def select_source(self):
        return 0

    def sample(self, s, T):
        infected = set([s])
        infection_order = [s]
        counter = len(self.graph)
        nodes = set(nx.neighbors(self.graph, s))
        for _ in range(1, T):
            jump = random.sample(nodes, 1)[0]
            nodes.remove(jump)
            if self.graph.degree[jump] == 1:
                neighbors = list(range(counter, counter+self.degree-1))
                counter = counter + self.degree - 1
                self.graph.add_nodes_from(neighbors)
                self.graph.add_edges_from([(jump, n) for n in neighbors])
            neighbors = set(nx.neighbors(self.graph, jump)) - infected
            nodes.update(neighbors)
            infected.add(jump)
            infection_order.append(jump)
        return infected, infection_order

    def radius(self, r):
        self.graph = nx.generators.ego_graph(self.graph, 0, radius=r)

class FromAdjacency(Graph):
    def __init__(self, N, A):
        self.graph = nx.from_numpy_matrix(A)
        super().__init__(N)

class EdgeList(Graph):
    def __init__(self, fname):
        self.graph = nx.readwrite.edgelist.read_edgelist(fname, delimiter=',', nodetype=int)
        super().__init__(self.graph.number_of_nodes(), False)

class PyEdgeList(Graph):
    def __init__(self, edges):
        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)
        super().__init__(self.graph.number_of_nodes(), False)

class GrowingNetwork(DirectedGraph):
    def __init__(self, N):
        self.graph = nx.gn_graph(N)
        super().__init__(N)

class ScaleFree(DirectedGraph):
    def __init__(self, N):
        self.graph = nx.scale_free_graph(N)
        super().__init__(N)

class WeightedCopy(WeightedGraph):
    def __init__(self, G):
        self.graph = G
        super().__init__(len(G))

class DirectedCopy(DirectedGraph):
    def __init__(self, G):
        self.graph = G.graph.to_directed()
        super().__init__(len(G.graph))

def addRandomWeights(G, a):
    graph = G.graph
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = powerlaw.rvs(a, 1)

    return WeightedCopy(graph)

def addConstantWeights(G):
    graph = G.graph
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = 1.0

    return WeightedCopy(graph)
