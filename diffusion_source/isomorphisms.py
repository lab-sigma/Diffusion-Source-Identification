import diffusion_source.graphs as graphs

import time
from collections import Counter
import networkx as nx
import numpy as np
import math
import itertools
import random
import queue
from decimal import *
from scipy.sparse.linalg import eigsh
from scipy import sparse


def get_iso(G, x):
    isomorphisms = []
    saved = set()
    saved_count = 0
    for u1 in x:
        if G.graph.degree[u1] == 0:
            continue
        curr = 0
        for u2 in x:
            n1 = set(G.graph.neighbors(u1))
            n2 = set(G.graph.neighbors(u2))
            if u2 in n1:
                n1.remove(u2)
                n2.remove(u1)
            if not u1 == u2 and n1 == n2:
                isomorphisms.append((u1, u2))
                saved.add(u2)
                curr = curr + 1
        if curr > 0 and not u1 in saved:
            saved_count = saved_count + curr
    return isomorphisms, saved_count

def get_iso_full(G):
    isomorphisms = []
    saved = set()
    saved_count = 0
    for u1 in list(G.graph.nodes):
        if G.graph.degree[u1] == 0:
            continue
        curr = 0
        for u2 in list(G.graph.nodes):
            n1 = set(G.graph.neighbors(u1))
            n2 = set(G.graph.neighbors(u2))
            if u2 in n1:
                n1.remove(u2)
                n2.remove(u1)
            if not u1 == u2 and n1 == n2:
                isomorphisms.append((u1, u2))
                saved.add(u2)
                curr = curr + 1
        if curr > 0 and not u1 in saved:
            saved_count = saved_count + curr
    return isomorphisms, saved_count

def first_order_iso_full(G):
    L = set()
    for u in list(G.graph.nodes):
        N14 = set(G.graph.neighbors(u))
        for i in range(3):
            for n in N14:
                N14 = N14.union(set(G.graph.neighbors(n)))
        if u in N14:
            N14.remove(u)
        for v in N14:
            if G.graph.degree[v] == G.graph.degree[u]:
                D1u = Counter([G.graph.degree[up] for up in G.graph.neighbors(u)])
                D1v = Counter([G.graph.degree[vp] for vp in G.graph.neighbors(v)])
                if D1u == D1v:
                    U12 = set(G.graph.neighbors(u))
                    V12 = set(G.graph.neighbors(v))
                    for n in U12:
                        U12 = U12.union(set(G.graph.neighbors(n)))
                    for n in V12:
                        V12 = V12.union(set(G.graph.neighbors(n)))
                    U12 = U12.difference(G.graph.neighbors(u))
                    U12.remove(u)
                    if v in U12:
                        U12.remove(v)
                    U12 = U12.difference(G.graph.neighbors(v))
                    V12 = V12.difference(G.graph.neighbors(v))
                    V12.remove(v)
                    if u in V12:
                        V12.remove(u)
                    V12 = V12.difference(G.graph.neighbors(u))
                    if U12 == V12:
                        Un = list(G.graph.neighbors(u))
                        if v in Un:
                            Un.remove(v)
                        Vn = list(G.graph.neighbors(v))
                        if u in Vn:
                            Vn.remove(u)
                        for p in itertools.permutations(Vn):
                            works = True
                            for i in range(len(Un)):
                                nU = set(G.graph.neighbors(Un[i]))
                                nV = set(G.graph.neighbors(p[i]))
                                if (not u in nV) and (not v in nU):
                                    nU.remove(u)
                                    nV.remove(v)
                                if not nU == nV:
                                    works = False
                                    break
                            if works:
                                L.add((u, v))
                                break
    return L

def first_order_iso(G, x):
    L = set()
    for u in x:
        N14 = set(G.graph.neighbors(u))
        for i in range(3):
            for n in N14:
                N14 = N14.union(set(G.graph.neighbors(n)))
        if u in N14:
            N14.remove(u)
        N14 = N14.intersection(set(x))
        for v in N14:
            if G.graph.degree[v] == G.graph.degree[u]:
                D1u = Counter([G.graph.degree[up] for up in G.graph.neighbors(u)])
                D1v = Counter([G.graph.degree[vp] for vp in G.graph.neighbors(v)])
                if D1u == D1v:
                    U12 = set(G.graph.neighbors(u))
                    V12 = set(G.graph.neighbors(v))
                    for n in U12:
                        U12 = U12.union(set(G.graph.neighbors(n)))
                    for n in V12:
                        V12 = V12.union(set(G.graph.neighbors(n)))
                    U12 = U12.difference(G.graph.neighbors(u))
                    U12.remove(u)
                    if v in U12:
                        U12.remove(v)
                    U12 = U12.difference(G.graph.neighbors(v))
                    V12 = V12.difference(G.graph.neighbors(v))
                    V12.remove(v)
                    if u in V12:
                        V12.remove(u)
                    V12 = V12.difference(G.graph.neighbors(u))
                    if U12 == V12:
                        Un = list(G.graph.neighbors(u))
                        if v in Un:
                            Un.remove(v)
                        Vn = list(G.graph.neighbors(v))
                        if u in Vn:
                            Vn.remove(u)
                        for p in itertools.permutations(Vn):
                            works = True
                            for i in range(len(Un)):
                                nU = set(G.graph.neighbors(Un[i]))
                                nV = set(G.graph.neighbors(p[i]))
                                if (not u in nV) and (not v in nU):
                                    nU.remove(u)
                                    nV.remove(v)
                                if not nU == nV:
                                    works = False
                                    break
                            if works:
                                L.add((u, v))
                                break
    return L

def first_order_iso_spectral(G, x):
    start = time.time()
    A = nx.adjacency_matrix(G.graph, weight=None)
    A = A.asfptype()
    D, U = eigsh(A, k=2)

    L = set()
    checked = set()
    for xi in x:
        checked.add(xi)
        for xj in x:
            if xj in checked:
                continue
            #if all([i == j for i, j in zip(U[xi], U[xj])]) or all([i == -j for i, j in zip(U[xi], U[xj])]):
            if np.allclose(U[xi], U[xj], 1e-10, 1e-10) or np.allclose(U[xi], -U[xj], 1e-10, 1e-10):
                L.add((xi, xj))

    return L

def distance_matching(neu, nev, U, P, swap):
    if not len(neu) == len(nev):
        return False, P

    nevc = nev.copy()

    for u in neu:
        Uu = U[u]
        passed = False
        if u in P:
            continue
        for v in nevc:
            if v in P:
                continue
            if np.allclose(swap*Uu/np.linalg.norm(Uu), U[v]/np.linalg.norm(U[v]), 1e-10, 1e-10):
                nevc.remove(v)
                passed = True
                P[u] = v
                P[v] = u
                break

        if not passed:
            return False, P

    return True, P

def distance_matching_stationary(neu, nev, U, P, swap):
    if not len(neu) == len(nev):
        return False, P

    nevc = nev.copy()

    for u in neu:
        Uu = U[u]
        passed = False
        if u in P:
            continue
        for v in nevc:
            if v in P:
                continue
            if np.isclose(Uu, U[v]):
                nevc.remove(v)
                passed = True
                P[u] = v
                P[v] = u
                break

        if not passed:
            return False, P

    return True, P

def check_permutation(G, P):
    for x in P.keys():
        for n in G.graph.neighbors(x):
            r = P[x]
            c = n
            if n in P.keys():
                c = P[n]
            if not G.graph.has_edge(r, c):
                return False
    return True

def neighborhoods(G, neu, nev):
    u_new = set()
    v_new = set()

    for u in neu:
        for nu in G.graph[u]:
            if not nu in neu:
                u_new.add(nu)

    for v in nev:
        for nv in G.graph[v]:
            if not nv in nev:
                v_new.add(nv)

    return u_new, v_new

def general_iso(G, x, K):
    start = time.time()
    A = nx.adjacency_matrix(G.graph, weight=None)
    A = A.asfptype()
    D, U = eigsh(A, k=A.shape[0]-1)
    roundD = np.around(D, 8)
    Dc = Counter(roundD)
    keep = []
    for i in range(len(D)):
        if Dc[roundD[i]] == 1:
            keep += [i]

    #U = np.take_along_axis(np.array(U), np.array(keep), axis=1).to_list()
    U = np.array(U)[:, np.array(keep)]

    L = list()
    checked = set()
    for u in x:
        checked.add(u)
        for v in x:
            if v in checked:
                continue
            u_sign = np.sign(U[u])
            v_sign = np.sign(U[v])
            swap = u_sign * v_sign
            swap[swap == 0] = 1
            if not np.allclose(swap*U[u]/np.linalg.norm(U[u]), U[v]/np.linalg.norm(U[v]), 1e-10, 1e-10):
                continue

            #P = sparse.lil_matrix(np.eye(len(G.graph)))
            P = {}
            P[u] = v
            P[v] = u
            #P[[u, v]] = P[[v, u]]

            neu = set([u])
            nev = set([v])

            for k in range(K):
                #if (A != A @ P).nnz == 0:
                if check_permutation(G, P):
                    L.append((u, v, P))
                    break

                u_new, v_new = neighborhoods(G, neu, nev)
                neu = u_new
                nev = v_new

                s, P = distance_matching(neu, nev, U, P, swap)
                if not s:
                    break

            #if (A != A @ P).nnz == 0:
            if not (u, v) in L and check_permutation(G, P):
                L.append((u, v, P))

    return L

def general_iso_stationary(G, x, K):
    start = time.time()
    A = nx.adjacency_matrix(G.graph, weight=None)
    A = A.asfptype()
    A = A/A.sum(axis=1)[:,None]

    evals, evecs = np.linalg.eig(Q.T)
    U = evecs[:, np.isclose(evals, 1)]

    U = U[:,0]
    U = U/U.sum()

    L = list()
    checked = set()
    for u in x:
        checked.add(u)
        for v in x:
            if v in checked:
                continue
            if not np.isclose(U[u], U[v]):
                continue

            #P = sparse.lil_matrix(np.eye(len(G.graph)))
            P = {}
            P[u] = v
            P[v] = u
            #P[[u, v]] = P[[v, u]]

            neu = set([u])
            nev = set([v])

            for k in range(K):
                #if (A != A @ P).nnz == 0:
                if check_permutation(G, P):
                    L.append((u, v, P))
                    break

                u_new, v_new = neighborhoods(G, neu, nev)
                neu = u_new
                nev = v_new

                s, P = distance_matching_stationary(neu, nev, U, P, swap)
                if not s:
                    break

            #if (A != A @ P).nnz == 0:
            if not (u, v) in L and check_permutation(G, P):
                L.append((u, v, P))

    return L

def iso_groups(L):
    groups = list()
    included = set()
    for l in L:
        g0 = set()
        g1 = set()
        if l[0] in included:
            i0 = next(g for g in range(len(groups)) if l[0] in groups[g])
            g0 = groups[i0]
            if not l[1] in g0:
                del groups[i0]
        if l[1] in included:
            i1 = next(g for g in range(len(groups)) if l[1] in groups[g])
            g1 = groups[i1]
            del groups[i1]

        g = g0.union(g1)
        g.add(l[0])
        g.add(l[1])
        groups.append(g)
        included.add(l[0])
        included.add(l[1])
    return groups

def compose_permutations(P1, P2):
    newP = {}
    permf = lambda P, s : P[s] if s in P else s
    for p in P1:
        newP[p] = permf(P2, permf(P1, p))
    for p in P2:
        if not p in newP:
            newP[p] = permf(P2, permf(P1, p))
    return newP

def compile_permutations(L, join_disjoint=True):
    groups = list()
    included = set()
    permutations = {}
    for l in L:
        permutations[(l[0], l[1])] = l[2]
        permutations[(l[1], l[0])] = l[2]
        g0 = set()
        g1 = set()
        if l[0] in included:
            i0 = next(g for g in range(len(groups)) if l[0] in groups[g])
            g0 = groups[i0]
            if not l[1] in g0:
                del groups[i0]
        if l[1] in included:
            i1 = next(g for g in range(len(groups)) if l[1] in groups[g])
            g1 = groups[i1]
            del groups[i1]

        g = g0.union(g1)
        g.add(l[0])
        g.add(l[1])
        groups.append(g)
        included.add(l[0])
        included.add(l[1])

    leaders = {}
    for g in groups:
        leader = g.pop()
        leaders[leader] = g

        if not join_disjoint:
            continue
        disjoint = queue.Queue()
        new = set()
        for s in g:
            if not (s, leader) in permutations:
                disjoint.put(s)
            else:
                new.add(s)

        while not disjoint.empty():
            s = disjoint.get()
            for n in new:
                if (s, n) in permutations:
                    newP = compose_permutations(permutations[(n, leader)], permutations[(s, n)])
                    permutations[(s, leader)] = newP
                    permutations[(leader, s)] = newP
                    new.add(s)
                    break
            if not s in new:
                disjoint.put(s)

    return leaders, permutations
