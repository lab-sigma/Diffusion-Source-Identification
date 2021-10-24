import graphs
import sys
import time
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import itertools
import random
from decimal import *

def add_pair(groups, included, l):
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
    return groups, included

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
    #groups = list()
    #included = set()
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
                                #groups, included = add_pair(groups, included, (u, v))
                                break
    #return groups
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
