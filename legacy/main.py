import graphs
from rumor_center import rumor_center, rumor_center_topk
from algorithms import Alg1, Alg2, Alg1V2, Alg1_filter_matrix, Alg1_iso, Alg1_compare, Alg1_final, Alg1_filtering, Alg1_filtering_intersect, Alg1_test_source, Alg1_faster_loss, Alg1_parallel, Alg1_parallel_fast, Alg1_directed, Alg1_weighted, Alg1_faster_loss_check_source
from isomorphisms import get_iso, first_order_iso, get_iso_full, first_order_iso_full, iso_groups
import time
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import itertools
import random
import pickle
from decimal import *
import os, psutil

degree = 4
N = 1365
T = 200
K = 1
height = 6
m = 10000

def L2_h(t_z, T):
    return 2

def ADT_h(t_z, T):
    return (T - t_z)/T

def ADT2_h(t_z, T):
    return ((T - t_z)/T)**2

def ADiT_h(t_z, T):
    return 1/t_z

def ADiT2_h(t_z, T):
    return 1/(t_z*t_z)

################################

def L2(G, x, Y, s):
    return len(x ^ Y)

def Ld(G, x, Y, s):
    k = 3
    count = 0
    for node in x ^ Y:
        if G.dist(node, s) <= k:
            count += 1
    return count

def Le(G, x, Y, s):
    loss = 0.0
    for node in x ^ Y:
        loss += np.exp(-G.dist(node, s))
    return loss

def Ll(G, x, Y, s):
    loss = 0.0
    for node in x ^ Y:
        loss += 1.0/G.dist(node, s)
    return loss

def Lm(G, x, Y, s):
    if x == Y:
        return 0.0
    return 1.0

def first_miss_loss(G, x, samples, s, weights = None):
    min_steps = []
    if not weights is None:
        for sample, weight in zip(samples, weights):
            for i in range(len(sample)):
                if not sample[i] in x:
                    min_steps += [weight*i]
                    break
    else:
        for sample in samples:
            for i in range(len(sample)):
                if not sample[i] in x:
                    min_steps += [i]
                    break
    return -np.mean(min_steps)

def avg_deviation_time(G, x, samples, s):
    avg_steps = list()
    T_l = len(samples[0])
    for sample in samples:
        avg_steps.append(np.mean([T_l - i*(not sample[i] in x) for i in range(T_l)]))
        #for i in range(T):
        #    if not sample[i] in x:
        #        diff_steps.append(T - i)
        #avg_steps.append(np.mean(diff_steps))
    return np.mean(avg_steps)

def avg_matching_time(G, x, samples, s):
    avg_match = list()
    T_l = len(samples[0])
    for sample in samples:
        #same_steps = list()
        #T = len(sample)
        #for i in range(T):
        #    if sample[i] in x:
        #        same_steps.append(i)
        #avg_match.append(np.mean(same_steps))
        avg_match.append(np.mean([i*(sample[i] in x) for i in range(T_l)]))
    return -np.mean(avg_match)

def distance_loss(G, x, samples, s):
    return sum([G.dist(i, s) for i in x])

def min_dist(G, x, samples, s):
    return -1*min([G.dist(i, s) for i in set(G.graph.nodes()) - set(x)])

def max_dist(G, x, samples, s):
    return max([G.dist(i, s) for i in x])

##############################################################

def is_connected(G, C):
    if len(C) == 0:
        return True
    Gsub = G.graph.subgraph(C)
    return nx.is_connected(Gsub)

def count_ends(T, counts, node, parent):
    neighbors = list(nx.neighbors(T, node))
    if parent in neighbors:
        neighbors.remove(parent)
    if len(neighbors) == 0:
        counts[node] = 1
    else:
        for n in neighbors:
            count_ends(T, counts, n, node)
        t = 1 + sum([counts[j] for j in neighbors])
        counts[node] = t

def general_center(G, x):
    center, ml, rcenter, vc, nodes = rumor_center(G, x)

    GN = G.subgraph(list(x))
    nodes = dict.fromkeys(GN.nodes(), 0)
    root = rcenter
    T = nx.bfs_tree(GN, root)
    count_ends(T, nodes, root, None)
    kappa = []
    boundary = [root]
    visited = set([root])
    while len(boundary) > 0:
        next_boundary = []
        for b in boundary:
            tmp = []
            max_val = -1
            children = list(nx.neighbors(GN, b))
            for c in children:
                if not c in visited:
                    if nodes[c] > max_val:
                        tmp = [c]
                        max_val = nodes[c]
                    elif nodes[c] == max_val:
                        tmp += [c]
                    visited.add(c)
            if max_val == 1:
                kappa += [b]
            next_boundary += tmp
        boundary = next_boundary
    return list(kappa)

def loss_creator(loss_func):
    def loss_wrapper(G, x, samples_Y, s, weights=None):
        if not weights is None:
            return sum([loss_func(G, x, Y, s)*weight for Y, weight in zip(samples_Y, weights)])
        return sum([loss_func(G, x, Y, s) for Y in samples_Y])
    return loss_wrapper

def display_sample(G, T):
    alps = [0.1]
    los = [loss_creator(L2)]
    lstring = ["L2"]
    ordered = [False]
    alpha = [0.1]

    source = G.select_source()
    x, x_l = G.sample(source, T)
    center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
    estimates, Cs, C21, C22, C23 = Alg1(G, x, m, los, ordered, T, alps, 0)
    d = Alg2(G, x)
    print("Rumor Center: ", rs)
    print("Distance Center: ", d)
    print("Point Estimate: ", estimates[0])
    print("Confidence Set: ", Cs[0])
    nodes = []
    colors = []
    for n in G.graph.nodes():
        p = False
        c = [0.0, 0.0, 0.0]
        if n in x:
            p = True
            c[0] += 1.0
        if n == source:
            c = [0.0, 1.0, 0.0]
        if n in Cs[0]:
            c[2] += 1.0
        if (not p) and any([(neighbor in x) for neighbor in G.graph.neighbors(n)]):
            p = True
            c = (0.5, 0.5, 0.5)
        if p:
            nodes += [n]
            colors += [c]
    nx.draw_networkx(G.graph.subgraph(nodes), nodelist=nodes, with_labels=True, node_color=colors)
    plt.show()

def display_infected(G, x):
    nodes = []
    colors = []
    for n in G.graph.nodes():
        p = False
        c = [0.0, 0.0, 0.0]
        if n in x:
            p = True
            c[0] += 1.0
        if (not p) and any([(neighbor in x) for neighbor in G.graph.neighbors(n)]):
            p = True
            c = (0.5, 0.5, 0.5)
        if p:
            nodes += [n]
            colors += [c]
    nx.draw_networkx(G.graph.subgraph(nodes), nodelist=nodes, with_labels=True, node_color=colors)
    plt.show()

def proper_len(L):
    s = set()
    for l in L:
        iso = l
        if l[0] > l[1]:
            iso = (l[1], l[0])
        if l[0] == l[1]:
            continue
        s.add(iso)
    return len(s)

def simplified_iso(L):
    s = set()
    for l in L:
        iso = l
        if l[0] > l[1]:
            iso = (l[1], l[0])
        if l[0] == l[1]:
            continue
        s.add(iso)
    return s

def total_iso_elements(L):
    s = set()
    for l in L:
        s.add(l[0])
        s.add(l[1])
    return s

def print_iso(G, name):
    s = G.select_source()
    ic = []
    fic = []
    d1x = []
    for k in range(K):
        x, x_l = G.sample(s, T)
        iso, iso_count = get_iso(G, x)
        ic = ic + [len(total_iso_elements(iso)) - len(iso_groups(iso))]
        f_iso = first_order_iso(G, x)
        fic = fic + [len(total_iso_elements(f_iso)) - len(iso_groups(f_iso))]
        d1x = d1x + [sum([G.graph.degree[v] == 1 for v in x])]

    print(name)
    iso_f, iso_count_f = get_iso_full(G)
    iso_f = simplified_iso(iso_f)
    f_iso_f = simplified_iso(first_order_iso_full(G))
    print("\tZero order iso: ", proper_len(iso_f))
    print("\tZero order iso infected: ", sum(ic)/K/T)
    print("\tFirst order iso: ", proper_len(f_iso_f))
    print("\tFirst order iso infected: ", sum(fic)/K/T)
    print("\tDegree 1: ", sum([G.graph.degree[v] == 1 for v in list(G.graph.nodes)]))
    print("\tDegree 1 infected: ", sum(ic)/K)
    return iso_f, f_iso_f

def display_only_sample(G, x, s):
    alps = [0.1]
    los = [loss_creator(L2)]
    lstring = ["L2"]
    ordered = [False]
    alpha = [0.1]

    source = s
    nodes = []
    colors = []
    for n in G.graph.nodes():
        p = False
        c = [0.0, 0.0, 0.0]
        if n in x:
            p = True
            c[0] += 1.0
        if n == source:
            c = [0.0, 1.0, 0.0]
        if (not p) and any([(neighbor in x) for neighbor in G.graph.neighbors(n)]):
            p = True
            c = (0.5, 0.5, 0.5)
        if p:
            nodes += [n]
            colors += [c]
    nx.draw_networkx(G.graph.subgraph(nodes), nodelist=nodes, with_labels=True, node_color=colors)
    plt.show()

def run_rs(G, T, name):
    b = 0
    bhops = []
    r = 0
    rhops = []
    for i in range(K):
        print("{} % Selecting source         ".format(i), end="\r")
        source = G.select_source()
        print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        d = Alg2(G, x)
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
    print()
    print(name)
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    return b/K, np.mean(bhops), r/K, np.mean(rhops)

def run_gc_topk(G, T, name):
    rc = 0
    rhops = []
    gc = 0
    ghops = []
    for i in range(K):
        print("{} % Selecting source         ".format(i), end="\r")
        source = G.select_uniform_source()
        print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        print("{} % Computing rumor center   ".format(i), end="\r")
        kappa = general_center(G.graph, x)
        topk = rumor_center_topk(G.graph, x, len(kappa))
        if source in topk:
            rc += 1
        rhops += [min([G.dist(d, source) for d in topk])]
        if source in kappa:
            gc += 1
        ghops += [min([G.dist(d, source) for d in kappa])]
    print()
    print(name)
    print("\tg: {}".format(gc/K))
    print("\tghops mean: {}".format(np.mean(ghops)))
    print("\tghops: {}".format(Counter(ghops)))
    print("\trs: {}".format(rc/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    return gc/K, np.mean(ghops), rc/K, np.mean(rhops)

def run_graph(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "TTD"]
    ordered = [False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]

    #los = [loss_creator(L2), first_miss_loss, avg_deviation_time, avg_matching_time]
    #lstring = ["L2", "TTD", "ADT", "AMT"]
    #ordered = [False, True, True, True]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    c21 = [0 for i in range(len(alps) * len(los))]
    c22 = [0 for i in range(len(alps) * len(los))]
    c23 = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]
    avg21 = [0 for i in range(len(alps) * len(los))]
    avg22 = [0 for i in range(len(alps) * len(los))]
    avg23 = [0 for i in range(len(alps) * len(los))]
    f = [0 for i in range(len(alps) * len(los))]
    bf = [0 for i in range(len(alps) * len(los))]
    st = 0
    lts = [0 for i in range(len(los))]
    iso_count = []
    source = G.select_source()
    for i in range(K):
        if len(G.graph) > 5000000:
            G.radius(2)
        #print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        #print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        #print("{} % Computing confidence set ".format(i), end="\r")
        estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1(G, x, m, los, ordered, T, alps, i)
        #print("{} % Computing distance center".format(i), end="\r")
        d = Alg2(G, x)
        pairs, isos = get_iso(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
            if source in C21[conf]:
                c21[conf] += 1
            avg21[conf] += len(C21[conf])
            f[conf] = f[conf] + filt[conf]
            bf[conf] = bf[conf] + bfilt[conf]

        st += sample_timing
        for j in range(len(los)):
            lts[j] += loss_timings[j]
        iso_count += [isos]
    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    print("\tisomorphisms: {}".format(np.mean(iso_count)))
    print("\tSampling Runtime: {}".format(st))
    for i in range(len(los)):
        print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
        print("\tLoss Runtime {}: {}".format(lstring[i], lts[i]))
        print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        print("\tLoss Runtime Comparison {}: {}".format(lstring[i], lts[i]/sum(lts)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
            print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
            print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
            print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))

    print_iso(G, name)
    return source

def large_run_graph(G, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "dist", "TTD"]
    ordered = [False, False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]
    source = G.select_source()

    for T in [25, 50]:
        a = [0 for i in range(len(los))]
        ahops = [[] for i in range(len(los))]
        b = 0
        bhops = []
        r = 0
        rhops = []
        c = [0 for i in range(len(alps) * len(los))]
        c21 = [0 for i in range(len(alps) * len(los))]
        c22 = [0 for i in range(len(alps) * len(los))]
        c23 = [0 for i in range(len(alps) * len(los))]
        avg = [0 for i in range(len(alps) * len(los))]
        avg21 = [0 for i in range(len(alps) * len(los))]
        avg22 = [0 for i in range(len(alps) * len(los))]
        avg23 = [0 for i in range(len(alps) * len(los))]
        f = [0 for i in range(len(alps) * len(los))]
        bf = [0 for i in range(len(alps) * len(los))]
        st = 0
        lts = [0 for i in range(len(los))]
        iso_count = []
        for i in range(K):
            #print("{} % Generating infected      ".format(i), end="\r")
            x, x_l = G.sample(source, T)
            #print("{} % Computing rumor center   ".format(i), end="\r")
            center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
            #print("{} % Computing confidence set ".format(i), end="\r")
            estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1(G, x, m, los, ordered, T, alps, i)
            #print("{} % Computing distance center".format(i), end="\r")
            d = Alg2(G, x)
            pairs, isos = get_iso(G, x)
            for e in range(len(estimates)):
                if estimates[e] == source:
                    a[e] += 1
                ahops[e] += [G.dist(estimates[e], source)]
            if d == source:
                b += 1
            bhops += [G.dist(d, source)]
            if rs == source:
                r += 1
            rhops += [G.dist(rs, source)]
            for conf in range(len(Cs)):
                if source in Cs[conf]:
                    c[conf] += 1
                avg[conf] += len(Cs[conf])
                if source in C21[conf]:
                    c21[conf] += 1
                avg21[conf] += len(C21[conf])
                f[conf] = f[conf] + filt[conf]
                bf[conf] = bf[conf] + bfilt[conf]

            st += sample_timing
            for j in range(len(los)):
                lts[j] += loss_timings[j]
            iso_count += [isos]
        print()
        print(name)
        print("T: ", T)
        print("N: ", G.graph.number_of_nodes())
        for j in range(len(los)):
            print("\ta {}: {}".format(lstring[j], a[j]/K))
            print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
            print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
        print("\tb: {}".format(b/K))
        print("\tbhops mean: {}".format(np.mean(bhops)))
        print("\tbhops: {}".format(Counter(bhops)))
        print("\trs: {}".format(r/K))
        print("\trhops mean: {}".format(np.mean(rhops)))
        print("\trhops: {}".format(Counter(rhops)))
        print("\tisomorphisms: {}".format(np.mean(iso_count)))
        for i in range(len(los)):
            print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
            print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
                print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
                print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
                print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
                print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))
    return source

def run_graph2(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, max_dist, min_dist]
    lstring = ["L2", "dist", "max_dist", "min_dist"]
    biased = [True, False, False, False]
    alpha = [0.1, 0.2]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]
    for i in range(K):
        print("{} % Selecting source         ".format(i), end="\r")
        source = G.select_source()
        print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        print("{} % Computing confidence set ".format(i), end="\r")
        estimates, Cs = Alg1V2(G, x, los, biased, T, alps, i)
        print("{} % Computing distance center".format(i), end="\r")
        d = Alg2(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    for i in range(len(los)):
        for u in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))
    return source

def small_run_graph(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "dist", "TTD"]
    ordered = [False, False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    c21 = [0 for i in range(len(alps) * len(los))]
    c22 = [0 for i in range(len(alps) * len(los))]
    c23 = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]
    avg21 = [0 for i in range(len(alps) * len(los))]
    avg22 = [0 for i in range(len(alps) * len(los))]
    avg23 = [0 for i in range(len(alps) * len(los))]
    f = [0 for i in range(len(alps) * len(los))]
    bf = [0 for i in range(len(alps) * len(los))]
    st = 0
    lts = [0 for i in range(len(los))]
    iso_count = []
    source = G.select_source()
    for i in range(5):
        #print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        #print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        #print("{} % Computing confidence set ".format(i), end="\r")
        estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1(G, x, m, los, ordered, T, alps, i)
        #print("{} % Computing distance center".format(i), end="\r")
        d = Alg2(G, x)
        pairs, isos = get_iso(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
            if source in C21[conf]:
                c21[conf] += 1
            avg21[conf] += len(C21[conf])
            f[conf] = f[conf] + filt[conf]
            bf[conf] = bf[conf] + bfilt[conf]

        st += sample_timing
        for j in range(len(los)):
            lts[j] += loss_timings[j]
        iso_count += [isos]
    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    print("\tisomorphisms: {}".format(np.mean(iso_count)))
    print("\tSampling Runtime: {}".format(st))
    for i in range(len(los)):
        print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
        print("\tLoss Runtime {}: {}".format(lstring[i], lts[i]))
        print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        print("\tLoss Runtime Comparison {}: {}".format(lstring[i], lts[i]/sum(lts)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
            print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
            print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
            print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))
    return source

def run_graph_source(G, T, source, name):
    alps = [0.1, 0.2]
    los = [first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["TTD"]
    ordered = [True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    c21 = [0 for i in range(len(alps) * len(los))]
    c22 = [0 for i in range(len(alps) * len(los))]
    c23 = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]
    avg21 = [0 for i in range(len(alps) * len(los))]
    avg22 = [0 for i in range(len(alps) * len(los))]
    avg23 = [0 for i in range(len(alps) * len(los))]
    f = [0 for i in range(len(alps) * len(los))]
    bf = [0 for i in range(len(alps) * len(los))]
    st = 0
    lts = [0 for i in range(len(los))]
    iso_count = []
    for i in range(K):
        #print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        #print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        #print("{} % Computing confidence set ".format(i), end="\r")
        estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1(G, x, m, los, ordered, T, alps, i)
        #print("{} % Computing distance center".format(i), end="\r")
        d = Alg2(G, x)
        pairs, isos = get_iso(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
            if source in C21[conf]:
                c21[conf] += 1
            avg21[conf] += len(C21[conf])
            f[conf] = f[conf] + filt[conf]
            bf[conf] = bf[conf] + bfilt[conf]

        st += sample_timing
        for j in range(len(los)):
            lts[j] += loss_timings[j]
        iso_count += [isos]
    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    print("\tisomorphisms: {}".format(np.mean(iso_count)))
    print("\tSampling Runtime: {}".format(st))
    for i in range(len(los)):
        print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
        print("\tLoss Runtime {}: {}".format(lstring[i], lts[i]))
        print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        print("\tLoss Runtime Comparison {}: {}".format(lstring[i], lts[i]/sum(lts)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
            print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
            print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
            print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))
    return source

def run_large_source(G, source, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "dist", "TTD"]
    ordered = [False, False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]

    for T in [25, 50]:
        a = [0 for i in range(len(los))]
        ahops = [[] for i in range(len(los))]
        b = 0
        bhops = []
        r = 0
        rhops = []
        c = [0 for i in range(len(alps) * len(los))]
        c21 = [0 for i in range(len(alps) * len(los))]
        c22 = [0 for i in range(len(alps) * len(los))]
        c23 = [0 for i in range(len(alps) * len(los))]
        avg = [0 for i in range(len(alps) * len(los))]
        avg21 = [0 for i in range(len(alps) * len(los))]
        avg22 = [0 for i in range(len(alps) * len(los))]
        avg23 = [0 for i in range(len(alps) * len(los))]
        f = [0 for i in range(len(alps) * len(los))]
        bf = [0 for i in range(len(alps) * len(los))]
        st = 0
        lts = [0 for i in range(len(los))]
        iso_count = []
        for i in range(K):
            #print("{} % Generating infected      ".format(i), end="\r")
            x, x_l = G.sample(source, T)
            #print("{} % Computing rumor center   ".format(i), end="\r")
            center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
            #print("{} % Computing confidence set ".format(i), end="\r")
            estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1(G, x, m, los, ordered, T, alps, i)
            #print("{} % Computing distance center".format(i), end="\r")
            d = Alg2(G, x)
            pairs, isos = get_iso(G, x)
            for e in range(len(estimates)):
                if estimates[e] == source:
                    a[e] += 1
                ahops[e] += [G.dist(estimates[e], source)]
            if d == source:
                b += 1
            bhops += [G.dist(d, source)]
            if rs == source:
                r += 1
            rhops += [G.dist(rs, source)]
            for conf in range(len(Cs)):
                if source in Cs[conf]:
                    c[conf] += 1
                avg[conf] += len(Cs[conf])
                if source in C21[conf]:
                    c21[conf] += 1
                avg21[conf] += len(C21[conf])
                f[conf] = f[conf] + filt[conf]
                bf[conf] = bf[conf] + bfilt[conf]

            st += sample_timing
            for j in range(len(los)):
                lts[j] += loss_timings[j]
            iso_count += [isos]
        print()
        print(name)
        print("T: ", T)
        print("N: ", G.graph.number_of_nodes())
        for j in range(len(los)):
            print("\ta {}: {}".format(lstring[j], a[j]/K))
            print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
            print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
        print("\tb: {}".format(b/K))
        print("\tbhops mean: {}".format(np.mean(bhops)))
        print("\tbhops: {}".format(Counter(bhops)))
        print("\trs: {}".format(r/K))
        print("\trhops mean: {}".format(np.mean(rhops)))
        print("\trhops: {}".format(Counter(rhops)))
        print("\tisomorphisms: {}".format(np.mean(iso_count)))
        for i in range(len(los)):
            print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
            print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
                print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
                print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
                print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
                print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))
    return source

def simple_run_large_source(G, source, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "dist", "TTD"]
    ordered = [False, False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]
    m_mat = []

    for T in [25, 50]:
        a = [0 for i in range(len(los))]
        ahops = [[] for i in range(len(los))]
        c = [0 for i in range(len(alps) * len(los))]
        c21 = [0 for i in range(len(alps) * len(los))]
        c21d6 = [0 for i in range(len(alps) * len(los))]
        c22 = [0 for i in range(len(alps) * len(los))]
        c23 = [0 for i in range(len(alps) * len(los))]
        avg = [0 for i in range(len(alps) * len(los))]
        avg21 = [0 for i in range(len(alps) * len(los))]
        avg21d6 = [0 for i in range(len(alps) * len(los))]
        avg22 = [0 for i in range(len(alps) * len(los))]
        avg23 = [0 for i in range(len(alps) * len(los))]
        f = [0 for i in range(len(alps) * len(los))]
        bf = [0 for i in range(len(alps) * len(los))]
        for i in range(K):
            #print("{} % Generating infected      ".format(i), end="\r")
            x, x_l = G.sample(source, T)
            #print("{} % Computing confidence set ".format(i), end="\r")
            estimates, Cs, C21, C21d6, C22, C23, filt, bfilt, sample_timing, loss_timings, p_list = Alg1_filter_matrix(G, x, m, source, los, ordered, T, alps, i)
            #print("{} % Computing distance center".format(i), end="\r")
            for e in range(len(estimates)):
                if estimates[e] == source:
                    a[e] += 1
                ahops[e] += [G.dist(estimates[e], source)]
            for conf in range(len(Cs)):
                if source in Cs[conf]:
                    c[conf] += 1
                avg[conf] += len(Cs[conf])
                if source in C21[conf]:
                    c21[conf] += 1
                avg21[conf] += len(C21[conf])
                if source in C21d6[conf]:
                    c21d6[conf] += 1
                avg21d6[conf] += len(C21d6[conf])
                f[conf] = f[conf] + filt[conf]
                bf[conf] = bf[conf] + bfilt[conf]
            m_mat += p_list

        print()
        print(name)
        print("T: ", T)
        print("N: ", G.graph.number_of_nodes())
        for j in range(len(los)):
            print("\ta {}: {}".format(lstring[j], a[j]/K))
            print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
            print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
                print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
                print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))
                print("\tc21d6 {} {}: {}".format(lstring[i], alps[j], c21d6[i*len(alps) + j]/K))
                print("\tavg21d6 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21d6[i*len(alps) + j]/K))

        for i in range(len(los)):
            for j in range(len(alps)):
                print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
                print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
                print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))
    return source, m_mat

def run_graph_multi_source(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "dist", "TTD"]
    ordered = [False, False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]

    los = [loss_creator(L2), first_miss_loss, avg_deviation_time, avg_matching_time]
    lstring = ["L2", "TTD", "ADT", "AMT"]
    ordered = [False, True, True, True]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    c21 = [0 for i in range(len(alps) * len(los))]
    c22 = [0 for i in range(len(alps) * len(los))]
    c23 = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]
    avg21 = [0 for i in range(len(alps) * len(los))]
    avg22 = [0 for i in range(len(alps) * len(los))]
    avg23 = [0 for i in range(len(alps) * len(los))]
    f = [0 for i in range(len(alps) * len(los))]
    bf = [0 for i in range(len(alps) * len(los))]
    st = 0
    lts = [0 for i in range(len(los))]
    iso_count = []
    source = G.select_source()
    for i in range(K):
        if i % 5 == 0:
            source = G.select_source()
        #print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        #print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        #print("{} % Computing confidence set ".format(i), end="\r")
        estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1(G, x, m, los, ordered, T, alps, i)
        #print("{} % Computing distance center".format(i), end="\r")
        d = Alg2(G, x)
        pairs, isos = get_iso(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
            if source in C21[conf]:
                c21[conf] += 1
            avg21[conf] += len(C21[conf])
            f[conf] = f[conf] + filt[conf]
            bf[conf] = bf[conf] + bfilt[conf]

        st += sample_timing
        for j in range(len(los)):
            lts[j] += loss_timings[j]
        iso_count += [isos]
    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    print("\tisomorphisms: {}".format(np.mean(iso_count)))
    print("\tSampling Runtime: {}".format(st))
    for i in range(len(los)):
        print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
        print("\tLoss Runtime {}: {}".format(lstring[i], lts[i]))
        print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        print("\tLoss Runtime Comparison {}: {}".format(lstring[i], lts[i]/sum(lts)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
            print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
            print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
            print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))
    return source

def run_graph_iso(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "dist", "TTD"]
    ordered = [False, False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alpha = [0.1, 0.2]

    #los = [loss_creator(L2), first_miss_loss, avg_deviation_time, avg_matching_time]
    #lstring = ["L2", "TTD", "ADT", "AMT"]
    #ordered = [False, True, True, True]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    c21 = [0 for i in range(len(alps) * len(los))]
    c22 = [0 for i in range(len(alps) * len(los))]
    c23 = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]
    avg21 = [0 for i in range(len(alps) * len(los))]
    avg22 = [0 for i in range(len(alps) * len(los))]
    avg23 = [0 for i in range(len(alps) * len(los))]
    f = [0 for i in range(len(alps) * len(los))]
    bf = [0 for i in range(len(alps) * len(los))]
    st = 0
    lts = [0 for i in range(len(los))]
    iso_count = []
    source = G.select_source()
    for i in range(K):
        #print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        #print("{} % Computing rumor center   ".format(i), end="\r")
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        #print("{} % Computing confidence set ".format(i), end="\r")
        estimates, Cs, C21, C22, C23, filt, bfilt, sample_timing, loss_timings = Alg1_iso(G, x, m, los, ordered, T, alps, i)
        #print("{} % Computing distance center".format(i), end="\r")
        d = Alg2(G, x)
        pairs, isos = get_iso(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
            if source in C21[conf]:
                c21[conf] += 1
            avg21[conf] += len(C21[conf])
            f[conf] = f[conf] + filt[conf]
            bf[conf] = bf[conf] + bfilt[conf]

        st += sample_timing
        for j in range(len(los)):
            lts[j] += loss_timings[j]
        iso_count += [isos]
    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    print("\tisomorphisms: {}".format(np.mean(iso_count)))
    print("\tSampling Runtime: {}".format(st))
    for i in range(len(los)):
        print("\tSampling Runtime Proportion {}: {}".format(lstring[i], st/(st + lts[i])))
        print("\tLoss Runtime {}: {}".format(lstring[i], lts[i]))
        print("\tLoss Runtime Proportion {}: {}".format(lstring[i], lts[i]/(st + lts[i])))
        print("\tLoss Runtime Comparison {}: {}".format(lstring[i], lts[i]/sum(lts)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc21 {} {}: {}".format(lstring[i], alps[j], c21[i*len(alps) + j]/K))
            print("\tavg21 confidence set size {} {}: {}".format(lstring[i], alps[j], avg21[i*len(alps) + j]/K))

    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tfiltered {} {}: {}".format(lstring[i], alps[j], f[i*len(alps) + j]))
            print("\tbadly filtered {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]))
            print("\tproportion bad {} {}: {}".format(lstring[i], alps[j], bf[i*len(alps) + j]/f[i*len(alps)+j]))

    print_iso(G, name)
    return source

def run_graph_compare(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), distance_loss, first_miss_loss]
    ordered = [False, True]
    alpha = [0.1, 0.2]

    l2_c = [0 for i in range(len(alps))]
    l2_avg = [0 for i in range(len(alps))]
    ttd_c = [0 for i in range(len(alps))]
    ttd_avg = [0 for i in range(len(alps))]
    min_c = [0 for i in range(len(alps))]
    min_avg = [0 for i in range(len(alps))]
    int_c = [0 for i in range(len(alps))]
    int_avg = [0 for i in range(len(alps))]

    source = G.select_source()
    for i in range(K):
        print("{} % Generating infected      ".format(i), end="\r")
        x, x_l = G.sample(source, T)
        print("{} % Computing confidence set ".format(i), end="\r")
        l1c, l2c, c_min, c_int = Alg1_compare(G, x, m, loss_creator(L2), first_miss_loss, False, True, T, alps, i)
        print("{} % Computing distance center".format(i), end="\r")

        for a in range(len(alps)):
            if source in l1c[a]:
                l2_c[a] += 1
            l2_avg[a] += len(l1c[a])

            if source in l2c[a]:
                ttd_c[a] += 1
            ttd_avg[a] += len(l2c[a])

            if source in c_min[a]:
                min_c[a] += 1
            min_avg[a] += len(c_min[a])

            if source in c_int[a]:
                int_c[a] += 1
            int_avg[a] += len(c_int[a])

    print()
    print(name)
    for j in range(len(alps)):
        print("\tL2 {}: {}".format(alps[j], l2_c[j]/K))
        print("\tavg size L2 {}: {}".format(alps[j], l2_avg[j]/K))

        print("\tTTD {}: {}".format(alps[j], ttd_c[j]/K))
        print("\tavg size TTD {}: {}".format(alps[j], ttd_avg[j]/K))

        print("\tMin {}: {}".format(alps[j], min_c[j]/K))
        print("\tavg size Min {}: {}".format(alps[j], min_avg[j]/K))

        print("\tInt {}: {}".format(alps[j], int_c[j]/K))
        print("\tavg size Int {}: {}".format(alps[j], int_avg[j]/K))

    return source

def run_graph_final_sources(G, T, name, sources):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [loss_creator(L2), first_miss_loss]
    #los = [loss_creator(L2)]
    lstring = ["L2", "TTD", "Intersect"]
    ordered = [False, True]
    #lstring = ["L2"]
    #ordered = [False]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    #los = [loss_creator(L2), first_miss_loss, avg_deviation_time, avg_matching_time]
    #lstring = ["L2", "TTD", "ADT", "AMT"]
    #ordered = [False, True, True, True]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    p = [0 for i in range(len(los))]
    phops = [[] for i in range(len(los))]
    pvals = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * (len(los)+1))]
    avg = [0 for i in range(len(alps) * (len(los)+1))]

    total_sts = []
    max_sts = []
    max_lts = [[] for i in range(len(los))]
    total_lts = [[] for i in range(len(los))]
    iso_timing = []
    rumor_timing = []

    iso_saved = []
    d1_saved = []

    for source in sources:
        for i in range(K):
            x, x_l = G.sample(source, T)
            start = time.time()
            center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
            end = time.time()
            #estimates, Cs, max_p = Alg1_filtering(G, x, m, los, ordered, T, alps)
            estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_filtering_intersect(G, x, m, los, ordered, T, alps)
            d = Alg2(G, x)
            for e in range(len(estimates)):
                if estimates[e] == source:
                    a[e] += 1
                ahops[e] += [G.dist(estimates[e], source)]
            for pi in range(len(max_p)):
                source_p = max_p[pi]
                if source_p[0] == source:
                    p[pi] += 1
                phops[pi] += [G.dist(source_p[0], source)]
                pvals[pi] += [source_p[1]]
            if d == source:
                b += 1
            bhops += [G.dist(d, source)]
            if rs == source:
                r += 1
            rhops += [G.dist(rs, source)]
            for conf in range(len(Cs)):
                if source in Cs[conf]:
                    c[conf] += 1
                avg[conf] += len(Cs[conf])

            total_sts += [sum(st + d1st)]
            max_sts += [max(st + d1st)]
            for l in range(len(los)):
                total_lts[l] += [sum(lt[l])]
                max_lts[l] += [max(lt[l])]
            rumor_timing += [end - start]
            iso_timing += [it]

            iso_saved += [iso_s]
            d1_saved += [d1_s]

    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tRumor Center Runtime: {}".format(np.mean(rumor_timing)))
    print("\tIso Runtime: {}".format(np.mean(iso_timing)))
    print("\tTotal Sampling Runtime: {}".format(np.mean(total_sts)))
    print("\tMax Sampling Runtime: {}".format(np.mean(max_sts)))
    for i in range(len(los)):
        print("\tTotal Loss Runtime {}: {}".format(lstring[i], np.mean(total_lts[i])))
        print("\tMax Loss Runtime {}: {}".format(lstring[i], np.mean(max_lts[i])))
    print("\tIso Saved: {}".format(np.mean(iso_saved)))
    print("\tD1 Saved: {}".format(np.mean(d1_saved)))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    for i in range(len(los)+1):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        print("\tp value success {}: {}".format(lstring[i], p[i]/K))
        print("\tp value {}: {} - {:.4f} - {}".format(lstring[i], min(pvals[i]), np.mean(pvals[i]), max(pvals[i])))

    print_iso(G, name)
    return source

def run_graph_test_d1(G, T, name):
    alps = [0.1, 0.2]
    los = [loss_creator(L2), first_miss_loss]
    lstring = ["L2", "TTD", "Intersect"]
    ordered = [False, True]
    alpha = [0.1, 0.2]

    c = [0 for i in range(len(alps) * (len(los)+1))]
    c_T = [0 for i in range(len(alps) * (len(los)+1))]
    c_p = [0 for i in range(len(alps) * (len(los)+1))]
    c_d1 = [0 for i in range(len(alps) * (len(los)+1))]

    source = G.select_source()
    for i in range(K):
        x, x_l = G.sample(source, T)
        Cs, Cs_T, Cs_p, Cs_d1 = Alg1_test_source(G, x, source, m, los, ordered, T, alps)
        for conf in range(len(Cs)):
            if Cs[conf]:
                c[conf] += 1
            if Cs_T[conf]:
                c_T[conf] += 1
            if Cs_p[conf]:
                c_p[conf] += 1
            if Cs_d1[conf]:
                c_d1[conf] += 1

    print()
    print(name)
    for i in range(len(los)+1):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tc T {} {}: {}".format(lstring[i], alps[j], c_T[i*len(alps) + j]/K))
            print("\tc p {} {}: {}".format(lstring[i], alps[j], c_p[i*len(alps) + j]/K))
            print("\tc d1 {} {}: {}".format(lstring[i], alps[j], c_d1[i*len(alps) + j]/K))
        print()

    return source

def run_graph_fast_loss(G, T, name, degree1=True, iso=True):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADT2_h, ADiT_h, ADiT2_h]
    lstring = ["L2", "ADT", "ADT2", "ADiT", "ADiT2"]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    p = [0 for i in range(len(los))]
    phops = [[] for i in range(len(los))]
    pvals = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]

    total_sts = []
    max_sts = []
    max_lts = [[] for i in range(len(los))]
    total_lts = [[] for i in range(len(los))]
    iso_timing = []
    rumor_timing = []
    alg1_timing = []

    iso_saved = []
    d1_saved = []

    source = G.select_source()

    K_tmp = K
    if T == 200:
        K_tmp = K_tmp//2

    for i in range(K_tmp):
        x, x_l = G.sample(source, T)
        start = time.time()
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        end = time.time()
        rumor_timing += [end - start]
        #estimates, Cs, max_p = Alg1_filtering(G, x, m, los, ordered, T, alps)
        start = time.time()
        estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_faster_loss(G, x, m, los, T, alps, degree1, iso)
        end = time.time()
        alg1_timing += [end - start]
        d = Alg2(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        for pi in range(len(max_p)):
            source_p = max_p[pi]
            if source_p[0] == source:
                p[pi] += 1
            phops[pi] += [G.dist(source_p[0], source)]
            pvals[pi] += [source_p[1]]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])

        total_sts += [sum(st + d1st)]
        max_sts += [max(st + d1st)]
        for l in range(len(los)):
            total_lts[l] += [sum(lt[l])]
            max_lts[l] += [max(lt[l])]
        iso_timing += [it]

        iso_saved += [iso_s]
        d1_saved += [d1_s]

    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tRumor Center Runtime: {}".format(np.mean(rumor_timing)))
    print("\tRumor Center Runtime: {}".format(np.mean(alg1_timing)))
    print("\tIso Runtime: {}".format(np.mean(iso_timing)))
    print("\tTotal Sampling Runtime: {}".format(np.mean(total_sts)))
    print("\tMax Sampling Runtime: {}".format(np.mean(max_sts)))
    for i in range(len(los)):
        print("\tTotal Loss Runtime {}: {}".format(lstring[i], np.mean(total_lts[i])))
        print("\tMax Loss Runtime {}: {}".format(lstring[i], np.mean(max_lts[i])))
    print("\tIso Saved: {}".format(np.mean(iso_saved)))
    print("\tD1 Saved: {}".format(np.mean(d1_saved)))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        print("\tp value success {}: {}".format(lstring[i], p[i]/K))
        print("\tp value {}: {} - {:.4f} - {}".format(lstring[i], min(pvals[i]), np.mean(pvals[i]), max(pvals[i])))

    #print_iso(G, name)
    return source

def run_graph_parallel(G, T, name, degree1=True, iso=True):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADT2_h, ADiT_h, ADiT2_h]
    lstring = ["L2", "ADT", "ADT2", "ADiT", "ADiT2"]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]

    rumor_timing = []
    alg1_timing = []

    source = G.select_source()

    for i in range(K):
        x, x_l = G.sample(source, T)
        start = time.time()
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        end = time.time()
        rumor_timing += [end - start]
        #estimates, Cs, max_p = Alg1_filtering(G, x, m, los, ordered, T, alps)
        start = time.time()
        Cs = Alg1_parallel_fast(G, x, m, los, T, alps, degree1, iso)
        end = time.time()
        alg1_timing += [end - start]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])

    print()
    print(name)
    #for j in range(len(los)):
    #    print("\ta {}: {}".format(lstring[j], a[j]/K))
    #    print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
    #    print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tRumor Center Runtime: {}".format(np.mean(rumor_timing)))
    print("\tParallel Runtime: {}".format(np.mean(alg1_timing)))
    #print("\tIso Saved: {}".format(np.mean(iso_saved)))
    #print("\tD1 Saved: {}".format(np.mean(d1_saved)))
    #print("\tb: {}".format(b/K))
    #print("\tbhops mean: {}".format(np.mean(bhops)))
    #print("\tbhops: {}".format(Counter(bhops)))
    #print("\trs: {}".format(r/K))
    #print("\trhops mean: {}".format(np.mean(rhops)))
    #print("\trhops: {}".format(Counter(rhops)))
    #for i in range(len(los)):
    #    for j in range(len(alps)):
    #        print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
    #        print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    #for i in range(len(los)):
    #    print("\tp value success {}: {}".format(lstring[i], p[i]/K))
    #    print("\tp value {}: {} - {:.4f} - {}".format(lstring[i], min(pvals[i]), np.mean(pvals[i]), max(pvals[i])))

    #print_iso(G, name)
    return source

def run_graph_connected(G, T, name, degree1=True, iso=True):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADT2_h, ADiT_h, ADiT2_h]
    lstring = ["L2", "ADT", "ADT2", "ADiT", "ADiT2"]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    p = [0 for i in range(len(los))]
    phops = [[] for i in range(len(los))]
    pvals = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]
    conn = [0 for i in range(len(alps) * (len(los)))]

    total_sts = []
    max_sts = []
    max_lts = [[] for i in range(len(los))]
    total_lts = [[] for i in range(len(los))]
    iso_timing = []
    rumor_timing = []
    alg1_timing = []

    iso_saved = []
    d1_saved = []

    source = G.select_source()

    for i in range(K):
        x, x_l = G.sample(source, T)
        start = time.time()
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        end = time.time()
        rumor_timing += [end - start]
        #estimates, Cs, max_p = Alg1_filtering(G, x, m, los, ordered, T, alps)
        start = time.time()
        estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_faster_loss(G, x, m, los, T, alps, degree1, iso)
        end = time.time()
        alg1_timing += [end - start]
        d = Alg2(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        for pi in range(len(max_p)):
            source_p = max_p[pi]
            if source_p[0] == source:
                p[pi] += 1
            phops[pi] += [G.dist(source_p[0], source)]
            pvals[pi] += [source_p[1]]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])
            if is_connected(G, Cs[conf]):
                conn[conf] += 1

        total_sts += [sum(st + d1st)]
        max_sts += [max(st + d1st)]
        for l in range(len(los)):
            total_lts[l] += [sum(lt[l])]
            max_lts[l] += [max(lt[l])]
        iso_timing += [it]

        iso_saved += [iso_s]
        d1_saved += [d1_s]

    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tRumor Center Runtime: {}".format(np.mean(rumor_timing)))
    print("\tRumor Center Runtime: {}".format(np.mean(alg1_timing)))
    print("\tIso Runtime: {}".format(np.mean(iso_timing)))
    print("\tTotal Sampling Runtime: {}".format(np.mean(total_sts)))
    print("\tMax Sampling Runtime: {}".format(np.mean(max_sts)))
    for i in range(len(los)):
        print("\tTotal Loss Runtime {}: {}".format(lstring[i], np.mean(total_lts[i])))
        print("\tMax Loss Runtime {}: {}".format(lstring[i], np.mean(max_lts[i])))
    print("\tIso Saved: {}".format(np.mean(iso_saved)))
    print("\tD1 Saved: {}".format(np.mean(d1_saved)))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))
            print("\tis connected {} {}: {}".format(lstring[i], alps[j], conn[i*len(alps) + j]/K))

    for i in range(len(los)):
        print("\tp value success {}: {}".format(lstring[i], p[i]/K))
        print("\tp value {}: {} - {:.4f} - {}".format(lstring[i], min(pvals[i]), np.mean(pvals[i]), max(pvals[i])))

    print_iso(G, name)
    return source

def run_graph_fast_loss_file(G, T, name, f, degree1=True, iso=True):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADiT_h]
    lstring = ["L2", "ADT", "ADiT"]
    #alps = [0.1, 0.2]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    p = [0 for i in range(len(los))]
    phops = [[] for i in range(len(los))]
    pvals = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]

    total_sts = []
    max_sts = []
    max_lts = [[] for i in range(len(los))]
    total_lts = [[] for i in range(len(los))]
    iso_timing = []
    rumor_timing = []
    alg1_timing = []

    iso_saved = []
    d1_saved = []

    source = G.select_uniform_source()

    for i in range(K):
        x, x_l = G.sample(source, T)
        start = time.time()
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        end = time.time()
        rumor_timing += [end - start]
        #estimates, Cs, max_p = Alg1_filtering(G, x, m, los, ordered, T, alps)
        start = time.time()
        estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_faster_loss(G, x, m, los, T, alps, degree1, iso)
        end = time.time()
        alg1_timing += [end - start]
        d = Alg2(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        for pi in range(len(max_p)):
            source_p = max_p[pi]
            if source_p[0] == source:
                p[pi] += 1
            phops[pi] += [G.dist(source_p[0], source)]
            pvals[pi] += [source_p[1]]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])

        total_sts += [sum(st + d1st)]
        max_sts += [max(st + d1st)]
        for l in range(len(los)):
            total_lts[l] += [sum(lt[l])]
            max_lts[l] += [max(lt[l])]
        iso_timing += [it]

        iso_saved += [iso_s]
        d1_saved += [d1_s]

    print(name, file=f)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K), file=f)
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])), file=f)
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])), file=f)
    print("\tRumor Center Runtime: {}".format(np.mean(rumor_timing)), file=f)
    print("\tRumor Center Runtime: {}".format(np.mean(alg1_timing)), file=f)
    print("\tIso Runtime: {}".format(np.mean(iso_timing)), file=f)
    print("\tTotal Sampling Runtime: {}".format(np.mean(total_sts)), file=f)
    print("\tMax Sampling Runtime: {}".format(np.mean(max_sts)), file=f)
    for i in range(len(los)):
        print("\tTotal Loss Runtime {}: {}".format(lstring[i], np.mean(total_lts[i])), file=f)
        print("\tMax Loss Runtime {}: {}".format(lstring[i], np.mean(max_lts[i])), file=f)
    print("\tIso Saved: {}".format(np.mean(iso_saved)), file=f)
    print("\tD1 Saved: {}".format(np.mean(d1_saved)), file=f)
    print("\tb: {}".format(b/K), file=f)
    print("\tbhops mean: {}".format(np.mean(bhops)), file=f)
    print("\tbhops: {}".format(Counter(bhops)), file=f)
    print("\trs: {}".format(r/K), file=f)
    print("\trhops mean: {}".format(np.mean(rhops)), file=f)
    print("\trhops: {}".format(Counter(rhops)), file=f)
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K), file=f)
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K), file=f)

    for i in range(len(los)):
        print("\tp value success {}: {}".format(lstring[i], p[i]/K), file=f)
        print("\tp value {}: {} - {:.4f} - {}".format(lstring[i], min(pvals[i]), np.mean(pvals[i]), max(pvals[i])), file=f)

    #print_iso(G, name)
    return source

def run_graph_directed(G, T, name):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADT2_h, ADiT_h, ADiT2_h]
    lstring = ["L2", "ADT", "ADT2", "ADiT", "ADiT2"]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]

    source = G.select_source()

    for i in range(K):
        x, x_l = G.sample(source, T)
        estimates, Cs = Alg1_directed(G, x, m, los, T, alps)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])

    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))
    return source

def run_graph_weighted(G, T, name):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADT2_h, ADiT_h, ADiT2_h]
    lstring = ["L2", "ADT", "ADT2", "ADiT", "ADiT2"]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]

    source = G.select_source()

    for i in range(K):
        x, x_l = G.sample(source, T)
        estimates, Cs = Alg1_weighted(G, x, m, los, T, alps)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])

    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))
    return source

def run_graph_fast_loss_source(G, T, source, name, degree1=True, iso=True, M=m):
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADT2_h, ADiT_h, ADiT2_h]
    lstring = ["L2", "ADT", "ADT2", "ADiT", "ADiT2"]
    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    p = [0 for i in range(len(los))]
    phops = [[] for i in range(len(los))]
    pvals = [[] for i in range(len(los))]
    b = 0
    bhops = []
    r = 0
    rhops = []
    c = [0 for i in range(len(alps) * (len(los)))]
    avg = [0 for i in range(len(alps) * (len(los)))]

    total_sts = []
    max_sts = []
    max_lts = [[] for i in range(len(los))]
    total_lts = [[] for i in range(len(los))]
    iso_timing = []
    rumor_timing = []
    alg1_timing = []

    iso_saved = []
    d1_saved = []

    #source = G.select_source()

    for i in range(K):
        x, x_l = G.sample(source, T)
        start = time.time()
        center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
        end = time.time()
        rumor_timing += [end - start]
        #estimates, Cs, max_p = Alg1_filtering(G, x, m, los, ordered, T, alps)
        start = time.time()
        #estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_faster_loss_check_source(G, x, source, M, los, T, alps, degree1, iso)
        estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_faster_loss(G, x, M, los, T, alps, degree1, iso)
        end = time.time()
        alg1_timing += [end - start]
        d = Alg2(G, x)
        for e in range(len(estimates)):
            if estimates[e] == source:
                a[e] += 1
            ahops[e] += [G.dist(estimates[e], source)]
        for pi in range(len(max_p)):
            source_p = max_p[pi]
            if source_p[0] == source:
                p[pi] += 1
            phops[pi] += [G.dist(source_p[0], source)]
            pvals[pi] += [source_p[1]]
        if d == source:
            b += 1
        bhops += [G.dist(d, source)]
        if rs == source:
            r += 1
        rhops += [G.dist(rs, source)]
        for conf in range(len(Cs)):
            if source in Cs[conf]:
                c[conf] += 1
            avg[conf] += len(Cs[conf])

        total_sts += [sum(st + d1st)]
        max_sts += [max(st + d1st)]
        for l in range(len(los)):
            total_lts[l] += [sum(lt[l])]
            max_lts[l] += [max(lt[l])]
        iso_timing += [it]

        iso_saved += [iso_s]
        d1_saved += [d1_s]

    print()
    print(name)
    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/K))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tRumor Center Runtime: {}".format(np.mean(rumor_timing)))
    print("\tRumor Center Runtime: {}".format(np.mean(alg1_timing)))
    print("\tIso Runtime: {}".format(np.mean(iso_timing)))
    print("\tTotal Sampling Runtime: {}".format(np.mean(total_sts)))
    print("\tMax Sampling Runtime: {}".format(np.mean(max_sts)))
    for i in range(len(los)):
        print("\tTotal Loss Runtime {}: {}".format(lstring[i], np.mean(total_lts[i])))
        print("\tMax Loss Runtime {}: {}".format(lstring[i], np.mean(max_lts[i])))
    print("\tIso Saved: {}".format(np.mean(iso_saved)))
    print("\tD1 Saved: {}".format(np.mean(d1_saved)))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    print("\trs: {}".format(r/K))
    print("\trhops mean: {}".format(np.mean(rhops)))
    print("\trhops: {}".format(Counter(rhops)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/K))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/K))

    for i in range(len(los)):
        print("\tp value success {}: {}".format(lstring[i], p[i]/K))
        print("\tp value {}: {} - {:.4f} - {}".format(lstring[i], min(pvals[i]), np.mean(pvals[i]), max(pvals[i])))

    #print_iso(G, name)
    return source

def final_results(G, name, t=150, source=None):
    if source is None:
        source = G.select_source()
    #for t in [50, 100, 150, 200]:
    #run_graph_weighted(G, tmp, name)
    #run_graph_fast_loss_source(G, t, source, name, False, False)
    #for mi in [8000, 12000, 16000, 20000, 24000, 28000, 32000, 40000]:
    for mi in [10000]:
        run_graph_parallel(G, t, name, True, True)
        run_graph_parallel(G, t, name, True, False)
        run_graph_parallel(G, t, name, False, True)
        run_graph_parallel(G, t, name, False, False)
        #run_graph_fast_loss_source(G, t, source, name, True, True, mi)
        #run_graph_fast_loss_source(G, t, source, name, True, False, mi)
        #run_graph_fast_loss_source(G, t, source, name, False, True, mi)
        #run_graph_fast_loss_source(G, t, source, name, False, False, mi)

    return source

def final_results_file(G, name, rf):
    sources = [G.select_source()]
    for t in [min(len(G.graph)//5, 150)]:
        run_graph_fast_loss_file(G, t, name, rf, True, True)
        #run_graph_fast_loss_file(G, t, name, rf, True, False)
        #run_graph_fast_loss_file(G, t, name, rf, False, True)
        #run_graph_fast_loss_file(G, t, name, rf, False, False)

    return sources

def final_results_T_file(G, name, t, rf):
    sources = [G.select_source()]
    #for t in [min(len(G.graph)//5, 150)]:
    #for t in [50, 100, 150, 200]:
    run_graph_fast_loss_file(G, t, name, rf, True, True)

    return sources

def save_graph(G, s, fname):
    pickle.dump(G, open("saved/" + fname + "_graph.p", "wb"))
    pickle.dump(s, open("saved/" + fname + "_source.p", "wb"))

def load_graph(fname):
    G = pickle.load(open("saved/" + fname + "_graph.p", "rb"))
    s = pickle.load(open("saved/" + fname + "_source.p", "rb"))
    return G, s
