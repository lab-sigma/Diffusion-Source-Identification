import graphs
from rumor_center import rumor_center, rumor_center_topk
from isomorphisms import first_order_iso, iso_groups
import time
from collections import Counter
from multiprocessing import Pool
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import itertools
import random
from decimal import *
import os, psutil

def analyze_group_d1(g, d1_children, x, G, m, T, losses, alphas, node_vals, temporal_loss, degree1):
    g_list = list(g)
    sp = g_list[0]
    tmp_conf = [[] for _ in range(len(alphas) * (len(losses)))]

    neighbors = set(G.graph.neighbors(sp))

    if degree1 and len(d1_children) > 0:
        samples_x_raw = [G.sample_d1(sp, T) for i in range(m)]
        samples_c_raw = [G.sample_d1(sp, T) for i in range(m)]
    else:
        samples_x_raw = [G.sample(sp, T) for i in range(m)]
        samples_c_raw = [G.sample(sp, T) for i in range(m)]

    samples_x = [xs[0] for xs in samples_x_raw]
    samples_x_list = [xs[1] for xs in samples_x_raw]
    samples_c = [c[0] for c in samples_c_raw]
    samples_c_list = [c[1] for c in samples_c_raw]

    for i in range(len(losses)):
        loss = losses[i]
        samples_x_tmp = node_vals(loss, samples_x_list, T)
        mu_x = temporal_loss(x, samples_x_tmp)
        avg = sum([temporal_loss(yi, samples_x_tmp) >= mu_x for yi in samples_c_list])/m
        for s in g:
            for a in range(len(alphas)):
                if avg > alphas[a]:
                    tmp_conf[i*len(alphas) + a].append(s)

    if degree1 and len(d1_children) > 0:
        sp = [sc for sc in G.graph.neighbors(sp) if G.graph.degree[sc] == 1][0]

        samples_x_degrees = [xs[2] for xs in samples_x_raw]
        samples_c_degrees = [xs[2] for xs in samples_c_raw]

        sample_x_ratios = list()
        for i in range(m):
            if not sp in samples_x[i]:
                sample_x_ratios.append(0)
                continue
            degree_list = samples_x_degrees[i]
            sample_list = samples_x_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_x_list[i] = sample_list
            sample_x_ratios.append(ratio)

        sample_c_ratios = list()
        for i in range(m):
            if not sp in samples_c[i]:
                sample_c_ratios.append(0)
                continue
            degree_list = samples_c_degrees[i] # Get degree list for sample
            sample_list = samples_c_list[i] # Get ordered sample
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp: # Reach d1 node
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k]) # Eqn 7
            samples_c_list[i] = sample_list
            sample_c_ratios.append(ratio)

        for i in range(len(losses)):
            loss = losses[i]
            samples_x_tmp = node_vals(loss, samples_x_list, T, sample_x_ratios)
            mu_x = temporal_loss(x, samples_x_tmp)/T
            avg = sum([ratio*(temporal_loss(yi, samples_x_tmp)/T >= mu_x) for yi, ratio in zip(samples_c_list, sample_c_ratios)])/(T*m)
            for s in g:
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        tmp_conf[i*len(alphas) + a].extend(d1_children)

    return tmp_conf

def analyze_group(g, x, G, m, T, losses, alphas, node_vals, temporal_loss, degree1):
    g_list = list(g)
    sp = g_list[0]
    tmp_conf = [[] for _ in range(len(alphas) * (len(losses)))]

    d1 = 0

    neighbors = set(G.graph.neighbors(sp))
    if degree1 and len(neighbors) == 1:
        return tmp_conf, d1, g, [], []

    d1 += 1

    if degree1 and any([G.graph.degree[n] == 1 for n in neighbors]):
        samples_x_raw = [G.sample_d1(sp, T) for i in range(m)]
        samples_c_raw = [G.sample_d1(sp, T) for i in range(m)]
        d1 += 1
    else:
        samples_x_raw = [G.sample(sp, T) for i in range(m)]
        samples_c_raw = [G.sample(sp, T) for i in range(m)]

    samples_x = [xs[0] for xs in samples_x_raw]
    samples_x_list = [xs[1] for xs in samples_x_raw]
    samples_c = [c[0] for c in samples_c_raw]
    samples_c_list = [c[1] for c in samples_c_raw]

    for i in range(len(losses)):
        loss = losses[i]
        samples_x_tmp = node_vals(loss, samples_x_list, T)
        mu_x = temporal_loss(x, samples_x_tmp)
        avg = sum([temporal_loss(yi, samples_x_tmp) >= mu_x for yi in samples_c_list])/m
        for a in range(len(alphas)):
            if avg > alphas[a]:
                tmp_conf[i*len(alphas) + a].extend(g_list)
    if d1 == 2:
        return tmp_conf, d1, g, samples_x_raw, samples_c_raw
    return tmp_conf, d1, [], [], []

def node_vals(h_t, samples, T, ratios=None):
    vals = {}
    if ratios is None:
        ratios = [1 for _ in range(len(samples))]
    for sample, ratio in zip(samples, ratios):
        for i, v in enumerate(sample):
            if not v in vals.keys():
                vals[v] = 0
            vals[v] += ratio * h_t(i+1, T)
    return vals

def temporal_loss(x, vals):
    Tx = 0
    for x_i in x:
        if x_i in vals.keys():
            Tx += vals[x_i]
    return -Tx

def collapse_degree1(G, x, groups):
    combined = []
    for g in groups:
        if G.graph.degree[list(g)[0]] == 1:
            continue
        d1_neighbors = list()
        for v_i in g:
            for n in G.graph.neighbors(v_i):
                if n in x and G.graph.degree[n] == 1:
                    d1_neighbors.append(n)
        combined.append((g, d1_neighbors))

    return combined


def Alg2(G, x):
    Dmin = -1
    amin = -1
    for i in x:
        Di = sum([G.dist(i, j) for j in x])
        if Di < Dmin or amin < 0:
            amin = i
            Dmin = Di
        elif Di == Dmin and random.random() < 0.5:
            amin = i
            Dmin = Di
    return amin

def add_groupless(x, groups):
    for v in x:
        if not any([v in g for g in groups]):
            groups.append({v})
    return groups

def Alg1(G, x, m, losses, ordered, T, alphas, progress=0, filter21=False, filter22=False, filter23=False):
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences21 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences22 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences23 = [[] for _ in range(len(alphas) * (len(losses)))]
    sample_timing = 0
    loss_timings = [0 for _ in range(len(losses))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    p = []
    GN = G.graph.subgraph(list(x))
    filtered = [0 for _ in range(len(alphas) * len(losses))]
    bad_filtered = [0 for _ in range(len(alphas) * len(losses))]
    for j in range(len(x_list)):
        s = x_list[j]
        neighbors = set(G.graph.neighbors(s))
        uninfected = set([v for v in neighbors if not v in x])
        infected = set([v for v in neighbors if v in x])
        #print("{:.2f} % Sampling                   ".format(progress + j/len(x)), end="\r")
        start = time.time()
        samples_x_raw = [G.sample(s, T) for i in range(m)]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c_raw = [G.sample(s, T) for i in range(m)]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]
        end = time.time()
        sample_timing += end - start
        #print("{:.2f} % Calcloss                   ".format(progress + j/len(x)), end="\r")

        if filter21:
            #if len(uninfected) > 0 and len(neighbors) <= 6:
            if len(uninfected) > 0:
                r = len(uninfected)
                du = len(neighbors)
                def dNxu(ui):
                    return len(neighbors.intersection(set(G.graph.neighbors(ui))))
                def comp_val(u1, u2):
                    return (r + dNxu(u1) + dNxu(u2))/(du + G.graph.degree(u1) + G.graph.degree(u2) - G.graph.has_edge(s, u1) - G.graph.has_edge(s, u2) - G.graph.has_edge(u1, u2))
                psi = 1 - r/du - sum([
                        (sum([comp_val(u1, u2)
                                for u2 in infected.union(set(GN.neighbors(u1))) if not (u2 == s or u2 == u1)])
                            + r + dNxu(u1))
                        /(du + G.graph.degree(u1) - 2) for u1 in infected])/du

        for i in range(len(losses)):
            loss = losses[i]
            if(ordered[i]):
                samples_x_tmp = samples_x_list
                samples_c_tmp = samples_c_list
            else:
                samples_x_tmp = samples_x
                samples_c_tmp = samples_c
            start = time.time()
            mu_x = loss(G, x, samples_x_tmp, s)
            if mu_x < mu_mins[i] or amins[i] == -1:
                amins[i] = s
                mu_mins[i] = mu_x
            avg = sum([loss(G, yi, samples_x_tmp, s) >= mu_x for yi in samples_c_tmp])/m
            end = time.time()
            loss_timings[i] += end - start
            p += [avg]
            for a in range(len(alphas)):
                if avg > alphas[a]:
                    confidences[i*len(alphas) + a].append(s)

            #if filter22:

            #if filter23:

            if filter21:
                #if len(uninfected) > 0 and len(neighbors) <= 6:
                if len(uninfected) > 0:
                    for a in range(len(alphas)):
                        if psi >= alphas[a]:
                            psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                            if psi_new > alphas[a]:
                                confidences21[i*len(alphas) + a].append(s)
                        else:
                            filtered[i*len(alphas) + a] = filtered[i*len(alphas) + a] + 1
                            psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                            if psi_new > alphas[a]:
                                bad_filtered[i*len(alphas) + a] = bad_filtered[i*len(alphas) + a] + 1
                else:
                    for a in range(len(alphas)):
                        if avg > alphas[a]:
                            confidences21[i*len(alphas) + a].append(s)
    return amins, confidences, confidences21, confidences22, confidences23, filtered, bad_filtered, sample_timing, loss_timings

def Alg1V2(G, x, losses, biased, T, alphas, progress=0):
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    p = []
    for j in range(len(x_list)):
        s = x_list[j]
        print("{:.2f} % Sampling                   ".format(progress + j/len(x)), end="\r")
        samples_x_raw = [G.sample(s, T) for i in range(m)]
        samples_x = [x[0] for x in samples_x_raw]
        samples_x_list = [x[1] for x in samples_x_raw]
        print("{:.2f} % Calcloss                   ".format(progress + j/len(x)), end="\r")

        for i in range(len(losses)):
            loss = losses[i]
            mu_x = loss(G, x, samples_x, s)/m
            if mu_x < mu_mins[i] or amins[i] < 0:
                amins[i] = s
                mu_mins[i] = mu_x
            if biased[i]:
                avg = sum([loss(G, yi, samples_x, s)/(m-1) >= mu_x for yi in samples_x])/m
            else:
                avg = sum([loss(G, yi, samples_x, s)/m >= mu_x for yi in samples_x])/m
            p += [avg]
            for a in range(len(alphas)):
                if avg > alphas[a]:
                    confidences[i*len(alphas) + a].append(s)

    return amins, confidences

def Alg1_filter_matrix(G, x, m, true_source, losses, ordered, T, alphas, progress=0, filter21=True, filter22=False, filter23=False):
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences21 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences21d6 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences22 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences23 = [[] for _ in range(len(alphas) * (len(losses)))]
    sample_timing = 0
    loss_timings = [0 for _ in range(len(losses))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    p = []
    p_val_list = []
    GN = G.graph.subgraph(list(x))
    filtered = [0 for _ in range(len(alphas) * len(losses))]
    bad_filtered = [0 for _ in range(len(alphas) * len(losses))]
    for j in range(len(x_list)):
        s = x_list[j]
        neighbors = set(G.graph.neighbors(s))
        uninfected = set([v for v in neighbors if not v in x])
        infected = set([v for v in neighbors if v in x])
        #print("{:.2f} % Sampling                   ".format(progress + j/len(x)), end="\r")
        start = time.time()
        samples_x_raw = [G.sample(s, T) for i in range(m)]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c_raw = [G.sample(s, T) for i in range(m)]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]
        end = time.time()
        sample_timing += end - start
        #print("{:.2f} % Calcloss                   ".format(progress + j/len(x)), end="\r")
        if filter21:
            #if len(uninfected) > 0 and len(neighbors) <= 6:
            if len(uninfected) > 0:
                r = len(uninfected)
                du = len(neighbors)
                def dNxu(ui):
                    return len(neighbors.intersection(set(G.graph.neighbors(ui))))
                def comp_val(u1, u2):
                    return (r + dNxu(u1) + dNxu(u2))/(du + G.graph.degree(u1) + G.graph.degree(u2) - G.graph.has_edge(s, u1) - G.graph.has_edge(s, u2) - G.graph.has_edge(u1, u2))
                psi = 1 - r/du - sum([
                        (sum([comp_val(u1, u2)
                                for u2 in infected.union(set(GN.neighbors(u1))) if not (u2 == s or u2 == u1)])
                            + r + dNxu(u1))
                        /(du + G.graph.degree(u1) - 2) for u1 in infected])/du

        for i in range(len(losses)):
            loss = losses[i]
            if(ordered[i]):
                samples_x_tmp = samples_x_list
                samples_c_tmp = samples_c_list
            else:
                samples_x_tmp = samples_x
                samples_c_tmp = samples_c
            start = time.time()
            mu_x = loss(G, x, samples_x_tmp, s)
            if mu_x < mu_mins[i] or amins[i] == -1:
                amins[i] = s
                mu_mins[i] = mu_x
            avg = sum([loss(G, yi, samples_x_tmp, s) >= mu_x for yi in samples_c_tmp])/m
            end = time.time()
            loss_timings[i] += end - start
            p += [avg]
            for a in range(len(alphas)):
                if avg > alphas[a]:
                    confidences[i*len(alphas) + a].append(s)

            #if filter22:

            #if filter23:

            if filter21:
                #if len(uninfected) > 0 and len(neighbors) <= 6:
                if len(uninfected) > 0:
                    if s == true_source:
                        samples_c_tmp_raw = [G.sample(s, T) for i in range(2000)]
                        samples_c_tmp = [c[0] for c in samples_c_tmp_raw]
                        p_val_list = [all([u not in y for u in uninfected]) for y in samples_c_tmp]
                    for a in range(len(alphas)):
                        if psi >= alphas[a]:
                            psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                            if psi_new > alphas[a]:
                                confidences21[i*len(alphas) + a].append(s)
                        else:
                            filtered[i*len(alphas) + a] = filtered[i*len(alphas) + a] + 1
                            psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                            if psi_new > alphas[a]:
                                bad_filtered[i*len(alphas) + a] = bad_filtered[i*len(alphas) + a] + 1
                    if len(neighbors) <= 6:
                        for a in range(len(alphas)):
                            if psi >= alphas[a]:
                                psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                                if psi_new > alphas[a]:
                                    confidences21d6[i*len(alphas) + a].append(s)
                else:
                    for a in range(len(alphas)):
                        if avg > alphas[a]:
                            confidences21[i*len(alphas) + a].append(s)

    return amins, confidences, confidences21, confidences21d6, confidences22, confidences23, filtered, bad_filtered, sample_timing, loss_timings, p_val_list

def Alg1_iso(G, x, m, losses, ordered, T, alphas, progress=0, filter21=True, filter22=False, filter23=False):
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences21 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences22 = [[] for _ in range(len(alphas) * (len(losses)))]
    confidences23 = [[] for _ in range(len(alphas) * (len(losses)))]
    sample_timing = 0
    loss_timings = [0 for _ in range(len(losses))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    p = []
    GN = G.graph.subgraph(list(x))
    filtered = [0 for _ in range(len(alphas) * len(losses))]
    bad_filtered = [0 for _ in range(len(alphas) * len(losses))]

    f_iso = first_order_iso(G, x)
    groups = add_groupless(x, iso_groups(f_iso))

    for g in groups:
        g_list = list(g)
        sp = g_list[0]
        neighbors = set(G.graph.neighbors(sp))
        uninfected = set([v for v in neighbors if not v in x])
        infected = set([v for v in neighbors if v in x])
        #print("{:.2f} % Sampling                   ".format(progress + j/len(x)), end="\r")
        start = time.time()
        samples_x_raw = [G.sample(sp, T) for i in range(m)]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c_raw = [G.sample(sp, T) for i in range(m)]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]
        end = time.time()
        sample_timing += end - start
        #print("{:.2f} % Calcloss                   ".format(progress + j/len(x)), end="\r")

        for s in g:

            if filter21:
                #if len(uninfected) > 0 and len(neighbors) <= 6:
                if len(uninfected) > 0:
                    r = len(uninfected)
                    du = len(neighbors)
                    def dNxu(ui):
                        return len(neighbors.intersection(set(G.graph.neighbors(ui))))
                    def comp_val(u1, u2):
                        return (r + dNxu(u1) + dNxu(u2))/(du + G.graph.degree(u1) + G.graph.degree(u2) - G.graph.has_edge(s, u1) - G.graph.has_edge(s, u2) - G.graph.has_edge(u1, u2))
                    psi = 1 - r/du - sum([
                            (sum([comp_val(u1, u2)
                                    for u2 in infected.union(set(GN.neighbors(u1))) if not (u2 == s or u2 == u1)])
                                + r + dNxu(u1))
                            /(du + G.graph.degree(u1) - 2) for u1 in infected])/du

            for i in range(len(losses)):
                loss = losses[i]
                if(ordered[i]):
                    samples_x_tmp = samples_x_list
                    samples_c_tmp = samples_c_list
                else:
                    samples_x_tmp = samples_x
                    samples_c_tmp = samples_c
                start = time.time()
                mu_x = loss(G, x, samples_x_tmp, s)
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                avg = sum([loss(G, yi, samples_x_tmp, s) >= mu_x for yi in samples_c_tmp])/m
                end = time.time()
                loss_timings[i] += end - start
                p += [avg]
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                #if filter22:

                #if filter23:

                if filter21:
                    #if len(uninfected) > 0 and len(neighbors) <= 6:
                    if len(uninfected) > 0:
                        for a in range(len(alphas)):
                            if psi >= alphas[a]:
                                psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                                if psi_new > alphas[a]:
                                    confidences21[i*len(alphas) + a].append(s)
                            else:
                                filtered[i*len(alphas) + a] = filtered[i*len(alphas) + a] + 1
                                psi_new = sum([all([u not in y for u in uninfected]) for y in samples_c])/m
                                if psi_new > alphas[a]:
                                    bad_filtered[i*len(alphas) + a] = bad_filtered[i*len(alphas) + a] + 1
                    else:
                        for a in range(len(alphas)):
                            if avg > alphas[a]:
                                confidences21[i*len(alphas) + a].append(s)

    return amins, confidences, confidences21, confidences22, confidences23, filtered, bad_filtered, sample_timing, loss_timings

def Alg1_compare(G, x, m, loss1, loss2, ordered1, ordered2, T, alphas, progress=0):
    num_losses = 4
    l1c = [[] for _ in range(len(alphas)*2)]
    l2c = [[] for _ in range(len(alphas)*2)]
    x_list = list(x)
    for j in range(len(x_list)):
        s = x_list[j]
        print("{:.2f} % Sampling                   ".format(progress + j/len(x)), end="\r")
        samples_x_raw = [G.sample(s, T) for i in range(m)]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c_raw = [G.sample(s, T) for i in range(m)]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]
        print("{:.2f} % Calcloss                   ".format(progress + j/len(x)), end="\r")

        if(ordered1):
            samples_x1 = samples_x_list
            samples_c1 = samples_c_list
        else:
            samples_x1 = samples_x
            samples_c1 = samples_c

        if(ordered2):
            samples_x2 = samples_x_list
            samples_c2 = samples_c_list
        else:
            samples_x2 = samples_x
            samples_c2 = samples_c
        mu_x1 = loss1(G, x, samples_x1, s)
        mu_x2 = loss2(G, x, samples_x2, s)
        avg1 = sum([loss1(G, yi, samples_x1, s) >= mu_x1 for yi in samples_c1])/m
        avg2 = sum([loss2(G, yi, samples_x2, s) >= mu_x2 for yi in samples_c2])/m
        for a in range(len(alphas)):
            if avg1 > alphas[a]:
                l1c[a].append(s)
            if avg2 > alphas[a]:
                l2c[a].append(s)

        for a in range(len(alphas)):
            if avg1 > alphas[a]/2:
                l1c[len(alphas) + a].append(s)
            if avg2 > alphas[a]/2:
                l2c[len(alphas) + a].append(s)

    min_c = [[] for _ in range(len(alphas))]
    int_c = [[] for _ in range(len(alphas))]

    for a in range(len(alphas)):
        c1 = l1c[len(alphas) + a]
        c2 = l2c[len(alphas) + a]
        if len(c1) < len(c2):
            min_c[a] = c1
        else:
            min_c[a] = c2

        int_c[a] = set(c1).intersection(set(c2))

    return l1c, l2c, min_c, int_c

def Alg1_final(G, x, m, losses, ordered, T, alphas):
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    max_p = [[-1, -1] for _ in range(len(losses))]
    for j in range(len(x_list)):
        s = x_list[j]

        samples_x_raw = [G.sample(s, T) for i in range(m)]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c_raw = [G.sample(s, T) for i in range(m)]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        for i in range(len(losses)):
            loss = losses[i]
            if(ordered[i]):
                samples_x_tmp = samples_x_list
                samples_c_tmp = samples_c_list
            else:
                samples_x_tmp = samples_x
                samples_c_tmp = samples_c
            mu_x = loss(G, x, samples_x_tmp, s)
            if mu_x < mu_mins[i] or amins[i] == -1:
                amins[i] = s
                mu_mins[i] = mu_x
            avg = sum([loss(G, yi, samples_x_tmp, s) >= mu_x for yi in samples_c_tmp])/m
            for a in range(len(alphas)):
                if avg > alphas[a]:
                    confidences[i*len(alphas) + a].append(s)

            if max_p[i][1] < avg:
                max_p[i] = [s, avg]

    return amins, confidences, max_p

def Alg1_filtering(G, x, m, losses, ordered, T, alphas, progress=0, degree1=True, iso=True):
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    max_p = [[-1, -1] for _ in range(len(losses))]

    f_iso = first_order_iso(G, x)
    groups = add_groupless(x, iso_groups(f_iso))
    if not iso:
        groups = set(x)

    d1_groups = list()
    d1_x_samples = {}
    d1_c_samples = {}
    d1_mapping = {}

    for g in groups:
        g_list = list(g)
        sp = g_list[0]
        neighbors = set(G.graph.neighbors(sp))
        if len(neighbors) == 1 and degree1:
            d1_groups.append(g)
            continue

        if any([G.graph.degree[n] == 1 for n in neighbors]):
            samples_x_raw = [G.sample_d1(sp, T) for i in range(m)]
            samples_c_raw = [G.sample_d1(sp, T) for i in range(m)]
            chosen = -1
            for n in neighbors:
                if G.graph.degree[n] == 1:
                    chosen = n
                    break
            for s in g:
                for n in G.graph.neighbors(s):
                    if G.graph.degree[n] == 1:
                        d1_x_samples[n] = samples_x_raw
                        d1_c_samples[n] = samples_c_raw
                        d1_mapping[n] = chosen
        else:
            samples_x_raw = [G.sample(sp, T) for i in range(m)]
            samples_c_raw = [G.sample(sp, T) for i in range(m)]

        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        for s in g:

            for i in range(len(losses)):
                loss = losses[i]
                if(ordered[i]):
                    samples_x_tmp = samples_x_list
                    samples_c_tmp = samples_c_list
                else:
                    samples_x_tmp = samples_x
                    samples_c_tmp = samples_c
                mu_x = loss(G, x, samples_x_tmp, s)
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                avg = sum([loss(G, yi, samples_x_tmp, s) >= mu_x for yi in samples_c_tmp])/m
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]

    for g in d1_groups:
        g_list = list(g)
        sp = d1_mapping[g_list[0]]

        samples_x_raw = d1_x_samples[sp]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_x_degrees = [xs[2] for xs in samples_x_raw]
        samples_c_raw = d1_c_samples[sp]
        samples_c = [xs[0] for xs in samples_c_raw]
        samples_c_list = [xs[1] for xs in samples_c_raw]
        samples_c_degrees = [xs[2] for xs in samples_c_raw]

        sample_x_ratios = list()
        for i in range(m):
            if not sp in samples_x[i]:
                sample_x_ratios.append(0)
            degree_list = samples_x_degrees[i]
            sample_list = samples_x_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    break
                #print(degree_list[k], sample_list[k+1], sp)
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_x_list[i] = sample_list
            sample_x_ratios.append(ratio)

        sample_c_ratios = list()
        for i in range(m):
            if not sp in samples_c[i]:
                sample_c_ratios.append(0)
            degree_list = samples_c_degrees[i]
            sample_list = samples_c_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_c_list[i] = sample_list
            sample_c_ratios.append(ratio)

        for s in g:

            for i in range(len(losses)):
                loss = losses[i]
                if(ordered[i]):
                    samples_x_tmp = samples_x_list
                    samples_c_tmp = samples_c_list
                else:
                    samples_x_tmp = samples_x
                    samples_c_tmp = samples_c
                mu_x = loss(G, x, samples_x_tmp, s, sample_x_ratios)
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                avg = sum([ratio*(loss(G, yi, samples_x_tmp, s, sample_x_ratios) >= mu_x) for yi, ratio in zip(samples_c_tmp, sample_c_ratios)])/m
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]

    return amins, confidences, max_p

def Alg1_filtering_intersect(G, x, m, losses, ordered, T, raw_alphas, progress=0, degree1=True, iso=True):
    alphas = raw_alphas.copy()
    alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    max_p = [[-1, -1] for _ in range(len(losses))]

    sample_timings = []
    d1_sample_timings = []
    loss_timings = [[] for _ in range(len(losses))]
    iso_saved = 0
    d1_saved = 0

    start = time.time()
    f_iso = first_order_iso(G, x)
    iso_timing = time.time() - start
    groups = add_groupless(x, iso_groups(f_iso))
    if not iso:
        groups = list([[x_i] for x_i in x])

    d1_groups = list()
    d1_x_samples = {}
    d1_c_samples = {}
    d1_mapping = {}

    for g in groups:
        g_list = list(g)
        sp = g_list[0]
        neighbors = set(G.graph.neighbors(sp))
        if len(neighbors) == 1 and degree1:
            d1_groups.append(g)
            continue

        start = time.time()

        if any([G.graph.degree[n] == 1 for n in neighbors]):
            samples_x_raw = [G.sample_d1(sp, T) for i in range(m)]
            samples_c_raw = [G.sample_d1(sp, T) for i in range(m)]
            end = time.time()
            d1_sample_timings += [end - start]
            chosen = -1
            for n in neighbors:
                if G.graph.degree[n] == 1:
                    chosen = n
                    break
            for s in g:
                for n in G.graph.neighbors(s):
                    if G.graph.degree[n] == 1:
                        d1_x_samples[n] = samples_x_raw
                        d1_c_samples[n] = samples_c_raw
                        d1_mapping[n] = chosen
        else:
            samples_x_raw = [G.sample(sp, T) for i in range(m)]
            samples_c_raw = [G.sample(sp, T) for i in range(m)]
            end = time.time()
            sample_timings += [end - start]

        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        iso_saved += len(g) - 1

        for i in range(len(losses)):
            loss = losses[i]
            if(ordered[i]):
                samples_x_tmp = samples_x_list
                samples_c_tmp = samples_c_list
            else:
                samples_x_tmp = samples_x
                samples_c_tmp = samples_c
            mu_x = loss(G, x, samples_x_tmp, sp)
            start = time.time()
            avg = sum([loss(G, yi, samples_x_tmp, sp) >= mu_x for yi in samples_c_tmp])/m
            end = time.time()
            loss_timings[i] += [end - start]
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]

    for g in d1_groups:
        g_list = list(g)
        sp = d1_mapping[g_list[0]]

        samples_x_raw = d1_x_samples[sp]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_x_degrees = [xs[2] for xs in samples_x_raw]
        samples_c_raw = d1_c_samples[sp]
        samples_c = [xs[0] for xs in samples_c_raw]
        samples_c_list = [xs[1] for xs in samples_c_raw]
        samples_c_degrees = [xs[2] for xs in samples_c_raw]

        d1_saved += len(g)

        sample_x_ratios = list()
        for i in range(m):
            if not sp in samples_x[i]:
                sample_x_ratios.append(0)
                continue
            degree_list = samples_x_degrees[i]
            sample_list = samples_x_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_x_list[i] = sample_list
            sample_x_ratios.append(ratio)

        sample_c_ratios = list()
        for i in range(m):
            if not sp in samples_c[i]:
                sample_c_ratios.append(0)
                continue
            degree_list = samples_c_degrees[i] # Get degree list for sample
            sample_list = samples_c_list[i] # Get ordered sample
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp: # Reach d1 node
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k]) # Eqn 7
            samples_c_list[i] = sample_list
            sample_c_ratios.append(ratio)


        for i in range(len(losses)):
            loss = losses[i]
            if(ordered[i]):
                samples_x_tmp = samples_x_list
                samples_c_tmp = samples_c_list
            else:
                samples_x_tmp = samples_x
                samples_c_tmp = samples_c
            mu_x = loss(G, x, samples_x_tmp, sp, sample_x_ratios)/T
            start = time.time()
            avg = sum([ratio*(loss(G, yi, samples_x_tmp, sp, sample_x_ratios)/T >= mu_x) for yi, ratio in zip(samples_c_tmp, sample_c_ratios)])/(T*m)
            end = time.time()
            loss_timings[i] += [end - start]
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]

    int_c = [[] for _ in range(len(alphas)//2)]

    for a in range(len(alphas)//2):
        c1 = confidences[len(alphas)//2 + a]
        c2 = confidences[3*len(alphas)//2 + a]

        int_c[a] = set(c1).intersection(set(c2))

    confidences = confidences[:len(alphas)//2] + confidences[len(alphas):3*len(alphas)//2] + int_c

    return amins, confidences, max_p, sample_timings, d1_sample_timings, loss_timings, iso_timing, iso_saved, d1_saved

def Alg1_test_source(G, x, s, m, losses, ordered, T, raw_alphas, progress=0):
    alphas = raw_alphas.copy()
    alphas.extend([a/2 for a in alphas])
    confidences = [False for _ in range(len(alphas) * (len(losses)))]
    confidences_T = [False for _ in range(len(alphas) * (len(losses)))]
    confidences_p = [False for _ in range(len(alphas) * (len(losses)))]
    confidences_d1 = [False for _ in range(len(alphas) * (len(losses)))]
    x_list = list(x)

    samples_x_raw = [G.sample(s, T) for i in range(m)]
    samples_c_raw = [G.sample(s, T) for i in range(m)]

    samples_x = [xs[0] for xs in samples_x_raw]
    samples_x_list = [xs[1] for xs in samples_x_raw]
    samples_c = [c[0] for c in samples_c_raw]
    samples_c_list = [c[1] for c in samples_c_raw]

    parent = list(G.graph.neighbors(s))[0]

    samples_x_raw_d1 = [G.sample_d1(parent, T) for i in range(m)]
    samples_c_raw_d1 = [G.sample_d1(parent, T) for i in range(m)]

    samples_x_d1 = [xs[0] for xs in samples_x_raw_d1]
    samples_x_list_d1 = [xs[1] for xs in samples_x_raw_d1]
    samples_x_degrees = [xs[2] for xs in samples_x_raw_d1]

    samples_c_d1 = [xs[0] for xs in samples_c_raw_d1]
    samples_c_list_d1 = [xs[1] for xs in samples_c_raw_d1]
    samples_c_degrees = [xs[2] for xs in samples_c_raw_d1]

    sample_x_ratios = list()
    for i in range(m):
        if not s in samples_x_d1[i]:
            sample_x_ratios.append(0)
            continue
        degree_list = samples_x_degrees[i]
        sample_list = samples_x_list_d1[i]
        ratio = 1
        for k in range(len(sample_list)):
            if sample_list[k+1] == s:
                sample_list.insert(0, sample_list.pop(k))
                ratio = ratio * degree_list[k]
                break
            ratio = ratio * 1/(1 - 1/degree_list[k])
        samples_x_list_d1[i] = sample_list
        sample_x_ratios.append(ratio)

    sample_c_ratios = list()
    for i in range(m):
        if not s in samples_c_d1[i]:
            sample_c_ratios.append(0)
            continue
        degree_list = samples_c_degrees[i] # Get degree list for sample
        sample_list = samples_c_list_d1[i] # Get ordered sample
        ratio = 1
        for k in range(len(sample_list)):
            if sample_list[k+1] == s: # Reach d1 node
                sample_list.insert(0, sample_list.pop(k))
                ratio = ratio * degree_list[k]
                break
            ratio = ratio * 1/(1 - 1/degree_list[k]) # Eqn 7
        samples_c_list_d1[i] = sample_list
        sample_c_ratios.append(ratio)

    for i in range(len(losses)):
        loss = losses[i]
        if(ordered[i]):
            samples_x_tmp = samples_x_list
            samples_c_tmp = samples_c_list
            samples_x_tmp_d1 = samples_x_list_d1
            samples_c_tmp_d1 = samples_c_list_d1
        else:
            samples_x_tmp = samples_x
            samples_c_tmp = samples_c
            samples_x_tmp_d1 = samples_x_d1
            samples_c_tmp_d1 = samples_c_d1
        mu_x = loss(G, x, samples_x_tmp, s)
        mu_x_d1 = loss(G, x, samples_x_tmp_d1, s, sample_x_ratios)/T
        avg = sum([loss(G, yi, samples_x_tmp, s) >= mu_x for yi in samples_c_tmp])/m
        avg_T = sum([(loss(G, yi, samples_x_tmp_d1, s, sample_x_ratios)/T >= mu_x_d1) for yi in samples_c_tmp])/m
        avg_p = sum([ratio*(loss(G, yi, samples_x_tmp, s) >= mu_x) for yi, ratio in zip(samples_c_tmp_d1, sample_c_ratios)])/(T*m)
        avg_d1 = sum([ratio*(loss(G, yi, samples_x_tmp_d1, s, sample_x_ratios)/T >= mu_x_d1) for yi, ratio in zip(samples_c_tmp_d1, sample_c_ratios)])/(T*m)
        for a in range(len(alphas)):
            if avg > alphas[a]:
                confidences[i*len(alphas) + a] = True
            if avg_T > alphas[a]:
                confidences_T[i*len(alphas) + a] = True
            if avg_p > alphas[a]:
                confidences_p[i*len(alphas) + a] = True
            if avg_d1 > alphas[a]:
                confidences_d1[i*len(alphas) + a] = True

    int_c = [False for _ in range(len(alphas)//2)]
    int_c_T = [False for _ in range(len(alphas)//2)]
    int_c_p = [False for _ in range(len(alphas)//2)]
    int_c_d1 = [False for _ in range(len(alphas)//2)]

    for a in range(len(alphas)//2):
        for conf, inter in zip([confidences, confidences_T, confidences_p, confidences_d1], [int_c, int_c_T, int_c_p, int_c_d1]):
            c1 = confidences[len(alphas)//2 + a]
            c2 = confidences[3*len(alphas)//2 + a]

            inter[a] = c1 and c2

    confidences = confidences[:len(alphas)//2] + confidences[len(alphas):3*len(alphas)//2] + int_c
    confidences_T = confidences_T[:len(alphas)//2] + confidences_T[len(alphas):3*len(alphas)//2] + int_c_T
    confidences_p = confidences_p[:len(alphas)//2] + confidences_p[len(alphas):3*len(alphas)//2] + int_c_p
    confidences_d1 = confidences_d1[:len(alphas)//2] + confidences_d1[len(alphas):3*len(alphas)//2] + int_c_d1

    return confidences, confidences_T, confidences_p, confidences_d1

def Alg1_faster_loss(G, x, m, losses, T, raw_alphas, degree1=True, iso=True):
    alphas = raw_alphas.copy()
    #alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    max_p = [[-1, -1] for _ in range(len(losses))]

    sample_timings = []
    d1_sample_timings = []
    loss_timings = [[] for _ in range(len(losses))]
    iso_saved = 0
    d1_saved = 0

    iso_timing = 0
    if iso:
        start = time.time()
        f_iso = first_order_iso(G, x)
        iso_timing = time.time() - start
        groups = add_groupless(x, iso_groups(f_iso))
        #groups = add_groupless(x, f_iso)
    if not iso:
        groups = list([[x_i] for x_i in x])

    d1_groups = list()
    d1_x_samples = {}
    d1_c_samples = {}
    d1_mapping = {}

    for g in groups:
        g_list = list(g)
        sp = g_list[0]
        neighbors = set(G.graph.neighbors(sp))
        if len(neighbors) == 1 and degree1:
            d1_groups.append(g)
            continue

        start = time.time()

        if any([G.graph.degree[n] == 1 for n in neighbors]):
            samples_x_raw = [G.sample_d1(sp, T) for i in range(m)]
            samples_c_raw = [G.sample_d1(sp, T) for i in range(m)]
            end = time.time()
            d1_sample_timings += [end - start]
            chosen = -1
            for n in neighbors:
                if G.graph.degree[n] == 1:
                    chosen = n
                    break
            for s in g:
                for n in G.graph.neighbors(s):
                    if G.graph.degree[n] == 1:
                        d1_x_samples[n] = samples_x_raw
                        d1_c_samples[n] = samples_c_raw
                        d1_mapping[n] = chosen
        else:
            samples_x_raw = [G.sample(sp, T) for i in range(m)]
            samples_c_raw = [G.sample(sp, T) for i in range(m)]
            end = time.time()
            sample_timings += [end - start]

        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        #process = psutil.Process(os.getpid())
        #print(process.memory_info().rss)
        #quit()

        iso_saved += len(g) - 1

        for i in range(len(losses)):
            loss = losses[i]
            start = time.time()
            samples_x_tmp = node_vals(loss, samples_x_list, T)
            mu_x = temporal_loss(x, samples_x_tmp)
            avg = sum([temporal_loss(yi, samples_x_tmp) >= mu_x for yi in samples_c_list])/m
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]
            end = time.time()
            loss_timings[i] += [end - start]

    for g in d1_groups:
        g_list = list(g)
        sp = d1_mapping[g_list[0]]

        samples_x_raw = d1_x_samples[sp]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_x_degrees = [xs[2] for xs in samples_x_raw]
        samples_c_raw = d1_c_samples[sp]
        samples_c = [xs[0] for xs in samples_c_raw]
        samples_c_list = [xs[1] for xs in samples_c_raw]
        samples_c_degrees = [xs[2] for xs in samples_c_raw]

        d1_saved += len(g)

        sample_x_ratios = list()
        for i in range(m):
            if not sp in samples_x[i]:
                sample_x_ratios.append(0)
                continue
            degree_list = samples_x_degrees[i]
            sample_list = samples_x_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_x_list[i] = sample_list
            sample_x_ratios.append(ratio)

        sample_c_ratios = list()
        for i in range(m):
            if not sp in samples_c[i]:
                sample_c_ratios.append(0)
                continue
            degree_list = samples_c_degrees[i] # Get degree list for sample
            sample_list = samples_c_list[i] # Get ordered sample
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp: # Reach d1 node
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k]) # Eqn 7
            samples_c_list[i] = sample_list
            sample_c_ratios.append(ratio)


        for i in range(len(losses)):
            loss = losses[i]
            start = time.time()
            samples_x_tmp = node_vals(loss, samples_x_list, T, sample_x_ratios)
            mu_x = temporal_loss(x, samples_x_tmp)/T
            avg = sum([ratio*(temporal_loss(yi, samples_x_tmp)/T >= mu_x) for yi, ratio in zip(samples_c_list, sample_c_ratios)])/(T*m)
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]
            end = time.time()
            loss_timings[i] += [end - start]

    return amins, confidences, max_p, sample_timings, d1_sample_timings, loss_timings, iso_timing, iso_saved, d1_saved

def Alg1_parallel(G, x, m, losses, T, raw_alphas, degree1=True, iso=True):
    alphas = raw_alphas.copy()
    #alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]

    f_iso = first_order_iso(G, x)
    groups = add_groupless(x, iso_groups(f_iso))
    if not iso:
        groups = list([[x_i] for x_i in x])

    d1_groups = list()
    d1_x_samples = {}
    d1_c_samples = {}
    d1_mapping = {}

    with Pool(20) as p:
        results = p.starmap(analyze_group, [(g, x, G, m, T, losses, alphas, node_vals, temporal_loss, degree1) for g in groups])
        p.close()
        p.join()
        for inclusion, d1, g, d1_x, d1_c in results:
            for i in range(len(losses)):
                for a in range(len(alphas)):
                    confidences[i*len(alphas) + a].extend(inclusion[i*len(alphas) + a])

            if d1 == 0:
                d1_groups.append(g)
            elif d1 == 2:
                neighbors = set(G.graph.neighbors(list(g)[0]))
                for n in neighbors:
                    if G.graph.degree[n] == 1:
                        chosen = n
                        break
                for s in g:
                    for n in G.graph.neighbors(s):
                        if G.graph.degree[n] == 1:
                            d1_x_samples[n] = d1_x
                            d1_c_samples[n] = d1_c
                            d1_mapping[n] = chosen

    for g in d1_groups:
        g_list = list(g)
        sp = d1_mapping[g_list[0]]

        samples_x_raw = d1_x_samples[sp]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_x_degrees = [xs[2] for xs in samples_x_raw]
        samples_c_raw = d1_c_samples[sp]
        samples_c = [xs[0] for xs in samples_c_raw]
        samples_c_list = [xs[1] for xs in samples_c_raw]
        samples_c_degrees = [xs[2] for xs in samples_c_raw]

        sample_x_ratios = list()
        for i in range(m):
            if not sp in samples_x[i]:
                sample_x_ratios.append(0)
                continue
            degree_list = samples_x_degrees[i]
            sample_list = samples_x_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_x_list[i] = sample_list
            sample_x_ratios.append(ratio)

        sample_c_ratios = list()
        for i in range(m):
            if not sp in samples_c[i]:
                sample_c_ratios.append(0)
                continue
            degree_list = samples_c_degrees[i] # Get degree list for sample
            sample_list = samples_c_list[i] # Get ordered sample
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp: # Reach d1 node
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k]) # Eqn 7
            samples_c_list[i] = sample_list
            sample_c_ratios.append(ratio)

        for i in range(len(losses)):
            loss = losses[i]
            samples_x_tmp = node_vals(loss, samples_x_list, T, sample_x_ratios)
            mu_x = temporal_loss(x, samples_x_tmp)/T
            avg = sum([ratio*(temporal_loss(yi, samples_x_tmp)/T >= mu_x) for yi, ratio in zip(samples_c_list, sample_c_ratios)])/(T*m)
            for s in g:
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

    return confidences

def Alg1_parallel_fast(G, x, m, losses, T, raw_alphas, degree1=True, iso=True):
    alphas = raw_alphas.copy()
    #alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]

    if iso:
        f_iso = first_order_iso(G, x)
        groups = add_groupless(x, iso_groups(f_iso))
        if degree1:
            groups = collapse_degree1(G, x, groups)
        else:
            groups = [(g, []) for g in groups]
    elif degree1:
        groups = list()
        for x_i in x:
            if G.graph.degree[x_i] == 1:
                continue
            d1_neighbors = list()
            for n in G.graph.neighbors(x_i):
                if n in x and G.graph.degree[n] == 1:
                    d1_neighbors += [n]
            groups.append(([x_i], d1_neighbors))
    else:
        groups = [([x_i], []) for x_i in x]

    with Pool(20) as p:
        results = p.starmap(analyze_group_d1, [(g[0], g[1], x, G, m, T, losses, alphas, node_vals, temporal_loss, degree1) for g in groups])
        p.close()
        p.join()
        for inclusion in results:
            for i in range(len(losses)):
                for a in range(len(alphas)):
                    confidences[i*len(alphas) + a].extend(inclusion[i*len(alphas) + a])

    return confidences

def Alg1_directed(G, x, m, losses, T, raw_alphas):
    alphas = raw_alphas.copy()
    #alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)

    groups = G.source_candidates(x)
    groups = [[xi] for xi in groups]

    for g in groups:
        g_list = list(g)
        sp = g_list[0]

        samples_x_raw = [G.sample(sp, T) for i in range(m)]
        samples_c_raw = [G.sample(sp, T) for i in range(m)]

        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        for i in range(len(losses)):
            loss = losses[i]
            samples_x_tmp = node_vals(loss, samples_x_list, T)
            mu_x = temporal_loss(x, samples_x_tmp)
            avg = sum([temporal_loss(yi, samples_x_tmp) >= mu_x for yi in samples_c_list])/m
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

    return amins, confidences

def Alg1_weighted(G, x, m, losses, T, raw_alphas):
    alphas = raw_alphas.copy()
    #alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)

    groups = [[xi] for xi in x]

    for g in groups:
        g_list = list(g)
        sp = g_list[0]

        samples_x_raw = [G.sample(sp, T) for i in range(m)]
        samples_c_raw = [G.sample(sp, T) for i in range(m)]

        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        for i in range(len(losses)):
            loss = losses[i]
            samples_x_tmp = node_vals(loss, samples_x_list, T)
            mu_x = temporal_loss(x, samples_x_tmp)
            avg = sum([temporal_loss(yi, samples_x_tmp) >= mu_x for yi in samples_c_list])/m
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

    return amins, confidences

def Alg1_faster_loss_check_source(G, x, source, m, losses, T, raw_alphas, degree1=True, iso=True):
    alphas = raw_alphas.copy()
    #alphas.extend([a/2 for a in alphas])
    confidences = [[] for _ in range(len(alphas) * (len(losses)))]
    amins = [-1 for _ in range(len(losses))]
    mu_mins = [-1 for _ in range(len(losses))]
    x_list = list(x)
    max_p = [[-1, -1] for _ in range(len(losses))]

    sample_timings = []
    d1_sample_timings = []
    loss_timings = [[] for _ in range(len(losses))]
    iso_saved = 0
    d1_saved = 0

    start = time.time()
    f_iso = first_order_iso(G, x)
    iso_timing = time.time() - start
    groups = add_groupless(x, iso_groups(f_iso))
    if not iso:
        groups = list([[x_i] for x_i in x])

    d1_groups = list()
    d1_x_samples = {}
    d1_c_samples = {}
    d1_mapping = {}

    for g in groups:
        if not source in g:
            continue
        g_list = list(g)
        sp = g_list[0]
        neighbors = set(G.graph.neighbors(sp))
        if len(neighbors) == 1 and degree1:
            d1_groups.append(g)
            continue

        start = time.time()

        if any([G.graph.degree[n] == 1 for n in neighbors]):
            samples_x_raw = [G.sample_d1(sp, T) for i in range(m)]
            samples_c_raw = [G.sample_d1(sp, T) for i in range(m)]
            end = time.time()
            d1_sample_timings += [end - start]
            chosen = -1
            for n in neighbors:
                if G.graph.degree[n] == 1:
                    chosen = n
                    break
            for s in g:
                for n in G.graph.neighbors(s):
                    if G.graph.degree[n] == 1:
                        d1_x_samples[n] = samples_x_raw
                        d1_c_samples[n] = samples_c_raw
                        d1_mapping[n] = chosen
        else:
            samples_x_raw = [G.sample(sp, T) for i in range(m)]
            samples_c_raw = [G.sample(sp, T) for i in range(m)]
            end = time.time()
            sample_timings += [end - start]

        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_c = [c[0] for c in samples_c_raw]
        samples_c_list = [c[1] for c in samples_c_raw]

        iso_saved += len(g) - 1

        for i in range(len(losses)):
            loss = losses[i]
            start = time.time()
            samples_x_tmp = node_vals(loss, samples_x_list, T)
            mu_x = temporal_loss(x, samples_x_tmp)
            avg = sum([temporal_loss(yi, samples_x_tmp) >= mu_x for yi in samples_c_list])/m
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]
            end = time.time()
            loss_timings[i] += [end - start]

    for g in d1_groups:
        g_list = list(g)
        sp = d1_mapping[g_list[0]]

        samples_x_raw = d1_x_samples[sp]
        samples_x = [xs[0] for xs in samples_x_raw]
        samples_x_list = [xs[1] for xs in samples_x_raw]
        samples_x_degrees = [xs[2] for xs in samples_x_raw]
        samples_c_raw = d1_c_samples[sp]
        samples_c = [xs[0] for xs in samples_c_raw]
        samples_c_list = [xs[1] for xs in samples_c_raw]
        samples_c_degrees = [xs[2] for xs in samples_c_raw]

        d1_saved += len(g)

        sample_x_ratios = list()
        for i in range(m):
            if not sp in samples_x[i]:
                sample_x_ratios.append(0)
                continue
            degree_list = samples_x_degrees[i]
            sample_list = samples_x_list[i]
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp:
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k])
            samples_x_list[i] = sample_list
            sample_x_ratios.append(ratio)

        sample_c_ratios = list()
        for i in range(m):
            if not sp in samples_c[i]:
                sample_c_ratios.append(0)
                continue
            degree_list = samples_c_degrees[i] # Get degree list for sample
            sample_list = samples_c_list[i] # Get ordered sample
            ratio = 1
            for k in range(len(sample_list)):
                if sample_list[k+1] == sp: # Reach d1 node
                    sample_list.insert(0, sample_list.pop(k))
                    ratio = ratio * degree_list[k]
                    break
                ratio = ratio * 1/(1 - 1/degree_list[k]) # Eqn 7
            samples_c_list[i] = sample_list
            sample_c_ratios.append(ratio)


        for i in range(len(losses)):
            loss = losses[i]
            start = time.time()
            samples_x_tmp = node_vals(loss, samples_x_list, T, sample_x_ratios)
            mu_x = temporal_loss(x, samples_x_tmp)/T
            avg = sum([ratio*(temporal_loss(yi, samples_x_tmp)/T >= mu_x) for yi, ratio in zip(samples_c_list, sample_c_ratios)])/(T*m)
            for s in g:
                if mu_x < mu_mins[i] or amins[i] == -1:
                    amins[i] = s
                    mu_mins[i] = mu_x
                for a in range(len(alphas)):
                    if avg > alphas[a]:
                        confidences[i*len(alphas) + a].append(s)

                if max_p[i][1] < avg:
                    max_p[i] = [s, avg]
            end = time.time()
            loss_timings[i] += [end - start]

    return amins, confidences, max_p, sample_timings, d1_sample_timings, loss_timings, iso_timing, iso_saved, d1_saved
