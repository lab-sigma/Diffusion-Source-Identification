import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sys

import pickle

import diffusion_source.graphs as graphs
from diffusion_source.infection_model import FixedTSI_Weighted, FixedTSI
from diffusion_source.discrepancies import L2_h, ADiT_h, ADT_h

def run_baidu_single(arg=0):

    truncs = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dates = list(range(1, 21, 1))
    threshes = [1, 2, 5, 10]
    print(len(truncs)*len(dates)*len(threshes))

    trunc = truncs[(int)(arg % len(truncs))]
    date = dates[(int)((arg / len(truncs)) % len(dates))]
    thresh = threshes[(int)(arg / (len(truncs) * len(dates)))]


    df = pd.read_csv("data/BaiduMobility/TrafficVol.csv", sep=',', index_col=0)
    mat = df.to_numpy()
    mat = np.nan_to_num(df)

    mat[abs(mat) < trunc] = 0.0

    GD = graphs.DirectedFromAdjacency(mat)
    G = graphs.FromAdjacency(mat)

    cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    #cc = pd.read_csv("data/BaiduMobility/InfectionStatus.csv", index_col=0)
    names = np.array(list(df.columns))
    #thresh = [10]
    s = np.where(names == 'Wuhan')[0][0]
    source = s

    losses = [L2_h, ADiT_h, ADT_h]

    x = set()
    for i in range(len(G.graph)):
        if cc.iloc[i, date] >= thresh:
            x.add(i)
    if len(x) <= 1:
        return

    IW = FixedTSI_Weighted(GD, losses, T=len(x)-1, m_l=2000, m_p=2000)
    I = FixedTSI(G, losses, T=len(x)-1, m_l=2000, m_p=2000, d1=False, iso=False)

    results = IW.p_values(x)
    pickle.dump(results, open("results/baidu/weighted_2000_{}.p".format(arg), "wb"))

    results = I.p_values(x)
    pickle.dump(results, open("results/baidu/unweighted_2000_{}.p".format(arg), "wb"))

def run_baidu():
    df = pd.read_csv("data/BaiduMobility/TrafficVol.csv", sep=',', index_col=0)
    mat = df.to_numpy()
    mat = np.nan_to_num(df)

    trunc = 1
    mat[abs(mat) < trunc] = 0.0

    GD = graphs.DirectedFromAdjacency(mat)
    G = graphs.FromAdjacency(mat)

    cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    #cc = pd.read_csv("data/BaiduMobility/InfectionStatus.csv", index_col=0)
    names = np.array(list(df.columns))
    dates = list(range(5, 20, 5))
    threshes = list(range(1, 10, 2))
    #thresh = [10]
    s = np.where(names == 'Wuhan')[0][0]
    source = s

    losses = [L2_h, ADiT_h]

    unweighted_results = {}
    weighted_results = {}

    for t in thresh:
        for d in dates:
            x = set()
            for i in range(len(G.graph)):
                if cc.iloc[i, d] >= t:
                    x.add(i)
            if len(x) <= 1:
                continue

            IW = FixedTSI_Weighted(GD, losses, T=len(x)-1, m_l=2000, m_p=2000)
            I = FixedTSI(G, losses, T=len(x)-1, m_l=2000, m_p=2000, d1=False, iso=False)

            results = I.p_values(x)

            unweighted_results[(t, d)] = results

            results = IW.p_values(x)

            weighted_results[(t, d)] = results

    pickle.dump(unweighted_results, open("results/baidu/unweighted_2000_{}.p".format(arg), "wb"))
    pickle.dump(weighted_results, open("results/baidu/weighted_2000_{}.p".format(arg), "wb"))

def graph_baidu():
    #adjacencies = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", skiprows=1, delimiter=',')
    #names = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    adjacencies = np.loadtxt("data/BaiduMobility/AdjacencyMatrix.NoNames.csv", skiprows=1, delimiter=',')
    names = np.loadtxt("data/BaiduMobility/AdjacencyMatrix.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    #adjacencies = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", skiprows=1, delimiter=',')
    #names = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    G = graphs.FromDirectedAdjacency(len(names), adjacencies)
    N = len(names)
    cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    #cc = pd.read_csv("data/BaiduMobility/InfectionStatus.csv", index_col=0)
    print(cc)
    dates = list(range(0, 25))
    #dates = [20]
    thresh = [1]
    s = np.where(names == "Wuhan4201")[0][0]
    pointr = []
    pointb = []
    rsr = []
    rsb = []
    distr = []
    distb = []
    C01r = []
    C01b = []
    C02r = []
    C02b = []
    overall = []
    colors = []
    x = set()
    for i in range(N):
        if cc.iloc[i, dates[0]] >= thresh[0]:
            x.add(i)
    for n in G.graph.nodes():
        if n in x:
            if n == s:
                colors += [(0.0, 1.0, 0.0)]
            else:
                colors += [(1.0, 0.0, 0.0)]
        else:
            colors += [(0.5, 0.5, 0.5)]
    nx.draw_networkx(G.graph, with_labels=False, node_size=25, width=0.25, node_list=list(G.graph.nodes()), node_color=colors)
    plt.show()

    los = [loss_creator(L2), first_miss_loss, avg_devation_time, avg_matching_time]
    lstring = ["L2", "TTD", "ADT", "AMT"]
    ordered = [False, True, True, True]
    alpha = [0.1, 0.2]

    alps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    los = [L2_h, ADT_h, ADiT_h]
    lstring = ["L2", "ADT", "ADiT"]
    alps = [0.1, 0.2]

    for d in dates:
        for t in thresh:
            x = set()
            for i in range(N):
                if cc.iloc[i, d] >= t:
                    x.add(i)
            if len(x) == 1:
                continue
            if d == 18:
                continue
            print()
            print()
            print(d)
            center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
            #estimates, Cs, C21, C22, C23, filt, bfilt = Alg1(G, x, m, los, ordered, len(x), alpha, progress=0, filter21=False)
            estimates, Cs, max_p, st, d1st, lt, it, iso_s, d1_s = Alg1_faster_loss(G, x, m, los, len(x), alpha, False, False)
            dc = Alg2(G, x)
            size = len(x)
            if estimates[0] == s:
                pointr.append((d, t, size))
            else:
                pointb.append((d, t, size))
            if rs == s:
                rsr.append((d, t, size))
            else:
                rsb.append((d, t, size))
            if dc == s:
                distr.append((d, t, size))
            else:
                distb.append((d, t, size))
            if s in Cs[0]:
                C01r.append((d, t, size, len(Cs[0])))
            else:
                C01b.append((d, t, size, len(Cs[0])))
            if s in Cs[1]:
                C02r.append((d, t, size, len(Cs[1])))
            else:
                C02b.append((d, t, size, len(Cs[1])))
            overall.append((d, t, size))

    plt.scatter([p[0] for p in pointr], [p[1] for p in pointr], c="Red")
    plt.scatter([p[0] for p in pointb], [p[1] for p in pointb], c="Blue")
    plt.xlabel("Date")
    plt.ylabel("Threshold")
    plt.show()
    plt.scatter([p[0] for p in C01r], [p[1] for p in C01r], c="Red")
    plt.scatter([p[0] for p in C01b], [p[1] for p in C01b], c="Blue")
    plt.xlabel("Date")
    plt.ylabel("Threshold")
    plt.show()
    plt.scatter([p[0] for p in C02r], [p[1] for p in C02r], c="Red")
    plt.scatter([p[0] for p in C02b], [p[1] for p in C02b], c="Blue")
    plt.xlabel("Date")
    plt.ylabel("Threshold")
    plt.show()
    print(overall)
    print("Point Success: ", len(pointr)/(len(pointr) + len(pointb)))
    print("RS Success: ", len(rsr)/(len(rsr) + len(rsb)))
    print("Dist Success: ", len(distr)/(len(distb) + len(distr)))
    print("C 0.1 Success: ", len(C01r)/(len(C01r) + len(C01b)))
    print("C 0.2 Success: ", len(C02r)/(len(C02r) + len(C02b)))
    print("pointr: ", pointr)
    print("pointb: ", pointb)
    print("C01r: ", C01r)
    print("C01b: ", C01b)
    print("C02r: ", C02r)
    print("C02b: ", C02b)
    print(rsr)
    print(distr)
    print("Average C 0.1 Coverage: ", np.mean([c[3]/c[2] for c in C01r] + [c[3]/c[2] for c in C01b]))
    print("Average C 0.2 Coverage: ", np.mean([c[3]/c[2] for c in C02r] + [c[3]/c[2] for c in C02b]))


if __name__ == "__main__":
    arg = (int(sys.argv[1]) - 1)
    run_baidu_single(arg)
