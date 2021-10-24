import numpy as np
import scipy.io as sio
from mat4py import loadmat
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import networkx as nx
import graphs
from collections import Counter
from main import Alg1, L2, first_miss_loss, avg_deviation_time, avg_matching_time, loss_creator, rumor_center, Alg2, run_graph, m, K

def run_baidu():
    #adjacencies = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", skiprows=1, delimiter=',')
    #names = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    adjacencies = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", skiprows=1, delimiter=',')
    names = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    G = graphs.FromAdjacency(len(names), adjacencies)
    N = len(names)
    #cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    cc = pd.read_csv("data/BaiduMobility/InfectionStatus.csv", index_col=0)
    print(list(cc.columns))
    dates = list(range(0, 20, 5))
    dates = [9]
    thresh = list(range(1, 10, 2))
    thresh = [1]
    s = np.where(names == "Wuhan4201")[0][0]
    source = s

    los = [loss_creator(L2), first_miss_loss, avg_deviation_time, avg_matching_time]
    lstring = ["L2", "TTD", "ADT", "AMT"]
    ordered = [False, True, True, True]
    alpha = [0.1, 0.2]
    alps = alpha

    a = [0 for i in range(len(los))]
    ahops = [[] for i in range(len(los))]
    b = 0
    bhops = []
    #r = 0
    #rhops = []
    c = [0 for i in range(len(alps) * len(los))]
    avg = [0 for i in range(len(alps) * len(los))]

    tested = 0

    for d in dates:
        for t in thresh:
            x = set()
            for i in range(N):
                if cc.iloc[i, d] >= t:
                    x.add(i)
            if len(x) <= 1:
                continue
            if d == 18:
                continue
            print()
            print(len(x))
            print()
            tested += 1
            #center, centrality, rs, rcentrality, nodes = rumor_center(G.graph, x)
            estimates, Cs, C21, C22, C23, filt, bfilt, st, lts = Alg1(G, x, m, los, ordered, len(x), alpha, progress=0, filter21=False)
            dc = Alg2(G, x)
            size = len(x)
            for e in range(len(estimates)):
                if estimates[e] == source:
                    a[e] += 1
                ahops[e] += [G.dist(estimates[e], source)]
            if dc == source:
                b += 1
            bhops += [G.dist(d, source)]
            #if rs == source:
            #    r += 1
            #rhops += [G.dist(rs, source)]
            for conf in range(len(Cs)):
                if source in Cs[conf]:
                    c[conf] += 1
                avg[conf] += len(Cs[conf])

    for j in range(len(los)):
        print("\ta {}: {}".format(lstring[j], a[j]/tested))
        print("\tahops mean {}: {}".format(lstring[j], np.mean(ahops[j])))
        print("\tahops {}: {}".format(lstring[j], Counter(ahops[j])))
    print("\tb: {}".format(b/K))
    print("\tbhops mean: {}".format(np.mean(bhops)))
    print("\tbhops: {}".format(Counter(bhops)))
    #print("\trs: {}".format(r/K))
    #print("\trhops mean: {}".format(np.mean(rhops)))
    #print("\trhops: {}".format(Counter(rhops)))
    for i in range(len(los)):
        for j in range(len(alps)):
            print("\tc {} {}: {}".format(lstring[i], alps[j], c[i*len(alps) + j]/tested))
            print("\tavg confidence set size {} {}: {}".format(lstring[i], alps[j], avg[i*len(alps) + j]/tested))

def graph_baidu():
    #adjacencies = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", skiprows=1, delimiter=',')
    #names = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    adjacencies = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", skiprows=1, delimiter=',')
    names = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    G = graphs.FromAdjacency(len(names), adjacencies)
    N = len(names)
    #cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    cc = pd.read_csv("data/BaiduMobility/InfectionStatus.csv", index_col=0)
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
            estimates, Cs, C21, C22, C23, filt, bfilt = Alg1(G, x, m, los, ordered, len(x), alpha, progress=0, filter21=False)
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

def use_baidu():
    #adjacencies = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", skiprows=1, delimiter=',')
    #names = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    adjacencies = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", skiprows=1, delimiter=',')
    names = np.loadtxt("data/BaiduMobility/NewBaiduAdjacency.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    G = graphs.FromAdjacency(len(names), adjacencies)
    N = len(names)

    for T in [10, 15, 20, 25, 30]:
        print("T: ", T)
        run_graph(G, T, "Baidu")
    #cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    s = np.where(names == "Wuhan4201")[0][0]

def run_facebook():
    for fname in listdir():
        adj = loadmat(fname)
        print(fname)
        print(adj)
        quit()

def run_baidu2():
    adjacencies = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", skiprows=1, delimiter=',')
    names = np.loadtxt("data/BaiduMobility/AdjacencyMatrix-degree96.NoNames.csv", max_rows=1, delimiter=',', dtype=str)
    G = graphs.FromAdjacency(len(names), adjacencies)
    N = len(names)
    cc = pd.read_csv("data/BaiduMobility/ConfirmedCases.csv", index_col=0)
    d = 10
    t = 10
    s = np.where(names == "Wuhan4201")[0][0]
    x = set()
    for i in range(N):
        if cc.iloc[i, d] >= t:
            x.add(i)
    size = len(x)
    estimates, Cs, C21, C22, C23, filt, bfilt = Alg1(G, x, m, [loss_creator(L2)], [False], len(x), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], progress=0, filter21=False)
    print()
    print()
    print(list(cc.iloc[:,d]))
    print(len(x))
    for c in Cs:
        print(len(c)/len(x))
        print([names[i] for i in c])
        print(s in c)

if __name__ == "__main__":
    #use_baidu()
    run_baidu()
    #run_facebook()
