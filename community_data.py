import numpy as np
import pandas as pd
import networkx as nx
import graphs
import pickle
import sys
from multiprocessing import Pool
from main import final_results_file
import sys
from os import path
import os, psutil

argument = int(sys.argv[1]) - 1
#networks = pickle.load(open("CommunityFitNet/Benchmark_updated/CommunityFitNet_missing.pickle", "rb"))
#networks = pickle.load(open("CommunityFitNet/Benchmark_updated/CommunityFitNet_updated.pickle", "rb"))
networks = pickle.load(open("CommunityFitNet/Benchmark_updated/CommunityFitNet_timing.pickle", "rb"))

index = argument % networks.shape[0]
run_index = argument // networks.shape[0]

edgelists = networks['edges_id']
network_titles = networks['title']

finished = 0
unfinished = 0


#for i in range(len(network_titles)):
#    G = graphs.PyEdgeList(edgelists.iloc[i])
#    if len(G.graph) < 30:
#        continue
#    if not path.exists("results/community/{}{}.txt".format(network_titles.iloc[i], i)):
#        print(len(G.graph))

#def output_graph(edges, title, i):
#    G = graphs.PyEdgeList(edges)
#
#    if len(G.graph) < 30:
#        return
#
#    result_file = open("results/community/{}{}.txt".format(title, i), 'w')
#    final_results_file(G, title, result_file)

#with Pool(4) as p:
#    p.starmap(output_graph, [(edgelists.iloc[i], network_titles.iloc[i], i) for i in range(len(network_titles)) if index*4 <= i and min(i < index*4 + 4, len(network_titles))])

#missing = []

#for i in range(networks.shape[0]):
#    edges = edgelists.iloc[i]

#    G = graphs.PyEdgeList(edges)

#    if len(G.graph) >= 200:
#        if path.exists("results/community_finished/{}{}.txt".format(network_titles.iloc[i], i)):
#            missing += [i]

#missing_networks = networks.loc[missing,]

#pickle.dump(missing_networks, open("CommunityFitNet/Benchmark_updated/CommunityFitNet_timing.pickle", "wb"))

edges = edgelists.iloc[index]

G = graphs.PyEdgeList(edges)

del networks

#process = psutil.Process(os.getpid())
#print(process.memory_info().rss)

result_file = open("results/community_finished_main/{}{}_{}.txt".format(network_titles.iloc[index], index, run_index), 'a')
final_results_file(G, network_titles.iloc[index], result_file)

#if len(G.graph) >= 200:
#    if not path.exists("results/community/{}{}.txt".format(network_titles.iloc[index], index)):
#        result_file = open("results/community/{}{}.txt".format(network_titles.iloc[index], index), 'w')
#        final_results_file(G, network_titles.iloc[index], result_file)
