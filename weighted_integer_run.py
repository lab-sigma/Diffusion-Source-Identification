import sys

import graphs
from infection_model import load_model

files = [
    "data/GlobalAirportTraffic/AirportFlightTraffic.txt",
    "data/StatisticianCitation/TotalCite.txt",
    "data/NorthAmericaHiring/BSchoolHiring.txt",
    "data/NorthAmericaHiring/ComputerScienceHiring.txt",
    "data/NorthAmericaHiring/HistoryHiring.txt",
    "data/NorthAmericaHiring/StatisticsHiring.txt"
]

names = [
    "AFT",
    "SatisticianCitations",
    "BSchoolHiring",
    "ComputerScienceHiring",
    "HistoryHiring",
    "StatisticsHiring"
]

arg = (int(sys.argv[1]) - 1)
index = arg % len(names)

f = files[index]
name = names[index]

I, s, x = load_model(name)

I.p_values(x, meta=(name, x, s))
print(I.results)
I.store_results("results/test/{}_{}.p".format(name, arg+1))
