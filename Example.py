from Definitions import City,citySubset,createDistances,readInStates, readInCities, State
from Clustering import clusterCenter,initialiseCenters, kMeans, showClustering
import Definitions
import random


readInStates("Population.csv")
readInStates("Area.csv")
readInCities("Cities.csv")

cities = citySubset(City.citySet,minPopulation=200000)
assignment = kMeans(cities,5)[0]
showClustering(assignment)
