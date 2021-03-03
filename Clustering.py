"""
Clustering method

Functions:
----------
clusterCenter : compute center of a set of cities
initialiseCenters : initialise a set of k locations
kMeans : assigns a set of cities to k clusters
showClustering : makes a scatter plot of a city assignment into clusters

"""
from Definitions import angularDistance, derivativesAngularDistance, Gaussian
import random
import matplotlib.pyplot as plt
import numpy as np


def clusterCenter(citySet,weights="uniform",error=10**(-3),learningRate=1):
    """
    Compute the center of the citySet cluster

    :param citySet: set with City elements
    :param weights: str or custom function City -> int
        "uniform" : (lambda city : 1) (default)
        "population" : (lambda city : city.population)
    :param error: float (default = 10**(-3))
    :param learningRate: float (default = 1)
    :return: (float,float)
        location of the cluster center with given error
    """
    numCities = len(citySet)

    if numCities == 0:
        return None
    if numCities == 1:
        return list(citySet)[0].location

    if weights == "uniform":
        weightFunction = lambda city: 1
    elif weights == "population":
        weightFunction = lambda city: city.population
    else:
        weightFunction = weights

    totalWeights = sum(weightFunction(city) for city in citySet)

    weightedLocations = {}
    for city in citySet:
        weightedLocations[city.location] = weightFunction(city)/totalWeights

    center = [0, 0]
    for location in weightedLocations:
        center[0] += location[0] * weightedLocations[location]
        center[1] += location[1] * weightedLocations[location]
    derivatives = float("infinity"), float("infinity")
    while abs(derivatives[0]) > error or abs(derivatives[1]) > error:
        derivatives = [0, 0]
        for location in weightedLocations:
            derivatives[0] += angularDistance(location, center) * derivativesAngularDistance(location, center)[0] * weightedLocations[location]
            derivatives[1] += angularDistance(location, center) * derivativesAngularDistance(location, center)[1] * weightedLocations[location]

        center = [center[0] - learningRate * derivatives[0], center[1] - learningRate * derivatives[1]]

    return round(center[0], 3), round(center[1], 3)


def initialiseCenters(citySet,k,method="random"):
    """
    Initialise a set of k cluster centers

    :param citySet: set with City elements
    :param k: int
    :param method: str
        "random" : cluster centers are randomly selected (default)
        "population" : cluster centers are k most populous cities
    :return: list of locations (float,float)
    """
    if method == "random":
        locations = [city.location for city in citySet]
        random.shuffle(locations)
        return locations[:k]

    if method == "population":
        locations = sorted([(city.location,city.population) for city in citySet], key=lambda x: x[1],reverse=True)
        return locations[:k]



def kMeans(citySet,k,maxIterations=200,emptySubstitute=False,initialise="random",weights="uniform"):
    """
    Given a set of cities and an integer k, computes a locally optimal assignment of the cities into k clusters

    :param citySet: set with City elements
    :param k: int
        number of clusters, k > 0
    :param maxIterations: int
        maximum number of iterations, default=200
    :param emptySubstitute: False or str
        How to replace vanishing clusters
        False : No replacement (default)
        "random" : Replacement with random city location
        "overall" : Replacement with overall center
    :param initialise: str
        method parameter of initialiseCenters, default="random"
    :param weights: str or custom function City -> int
        weigths parameter of clusterCenter, default="uniform"
    :return: list of sets with City elements
        assignment of cities to the clusters
    """


    if k < 1 or type(k) != int:
        raise ValueError("k must be a positive integer!")

    overallCenter = clusterCenter(citySet)

    numIterations = 0
    centers = initialiseCenters(citySet,k,method=initialise)
    oldAssignment = None
    newAssignment = None

    while numIterations < maxIterations:

        newAssignment = []
        for i in range(k):
            newAssignment.append(set())
        for city in citySet:
            minDistance = float("infinity")
            newCluster = None
            for i in range(k):
                if centers[i]:
                    distance = angularDistance(city.location, centers[i])
                    if distance < minDistance:
                        minDistance, newCluster = distance, i
            newAssignment[newCluster].add(city)

        if newAssignment == oldAssignment:
            break
        numIterations += 1
        centers = [clusterCenter(cluster,weights=weights) for cluster in newAssignment]

        if emptySubstitute:
            if emptySubstitute == "overall":
                centers = list(map(lambda x: x if x else overallCenter,centers))
            if emptySubstitute == "random":
                centers = list(map(lambda x: x if x else random.choice(citySet).location,centers))


        oldAssignment = newAssignment


    return newAssignment


def initaliseGauss(citySet,k,method="random"):
    """
    Initialise a set of k cluster centers and a covariance of a standard distribution

    :param citySet: set with City elements
    :param k: int
        number of clusters, k > 0
    :param method: str
        "random" : cluster centers are randomly selected (default)
        "population" : cluster centers are k most populous cities
    :return: (np.array, np.array)
        0 : array of k means of length two
        1 : array of k covariances of length two x two
    """

    import statistics

    north_var = statistics.variance({city.north for city in citySet})
    west_var = statistics.variance({city.west for city in citySet})

    means = initialiseCenters(citySet,k,method=method)
    cov = np.array([[north_var/k,0],[0,west_var/k]]) * k
    covariances = []
    for _ in range(k):
        covariances.append(cov)

    return np.array(means), covariances


def EMGaussian(citySet, means, covariances, fullCov="full"):
    """
    Computes the probabilities of a set of cities to be in either of k clusters
    Updates the means and covariances of the clusters

    Add description
    
    :param citySet: set with City elements
    :param means: list of k locations of length two
        centers of the standard distributions
    :param covariances: list of k matrices of length two x two
        covariances of the standard distributions
    :param fullCov: str
        full : covariances are generic (default)
        diag : covariances are diagonal
        single : covariances are multiples of the identity
    :return: (dict, list, list)
        0 : key-value pairs of the form City : list of floats of length k
            probabilities of the cities to be in each of the clusters
        1 : list of k locations
            updated means of the clusters
        2 : list of k covariances
            updated covariances of the clusters
    """

    k = len(means)


    newProbs = {}
    for city in citySet:
        loc = [city.north,city.west]
        probs = []
        for cluster in range(k):
            probs.append(Gaussian(loc,means[cluster],cov=covariances[cluster]))
        sumProbs = sum(probs)
        newProbs[str(city)] = list(map(lambda x: x/sumProbs,probs))

    newNormalisers = []
    for cluster in range(k):
        norm = 0
        for city in citySet:
            norm += newProbs[str(city)][cluster]
        newNormalisers.append(norm)

    newMeans = []
    for cluster in range(k):
        meanN = 0
        meanW = 0
        for city in citySet:
            meanN += newProbs[str(city)][cluster] * city.north
            meanW += newProbs[str(city)][cluster] * city.west
        newMeans.append([meanN / newNormalisers[cluster], meanW / newNormalisers[cluster]])

    newCovariances = []
    for cluster in range(k):
        covNN = 0
        covWW = 0
        covNW = 0
        for city in citySet:
            covNN += newProbs[str(city)][cluster] * (city.north - newMeans[cluster][0]) ** 2
            covWW += newProbs[str(city)][cluster] * (city.west - newMeans[cluster][1]) ** 2
            if fullCov == "full":
                covNW += newProbs[str(city)][cluster] * (city.north - newMeans[cluster][0]) * (city.west - newMeans[cluster][1])
            elif fullCov == "single":
                covNN, covWW = (covNN + covWW) / 2, (covNN + covWW) / 2
            elif fullCov == "diag":
                pass
            else:
                raise ValueError("No correct variance input!")
        newCovariances.append([[covNN / newNormalisers[cluster], covNW / newNormalisers[cluster]],
                               [covNW / newNormalisers[cluster], covWW / newNormalisers[cluster]]])

    return newProbs, newMeans, newCovariances


def BayesianGauss(citySet,k,numIter = 100,initialise="random",fullCov="full"):
    """

    :param citySet: set with City elements
    :param k: int
        number of clusters, k > 0
    :param initialise: str
        "random" : cluster centers are randomly selected (default)
        "population" : cluster centers are k most populous cities
    :param numIter: int
        number of iterations, default=100
    :param fullCov: str
        full : covariances are generic (default)
        diag : covariances are diagonal
        single : covariances are multiples of the identity
    :return: list of sets with City elements
        assignment of cities to the clusters
    """

    means, cov = initaliseGauss(citySet,k,method=initialise)
    probs = None

    for _ in range(numIter):
        probs, means, cov = EMGaussian(citySet,means,cov,fullCov)

    assignment=[]
    for cluster in range(k):
        assignment.append(set())
    for city in citySet:
        cluster = probs[str(city)].index(max(probs[str(city)]))
        assignment[cluster].add(city)

    return assignment


def error(assignment, centerWeights = "uniform"):

    """
    Computes the error of a given assignment

    :param assignment: list of k sets with City elements
    :param centerWeights: str or custom function City -> int
        "uniform" : (lambda city : 1) (default)
        "population" : (lambda city : city.population)
    :return: float
        error of the assignment
    """


    k = len(assignment)
    numCities = sum(len(assignment[cluster]) for cluster in range(k))

    centers = []
    for cluster in range(k):
        centers.append(clusterCenter(assignment[cluster],weights=centerWeights))

    error = 0
    for cluster in range(k):
        for city in assignment[cluster]:
            error += angularDistance(city.location, centers[cluster]) / numCities

    return error

def showClustering(assignment, colours=None):
    """
    Makes a scatter plot of a city assignment into clusters

    :param assignment: list of sets with City elements
        city assignment to be shown
    :param colours: list of colours
        default is a set of 15 standard colours
    :return: None
    """

    k = len(assignment)

    if not colours:
        colours = ["black","red","brown","darkorange","grey","olive","yellow","lime","turquoise","cyan","dodgerblue","midnightblue","purple","fuchsia","crimson"]
    else:
        colours = colours

    if len(colours) < k:
        raise ValueError("Not enough colours specified!")


    for i in range(k):
        north = [city.north for city in assignment[i]]
        west = [-city.west for city in assignment[i]]
        plt.scatter(west,north,c=colours[i])

    plt.show()


