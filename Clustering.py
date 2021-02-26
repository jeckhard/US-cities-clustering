"""
Clustering method

Functions:
----------
clusterCenter : compute center of a set of cities
initialiseCenters : initialise a set of k locations
kMeans : assigns a set of cities to k clusters

"""
from Definitions import angularDistance, derivativesAngularDistance
import random
import matplotlib.pyplot as plt

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


def initialiseCenters(k,citySet,method="random"):
    """
    Initialise a set of k cluster centers

    :param k: int
    :param citySet: set with City elements
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
    Given a set of cities and an integer k, computes the optimal assignment of the cities into k clusters

    :param citySet: set with City elements
    :param k: int
        number of clusters, k > 0
    :param maxIterations: int
        maximum number of iterations
    :param emptySubstitute: False or str
        How to replace vanishing clusters
        False : No replacement (default)
        "random" : Replacement with random city location
        "overall" : Replacement with overall center
    :param initialise: str
        method parameter of initialiseCenters
    :param weights: str or custom function City -> int
        weigths parameter of clusterCenter
    :return: (list of sets with City elements,int)
        0 : assignment of cities to the clusters
        1 : error of the assignment
    """


    if k < 1 or type(k) != int:
        raise ValueError("k must be a positive integer!")

    overallCenter = clusterCenter(citySet)

    numIterations = 0
    centers = initialiseCenters(k,citySet,method=initialise)
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

    if not centers:
        raise ValueError("There are no centers!")

    error = 0
    for i in range(k):
        for city in newAssignment[i]:
            error += angularDistance(city.location,centers[i]) / len(citySet)

    return newAssignment,error


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


