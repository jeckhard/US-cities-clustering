"""
Definitions module

Classes:
---------
State : A class representing a US state
City : A class representing a US city


Functions:
----------
readInStates: Reads in states and its properties
readInCities: Reads in cities and its properties
citySubset : Returns subset of cities in population interval and in given states
angularDistance : Computes angular distance between two locations
derivativesAngularDistance : Computes the derivatives of the angular distance at two locations
createDistances : Creates a dictionary with all distances between given cities
"""

import csv
import math
from math import sin,cos,sqrt

class State:
    """
    A class representing a US state

    Attributes:
    ------------
    name : str
        name of the state
    population : int
        population of the state
    area : int
        area of the state, recommended unit km^2
    region : str
        region of the string
    capital : City
        capital city of the state or None if unknown
    abbreviation : str
        abbreviation of the state, represented by two capital letters
    cities : set
        set of all known cities in the state


    ClassAttributes:
    ------------
    stateSet : set
        the set of all known states
    nameToRef : dict
        key-value pairs of the form str(self) : self


    Methods:
    ------------
    __str__ : str
        returns the name of the state
    """

    stateSet = set()
    nameToRef = {}

    def __init__(self, name, population=0, area=0, region=None, capital=None, abbreviation=None):
        """
        Initialise a State instance

        :param name: str
            name of the state
        :param population: int
            population of the state, default=0
        :param area: int
            area of the state, default=0
        :param region: str
            region of the state, default=None
        :param capital: City
            capital of the state, default=None
        :param abbreviation: str
            abbreviation of the state, represented by two capital letters
        """

        if not abbreviation:
            abbreviation = name

        self.name = name
        self.population = population
        self.area = area
        self.region = region
        self.capital = capital
        self.abbreviation = abbreviation
        self.cities = set()

        State.stateSet.add(self)
        State.nameToRef[self.name] = self

    def __str__(self):
        """
        Representation of a State instance

        :return: str
            name of the state
        """
        return self.name


class City:
    """
        A class representing a US city

        Attributes:
        ------------
        name : str
            name of the city
        state : State
            state the city is in, default=None
        population : int
            population of the city
        capital : bool
            whether the city is the capital of its state
        north : float
            location of the city in northern direction, measured in degree
        west : float
            location of the city in western direction, measured in degree
        location : tuple
            location of the city in northern and western direction, measured in degree

        ClassAttributes:
        ------------
        citySet : set
            the set of all known cities
        nameToRef : dict
            key-value pairs of the form str(self) : self


        Methods:
        ------------
        __str__ : str
            returns the name of the city together with the abbreviation of its state
        setState(state) : None
            sets the state of the city
        """

    citySet = set()
    nameToRef = {}

    def __init__(self,cityName,stateName,population=0,capital=False,location=None):
        """
        Initialise a City instance

        :param cityName: str
            name of the city
        :param stateName: str
            name of the state the city is in
        :param population: int
            population of the city, default=0
        :param capital: bool
            whether the city is the capital of its state
        :param location: iterative of length two
            northern and western coordinate of the city in degree

        Adds city to the cityList of its state
        If applicable, updates the capital of the state
        """
        self.name = cityName
        self.capital = capital
        self.state = None
        self.setState(stateName)
        self.population = population
        self.north = location[0]
        self.west = location[1]
        self.location = (self.north,self.west)
        City.citySet.add(self)

    def __str__(self):
        """
        Representation of a City instance

        :return: str
            name of the city and abbreviation of its state
        """
        return self.name + ", " + self.state.abbreviation

    def setState(self,state):
        """
        Sets the state of a city
        :param state: State or str
        :return: None
        """
        if type(state) == State:
            self.state = state
        elif state in State.nameToRef:
            self.state = State.nameToRef[state]
        else:
            raise ValueError("State not defined")

        if self.capital:
            self.state.capital = self

        self.state.cities.add(self)

        City.nameToRef[self.name + ", " + self.state.abbreviation] = self


def readInStates(fileName):
    """
    Reads in a csv file, creating or updating State instances

    :param fileName: str
        name of the file containing states
    :return: None

    Description of the file structure:
        Optional columns with following headers:
        Name(mandatory) : str
        Population : int, may contain "," as separator
        Area : int, may contain "," as separator
        Region : str
        Abbreviation : str, recommended of length two
    """

    supportedHeaders = {"name","population","area","region","abbreviation"}

    with open(fileName) as file:
        reader = csv.reader(file)
        headerPositions = {}
        for row in reader:
            position = 0
            for header in row:
                if header.lower().strip() in supportedHeaders:
                    headerPositions[header.lower().strip()] = position
                position += 1
            break

        if "name" not in headerPositions:
            raise ValueError("State name not included in file!")

        for row in reader:
            name = row[headerPositions["name"]].strip()
            if name in State.nameToRef:
                state = State.nameToRef[name]
            else:
                state = State(name)

            if "population" in headerPositions:
                state.population = int(row[headerPositions["population"]].replace(",","").strip())
            if "area" in headerPositions:
                state.area = int(row[headerPositions["area"]].replace(",","").strip())
            if "region" in headerPositions:
                state.region = row[headerPositions["region"]].strip()
            if "abbreviation" in headerPositions:
                state.abbreviation = row[headerPositions["abbreviation"]].strip()


def readInCities(fileName):
    """
    Reads in a csv file, creating City instances

    :param fileName: str
        name of the file containing cities
    :return: None

    Description of the file structure:
        Optional columns with following headers:
        Name(mandatory) : str
        Population(mandatory) : int, may contain "," as separator
        State(mandatory) : str
        Location(mandatory) : str of the form "{float}˚N {float}˚W"
        Capital : bool
    """

    supportedHeaders = {"name","population","state","capital","location"}

    with open(fileName) as file:
        reader = csv.reader(file)
        headerPositions = {}
        for row in reader:
            position = 0
            for header in row:
                if header.lower().strip() in supportedHeaders:
                    headerPositions[header.lower().strip()] = position
                position += 1
            break

        if "name" not in headerPositions:
            raise ValueError("City name not included in file!")
        if "state" not in headerPositions:
            raise ValueError("City state not included in file!")
        if "population" not in headerPositions:
            raise ValueError("City population not included in file!")
        if "location" not in headerPositions:
            raise ValueError("City location not included in file!")

        for row in reader:

            if "[" in row[headerPositions["name"]]:
                name = row[headerPositions["name"]][:row[headerPositions["name"]].index("[")].strip()
            else:
                name = row[headerPositions["name"]].strip()

            population = int(row[headerPositions["population"]].replace(",","").strip())
            state = row[headerPositions["state"]].replace(",","").strip()
            north = float(row[headerPositions["location"]][:row[headerPositions["location"]].index("°N")].strip())
            west = float(row[headerPositions["location"]][row[headerPositions["location"]].index("°N") + 2:row[headerPositions["location"]].index("°W")].strip())
            capital = False
            if "capital" in headerPositions and row[headerPositions["capital"]].lower() == "true":
                capital = True

            city = City(name, state, population, capital, (north, west))




def citySubset(citySet,minPopulation=0,maxPopulation=float("infinity"),states=State.stateSet):
    """
    Returns subset of cities in population interval and in given states

    :param citySet: set with City elements
    :param minPopulation: int, default=0
    :param maxPopulation: int, default=infinity
    :param states: set with State elements, default=State.stateSet
    :return: set with City elements
    """
    subset = set()
    for city in citySet:
        if minPopulation <= city.population <= maxPopulation and (city.state in states or city.state.name in states):
            subset.add(city)
    return subset


def angularDistance(location1,location2):
    """
    Compute angular distance between two locations

    :param location1, location2: (float,float)
        (northern,western) coordinates of the two locations
    :return: float
        angular distance in degree
    """

    if location1 == location2:
        return float(0)

    theta1 = location1[0] * math.pi / 180
    theta2 = location2[0] * math.pi / 180
    deltaphi = (location1[1] - location2[1]) * math.pi / 180

    return round(math.acos(sin(theta1) * sin(theta2) + cos(theta1) * cos(theta2) * cos(deltaphi)) * 180 / math.pi, 4)


def derivativesAngularDistance(location1,location2):
    """
        Compute derivatives of the angular distance given two locations

        :param location1, location2: (float,float)
            (northern,western) coordinates of the two locations
        :return: (float,float)
            derivatives of the angular distance w.r.t. theta and phi in degree
        """
    theta1 = location1[0] * math.pi / 180
    theta2 = location2[0] * math.pi / 180
    deltaphi = (location1[1] - location2[1]) * math.pi / 180

    denumerator = sqrt(1 - (sin(theta1) * sin(theta2) + cos(theta1) * cos(theta2) * cos(deltaphi)) ** 2)
    numeratorTheta = -sin(theta1) * cos(theta2) + cos(theta1) * sin(theta2) * cos(deltaphi)
    numeratorPhi = -cos(theta1) * cos(theta2) * sin(deltaphi)

    return numeratorTheta / denumerator, numeratorPhi / denumerator


def createDistances(citySet,reference=None):
    """
    Creates a dictionary of distances between cities

    :param citySet: set with City elements
    :param reference: City, default=None
        if City: compute distances between reference and cities in citySet
        if None: compute distances between all cities in citySet
    :return: dict
        if reference=None: key-value pairs (str(city1),str(city2)) : distance
        if reference=City: key-value pairs str(city) : distance
    """
    distances = {}
    if not reference:
        for city1 in citySet:
            for city2 in citySet:
                if city1 != city2:
                    distances[(str(city1),str(city2))] = angularDistance(city1.location,city2.location)

    else:
        for city in citySet:
            if city != reference:
                distances[str(city)] = angularDistance(city.location,reference.location)

    return distances