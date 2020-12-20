import sys
import math
from collections import OrderedDict 
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygraphviz as pgv
from match import *

# Load the data to a pandas dataframe
dataframe = pd.read_csv("resources/film2.csv", sep=";", header=0, encoding='latin-1')
minYear = dataframe["Year"].min()
maxYear = dataframe["Year"].max()
gapYear = maxYear - minYear
minLength = dataframe["Length"].min()
maxLength = dataframe["Length"].max()
gapLength = maxLength - minLength
print(minYear, maxYear, minLength, maxLength, dataframe["Popularity"].max())

# general info on the dataframe
print('---\ngeneral info on the dataframe')
print(dataframe.info())

# get the number of films
nb_movies = len(dataframe.index)

# general info on the dataframe
print('---\ngeneral info on the dataframe')
print(dataframe.info())

# print the columns of the dataframe
print('---\ncolumns of the dataset')
print(dataframe.columns)

# print the first 10 lines of the dataframe
print('---\nfirst lines')
print(dataframe.head(10))

# print the correlation matrix of the dataset
print('---\nCorrelation matrix')
print(dataframe.corr())

# print the standard deviation
print('---\nStandard deviation')
print(dataframe.std())

# get specific values in the dataframe
movie_id = 0
print('---\nall info on movie ' + str(movie_id))
print(dataframe.loc[movie_id])
print(dataframe.loc[movie_id][5])
print(dataframe.loc[movie_id][6])

def match_strings(string1, string2):
    if string1 == string2:
        return 0
    # value1 = string1
    # value2 = string2
    # if (type(value1) is str and type(value2) is str):
    #     list1 = value1.split(", ")
    #     list2 = value2.split(", ")
    #     max = len(list2) if len(list2) > len(list1) else len(list1)
    #     return len(set(list1).intersection(list2)) / max if value1 != '' else 1
    return 0.2

def compute_dissimilarity(movie_1_id, movie_2_id):
    """
        Compute  dissimilarity betwwen two movies
        based on their id.

        The meal is not a quantitative attribute.
        It is called a categorical variable.
        We must handle it differently than quantitative
        attributes.
        0 = Year, 1 = Length, 2 = Title, 3 = Subject, 4 = Actor,
        5 = Actress, 6 = Director, 7 = Popularity, 8 = Awards, 9 = Images
    """
    movie_1_year = dataframe.loc[movie_1_id][0]
    movie_2_year = dataframe.loc[movie_2_id][0]

    # movie_1_length = dataframe.loc[movie_1_id][1]
    # movie_2_length = dataframe.loc[movie_2_id][1]

    # movie_1_title = dataframe.loc[movie_1_id][2]
    # movie_2_title = dataframe.loc[movie_2_id][2]

    # movie_1_subject = dataframe.loc[movie_1_id][3]
    # movie_2_subject = dataframe.loc[movie_2_id][3]

    movie_1_actor = dataframe.loc[movie_1_id][4]
    movie_2_actor = dataframe.loc[movie_2_id][4]

    movie_1_actress = dataframe.loc[movie_1_id][5]
    movie_2_actress = dataframe.loc[movie_2_id][5]

    movie_1_director = dataframe.loc[movie_1_id][6]
    movie_2_director = dataframe.loc[movie_2_id][6]

    movie_1_popularity = dataframe.loc[movie_1_id][7]
    movie_2_popularity = dataframe.loc[movie_2_id][7]

    # movie_1_awards = dataframe.loc[movie_1_id][8]
    # movie_2_awards = dataframe.loc[movie_2_id][8]

    # movie_1_image = dataframe.loc[movie_1_id][9]
    # movie_2_image = dataframe.loc[movie_2_id][9]

    # we build a hybrid dissimilarity
    dissimilarity = math.sqrt(
        ((movie_1_year - movie_2_year) / gapYear)**2
        + match_strings(movie_1_actor, movie_2_actor)**2
        + match_strings(movie_1_actress, movie_2_actress)**2
        + match_strings(movie_1_director, movie_2_director)**2
        + ((movie_1_popularity - movie_2_popularity) / 100)**2
        # + ((movie_1_length - movie_2_length) / gapLength)
        # + match_strings(movie_1_title, movie_2_title)**2
        # + match_strings(movie_1_subject, movie_2_subject)**2
        # + match_strings(movie_1_awards, movie_2_awards)
        # + match_strings(movie_1_image, movie_2_image)
        )

    # print("----")
    # print(f"movie 1 {movie_1_title}, movie 2 {movie_2_title}, dissimilarity: {dissimilarity}")
    return dissimilarity


# build a dissimilarity matrix
dissimilarity_matrix = np.zeros((nb_movies, nb_movies))
print("compute dissimilarities")
for movie_1_id in range(nb_movies):
    for movie_2_id in range(nb_movies):
        dissimilarity = compute_dissimilarity(movie_1_id, movie_2_id)
        dissimilarity_matrix[movie_1_id, movie_2_id] = dissimilarity

print("dissimilarity matrix")
# np.set_printoptions(threshold=sys.maxsize)
print(dissimilarity_matrix)

# plt.hist(dissimilarity_matrix.flatten(), bins=100)
# plt.show()

# threshold = 0.45
threshold = 0.34

# build a graph from the dissimilarity
dot = pgv.AGraph(comment='Graph created from complex data',
            strict=True, overlap=False)

# useful list
list_of_all_nodes = []
list_of_all_edges = []
list_of_nodes_with_edge = []
list_of_nodes_without_edge = []

# useful dict
dict_of_edges_per_node = {}

for movie_id in range(nb_movies):
    movie_name = dataframe.loc[movie_id][2]
    node_name = bytes(movie_name, 'utf-8').decode('utf-8', 'ignore')
    list_of_all_nodes.append(node_name)
    dot.add_node(node_name, fixedsize=False)
    

for movie_1_id in range (nb_movies):
    # we use an undirected graphmovie_1_id in ran so we do not need
    # to take the potential reciprocal edge
    # into account
    for movie_2_id in range(nb_movies):

        # no self loops
        if not movie_1_id == movie_2_id:
            movie_1_name = dataframe.loc[movie_1_id][2]
            movie_2_name = dataframe.loc[movie_2_id][2]

            # use the threshold condition
            if dissimilarity_matrix[movie_1_id, movie_2_id] <= threshold:
                # build list of nodes with edge
                if movie_1_name not in list_of_nodes_with_edge:
                    list_of_nodes_with_edge.append(movie_1_name)
                if movie_2_name not in list_of_nodes_with_edge:
                    list_of_nodes_with_edge.append(movie_2_name)

                #create edges in graph
                edge = (movie_1_name, movie_2_name)
                list_of_all_edges.append(edge)
                dot.add_edge(movie_1_name,
                         movie_2_name,
                         color='darkolivegreen4',
                         penwidth='1.1')

# visualize the graph
# set some default node attributes
dot.node_attr["style"] = "filled"
dot.node_attr["shape"] = "box"
dot.node_attr["fixedsize"] = "true"
dot.node_attr["fontcolor"] = "#000000"

def add_tuple_in_dict(dic, tup):
    if tup[0] in dic.keys():
        if tup[1] not in dic[tup[0]]:
            dic[tup[0]].append(tup[1])
    else:
        dic[tup[0]] = [tup[1]]
    return dic

#create dict of edges
dict_of_edges_per_node = {}
for node in list_of_nodes_with_edge:
    for tup in dot.edges(node):
        dict_of_edges_per_node = add_tuple_in_dict(dict_of_edges_per_node, tup)
        dict_of_edges_per_node = add_tuple_in_dict(dict_of_edges_per_node, (tup[1], tup[0]))
# for k,v in dict_of_edges_per_node.items():
#     print(k)
#     print(v)
#     print()


#calculate list with of nodes without edge and remove them
list_of_nodes_without_edge = list_of_all_nodes.copy()
for node in list_of_nodes_with_edge:
    list_of_nodes_without_edge.remove(node)
dot = match(dot, dict_of_edges_per_node)
# print(dict_of_edges_per_node)
        
dot.remove_nodes_from(list_of_nodes_without_edge)
dot.graph_attr.update(label=f"threshold {threshold}", fontsize='20')
dot.draw("graph1.png", prog="circo")
