#Create dict key=nom, value=listofattribute
import sys
import math
from collections import OrderedDict 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygraphviz as pgv

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
        # + ((movie_1_length - movie_2_length) / gapLength)
        # + match_strings(movie_1_title, movie_2_title)**2
        # + match_strings(movie_1_subject, movie_2_subject)**2

        + match_strings(movie_1_actor, movie_2_actor)**2
        + match_strings(movie_1_actress, movie_2_actress)**2
        + match_strings(movie_1_director, movie_2_director)**2
        + ((movie_1_popularity - movie_2_popularity) / 100)**2

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
for node in list_of_nodes_with_edge:
    for tup in dot.edges(node):
        dict_of_edges_per_node = add_tuple_in_dict(dict_of_edges_per_node, tup)
        dict_of_edges_per_node = add_tuple_in_dict(dict_of_edges_per_node, (tup[1], tup[0]))
# for k,v in dict_of_edges_per_node.items():
#     print(k)
#     print(v)
#     print()

def DFS_visitor(node, marked_vertices, dict_of_edges_per_node,
            heuristic_depth, heuristic, visitor_depth, stack):
    visitor_depth += 1
    for neighbor in dict_of_edges_per_node[node]:
        if marked_vertices[neighbor] is False:
            marked_vertices[neighbor] = True
            heuristic[heuristic_depth].append(neighbor)
            (marked_vertices, heuristic, visitor_depth, stack) = DFS_visitor(neighbor, 
                                                                                marked_vertices, 
                                                                                dict_of_edges_per_node, 
                                                                                heuristic_depth, 
                                                                                heuristic, 
                                                                                visitor_depth,
                                                                                stack)
    visitor_depth -= 1
    return (marked_vertices, heuristic, visitor_depth, stack)
    

def depth_first_search(marked_vertices, dict_of_edges_per_node,
                       heuristic_depth, heuristic):
    for node in dict_of_edges_per_node.keys():
        stack = []
        if marked_vertices[node] is False:
            stack.append(node)
            visitor_depth = 0
            heuristic_depth += 1
            heuristic.append([node])
            marked_vertices[node] = True
            (marked_vertices, heuristic, visitor_depth, stack) = DFS_visitor(node, marked_vertices, 
                                                                                dict_of_edges_per_node, 
                                                                                heuristic_depth, 
                                                                                heuristic, 
                                                                                visitor_depth,
                                                                                stack)
    return heuristic_depth, heuristic


def BFS_visitor(node, marked_vertices, dict_of_edges_per_node, max_stack, node_to_visit, visited_node, link_to_parent):
    # print("\nNEW DEPTH %s" % (node))
    # marked_vertices[node] = True
    if node not in visited_node:
        for neighbor in dict_of_edges_per_node[node]:
            # print("2 %s" % (neighbor))
            if marked_vertices[neighbor] is False:
                # print("MARKED")
                node_to_visit.append(neighbor)
    visited_node.append(node)
    for value in node_to_visit.copy():
        # print("3 %s" % (value))
        # print("visited_node")
        # print(visited_node)
        # print("node to visit")
        # print(node_to_visit)
        # print("link to parent")
        # print(link_to_parent)
        if value not in link_to_parent.keys() and value in dict_of_edges_per_node[node] and next(reversed(link_to_parent)) in dict_of_edges_per_node[value]:
            # print("VISITING")
            node_to_visit.remove(value)
            link_to_parent[value] = node
            # print("visited_node")
            # print(visited_node)
            # print("node to visit")
            # print(node_to_visit)
            # print("link to parent")
            # print(link_to_parent)
            (visited_node, node_to_visit, link_to_parent) = BFS_visitor(value, marked_vertices.copy(), 
                                                                        dict_of_edges_per_node, 
                                                                        max_stack, node_to_visit.copy(),
                                                                        visited_node.copy(),
                                                                        link_to_parent.copy())
    return (visited_node, node_to_visit, link_to_parent)

    

def breadth_first_search(node, marked_vertices, dict_of_edges_per_node):
    node_to_visit = []
    visited_node = []
    link_to_parent = OrderedDict()
    max_stack = []
    link_to_parent[node] = "0"
    # print("HERE %s" % (node))
    # marked_vertices[node] = True
    if node not in visited_node:
        for neighbor in dict_of_edges_per_node[node]:
            # print("2 %s" % (neighbor))
            if marked_vertices[neighbor] is False:
                node_to_visit.append(neighbor)
    visited_node.append(node)
    # print(node_to_visit)
    for value in node_to_visit.copy():
        # print("3 %s" % (value))
        # print("visited_node")
        # print(visited_node)
        # print("node to visit")
        # print(node_to_visit)
        # print("link to parent")
        # print(link_to_parent)
        if value not in link_to_parent.keys() and value in dict_of_edges_per_node[node]:
            # print("VISITING")
            node_to_visit.remove(value)
            link_to_parent[value] = node
            # print("visited_node")
            # print(visited_node)
            # print("node to visit")
            # print(node_to_visit)
            # print("link to parent")
            # print(link_to_parent)
            (visited_node, node_to_visit, link_to_parent) = BFS_visitor(value, marked_vertices.copy(), 
                                                                        dict_of_edges_per_node,
                                                                        max_stack, node_to_visit.copy(),
                                                                        visited_node.copy(),
                                                                        link_to_parent.copy())
    # print("\nEND\nvisited_node")
    # print(visited_node)
    # print("node to visit")
    # print(node_to_visit)
    # print("link to parent")
    # print(link_to_parent)
    max_path = []
    last = list(link_to_parent.keys())[-1]
    value = "1"
    buffer = last
    # print("last")
    # print(last)
    while value != "0":
        value = link_to_parent[last]
        # print("\nvalue is %s" % (value))
        # print("last is %s" % (last))
        # print("buffer is %s" % (buffer))
        # print(dict_of_edges_per_node[buffer])
        if value in dict_of_edges_per_node[buffer]:
            # print("IF %s" % (value))
            max_path.append(buffer)
            buffer = value
        last = value
    max_path.append(buffer)
    return max_path


# def BFS_visitor(node, marked_vertices, dict_of_edges_per_node, max_stack, 
#                 node_to_visit, visited_node, link_to_parent):
#     marked_vertices[node] = True
#     for neighbor in dict_of_edges_per_node[node]:
#         if neighbor not in link_to_parent.keys() and neighbor in dict_of_edges_per_node[node] and next(reversed(link_to_parent)) in dict_of_edges_per_node[neighbor]:
#             # node_to_visit.remove(neighbor)
#             link_to_parent[neighbor] = node
#             node_to_visit.append(neighbor)
#             (visited_node, node_to_visit,
#                 link_to_parent) = BFS_visitor(neighbor, marked_vertices.copy(), 
#                                                 dict_of_edges_per_node, 
#                                                 max_stack, node_to_visit,
#                                                 visited_node.copy(),
#                                                 link_to_parent.copy())
#             if len(max_stack.keys()) < len(link_to_parent.keys()):
#                 max_stack = link_to_parent.copy()
#     marked_vertices[node] = False
#     link_to_parent.popitem()
#     return (visited_node, node_to_visit, link_to_parent)

    

# def breadth_first_search(node, marked_vertices, dict_of_edges_per_node):
#     node_to_visit = []
#     visited_node = []
#     link_to_parent = OrderedDict()
#     max_stack = OrderedDict()
#     link_to_parent[node] = "0"
#     marked_vertices[node] = True
#     for value in dict_of_edges_per_node[node]:
#         if value not in link_to_parent.keys() and value in dict_of_edges_per_node[node]:
#             # node_to_visit.remove(value)
#             node_to_visit.append(value)
#             link_to_parent[value] = node
#             (visited_node, node_to_visit, 
#                 link_to_parent) = BFS_visitor(value, marked_vertices.copy(), 
#                                                 dict_of_edges_per_node,
#                                                 max_stack, node_to_visit,
#                                                 visited_node.copy(),
#                                                 link_to_parent.copy())
#             if len(max_stack.keys()) < len(link_to_parent.keys()):
#                 max_stack = link_to_parent.copy()
#     max_path = []
#     last = list(max_stack.keys())[-1]
#     value = "1"
#     buffer = last
#     while value != "0":
#         value = max_stack[last]
#         if value in dict_of_edges_per_node[buffer]:
#             # print("IF %s" % (value))
#             max_path.append(buffer)
#             buffer = value
#         last = value
#     max_path.append(buffer)
#     return max_path





marked_vertices = {}
for k in dict_of_edges_per_node.keys():
    marked_vertices[k] = False

heuristic_depth = -1
heuristic = []
# print("****************************************************************************************")
heuristic_depth, heuristic = depth_first_search(marked_vertices.copy(), dict_of_edges_per_node, heuristic_depth, heuristic)
max_paths = []
for nodes in heuristic:
    length = 0
    # print("\nHERE reset")
    for node in nodes:
        # print("\nnode =%s" % (node))
        res = breadth_first_search(node, marked_vertices.copy(), dict_of_edges_per_node)
        # print(res)
        if length == 0:
            # print("HERE once")
            # print(res)
            max_paths.append(res)
            length = len(max_paths[-1])
        elif length < len(res):
            # print("HERE all")
            # print("to replace %d" % (length))
            # print(max_paths[-1])
            # print("by %d" % (len(res)))
            # print(res)
            max_paths[-1] = res
            length = len(res)

# print("\nHEURISTIQUE")
# for i in heuristic:
#     print(i)
# print("\nMAX_PATH")
# for i in max_paths:
#     print(i)

def compare_tuple_with_tuplelist(tup1, tuplelist):
    for tup2 in tuplelist:
        # print("%s | %s; %s | %s" % (tup1[0], tup2[0], tup1[1], tup2[1]))
        if tup1[0] == tup2[0]:
            if tup1[1] == tup2[1]:
                return True
    return False

def find_neighbor2_with_red_edge(node, list_of_red_edges):
    for node_neighbor in dict_of_edges_per_node[node]:
        # print("LOOP 1 %s | %s" % (node, node_neighbor))
        if (node, node_neighbor) in list_of_red_edges:
            return False
        for node_neighbor_neighbor in dict_of_edges_per_node[node_neighbor]:
            # print("LOOP 2 %s | %s" % (node_neighbor, node_neighbor_neighbor))
            if (node_neighbor, node_neighbor_neighbor) in list_of_red_edges:
                if node_neighbor != node and node_neighbor_neighbor != node:
                    # print("TRUE by %s : %s | %s" % (node, node_neighbor, node_neighbor_neighbor))
                    return True
    # print("FALSE by %s : %s | %s" % (node, node_neighbor, node_neighbor_neighbor))
    return False

def find_neighbor1_with_red_edge(node, list_of_red_edges):
    for node_neighbor in dict_of_edges_per_node[node]:
        # print("LOOP 1 %s | %s" % (node, node_neighbor))
        if (node, node_neighbor) in list_of_red_edges:
            return True
    # print("FALSE by %s : %s" % (node, node_neighbor))
    return False


#printing purposes
#calculate list with of nodes without edge and remove them
list_of_nodes_without_edge = list_of_all_nodes.copy()
for node in list_of_nodes_with_edge:
    list_of_nodes_without_edge.remove(node)
depth = 0
list_of_red_edges = []
for path in max_paths:
    i = 1
    list_of_red_edges.append([])
    while i < len(path):
        node = path[i - 1]
        node_next = path[i]
        list_of_red_edges[depth].append((node, node_next))
        list_of_red_edges[depth].append((node_next, node))
        edge = dot.get_edge(node, node_next)
        edge.attr["color"] = "red"
        edge.attr["penwidth"] = 2.4
        i += 2
    depth += 1
depth = 0
for nodes in heuristic:
    for node in nodes:
        # print("HERE2 = %s" % (node))
        for neighbor in dict_of_edges_per_node[node]:
            if compare_tuple_with_tuplelist((node, neighbor), list_of_red_edges[depth]):
                # print("WHY %s | %s" % (node, neighbor))
            elif find_neighbor2_with_red_edge(node, list_of_red_edges[depth]) is True:
                if find_neighbor1_with_red_edge(neighbor, list_of_red_edges[depth]) is False and find_neighbor1_with_red_edge(node, list_of_red_edges[depth]) is False:
                    # print("HEREEEEEEEEE%s and %s" % (node, neighbor))
                    list_of_red_edges[depth].append((node, neighbor))
                    list_of_red_edges[depth].append((neighbor, node))
                    edge = dot.get_edge(node, neighbor)
                    edge.attr["color"] = "red"
                    edge.attr["penwidth"] = 2.4
    depth += 1

# print(dict_of_edges_per_node)
        
dot.remove_nodes_from(list_of_nodes_without_edge)
dot.graph_attr.update(label=f"threshold {threshold}", fontsize='20')
dot.draw("graph1.png", prog="circo")
