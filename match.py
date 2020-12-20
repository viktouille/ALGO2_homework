import sys
import math
from collections import OrderedDict 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygraphviz as pgv


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




def compare_tuple_with_tuplelist(tup1, tuplelist):
    for tup2 in tuplelist:
        # print("%s | %s; %s | %s" % (tup1[0], tup2[0], tup1[1], tup2[1]))
        if tup1[0] == tup2[0]:
            if tup1[1] == tup2[1]:
                return True
    return False

def find_neighbor2_with_red_edge(node, list_of_red_edges, dict_of_edges_per_node):
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

def find_neighbor1_with_red_edge(node, list_of_red_edges, dict_of_edges_per_node):
    for node_neighbor in dict_of_edges_per_node[node]:
        # print("LOOP 1 %s | %s" % (node, node_neighbor))
        if (node, node_neighbor) in list_of_red_edges:
            return True
    # print("FALSE by %s : %s" % (node, node_neighbor))
    return False

def find_maxpaths_and_subgraphs(dict_of_edges_per_node):
    heuristic_depth = -1
    heuristic = []
    marked_vertices = {}
    for k in dict_of_edges_per_node.keys():
        marked_vertices[k] = False
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
    return max_paths, heuristic, heuristic_depth

    # print("\nHEURISTIQUE")
    # for i in heuristic:
    #     print(i)
    # print("\nMAX_PATH")
    # for i in max_paths:
    #     print(i)


def match(dot, dict_of_edges_per_node):
    max_paths, heuristic, heuristic_depth = find_maxpaths_and_subgraphs(dict_of_edges_per_node)
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
                    pass
                elif find_neighbor2_with_red_edge(node, list_of_red_edges[depth], dict_of_edges_per_node) is True:
                    if find_neighbor1_with_red_edge(neighbor, list_of_red_edges[depth], dict_of_edges_per_node) is False and find_neighbor1_with_red_edge(node, list_of_red_edges[depth], dict_of_edges_per_node) is False:
                        # print("HEREEEEEEEEE%s and %s" % (node, neighbor))
                        list_of_red_edges[depth].append((node, neighbor))
                        list_of_red_edges[depth].append((neighbor, node))
                        edge = dot.get_edge(node, neighbor)
                        edge.attr["color"] = "red"
                        edge.attr["penwidth"] = 2.4
        depth += 1
    return dot
