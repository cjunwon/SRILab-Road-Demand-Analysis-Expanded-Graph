import pickle
import numpy as np
import networkx as nx
import pandas as pd
import time
from collections import defaultdict
import random
import multiprocessing

print("Importing nodes_edges_ucla_big_graph.pickle...")
with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

print("Importing intermediate_files/Node_to_Node_pairs.pickle...")
with open(r'intermediate_files/Node_to_Node_pairs.pickle', 'rb') as handle:
    Node_to_Node_pairs = pickle.load(handle)

print("Creating graph using B_matrix_str_sliced...")
time_start = time.time()

G=nx.DiGraph()

for i in range(len(B_matrix_sliced)):
    oneway=B_matrix_str_sliced[i,2]
    
    if oneway==None:        
        G.add_edge(int(B_matrix_sliced[i,0]),int(B_matrix_sliced[i,1]),sect_id=str(B_matrix_str_sliced[i,3]),distance=int(B_matrix_sliced[i,2]),speed=int(B_matrix_sliced[i,3]),weight=B_matrix_sliced[i,4],walk_min=B_matrix_sliced[i,5],name=str(B_matrix_str_sliced[i,0]),roadtype=str(B_matrix_str_sliced[i,1]))
        G.add_edge(int(B_matrix_sliced[i,1]),int(B_matrix_sliced[i,0]),sect_id=str(B_matrix_str_sliced[i,3]),distance=int(B_matrix_sliced[i,2]),speed=int(B_matrix_sliced[i,3]),weight=B_matrix_sliced[i,4],walk_min=B_matrix_sliced[i,5],name=str(B_matrix_str_sliced[i,0]),roadtype=str(B_matrix_str_sliced[i,1]))
    elif oneway=="FT":        
        G.add_edge(int(B_matrix_sliced[i,0]),int(B_matrix_sliced[i,1]),sect_id=str(B_matrix_str_sliced[i,3]),distance=int(B_matrix_sliced[i,2]),speed=int(B_matrix_sliced[i,3]),weight=B_matrix_sliced[i,4],walk_min=B_matrix_sliced[i,5],name=str(B_matrix_str_sliced[i,0]),roadtype=str(B_matrix_str_sliced[i,1]))
    elif oneway=="TF":        
        G.add_edge(int(B_matrix_sliced[i,1]),int(B_matrix_sliced[i,0]),sect_id=str(B_matrix_str_sliced[i,3]),distance=int(B_matrix_sliced[i,2]),speed=int(B_matrix_sliced[i,3]),weight=B_matrix_sliced[i,4],walk_min=B_matrix_sliced[i,5],name=str(B_matrix_str_sliced[i,0]),roadtype=str(B_matrix_str_sliced[i,1]))
    else:
        print("Error")

time_end = time.time()
print("Time taken to create graph:", (time_end-time_start)/60, "minutes")

print("Subsetting Node_to_Node_pairs...")
origin_nodes_list = [t[0] for t in Node_to_Node_pairs]
origin_nodes_list = list(dict.fromkeys(origin_nodes_list))

random.seed(0)
origin_nodes_list_subset = random.sample(origin_nodes_list, 10)
Node_to_Node_pairs_subset = [t for t in Node_to_Node_pairs if t[0] in origin_nodes_list_subset]

print("Using single source dijkstra method to find shortest paths for ", len(Node_to_Node_pairs_subset), "pairs of nodes...")

# Single Source Method

All_shortest_paths_dict = {}
Node_to_Node_shortest_paths_dict = {}

time_start = time.time()

for pairing in Node_to_Node_pairs_subset:
    node_1 = pairing[0]
    node_2 = pairing[1]

    if node_2 in All_shortest_paths_dict and node_1 in All_shortest_paths_dict[node_2]:
        shortest_path = All_shortest_paths_dict[node_2].get(node_1, None)
        if shortest_path is not None:
            shortest_path = shortest_path[::-1] # reverse the list
        Node_to_Node_shortest_paths_dict[pairing] = shortest_path
    elif node_1 in All_shortest_paths_dict:
        shortest_path = All_shortest_paths_dict[node_1].get(node_2, None)
        Node_to_Node_shortest_paths_dict[pairing] = shortest_path
    else:
        All_shortest_paths_dict[node_1] = nx.single_source_dijkstra_path(G, node_1, weight='weight')
        shortest_path = All_shortest_paths_dict[node_1].get(node_2, None)
        Node_to_Node_shortest_paths_dict[pairing] = shortest_path


time_end = time.time()

print("Time taken for Single Source Method: ", (time_end - time_start)/60, "minutes")
print("Estimated time for all pairs: ", (time_end - time_start)/60 * len(Node_to_Node_pairs)/len(Node_to_Node_pairs_subset) / 60 / 24, "days")