import pickle
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import random
import igraph as ig


print("Importing nodes_edges_ucla_big_graph.pickle...")
with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)


print("Importing intermediate_files/Node_to_Node_pairs.pickle...")
with open(r'intermediate_files/Node_to_Node_pairs.pickle', 'rb') as handle:
    Node_to_Node_pairs = pickle.load(handle)

# convert all values in B_matrix_sliced from float to int
B_matrix_sliced = B_matrix_sliced.astype(int)

print("Creating graph using B_matrix_sliced...")
time_start = time.time()

g = ig.Graph(directed=True)
g.add_vertices(np.max(B_matrix_sliced[:, :2]) + 1)  # assuming nodes are labeled 0 to n-1
edges = [(int(B_matrix_sliced[i, 0]), int(B_matrix_sliced[i, 1])) for i in range(len(B_matrix_sliced))]
g.add_edges(edges)
g.es['weight'] = B_matrix_sliced[:, 4]

time_end = time.time()
print("Time taken to create graph:", (time_end - time_start) / 60, "minutes")

del B_matrix_sliced, B_matrix_str_sliced, nodes_coordinates_array

print("Creating Node_to_Node_pairs_dict...")
Node_to_Node_pairs_dict = defaultdict(list)

for key, value in Node_to_Node_pairs:
    Node_to_Node_pairs_dict[key].append(value)

Node_to_Node_pairs_len = len(Node_to_Node_pairs)
del Node_to_Node_pairs

# print("Subsetting origin_nodes_list...")
# origin_nodes_list = list(Node_to_Node_pairs_dict.keys())

# random.seed(123)
# origin_nodes_list_subset = random.sample(origin_nodes_list, 50)

print("Creating Node_to_Node_pairs_dict_subset...")
# Node_to_Node_pairs_dict_subset = {k: v for k, v in Node_to_Node_pairs_dict.items() if k in origin_nodes_list_subset}
Node_to_Node_pairs_dict_subset = Node_to_Node_pairs_dict

# count the total number of pairs in Node_to_Node_pairs_dict_subset from keys and values
total_subset_pairs = 0

for key, value in Node_to_Node_pairs_dict_subset.items():
    total_subset_pairs += len(value)

print("Using single source dijkstra algorithm in igraph to find shortest paths for ", total_subset_pairs, "pairs...")

time_start = time.time()

shortest_path_results = {}

for origin, destinations in Node_to_Node_pairs_dict_subset.items():
    shortest_paths = g.get_shortest_paths(origin, to=destinations, weights='weight', output="vpath", algorithm="dijkstra")
    to_destination_dict = {}
    for destination, path in zip(destinations, shortest_paths):
        to_destination_dict[destination] = path

    shortest_path_results[origin] = to_destination_dict

time_end = time.time()

print("Time taken for Single Source Method: ", (time_end - time_start)/60, "minutes")
print("Estimated time for all pairs: ", (time_end - time_start)/60 * Node_to_Node_pairs_len/total_subset_pairs / 60, "hours")

print("Converting shortest_path_results to tuple keys...")

time_start = time.time()

shortest_path_tuple_keys = {(origin, destination): path 
                            for origin, destinations in shortest_path_results.items() 
                            for destination, path in destinations.items()}

time_end = time.time()

print("Time taken to convert shortest_path_results to tuple keys: ", (time_end - time_start), "seconds")

# count the number of values that are empty lists

empty_paths = 0

for key, value in shortest_path_tuple_keys.items():
    if len(value) == 0:
        empty_paths += 1

print("Number of empty paths: ", empty_paths)
print("Percentage of empty paths: ", empty_paths/total_subset_pairs * 100, "%")

print("Exporting shortest_path_results to intermediate_files/shortest_path_results.pickle...")

with open(r'intermediate_files/shortest_path_results.pickle', 'wb') as handle:
    pickle.dump(shortest_path_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Exporting shortest_path_tuple_keys to intermediate_files/shortest_path_tuple_keys.pickle...")

with open(r'intermediate_files/shortest_path_tuple_keys.pickle', 'wb') as handle:
    pickle.dump(shortest_path_tuple_keys, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")