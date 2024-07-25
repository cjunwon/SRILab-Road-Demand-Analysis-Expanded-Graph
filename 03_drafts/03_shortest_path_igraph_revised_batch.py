import pickle
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import random
import igraph as ig


print("Importing nodes_edges_ucla_access.pickle...")
with open(r'nodes_edges_ucla_access.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)


print("Importing intermediate_files/Node_to_Node_pairs.pickle...")
with open(r'intermediate_files/Node_to_Node_pairs.pickle', 'rb') as handle:
    Node_to_Node_pairs = pickle.load(handle)

# convert all values in B_matrix_sliced from float to int
B_matrix_sliced = B_matrix_sliced.astype(int)

print("Creating graph using B_matrix_sliced...")
time_start = time.time()

# g = ig.Graph(directed=True)
# g.add_vertices(np.max(B_matrix_sliced[:, :2]) + 1)
# edges = [(int(B_matrix_sliced[i, 0]), int(B_matrix_sliced[i, 1])) for i in range(len(B_matrix_sliced))]
# g.add_edges(edges)
# g.es['weight'] = B_matrix_sliced[:, 4]

# Create an empty directed graph
g = ig.Graph(directed=True)

vertices = set(B_matrix_sliced[:, 0]).union(set(B_matrix_sliced[:, 1]))
g.add_vertices(list(vertices))

for i in range(len(B_matrix_sliced)):
    oneway = B_matrix_str_sliced[i, 2]
    edge_attrs = {
        "sect_id": str(B_matrix_str_sliced[i, 3]),
        "distance": int(B_matrix_sliced[i, 2]),
        "speed": int(B_matrix_sliced[i, 3]),
        "weight": B_matrix_sliced[i, 4],
        "walk_min": B_matrix_sliced[i, 5],
        "name": str(B_matrix_str_sliced[i, 0]),
        "roadtype": str(B_matrix_str_sliced[i, 1])
    }

    if oneway is None:
        g.add_edge(int(B_matrix_sliced[i, 0]), int(B_matrix_sliced[i, 1]), **edge_attrs)
        g.add_edge(int(B_matrix_sliced[i, 1]), int(B_matrix_sliced[i, 0]), **edge_attrs)
    elif oneway == "FT":
        g.add_edge(int(B_matrix_sliced[i, 0]), int(B_matrix_sliced[i, 1]), **edge_attrs)
    elif oneway == "TF":
        g.add_edge(int(B_matrix_sliced[i, 1]), int(B_matrix_sliced[i, 0]), **edge_attrs)
    else:
        print("Error")

time_end = time.time()
print("Time taken to create graph:", (time_end - time_start) / 60, "minutes")

# export the graph to a pickle file

# print("Exporting graph to intermediate_files/graph.pickle...")
# with open(r'intermediate_files/graph.pickle', 'wb') as handle:
#     pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)

del B_matrix_sliced, B_matrix_str_sliced, nodes_coordinates_array

print("Creating Node_to_Node_pairs_dict...")
Node_to_Node_pairs_dict = defaultdict(list)

for key, value in Node_to_Node_pairs:
    Node_to_Node_pairs_dict[key].append(value)

# Node_to_Node_pairs_len = len(Node_to_Node_pairs)
# del Node_to_Node_pairs

# print("Subsetting origin_nodes_list...")
# origin_nodes_list = list(Node_to_Node_pairs_dict.keys())
# random.seed(123)
# origin_nodes_list_subset = random.sample(origin_nodes_list, 51)
# del origin_nodes_list

# print("Creating Node_to_Node_pairs_dict_subset...")
# Node_to_Node_pairs_dict_subset = {k: v for k, v in Node_to_Node_pairs_dict.items() if k in origin_nodes_list_subset}
Node_to_Node_pairs_dict_subset = Node_to_Node_pairs_dict

del Node_to_Node_pairs_dict

# count the total number of pairs in Node_to_Node_pairs_dict_subset from keys and values
total_subset_pairs = 0

for key, value in Node_to_Node_pairs_dict_subset.items():
    total_subset_pairs += len(value)

def chunks(data, SIZE=1000):
    #it = iter(data)
    for i in range(0, len(data), SIZE):
        it = iter(data)
        if isinstance(data, dict):
            yield {k: data[k] for k in list(it)[i:i + SIZE]}
        else:
            yield data[i:i + SIZE]

SIZE = 1000
# SIZE = 10

print("Using single source dijkstra algorithm in igraph to find shortest paths for ", total_subset_pairs, "pairs...")

print("Creating chunks of Node_to_Node_pairs_dict_subset...")
Node_to_Node_pairs_dict_subset_chunks = list(chunks(Node_to_Node_pairs_dict_subset, SIZE))

print("Number of chunks to process: ", len(Node_to_Node_pairs_dict_subset_chunks))

missing_paths = 0

time_start = time.time()

for i, chunk in enumerate(Node_to_Node_pairs_dict_subset_chunks):

    chunk_time_start = time.time()

    shortest_path_results = {}
    print(f"Processing chunk {i+1}/{len(Node_to_Node_pairs_dict_subset_chunks)}...")
    for origin, destinations in chunk.items():
        shortest_paths = g.get_shortest_paths(origin, to=destinations, weights='weight', output="vpath", algorithm="dijkstra")
        to_destination_dict = {}
        for destination, path in zip(destinations, shortest_paths):
            if len(path) == 0:
                missing_paths += 1
            else:
                to_destination_dict[destination] = path

        shortest_path_results[origin] = to_destination_dict

    shortest_path_tuple_keys = {(origin, destination): path 
                            for origin, destinations in shortest_path_results.items() 
                            for destination, path in destinations.items()}

    output_file_1 = f'intermediate_files/shortest_path_results/shortest_path_results_{i+1}.pickle'

    print("Exporting shortest_path_results to " + output_file_1 + "...")

    with open(output_file_1, 'wb') as handle:
        pickle.dump(shortest_path_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    output_file_2 = f'intermediate_files/shortest_path_results/shortest_path_tuple_keys_{i+1}.pickle'

    print("Exporting shortest_path_tuple_keys to " + output_file_2 + "...")

    with open(output_file_2, 'wb') as handle:
        pickle.dump(shortest_path_tuple_keys, handle, protocol=pickle.HIGHEST_PROTOCOL)

    chunk_time_end = time.time()

    del shortest_path_results, shortest_path_tuple_keys

    print("Time taken for chunk ", i+1, ": ", (chunk_time_end - chunk_time_start)/60, "minutes")

time_end = time.time()

print("Time taken for all chunks: ", (time_end - time_start)/60, "minutes")
# print("Estimated time for all pairs: ", (time_end - time_start)/60 * Node_to_Node_pairs_len/total_subset_pairs / 60, "hours")

print("Number of missing paths: ", missing_paths)
percentage_missing_paths = missing_paths/total_subset_pairs * 100
print("Percentage of missing paths: " + str(percentage_missing_paths) + "%")

# count the number of values that are empty lists

# empty_paths = 0

# for key, value in shortest_path_tuple_keys.items():
#     if len(value) == 0:
#         empty_paths += 1

# print("Number of empty paths: ", empty_paths)
# print("Percentage of empty paths: ", empty_paths/total_subset_pairs * 100, "%")

print("Done!")