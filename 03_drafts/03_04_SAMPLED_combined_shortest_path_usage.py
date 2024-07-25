import numpy as np
import pandas as pd
import time
from collections import defaultdict
import random
import igraph as ig
import pickle
import zipfile
import os


print("Importing intermediate_files/Node_to_Node_pairs.pickle...")
with open(r'intermediate_files/Node_to_Node_pairs.pickle', 'rb') as handle:
    Node_to_Node_pairs = pickle.load(handle)

print("Importing intermediate_files/igraph.pickle...")
with open(r'intermediate_files/igraph.pickle', 'rb') as handle:
    g = pickle.load(handle)

print("Importing intermediate_files/nodes_edges_ucla_big_graph.pickle...")
with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

print("Importing intermediate_files/LODES_adjusted_block_pairing_count_list.pickle...")
with open(r'intermediate_files/LODES_adjusted_block_pairing_count_list.pickle', 'rb') as handle:
    Block_to_Block_Pairs, LODES_adjusted, block_pairing_count_list = pickle.load(handle)


expanded_LODES = [value for value, count in zip(LODES_adjusted, block_pairing_count_list) for _ in range(count)]
expanded_LODES = np.array(expanded_LODES, dtype='d')


lodes_lookup = {node_pair: lodes_val for node_pair, lodes_val in zip(Node_to_Node_pairs, expanded_LODES)}

def get_lodes_from_node_pair(node_pair):
    # Directly return the LODES_vals using the node_pair as the key
    return lodes_lookup.get(node_pair, None)


# Initialize array of 0s with 10 columns and append to B_matrix_sliced
zero_columns = np.zeros((B_matrix_sliced.shape[0], 10))
B_matrix_weighted_array = np.hstack((B_matrix_sliced, zero_columns))
B_matrix_weighted_array = B_matrix_weighted_array.astype(float)


# Converting B_matrix_weighted to a dictionary for faster lookups (O(1) lookups, faster)
B_matrix_weighted_dict = {(row[0].astype(int), row[1].astype(int)): row for row in B_matrix_weighted_array}


print("Creating Node_to_Node_pairs_dict...")
Node_to_Node_pairs_dict = defaultdict(list)

for key, value in Node_to_Node_pairs:
    Node_to_Node_pairs_dict[key].append(value)

Node_to_Node_pairs_len = len(Node_to_Node_pairs)


print("Subsetting origin_nodes_list...")
origin_nodes_list = list(Node_to_Node_pairs_dict.keys())
random.seed(123)
origin_nodes_list_subset = random.sample(origin_nodes_list, 99)
del origin_nodes_list

print("Creating Node_to_Node_pairs_dict_subset...")
Node_to_Node_pairs_dict_subset = {k: v for k, v in Node_to_Node_pairs_dict.items() if k in origin_nodes_list_subset}
# Node_to_Node_pairs_dict_subset = Node_to_Node_pairs_dict

del Node_to_Node_pairs_dict


# count the total number of pairs in Node_to_Node_pairs_dict_subset from keys and values
total_subset_pairs = 0

for key, value in Node_to_Node_pairs_dict_subset.items():
    total_subset_pairs += len(value)

def chunks(data, SIZE=10):
    #it = iter(data)
    for i in range(0, len(data), SIZE):
        it = iter(data)
        if isinstance(data, dict):
            yield {k: data[k] for k in list(it)[i:i + SIZE]}
        else:
            yield data[i:i + SIZE]

# SIZE = 1000
SIZE = 10


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

    chunk_time_end = time.time()

    print("Time taken to find shortest paths for chunk ", i+1, ": ", (chunk_time_end - chunk_time_start)/60, "minutes")

    B_matrix_update_time_start = time.time()

    # Initialize variables to track time spent in different parts of the loop
    time_getting_path_and_lodes = 0
    time_updating_matrix = 0

    for key in shortest_path_tuple_keys:
        start_time = time.time()
        path = shortest_path_tuple_keys[key]
        lodes_values = get_lodes_from_node_pair(key)
        time_getting_path_and_lodes += time.time() - start_time

        start_time = time.time()
        for j in range(len(path) - 1):
            pair = (path[j], path[j+1])
            reverse_pair = (path[j+1], path[j])

            if pair in B_matrix_weighted_dict:
                B_matrix_weighted_dict[pair][6:16] += lodes_values
            elif reverse_pair in B_matrix_weighted_dict:
                B_matrix_weighted_dict[reverse_pair][6:16] += lodes_values
        
        time_updating_matrix += time.time() - start_time

    B_matrix_update_time_end = time.time()

    # print("Time taken to update B_matrix_weighted_array for chunk ", i+1, ": ", (B_matrix_update_time_end - B_matrix_update_time_start)/60, "minutes")
    # Print the total time and the time spent in each part
    print(f"Total time for B_matrix update: {B_matrix_update_time_end - B_matrix_update_time_start} seconds")
    print(f"Time getting path and lodes values: {time_getting_path_and_lodes} seconds")
    print(f"Time updating matrix: {time_updating_matrix} seconds")

    del shortest_path_tuple_keys

    # find number of rows in B_matrix_weighted_array where the last 10 columns are not 0
    non_zero_rows = np.where(B_matrix_weighted_array[:,-10:].any(axis=1))[0]
    print("Percent of B_matrix_weighted_array updated: ", len(non_zero_rows)/B_matrix_weighted_array.shape[0] * 100, "%")

time_end = time.time()

print("Time taken for all chunks: ", (time_end - time_start)/60, "minutes")
print("Estimated time for all pairs: ", (time_end - time_start)/60 * Node_to_Node_pairs_len/total_subset_pairs / 60, "hours")

print("Number of missing paths: ", missing_paths)
percentage_missing_paths = missing_paths/total_subset_pairs * 100
print("Percentage of missing paths: " + str(percentage_missing_paths) + "%")



# Export B_matrix_weighted_array to a pickle file

print("Exporting B_matrix_weighted_array to intermediate_files/B_matrix_weighted_array.pickle...")
with open(r'intermediate_files/B_matrix_weighted_array.pickle', 'wb') as handle:
    pickle.dump(B_matrix_weighted_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


