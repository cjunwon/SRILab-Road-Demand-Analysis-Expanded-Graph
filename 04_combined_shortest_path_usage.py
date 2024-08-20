print("Importing libraries...")

import numpy as np
import pandas as pd
import time
from collections import defaultdict
import igraph as ig
import pickle
from itertools import islice


################### IMPORTANT SPECIFICATION TO CONTROL RUN SIZE ###################

# There are len(Node_to_Node_pairs_dict) = 156244 origin nodes to compute.
# You can specify how much to compute for each script run.

START_INDEX = 0 # CHANGE AS NEEDED
END_INDEX = 156244 # CHANGE AS NEEDED
# END_INDEX = 156244

# To run all in one go, set START_INDEX = 0, END_INDEX = 156244

# You can also specify the desired chunk size here.
CHUNK_SIZE = 10

###################################################################################


# ---------------------------------FILE IMPORT---------------------------------

print("Importing intermediate_files/Node_to_Node_pairs.pickle...")
with open(r'intermediate_files/Node_to_Node_pairs.pickle', 'rb') as handle:
    Node_to_Node_pairs = pickle.load(handle)

print("Importing intermediate_files/igraph.pickle...")
with open(r'intermediate_files/igraph.pickle', 'rb') as handle:
    g = pickle.load(handle)

print("Importing intermediate_files/nodes_edges_ucla_access.pickle...")
with open(r'nodes_edges_ucla_access.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

print("Importing intermediate_files/LODES_adjusted_block_pairing_count_list.pickle...")
with open(r'intermediate_files/LODES_adjusted_block_pairing_count_list.pickle', 'rb') as handle:
    Block_to_Block_Pairs, LODES_adjusted, block_pairing_count_list = pickle.load(handle)


# ---------------------------------PREPROCESSING---------------------------------



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

# ---------------------------------SUBSETTING & BATCH SETUP---------------------------------

# count the total number of pairs in Node_to_Node_pairs_dict from keys and values
total_subset_pairs = 0

for key, value in Node_to_Node_pairs_dict.items():
    total_subset_pairs += len(value)

def chunks(data, CHUNK_SIZE=10):
    #it = iter(data)
    for i in range(0, len(data), CHUNK_SIZE):
        it = iter(data)
        if isinstance(data, dict):
            yield {k: data[k] for k in list(it)[i:i + CHUNK_SIZE]}
        else:
            yield data[i:i + CHUNK_SIZE]

Node_to_Node_pairs_subset_dict = dict(islice(Node_to_Node_pairs_dict.items(), START_INDEX, END_INDEX))


# ---------------------------------MAIN ALGORITHM & B_MATRIX UPDATE---------------------------------


print("Using single source dijkstra algorithm in igraph to find shortest paths for ", total_subset_pairs, "pairs...")

print("Creating chunks of Node_to_Node_pairs_dict_subset...")
Node_to_Node_pairs_subset_dict_chunks = list(chunks(Node_to_Node_pairs_subset_dict, CHUNK_SIZE))

print("Number of chunks to process: ", len(Node_to_Node_pairs_subset_dict_chunks))

missing_paths = 0

time_start = time.time()

for i, chunk in enumerate(Node_to_Node_pairs_subset_dict_chunks):

    chunk_time_start = time.time()

    shortest_path_results = {}
    print(f"Processing chunk {i+1}/{len(Node_to_Node_pairs_subset_dict_chunks)}...")
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

# convert B_matrix_weighted_dict back to B_matrix_weighted_array

print("Converting B_matrix_weighted_dict back to B_matrix_weighted_array...")

convert_start = time.time()

del B_matrix_weighted_array

# use keys and values from B_matrix_weighted_dict to create B_matrix_weighted_array

B_matrix_weighted_array = np.array(list(B_matrix_weighted_dict.values()))

# find number of rows in B_matrix_weighted_array where the last 10 columns are not 0
non_zero_rows = np.where(B_matrix_weighted_array[:,-10:].any(axis=1))[0]
print("Percent of B_matrix_weighted_array updated: ", len(non_zero_rows)/B_matrix_weighted_array.shape[0] * 100, "%")

convert_end = time.time()

print("Time taken to convert B_matrix_weighted_dict back to B_matrix_weighted_array: ", (convert_end - convert_start)/60, "minutes")

time_end = time.time()

print("Time taken for all chunks: ", (time_end - time_start)/60, "minutes")
print("Estimated time for all pairs: ", (time_end - time_start)/60 * Node_to_Node_pairs_len/total_subset_pairs / 60, "hours")

print("Number of missing paths: ", missing_paths)
percentage_missing_paths = missing_paths/total_subset_pairs * 100
print("Percentage of missing paths: " + str(percentage_missing_paths) + "%")


# ---------------------------------OUTPUT EXPORT---------------------------------

# Export B_matrix_weighted_dict to a pickle file

# filename = f"B_matrix_weighted_dict_{START_INDEX}_to_{END_INDEX-1}.pickle"

# print("Exporting B_matrix_weighted_dict to intermediate_files/B_matrix_weighted_dict.pickle...")

# with open(f'intermediate_files/{filename}', 'wb') as handle:
#     pickle.dump(B_matrix_weighted_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Export B_matrix_weighted_array to a pickle file

filename = f"B_matrix_weighted_array_{START_INDEX}_to_{END_INDEX-1}.pickle"

print("Exporting B_matrix_weighted_array to intermediate_files/B_matrix_weighted_array.pickle...")
with open(f'intermediate_files/{filename}', 'wb') as handle:
    pickle.dump(B_matrix_weighted_array, handle, protocol=pickle.HIGHEST_PROTOCOL)