import pickle
import numpy as np
import networkx as nx
import pandas as pd
import time
from collections import defaultdict
import random
import multiprocessing
from functools import partial
import tqdm

# Node_to_Node_shortest_paths_dict = {}

def compute_shortest_path(pairing, all_shortest_paths_dict, G):
    node_1, node_2 = pairing

    if node_2 in all_shortest_paths_dict and node_1 in all_shortest_paths_dict[node_2]:
        shortest_path = all_shortest_paths_dict[node_2][node_1][::-1] # reverse the list
    else:
        if node_1 not in all_shortest_paths_dict:
            all_shortest_paths_dict[node_1] = nx.single_source_dijkstra_path(G, node_1, weight='weight')
        shortest_path = all_shortest_paths_dict[node_1].get(node_2)
    return (pairing, shortest_path)

if __name__ == '__main__':

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

    del B_matrix_sliced, B_matrix_str_sliced, nodes_coordinates_array

    print("Subsetting Node_to_Node_pairs...")
    # origin_nodes_list = [t[0] for t in Node_to_Node_pairs]
    # origin_nodes_list = list(dict.fromkeys(origin_nodes_list))
    origin_nodes_list = list(dict.fromkeys([t[0] for t in Node_to_Node_pairs]))

    random.seed(0)
    origin_nodes_list_subset = random.sample(origin_nodes_list, 10)
    Node_to_Node_pairs_subset = [t for t in Node_to_Node_pairs if t[0] in origin_nodes_list_subset]

    Node_to_Node_pairs_len = len(Node_to_Node_pairs)
    del Node_to_Node_pairs

    core_count = multiprocessing.cpu_count()
    print("This machine has ", core_count, "cores.")
    print("Using single source dijkstra method to find shortest paths for ", len(Node_to_Node_pairs_subset), "pairs of nodes...")

    # All_shortest_paths_dict = {}

    # time_start = time.time()

    # with multiprocessing.Manager() as manager:
    #     All_shortest_paths_dict = manager.dict()
    #     # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     # with multiprocessing.Pool(processes=2) as pool:
    #     with multiprocessing.Pool() as pool:
    #         results = pool.starmap(compute_shortest_path, [(pairing, All_shortest_paths_dict, G) for pairing in Node_to_Node_pairs_subset])

    # Node_to_Node_shortest_paths_dict = dict(results)

    # time_end = time.time()

    time_start = time.time()

    with multiprocessing.Manager() as manager:
        all_shortest_paths_dict = manager.dict()
        progress_counter = manager.Value('i', 0)

        compute_shortest_path_partial = partial(compute_shortest_path, all_shortest_paths_dict=all_shortest_paths_dict, G=G, progress_counter=progress_counter)

        total_pairs = len(Node_to_Node_pairs_subset)
        
        with multiprocessing.Pool(processes=core_count) as pool:
            results = pool.imap_unordered(compute_shortest_path_partial, Node_to_Node_pairs_subset)

            Node_to_Node_shortest_paths_dict = {}
            with tqdm.tqdm(total=total_pairs) as pbar:
                for result in results:
                    Node_to_Node_shortest_paths_dict.update([result])
                    pbar.update(progress_counter.value - pbar.n)

    time_end = time.time()

    print("Time taken for Single Source Method: ", (time_end - time_start)/60, "minutes")
    print("Estimated time for all pairs: ", (time_end - time_start)/60 * Node_to_Node_pairs_len/len(Node_to_Node_pairs_subset) / 60 / 24, "days")