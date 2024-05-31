import subprocess
import os
import time

# 00 - Required Packages & LODES Dataset Update

'''

Before running this script, make sure you are in the 'Importance Update' directory.

'''

required_packages = [
    'numpy',
    'matplotlib',
    'requests',
    'networkx',
    'scipy',
    'pandas',
    'geopandas',
    'shapely',
    'rtree',
    'scikit-gstat'
]


subprocess.run(['python', 'dataset_download/lodes_check_download.py'])


#######################################################################################################################

# Running each script in sequence:

# 01 -  Block Determination
'''
Description: This script determines which block each node is in.

Input:
    - blocks/tl_2020_06037_tabblock20.shp
    - graph_centrality_codes/nodes_edges_weighted.pickle

Output:
    - intermediate_files/Node_Block.pkl

'''
# Check if input files are present
if not os.path.isfile('blocks/tl_2020_06037_tabblock20.shp'):
    print('tl_2020_06037_tabblock20.shp is missing. Please contact SRILab for the shp file and its dependencies.')
    exit()
if not os.path.isfile('graph_centrality_codes/nodes_edges_weighted.pickle'):
    print('nodes_edges_weighted.pickle is missing.')
    exit()

subprocess.run(['python', '01_block_determination.py'])

# 02 - Block Node Pairing

# pull in file name as string from dataset_download/lodes_version.txt

with open('dataset_download/lodes_version.txt', 'r') as f:
    recent_lodes_version = f.read()


'''
Description: This script combines results from 01_block_determination.py with the Origin-Destination dataset to append job/worker quantity for each node.

Input:
    - lodes_od_data/ca_od_main_JT00_{yyyy} *recent_lodes_version*
    - intermediate_files/Node_Block.pkl

Output:
    - intermediate_files/Origin_Destination_Node_Added.pkl

'''


# Check if input files are present
if not os.path.isfile('lodes_od_data/' + recent_lodes_version):
    print(recent_lodes_version + ' is missing. Please run the data download file to optain the latest the LODES dataset.')
    exit()
if not os.path.isfile('intermediate_files/Node_Block.pkl'):
    print('Node_Block.pkl is missing. The 01_block_determination.py script should have produced this file.')
    exit()

subprocess.run(['python', '02_block_node_pairing.py'])

# 03 - Shortest Path
'''
Description: This script implements shortest path (single_source_dijkstra_path) algorithm to approximate optimal traveling path between two nodes.

Input:
    - graph_centrality_codes/nodes_edges_weighted.pickle
    - graph_centrality_codes/distance.pickle
    - intermediate_files/Origin_Destination_Node_Added.pkl

Output:
    - intermediate_files/Shortest_Path_Results.pkl

'''
# Check if input files are present
if not os.path.isfile('graph_centrality_codes/nodes_edges_weighted.pickle'):
    print('nodes_edges_weighted.pickle is missing. Please run the graph_centrality_codes/01_create_graph.py script.')
    exit()
if not os.path.isfile('graph_centrality_codes/distance.pickle'):
    print('distance.pickle is missing. Please run the graph_centrality_codes/01_create_graph.py script.')
    exit()
if not os.path.isfile('intermediate_files/Origin_Destination_Node_Added.pkl'):
    print('Origin_Destination_Node_Added.pkl is missing. The 02_block_node_pairing.py script should have produced this file.')
    exit()

subprocess.run(['python', '03_shortest_path.py'])

# 04 - Path Usage
'''
Description: This script computes road demand based on results from 03_shortest_path.py by couting pairing frequency of nodes.

Input:
    - graph_centrality_codes/nodes_edges_weighted.pickle
    - intermediate_files/Shortest_Path_Results.pkl

Output:
    - intermediate_files/B_matrix_weighted_updated.pickle

'''
# Check if input files are present
if not os.path.isfile('graph_centrality_codes/nodes_edges_weighted.pickle'):
    print('nodes_edges_weighted.pickle is missing.')
    exit()
if not os.path.isfile('intermediate_files/Shortest_Path_Results.pkl'):
    print('Shortest_Path_Results.pkl is missing. The 03_shortest_path.py script should have produced this file.')
    exit()

subprocess.run(['python', '04_path_usage.py'])

# 05 - LODES to UDF

'''
Description: This script updates UDF from S3 with LODES data.

Input:
    - intermediate_files/B_matrix_weighted_updated.pickle
    - udf/hillside_inventory_LA_centrality_full_new_evacmidnorth_lodes.geojson

Output:
    - udf/hillside_inventory_LA_centrality_full_new_evacmidnorth_lodes.geojson

'''




# 06 - Kriging Update
'''
Description: This script updates missing road demand values using kriging.

Input:
    - udf/hillside_inventory_LA_centrality_full_new_evacmidnorth_lodes.geojson

Output:
    - udf/hillside_inventory_LA_centrality_full_new_evacmidnorth_lodes_kriging.geojson

'''
# Check if input files are present
if not os.path.isfile('udf/hillside_inventory_LA_centrality_full_new_evacmidnorth_lodes.geojson'):
    print('hillside_inventory_LA_centrality_full_new_evacmidnorth_lodes.geojson is missing. Please contact SRILab for the geojson file and its dependencies.')
    exit()


