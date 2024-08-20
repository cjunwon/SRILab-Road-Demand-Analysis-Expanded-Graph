print("Importing libraries...")

import pickle
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import Point
from rtree import index
from tqdm import tqdm
from shapely.geometry import Point

# with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
#     B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle) <-- OLD

print("Loading data...")

with open(r'nodes_edges_ucla_access.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

# blocks=gpd.read_file(r"blocks_epsg4326.geojson") <-- OLD

blocks=gpd.read_file(r"blocks_epsg3857.geojson")

# Subset block data

print("Subsetting block data...")

blocknumbers = np.array(blocks["GEOID20"],dtype=str)
blockboundaries = np.array(blocks["geometry"])

print("Separating node coordinates...")

Hillside_NodeCoordinate = nodes_coordinates_array[:,0:2]

# Create a dictionary mapping block boundaries to their indices

print("Creating dictionary mapping block boundaries to their indices...")

block_to_index = {boundary: i for i, boundary in enumerate(blockboundaries)}

# Use a set to keep track of identified nodes
identified_nodes = set()

# Use a DataFrame for Node_Block
Node_Block = pd.DataFrame(columns=['Node', 'BlockNumber', 'BlockBoundary'])

# Create an R-tree index

print("Creating R-tree index...")

idx = index.Index()
for i, boundary in enumerate(blockboundaries):
    idx.insert(i, boundary.bounds)

# Identify nodes that are in a block using R-tree

print("Identifying nodes that are in a block using R-tree...")

for i in tqdm(range(len(Hillside_NodeCoordinate)), desc="Filtering nodes"):
    coord = Hillside_NodeCoordinate[i]
    possible_blocks_indices = list(idx.intersection((coord[0], coord[1], coord[0], coord[1])))
    for block_index in possible_blocks_indices:
        block = blockboundaries[block_index]
        if block.contains(Point(coord[0], coord[1])):
            new_row = pd.DataFrame({'Node': [i], 'BlockNumber': [blocknumbers[block_index]], 'BlockBoundary': [block]})
            Node_Block = pd.concat([Node_Block, new_row], ignore_index=True)
            identified_nodes.add(i)

# Identify nodes that are not in any block
unidentified_nodes = set(range(len(Hillside_NodeCoordinate))) - identified_nodes

# Identify the closest block for each node that is not in any block

print("Identifying the closest block for each node that is not in any block...")

for i in tqdm(unidentified_nodes, desc="Identifying closest blocks"):
    closest_block = min(blockboundaries, key=lambda x: x.distance(Point(Hillside_NodeCoordinate[i][0], Hillside_NodeCoordinate[i][1])))
    new_row = pd.DataFrame({'Node': [i], 'BlockNumber': [blocknumbers[block_to_index[closest_block]]], 'BlockBoundary': [closest_block]})
    Node_Block = pd.concat([Node_Block, new_row], ignore_index=True)

# Total nodes identified

print("Identification complete.")

total_nodes_identified = len(Node_Block)
print("Total nodes identified: " + str(total_nodes_identified))

# Save Node_Block as a pickle file

print("Saving Node_Block as a pickle file...")

with open(r'intermediate_files/Node_Block.pkl', 'wb') as f:
    pickle.dump(Node_Block, f)