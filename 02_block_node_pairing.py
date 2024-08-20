print("Importing libraries...")

import pickle
import numpy as np
import pandas as pd
import pickle
from itertools import product

# import Node_Block.pkl

print("Loading files...")
with open(r'intermediate_files/Node_Block.pkl', 'rb') as f:
    Node_Block = pickle.load(f)

Origin_Destination = pd.read_csv('lodes_od_data/ca_od_main_JT00_2021.csv')

print("Number of unique nodes: ", len(Node_Block['Node'].unique()))

Initial_Origin_Destination_Count = len(Origin_Destination)

# Create dictionary for block and node pairings (identified from BlockDetermination.ipynb)
print("Initializing dictionary...")
block_node_dict = {}

print("Mapping blocks to nodes...")
for _, row in Node_Block.iterrows():
    node_id = row['Node']
    block_id = row['BlockNumber']
    # block_coord = row['BlockBoundary']
    if int(block_id) not in block_node_dict:
        block_node_dict[int(block_id)] = []
    block_node_dict[int(block_id)].append(node_id)

Origin_Destination_Node_Added = Origin_Destination.copy()

Origin_Destination_Node_Added = Origin_Destination_Node_Added.drop(columns=['createdate'])

# Map nodes to work and home blocks

print("Mapping nodes to work and home blocks...")

Origin_Destination_Node_Added['w_node_id'] = Origin_Destination_Node_Added['w_geocode'].map(block_node_dict)
Origin_Destination_Node_Added['h_node_id'] = Origin_Destination_Node_Added['h_geocode'].map(block_node_dict)

# Remove rows where the node columns are missing (i.e. the block is not in the node dictionary)

print("Removing rows where the node columns are missing...")

Origin_Destination_Node_Added = Origin_Destination_Node_Added[Origin_Destination_Node_Added['w_node_id'].notnull()]
Origin_Destination_Node_Added = Origin_Destination_Node_Added[Origin_Destination_Node_Added['h_node_id'].notnull()]

Origin_Destination_Node_Added_array = Origin_Destination_Node_Added.to_numpy()

# extract only columns 0 and 1 from Origin_Destination_Node_Added_array
Block_to_Block_Pairs = Origin_Destination_Node_Added_array[:, 0:2]

# check if each row in Block_to_Block_Pairs array is a unique pair

print('Checking if each row in Block_to_Block_Pairs array is a unique pair...')

Block_to_Block_Pairs = Block_to_Block_Pairs.astype(float)  # or int, depending on your data
unique_rows = np.unique(Block_to_Block_Pairs, axis=0)
is_all_unique = len(Block_to_Block_Pairs) == len(unique_rows)

print('All Block_to_Block_Pairs are unique: ', is_all_unique)

# extract only columns 2 to 11 from Origin_Destination_Node_Added_array
LODES_info_to_be_adjusted = Origin_Destination_Node_Added_array[:, 2:12]

# Origin_Destination_Node_Added_array[:, 12:14]

block_pairing_count_list = []

for nodes in Origin_Destination_Node_Added_array[:, 12:14]:
    block_pairing_count = len(nodes[0]) * len(nodes[1])
    block_pairing_count_list.append(block_pairing_count)

# Adjust LODES info by dividing by the number of block pairings
print('Adjusting LODES info by dividing by the number of block pairings...')
for i in range(len(LODES_info_to_be_adjusted)):
    LODES_info_to_be_adjusted[i] = LODES_info_to_be_adjusted[i] / block_pairing_count_list[i]

LODES_adjusted = LODES_info_to_be_adjusted

Node_to_Node_lists = Origin_Destination_Node_Added_array[:, 12:14]

Node_to_Node_pairs = []

print('Creating Node_to_Node_pairs...')
for nodes in Node_to_Node_lists:
    Node_to_Node_pairs.extend(product(nodes[0], nodes[1]))

# print(Node_to_Node_pairs)

# Dump Node_to_Node_pairs as a pickle file
print('Dumping Node_to_Node_pairs as a pickle file...')
with open('intermediate_files/Node_to_Node_pairs.pickle', 'wb') as handle:
    pickle.dump(Node_to_Node_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Dump Origin_Destination_Node_Added_array as a pickle file
print('Dumping Origin_Destination_Node_Added_array as a pickle file...')
with open('intermediate_files/Origin_Destination_Node_Added_array.pickle', 'wb') as handle:
    pickle.dump(Origin_Destination_Node_Added_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Export block_pairing_count_list to a pickle file
print('Exporting block_pairing_count_list to a pickle file...')
with open('intermediate_files/block_pairing_count_list.pickle', 'wb') as handle:
    pickle.dump(block_pairing_count_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Export LODES_adjusted to a pickle file
print('Exporting Block_to_Block_Pairs, LODES_adjusted, block_pairing_count_list to a pickle file together...')
with open('intermediate_files/LODES_adjusted_block_pairing_count_list.pickle', 'wb') as handle:
    pickle.dump([Block_to_Block_Pairs, LODES_adjusted, block_pairing_count_list], handle, protocol=pickle.HIGHEST_PROTOCOL)