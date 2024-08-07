import pickle
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

with open('intermediate_files/igraph.pickle', 'rb') as f:
    g = pickle.load(f)

with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)


with open(r'intermediate_files/Node_to_Node_pairs.pickle', 'rb') as handle:
    Node_to_Node_pairs = pickle.load(handle)


# find all tuple pairs in Node_to_Node_pairs where the first element is 118106
Node_to_Node_pairs_118106 = [pair for pair in Node_to_Node_pairs if pair[0] == 118106]


Node_to_Node_pairs_118106_destinations = [pair[1] for pair in Node_to_Node_pairs_118106]

shortest_paths = g.get_shortest_paths(118106, to=Node_to_Node_pairs_118106_destinations, weights='weight', output="vpath", algorithm="dijkstra")

g_networkx = g.to_networkx()


Node_to_Node_pairs_118106_destinations


# Extract nodes and edges from the shortest paths
nodes_in_paths = set()
edges_in_paths = set()
for path in shortest_paths:
    nodes_in_paths.update(path)
    edges_in_paths.update(zip(path[:-1], path[1:]))

# Create a larger figure
fig, ax = plt.subplots(1, figsize=(100, 90))

# Draw the entire graph with a base style
pos = {i: (nodes_coordinates_array[i, 0], nodes_coordinates_array[i, 1]) for i in range(len(nodes_coordinates_array))}
nx.draw(g_networkx.to_undirected(), pos=pos, node_size=1, node_color="blue", with_labels=False, ax=ax)

# Highlight the shortest paths
nx.draw_networkx_nodes(g_networkx, pos, nodelist=nodes_in_paths, node_color='red', node_size=10, ax=ax)
nx.draw_networkx_edges(g_networkx, pos, edgelist=edges_in_paths, edge_color='red', width=2, ax=ax)

# Highlight the node 118106 as a green star
nx.draw_networkx_nodes(g_networkx, pos, nodelist=[118106], node_color='green', node_shape='*', node_size=5000, ax=ax)

# Highlight the nodes in Node_to_Node_pairs_118106_destinations as yellow stars
nx.draw_networkx_nodes(g_networkx, pos, nodelist=Node_to_Node_pairs_118106_destinations, node_color='yellow', node_shape='*', node_size=5000,  edgecolors='black', linewidths=5, ax=ax)

# Show the plot
plt.show()


# Find the shortest paths using NetworkX's Dijkstra algorithm
shortest_paths_networkx = []
for destination in Node_to_Node_pairs_118106_destinations:
    path = nx.shortest_path(g_networkx, source=118106, target=destination, weight='weight', method='dijkstra')
    shortest_paths_networkx.append(path)


len(shortest_paths_networkx)


# Extract nodes and edges from the shortest paths
nodes_in_paths = set()
edges_in_paths = set()
for path in shortest_paths_networkx:
    nodes_in_paths.update(path)
    edges_in_paths.update(zip(path[:-1], path[1:]))

# Create a larger figure
fig, ax = plt.subplots(1, figsize=(100, 90))

# Draw the entire graph with a base style
pos = {i: (nodes_coordinates_array[i, 0], nodes_coordinates_array[i, 1]) for i in range(len(nodes_coordinates_array))}
nx.draw(g_networkx.to_undirected(), pos=pos, node_size=1, node_color="blue", with_labels=False, ax=ax)

# Highlight the shortest paths
nx.draw_networkx_nodes(g_networkx, pos, nodelist=nodes_in_paths, node_color='red', node_size=10, ax=ax)
nx.draw_networkx_edges(g_networkx, pos, edgelist=edges_in_paths, edge_color='red', width=2, ax=ax)

# Highlight the node 118106 as a green star
nx.draw_networkx_nodes(g_networkx, pos, nodelist=[118106], node_color='green', node_shape='*', node_size=5000, ax=ax)

# Highlight the nodes in Node_to_Node_pairs_118106_destinations as yellow stars
nx.draw_networkx_nodes(g_networkx, pos, nodelist=Node_to_Node_pairs_118106_destinations, node_color='yellow', node_shape='*', node_size=5000,  edgecolors='black', linewidths=5, ax=ax)

# Show the plot
plt.show()


