import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

G_nx = nx.Graph()
edges = [
    (0, 1, 2),
    (0, 2, 0),
    (1, 2, 3),
    (1, 3, 0),
    (2, 3, 1),
    (2, 4, 0),
    (3, 4, 2),
    (3, 5, 0),
    (4, 5, 1)
]
G_nx.add_weighted_edges_from(edges)

all_paths_nx = list(nx.all_shortest_paths(G_nx, source=0, target=5, weight='weight'))
print("NetworkX all shortest paths:", all_paths_nx)

# NetworkX all shortest paths: [[0, 2, 4, 5], [0, 2, 4, 5], [0, 2, 3, 5], [0, 2, 3, 5]]

G_ig = ig.Graph(directed=False)
G_ig.add_vertices(6)
edges = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 4),
    (3, 5),
    (4, 5)
]
weights = [2, 0, 3, 0, 1, 0, 2, 0, 1]
G_ig.add_edges(edges)
G_ig.es['weight'] = weights

all_paths_ig = G_ig.get_all_shortest_paths(0, to=5, weights='weight')
print("igraph all shortest paths:", all_paths_ig)

# igraph all shortest paths: [[0, 2, 4, 5]]

# Visualize Graph
pos = nx.shell_layout(G_nx)
nx.draw_networkx_nodes(G_nx, pos, node_size=700)
nx.draw_networkx_labels(G_nx, pos, font_size=20, font_family="sans-serif")
nx.draw_networkx_edges(G_nx, pos, edgelist=G_nx.edges(), width=2)
edge_labels = nx.get_edge_attributes(G_nx, 'weight')
nx.draw_networkx_edge_labels(G_nx, pos, edge_labels=edge_labels)

plt.show()