import pickle
import time
import igraph as ig
from tqdm import tqdm

print("Importing nodes_edges_ucla_access.pickle...")
with open(r'nodes_edges_ucla_access.pickle', 'rb') as handle:
# with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

print("Creating graph using B_matrix_sliced...")
time_start = time.time()

g = ig.Graph(directed=True)

vertices = set(B_matrix_sliced[:, 0].astype(int)).union(set(B_matrix_sliced[:, 1].astype(int)))
g.add_vertices(list(vertices))

for i in tqdm(range(len(B_matrix_sliced)), desc="Adding edges"):
    oneway = B_matrix_str_sliced[i, 2]
    edge_attrs = {
        "sect_id": str(B_matrix_str_sliced[i, 3]),
        "distance": B_matrix_sliced[i, 2],
        "speed": B_matrix_sliced[i, 3],
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

print("Exporting graph to intermediate_files/igraph.pickle...")
with open(r'intermediate_files/igraph.pickle', 'wb') as handle:
    pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)

del B_matrix_sliced, B_matrix_str_sliced, nodes_coordinates_array