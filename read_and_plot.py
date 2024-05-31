####################Read and PLot


import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd

from shapely.geometry import Point, Polygon, LineString,MultiLineString

with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

udf=gpd.read_file(r"ucla_large_centrality_roi.geojson")

blocks=gpd.read_file(r"blocks_epsg4326.geojson")


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



fig,ax=plt.subplots(1)
blocks.plot(ax=ax,color="green",alpha=0.3)
udf.plot(ax=ax,color="gray",linestyle="--")
nx.draw(G.to_undirected(),pos=nodes_coordinates_array[:,0:2],node_size=1,node_color="blue", with_labels = False)


