import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString,MultiLineString
import shapely.geometry
import pandas as pd
import pickle
import fiona
from shapely.ops import linemerge
import pyproj
from functools import partial
from shapely.ops import transform
from geopy import distance
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
from collections import defaultdict

import pandas as pd
# import utm
from scipy.spatial import KDTree


import requests as req
import re

hillside_layer=gpd.read_file("D:/fire_paper/network/filestouse/1/hillside_inventory_LA_centrality_full_new_evacmidnorth.geojson")

uclafile=gpd.read_file("D:/fire_paper/network/filestouse/1/ucla_streets_latlon.geojson")
ordinance=gpd.read_file("D:/fire_paper/network/filestouse/1/Hillside_Ordinance.geojson")

calscreen=gpd.read_file("D:/fire_paper/network/filestouse/1/CalEnviroscreen_4.0.geojson")

# medianincome=gpd.read_file("D:/eq_paper_graph/filestouse/equity_data/Median_Income_and_AMI_(census_tract).geojson")

# novehicle=gpd.read_file("D:/eq_paper_graph/filestouse/equity_data/Without_Vehicle_(census_tract).geojson")


population=np.array(calscreen["Population"])




speed=np.array(uclafile["SPEED"],dtype=float)
drivemin=np.array(uclafile["drive_mins"],dtype=float)
walkmin=np.array(uclafile["walk_mins"],dtype=float)

idx=(speed>1) & (drivemin<walkmin)

geometries=uclafile["geometry"]
centroids_lat=np.zeros((len(uclafile),1))
centroids_lon=np.zeros((len(uclafile),1))

for i,g in enumerate(geometries):
    centroids_lat[i]=np.array(g.centroid.coords)[0][1]
    centroids_lon[i]=np.array(g.centroid.coords)[0][0]


fig,ax=plt.subplots(1)
uclafile.plot(ax=ax,color="gray")
ordinance.plot(ax=ax,color="red",alpha=0.99)
# extract_polygon.plot()



# nx.draw(G,pos=node_coordinates_weighted[:,0:2],node_size=1,node_color="red")
# Function to print mouse click event coordinates
def onclick(event):
    print([event.xdata, event.ydata])

# Bind the button_press_event with the onclick() method
fig.canvas.mpl_connect('button_press_event', onclick)

# Display the plot
plt.show()




# roi_sparse=Polygon([[-118.67588076315292, 34.36700029018994],
# [-118.14097916475816, 34.366015082869914],
# [-118.14097916475816, 33.67735516617017],
# [-118.67706943337157, 33.676369958850145]])


roi_sparse=Polygon([[-118.68043253043444, 34.50929226141839],
[-118.20339839330889, 34.50929226141839],
[-118.20339839330889, 33.69381911535673],
[-118.68043253043444, 33.69381911535673]])





# big_pol= ordinance["geometry"][0]




# big_pol_buffer = big_pol.buffer(0.004)


# extra_pol= Polygon([(-118.41934,34.11299), (-118.41237, 34.11299), (-118.41237, 34.10409), (-118.41934, 34.10409)])

# extra_pol2=Polygon([[-118.55224132326056, 34.04146560349692],
# [-118.54555231641692, 34.049483864628485],
# [-118.51752219250068, 34.05349299519426],
# [-118.50713830568627, 34.060245215094525],
# [-118.49853815403014, 34.06351582160872],
# [-118.49739146714266, 34.057607629195985],
# [-118.49739146714266, 34.0561305810928],
# [-118.4990477926468, 34.054759036425565],
# [-118.49879297333848, 34.053915008938034],
# [-118.4998122505718, 34.052226953962965],
# [-118.50688348637793, 34.04299540331807],
# [-118.51331767391325, 34.03434412157086],
# [-118.51777701180902, 34.02996572897928],
# [-118.52045261454647, 34.0297547221074],
# [-118.52586752484846, 34.03228680457],
# [-118.53236541721087, 34.03476613531463],
# [-118.54026481576908, 34.03967204508591],
# [-118.5477819853648, 34.04030506570156],
# [-118.55523545013342, 34.04093808631721]])


# extract_polygon=Polygon([[-118.5729436813221, 34.12274207687375],
#                         [-118.56281964154134, 34.11299670740343],
#                         [-118.58091636264945, 34.0785211530407],
#                         [-118.5804101606604, 34.06039267004753],
#                         [-118.611921234478, 34.06217408167113],
#                         [-118.60939022453282, 34.12567616660675]])







idx=np.zeros(len(uclafile))

ucla_geo=uclafile["geometry"]

for i,geo in enumerate(ucla_geo):
    print(i)
    if geo.intersects(roi_sparse):
        idx[i]=1
    
idx=(speed>1) & (drivemin<walkmin) & (idx==1)



udf=uclafile[idx]

fig,ax=plt.subplots(1)

uclafile.plot(ax=ax,color="gray",linestyle="--")
ordinance.plot(ax=ax,color="red",alpha=0.3)
udf.plot(ax=ax,color="black")




udf.to_file(r"C:\Users\srila\Desktop\task_3_junwon\ucla_large_centrality_roi.geojson", driver='GeoJSON')


udf=gpd.read_file(r"C:\Users\srila\Desktop\task_3_junwon\ucla_large_centrality_roi.geojson")



fig,ax=plt.subplots(1)
udf.plot(ax=ax,color="red")
# plt.scatter(coordinates_start[0],coordinates_start[1],color="blue")
# plt.scatter(coordinates_end[0],coordinates_end[1],color="blue")

geometries=udf["geometry"].reset_index(drop=True)
sect_ids=udf["ObjectID"].reset_index(drop=True)





# ##########################################ADD WIDTH

# hillside_layer=gpd.read_file("C:/Users/srila/Desktop/accessibility_study/hillside_inventory_LA_centrality_full_new_evacmidnorth_EPGS3857.geojson")
# udf=gpd.read_file(r"C:/Users/srila/Desktop/accessibility_study/uclaaccess_roi_EPGS3857.geojson")



# fig,ax=plt.subplots(1)

# udf.plot(ax=ax,color="red",linestyle="--")
# hillside_layer.plot(ax=ax,color="black")






# geometries=hillside_layer["geometry"]
# sect_ids=hillside_layer["SECT_ID"]

# points=np.zeros([10000000,2])
# labels=np.zeros([10000000,1])
# counter=0

# for i in range(len(geometries)):
#     print(i)
#     if type(geometries[i])==MultiLineString:
#         for z in range(len(geometries[i].geoms)):
#             line=np.array(geometries[i].geoms[z].coords) 
#             sect_id=sect_ids[i]
#             for j in range(1,line.shape[0]):
#                 point1=line[j-1,:]
#                 point2=line[j,:]
#                 # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#                 #print(dist)
#                 dist=math.dist(point1, point2)
#                 lats=np.linspace(point1[0],point2[0],int(dist))
#                 lons=np.linspace(point1[1],point2[1],int(dist))
#                 new_points=np.transpose(np.array([lats,lons]))
#                 points[counter:counter+new_points.shape[0],:]=new_points
#                 labels[counter:counter+new_points.shape[0]]=sect_id
#                 counter=counter+new_points.shape[0]
#     else:
#         line=np.array(geometries[i].coords) 
#         sect_id=sect_ids[i]
#         for j in range(1,line.shape[0]):
#             point1=line[j-1,:]
#             point2=line[j,:]
#             # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#             #print(dist)
#             dist=math.dist(point1, point2)
#             lats=np.linspace(point1[0],point2[0],int(dist))
#             lons=np.linspace(point1[1],point2[1],int(dist))
#             new_points=np.transpose(np.array([lats,lons]))
#             points[counter:counter+new_points.shape[0],:]=new_points
#             labels[counter:counter+new_points.shape[0]]=sect_id
#             counter=counter+new_points.shape[0]

# points_hillside=points[0:counter,:]
# labels_hillside=labels[0:counter]
# # kdpoints=KDTree(points)
# # dumpdata=[points,labels]




# geometries=udf["geometry"].reset_index(drop=True)
# sect_ids=udf["ObjectID"].reset_index(drop=True)

# points=np.zeros([60000000,2])
# labels=np.zeros([60000000,1])
# counter=0

# for i in range(len(geometries)):
#     print(i)
#     if type(geometries[i])==MultiLineString:
#         for z in range(len(geometries[i].geoms)):
#             line=np.array(geometries[i].geoms[z].coords) 
#             sect_id=sect_ids[i]
#             for j in range(1,line.shape[0]):
#                 point1=line[j-1,:]
#                 point2=line[j,:]
#                 # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#                 #print(dist)
#                 dist=math.dist(point1, point2)
#                 lats=np.linspace(point1[0],point2[0],int(dist))
#                 lons=np.linspace(point1[1],point2[1],int(dist))
#                 new_points=np.transpose(np.array([lats,lons]))
#                 points[counter:counter+new_points.shape[0],:]=new_points
#                 labels[counter:counter+new_points.shape[0]]=sect_id
#                 counter=counter+new_points.shape[0]
#     else:
#         line=np.array(geometries[i].coords) 
#         sect_id=sect_ids[i]
#         for j in range(1,line.shape[0]):
#             point1=line[j-1,:]
#             point2=line[j,:]
#             # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#             #print(dist)
#             dist=math.dist(point1, point2)
#             lats=np.linspace(point1[0],point2[0],int(dist))
#             lons=np.linspace(point1[1],point2[1],int(dist))
#             new_points=np.transpose(np.array([lats,lons]))
#             points[counter:counter+new_points.shape[0],:]=new_points
#             labels[counter:counter+new_points.shape[0]]=sect_id
#             counter=counter+new_points.shape[0]

# pointsu=points[0:counter,:]
# labelsu=labels[0:counter]





# kdpoints=KDTree(points_hillside)




# allocationsid=np.zeros_like(labelsu)

# for i in range(len(labelsu)):
#     print(i)
#     point=pointsu[i,:]
    
#     distance, closest_line_index = kdpoints.query(point,k=1)
    
#     if distance<15:
#         allocationsid[i]=labels_hillside[closest_line_index]






# unique_obj_ids=np.unique(labelsu)

# sidobjid=np.zeros([len(unique_obj_ids),2])

# list_of_sectids=[]



# from collections import Counter
# for i in range(len(unique_obj_ids)):
#     print(i)
#     sid=unique_obj_ids[i]
#     sidobjid[i,0]=sid
#     allocs=allocationsid[labelsu==sid]
#     allocsnonzero=allocs[allocs!=0]
    
#     if len(allocsnonzero)>=0.5*len(allocs):
#         values, counts = np.unique(allocs, return_counts=True)

#         # Find the value with the highest count
#         most_common_value = values[np.argmax(counts)]
#         sidobjid[i,1]=most_common_value
#         value_counts = Counter(allocs)
#         values_more_than_ten = [value for value, count in value_counts.items() if count > 10]
#         list_of_sectids.append(values_more_than_ten)
#     else:
#         list_of_sectids.append([0])
# list_of_sectids_filtered=[]

# for l in list_of_sectids:
#     filtered_list = [num for num in l if num != 0]
#     list_of_sectids_filtered.append(filtered_list)
    


# objids_udf=np.array(udf["ObjectID"],dtype=int)

# sids_hillside=np.array(hillside_layer["SECT_ID"],dtype=int)
# widths_hillside=np.array(hillside_layer["ST_WIDTH"],dtype=float)
# widths_udf=[]


# errorcount=0
# for i in range(len(objids_udf)):
#     objid=objids_udf[i]
#     try:
#         sididx=np.where(sidobjid[:,0]==objid)[0][0]
#         sid=sidobjid[sididx,1]
#         if sid!=0:
#             wid=np.where(sids_hillside==sid)[0][0]
#             widths_udf.append(widths_hillside[wid])
#         else:
#             widths_udf.append(999)
#     except IndexError:
#         errorcount+=1
#         widths_udf.append(999)
        
# udf["ST_WIDTH"]=widths_udf




# fig,ax=plt.subplots(1)

# udf.plot(ax=ax,color="gray",linestyle="--")
# udf[np.array(widths_udf)<20].plot(ax=ax,color="red",linestyle="--")
# hillside_layer[widths_hillside<20].plot(ax=ax,color="green")
# # plt.scatter(np.array(nodes_coordinates)[:,0],np.array(nodes_coordinates)[:,1],s=4)




# fig,ax=plt.subplots(1)

# hillside_layer.plot(ax=ax,color="gray",linestyle="--")

# # hillside_layer.plot(ax=ax,color="gray",linestyle="--")
# hillside_layer[np.array(widths_hillside)<20].plot(ax=ax,color="red",linestyle="--")





# udf.to_file(r"C:\Users\srila\Desktop\accessibility_study\uclaaccess_roi_widths.geojson", driver='GeoJSON')


# udf=gpd.read_file(r"C:\Users\srila\Desktop\accessibility_study\uclaaccess_roi_widths.geojson")








#######################################################################################################Create objidsid



# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import math
# import numpy as np
# from shapely.geometry import Point, Polygon, LineString,MultiLineString
# import shapely.geometry
# import pandas as pd
# import pickle
# import fiona
# from shapely.ops import linemerge
# import pyproj
# from functools import partial
# from shapely.ops import transform
# from geopy import distance
# # gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# from collections import defaultdict

# import pandas as pd
# # import utm
# from scipy.spatial import KDTree


# import requests as req
# import re



# hillside_layer=gpd.read_file("C:/Users/srila/Desktop/accessibility_study/hillside_inventory_LA_centrality_full_new_evacmidnorth_EPGS3857.geojson")
# udf=gpd.read_file(r"C:/Users/srila/Desktop/accessibility_study/uclaaccess_roi_EPGS3857.geojson")






# geometries=hillside_layer["geometry"]
# sect_ids=hillside_layer["SECT_ID"]

# points=np.zeros([10000000,2])
# labels=np.zeros([10000000,1])
# counter=0

# for i in range(len(geometries)):
#     print(i)
#     if type(geometries[i])==MultiLineString:
#         for z in range(len(geometries[i].geoms)):
#             line=np.array(geometries[i].geoms[z].coords) 
#             sect_id=sect_ids[i]
#             for j in range(1,line.shape[0]):
#                 point1=line[j-1,:]
#                 point2=line[j,:]
#                 # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#                 #print(dist)
#                 dist=math.dist(point1, point2)
#                 lats=np.linspace(point1[0],point2[0],int(dist))
#                 lons=np.linspace(point1[1],point2[1],int(dist))
#                 new_points=np.transpose(np.array([lats,lons]))
#                 points[counter:counter+new_points.shape[0],:]=new_points
#                 labels[counter:counter+new_points.shape[0]]=sect_id
#                 counter=counter+new_points.shape[0]
#     else:
#         line=np.array(geometries[i].coords) 
#         sect_id=sect_ids[i]
#         for j in range(1,line.shape[0]):
#             point1=line[j-1,:]
#             point2=line[j,:]
#             # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#             #print(dist)
#             dist=math.dist(point1, point2)
#             lats=np.linspace(point1[0],point2[0],int(dist))
#             lons=np.linspace(point1[1],point2[1],int(dist))
#             new_points=np.transpose(np.array([lats,lons]))
#             points[counter:counter+new_points.shape[0],:]=new_points
#             labels[counter:counter+new_points.shape[0]]=sect_id
#             counter=counter+new_points.shape[0]

# points_hillside=points[0:counter,:]
# labels_hillside=labels[0:counter]
# # kdpoints=KDTree(points)
# # dumpdata=[points,labels]




# geometries=udf["geometry"].reset_index(drop=True)
# sect_ids=udf["ObjectID"].reset_index(drop=True)

# points=np.zeros([60000000,2])
# labels=np.zeros([60000000,1])
# counter=0

# for i in range(len(geometries)):
#     print(i)
#     if type(geometries[i])==MultiLineString:
#         for z in range(len(geometries[i].geoms)):
#             line=np.array(geometries[i].geoms[z].coords) 
#             sect_id=sect_ids[i]
#             for j in range(1,line.shape[0]):
#                 point1=line[j-1,:]
#                 point2=line[j,:]
#                 # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#                 #print(dist)
#                 dist=math.dist(point1, point2)
#                 lats=np.linspace(point1[0],point2[0],int(dist))
#                 lons=np.linspace(point1[1],point2[1],int(dist))
#                 new_points=np.transpose(np.array([lats,lons]))
#                 points[counter:counter+new_points.shape[0],:]=new_points
#                 labels[counter:counter+new_points.shape[0]]=sect_id
#                 counter=counter+new_points.shape[0]
#     else:
#         line=np.array(geometries[i].coords) 
#         sect_id=sect_ids[i]
#         for j in range(1,line.shape[0]):
#             point1=line[j-1,:]
#             point2=line[j,:]
#             # dist=np.ceil(distance.distance(point1[::-1], point2[::-1]).meters)
#             #print(dist)
#             dist=math.dist(point1, point2)
#             lats=np.linspace(point1[0],point2[0],int(dist))
#             lons=np.linspace(point1[1],point2[1],int(dist))
#             new_points=np.transpose(np.array([lats,lons]))
#             points[counter:counter+new_points.shape[0],:]=new_points
#             labels[counter:counter+new_points.shape[0]]=sect_id
#             counter=counter+new_points.shape[0]

# pointsu=points[0:counter,:]
# labelsu=labels[0:counter]





# kdpoints=KDTree(pointsu)




# allocationobjid=np.zeros_like(labels_hillside)

# for i in range(len(labels_hillside)):
#     print(i)
#     point=points_hillside[i,:]
    
#     distance, closest_line_index = kdpoints.query(point,k=1)
    
#     if distance<15:
#         allocationobjid[i]=labelsu[closest_line_index]






# unique_sect_ids=np.unique(labels_hillside)

# sidobjid=np.zeros([len(unique_sect_ids),2])

# list_of_objids=[]



# from collections import Counter
# for i in range(len(unique_sect_ids)):
#     print(i)
#     sid=unique_sect_ids[i]
#     sidobjid[i,0]=sid
#     allocs=allocationobjid[labels_hillside==sid]
#     allocsnonzero=allocs[allocs!=0]
    
#     if len(allocsnonzero)>=0.5*len(allocs):
#         values, counts = np.unique(allocs, return_counts=True)

#         # Find the value with the highest count
#         most_common_value = values[np.argmax(counts)]
#         sidobjid[i,1]=most_common_value
#         value_counts = Counter(allocs)
#         values_more_than_ten = [value for value, count in value_counts.items() if count > 10]
#         list_of_objids.append(values_more_than_ten)
#     else:
#         list_of_objids.append([0])
# list_of_objids_filtered=[]

# for l in list_of_objids:
#     filtered_list = [num for num in l if num != 0]
#     list_of_objids_filtered.append(filtered_list)
    



# with open(r'D:\fire_paper\network\filestouse\allocation_ucla_access.pickle', 'wb') as handle:
#     pickle.dump([sidobjid,list_of_objids_filtered], handle, protocol=pickle.HIGHEST_PROTOCOL)






#################################################################################





import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString,MultiLineString
import shapely.geometry
import pandas as pd
import pickle
import fiona
from shapely.ops import linemerge
import pyproj
from functools import partial
from shapely.ops import transform
from geopy import distance
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
from collections import defaultdict

import pandas as pd
# import utm
from scipy.spatial import KDTree


import requests as req
import re



udf=gpd.read_file(r"C:/Users/srila/Desktop/task_3_junwon/ucla_large_centrality_roi.geojson")


udf = udf.reset_index(drop=True)



geometries=udf["geometry"]



points=np.zeros([30000000,2])
labels=np.zeros([30000000,1])
counter=0

for i in range(len(geometries)):
    print(i)
    if type(geometries[i])==MultiLineString:
        for z in range(len(geometries[i].geoms)):
            line=np.array(geometries[i].geoms[z].coords) 
            sect_id=sect_ids[i]
            points[counter:counter+line.shape[0],:]=line

            labels[counter:counter+line.shape[0]]=sect_id
            counter=counter+line.shape[0]
    else:
        line=np.array(geometries[i].coords) 
        sect_id=sect_ids[i]
        points[counter:counter+line.shape[0],:]=line

        labels[counter:counter+line.shape[0]]=sect_id
        counter=counter+line.shape[0]
       

points=points[0:counter,:]
labels=labels[0:counter]

unq, unq_idx, unq_cnt = np.unique(points,return_inverse=True, axis=0, return_counts=True)

nodes_labels=[]
nodes_coordinates=[]

for i in range(len(unq_cnt)):
    print(i)
    cnt=unq_cnt[i]
    
    if cnt>1:
        coords=unq[i,:]
        result=np.unique(np.where(points[:,0:2]==unq[i,:])[0])
        result_labels=labels[np.unique(np.where(points[:,0:2]==unq[i,:])[0])]
        nodes_labels.append(result_labels)
        nodes_coordinates.append(coords)

fig,ax=plt.subplots(1)
udf.plot(ax=ax,color="red")
plt.scatter(np.array(nodes_coordinates)[:,0],np.array(nodes_coordinates)[:,1])





point_array=np.array(nodes_coordinates)

nodes_labels_orig=nodes_labels
nodes_coordinates_orig=nodes_coordinates

for i in range(len(geometries)):
    print(i)
    if type(geometries[i])==MultiLineString:
        for z in range(len(geometries[i].geoms)):
            line=np.array(geometries[i].geoms[z].coords) 
            start=line[0,:]
            end=line[-1,:]
            sect_id=sect_ids[i]
            if len(np.where(point_array[:,0:2]==start)[0])==0:
                nodes_coordinates.append(start)
                nodes_labels.append(sect_id)
            if len(np.where(point_array[:,0:2]==end)[0])==0:
                nodes_coordinates.append(end)
                nodes_labels.append(sect_id)
    else:
        line=np.array(geometries[i].coords)
        start=line[0,:]
        end=line[-1,:]
        sect_id=sect_ids[i]
        if len(np.where(point_array[:,0:2]==start)[0])==0:
            nodes_coordinates.append(start)
            nodes_labels.append(sect_id)
        if len(np.where(point_array[:,0:2]==end)[0])==0:
            nodes_coordinates.append(end)
            nodes_labels.append(sect_id)
       


fig,ax=plt.subplots(1)
udf.plot(ax=ax,color="red")
plt.scatter(np.array(nodes_coordinates)[:,0],np.array(nodes_coordinates)[:,1])
# ordinance[0:1].plot(ax=ax,color="red",alpha=0.3)




nodes_coordinates_array=np.array(nodes_coordinates)


with open(r'C:/Users/srila/Desktop/task_3_junwon/nodes_ucla_roi_big.pickle', 'wb') as handle:
    pickle.dump([nodes_coordinates_array,nodes_labels], handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(r'C:/Users/srila/Desktop/task_3_junwon/nodes_ucla_roi_big.pickle', 'rb') as handle:
    nodes_coordinates_array,nodes_labels = pickle.load(handle)









geometries=udf["geometry"].reset_index(drop=True)
sect_ids=udf["ObjectID"].reset_index(drop=True)
names=udf["NAME"].reset_index(drop=True)
types=udf["TYPE"].reset_index(drop=True)
oneways=udf["ONE_WAY"].reset_index(drop=True)
lengths=udf["Meters"].reset_index(drop=True)
speeds=udf["SPEED"].reset_index(drop=True)
drives=udf["drive_mins"].reset_index(drop=True)
walks=udf["walk_mins"].reset_index(drop=True)




B_matrix=np.zeros([10000000,6])
B_matrix_str=np.zeros([10000000,4],dtype=object)#.dtype(object)

counter=0
status=1
nodes_coordinates_array=np.array(nodes_coordinates)
i=0
for i in range(len(geometries)):
    print(i)
    if type(geometries[i])==MultiLineString:
        for z in range(len(geometries[i].geoms)):
            line=np.array(geometries[i].geoms[z].coords) 
            start=line[0,:]
            end=line[-1,:]
            
            sect_id=sect_ids[i]
            name=names[i]
            roadtype=types[i]
            oneway=oneways[i]
            length=lengths[i]
            speed=speeds[i]
            drive=drives[i]
            walk=walks[i]

            
            
            
            
            
            
            node_start=np.where(nodes_coordinates_array[:,0:2]==start)[0][0]
            for internode in line[1:-1]:
                #print(internode)
                if len(np.where(nodes_coordinates_array[:,0:2]==internode)[0])!=0:
                    print("internode")
                    node_end=np.where(nodes_coordinates_array[:,0:2]==internode)[0][0]
                    B_matrix[counter,:]=np.array([node_start,node_end,length,speed,drive,walk])
                    B_matrix_str[counter,:]=np.array([name,roadtype,oneway,str(sect_id)])
                    counter+=1
                    node_start=np.where(nodes_coordinates_array[:,0:2]==internode)[0][0]
                    
            node_end=np.where(nodes_coordinates_array[:,0:2]==end)[0][0]        
            B_matrix[counter,:]=np.array([node_start,node_end,length,speed,drive,walk])
            B_matrix_str[counter,:]=np.array([name,roadtype,oneway,str(sect_id)])
            counter+=1
    else:
        line=np.array(geometries[i].coords)
        start=line[0,:]
        end=line[-1,:]
        sect_id=sect_ids[i]
        name=names[i]
        roadtype=types[i]
        oneway=oneways[i]
        length=lengths[i]
        speed=speeds[i]
        drive=drives[i]
        walk=walks[i]
        node_start=np.where(nodes_coordinates_array[:,0:2]==start)[0][0]
        for internode in line[1:-1]:
            
            if len(np.where(nodes_coordinates_array[:,0:2]==internode)[0])!=0:
                print("internode")
                node_end=np.where(nodes_coordinates_array[:,0:2]==internode)[0][0]
                B_matrix[counter,:]=np.array([node_start,node_end,length,speed,drive,walk])
                B_matrix_str[counter,:]=np.array([name,roadtype,oneway,str(sect_id)])
                counter+=1
                node_start=np.where(nodes_coordinates_array[:,0:2]==internode)[0][0]
                
        node_end=np.where(nodes_coordinates_array[:,0:2]==end)[0][0]        
        B_matrix[counter,:]=np.array([node_start,node_end,length,speed,drive,walk])
        B_matrix_str[counter,:]=np.array([name,roadtype,oneway,str(sect_id)])
        counter+=1
        
        
B_matrix_sliced=B_matrix[0:counter,:]
B_matrix_str_sliced=B_matrix_str[0:counter,:]
# B_matrix_sliced[70971:,4]=36











with open(r'C:/Users/srila/Desktop/task_3_junwon/nodes_edges_ucla_big_graph.pickle', 'wb') as handle:
    pickle.dump([B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array], handle, protocol=pickle.HIGHEST_PROTOCOL)








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
# udf.plot(ax=ax,color="gray",linestyle="--")
nx.draw(G.to_undirected(),pos=nodes_coordinates_array[:,0:2],node_size=1,node_color="blue", with_labels = False)




####################Read and PLot





import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd

from shapely.geometry import Point, Polygon, LineString,MultiLineString



with open(r'C:/Users/srila/Desktop/task_3_junwon/nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

udf=gpd.read_file(r"C:/Users/srila/Desktop/task_3_junwon/ucla_large_centrality_roi.geojson")

blocks=gpd.read_file(r"C:/Users/srila/Desktop/task_3_junwon/blocks_epsg4326.geojson")


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


