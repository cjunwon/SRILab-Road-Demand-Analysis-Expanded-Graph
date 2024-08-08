import pickle
import numpy as np
import geopandas as gpd
import numpy as np
import pickle
import boto3

print('Importing B_matrix_weighted_array_0_to_141592.pickle...')

with open(r'intermediate_files/B_matrix_weighted_array_0_to_141592.pickle', 'rb') as handle:
    B_matrix_weighted_array = pickle.load(handle) 

print('Importing nodes_edges_ucla_big_graph.pickle...')

with open(r'nodes_edges_ucla_big_graph.pickle', 'rb') as handle:
    B_matrix_sliced,B_matrix_str_sliced,nodes_coordinates_array = pickle.load(handle)

print('Importing B_matrix_weighted_array.pickle...')

with open(r'intermediate_files/B_matrix_weighted_array.pickle', 'rb') as handle:
    B_matrix_weighted_array_ORIGINAL = pickle.load(handle)

print('Importing ucla_large_centrality_roi.geojson...')

udf = gpd.read_file('ucla_large_centrality_roi.geojson')


count=0

for i in range(B_matrix_weighted_array.shape[0]):
    if np.count_nonzero(B_matrix_weighted_array[i,6:])==0:
        count+=1

percent_updated_B_matrix = (1 - (count / B_matrix_weighted_array.shape[0]))*100

print('Percent of B_matrix updated from script 04_combined_shortest_path_usage.py: ', round(percent_updated_B_matrix, 2), '%')


oids = np.array(udf['ObjectID']).astype(int)
allocated_B_matrix=np.zeros((len(oids),10))


print('Allocating B_matrix values...')

# Create a dictionary to map oid values to their indices
oid_to_index = {int(oid): idx for idx, oid in enumerate(B_matrix_str_sliced[:,3].astype(int))}

exceptioncounter = 0
for i in range(len(oids)):
    oid = oids[i]
    try:
        idx = oid_to_index[oid]
        allocated_B_matrix[i,:] = B_matrix_weighted_array[idx,6:16]
    except KeyError:
        exceptioncounter += 1

print('Updating normalized values in udf_UPDATED_LODES...')

udf_UPDATED_LODES = udf.copy()

udf_UPDATED_LODES["S000_adjusted"]=allocated_B_matrix[:,0]/np.max(allocated_B_matrix[:,0])
udf_UPDATED_LODES["SA01_adjusted"]=allocated_B_matrix[:,1]/np.max(allocated_B_matrix[:,1])
udf_UPDATED_LODES["SA02_adjusted"]=allocated_B_matrix[:,2]/np.max(allocated_B_matrix[:,2])
udf_UPDATED_LODES["SA03_adjusted"]=allocated_B_matrix[:,3]/np.max(allocated_B_matrix[:,3])
udf_UPDATED_LODES["SE01_adjusted"]=allocated_B_matrix[:,4]/np.max(allocated_B_matrix[:,4])
udf_UPDATED_LODES["SE02_adjusted"]=allocated_B_matrix[:,5]/np.max(allocated_B_matrix[:,5])
udf_UPDATED_LODES["SE03_adjusted"]=allocated_B_matrix[:,6]/np.max(allocated_B_matrix[:,6])
udf_UPDATED_LODES["SI01_adjusted"]=allocated_B_matrix[:,7]/np.max(allocated_B_matrix[:,7])
udf_UPDATED_LODES["SI02_adjusted"]=allocated_B_matrix[:,8]/np.max(allocated_B_matrix[:,8])
udf_UPDATED_LODES["SI03_adjusted"]=allocated_B_matrix[:,9]/np.max(allocated_B_matrix[:,9])

print('Exporting udf_UPDATED_LODES to ucla_large_centrality_roi_UPDATED_LODES_2021.geojson...')

udf_UPDATED_LODES.to_file(r"ucla_large_centrality_roi_UPDATED_LODES_2021.geojson", driver='GeoJSON')


# count how many values in udf_UPDATED_LODES['S000_adjusted'] are 0

S000_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['S000_adjusted']==0])
S000_no_usage_percent = (S000_adjusted_zero_count / len(udf_UPDATED_LODES['S000_adjusted']))*100

SA01_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SA01_adjusted']==0])
SA01_no_usage_percent = (SA01_adjusted_zero_count / len(udf_UPDATED_LODES['SA01_adjusted']))*100

SA02_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SA02_adjusted']==0])
SA02_no_usage_percent = (SA02_adjusted_zero_count / len(udf_UPDATED_LODES['SA02_adjusted']))*100

SA03_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SA03_adjusted']==0])
SA03_no_usage_percent = (SA03_adjusted_zero_count / len(udf_UPDATED_LODES['SA03_adjusted']))*100

SE01_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SE01_adjusted']==0])
SE01_no_usage_percent = (SE01_adjusted_zero_count / len(udf_UPDATED_LODES['SE01_adjusted']))*100

SE02_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SE02_adjusted']==0])
SE02_no_usage_percent = (SE02_adjusted_zero_count / len(udf_UPDATED_LODES['SE02_adjusted']))*100

SE03_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SE03_adjusted']==0])
SE03_no_usage_percent = (SE03_adjusted_zero_count / len(udf_UPDATED_LODES['SE03_adjusted']))*100

SI01_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SI01_adjusted']==0])
SI01_no_usage_percent = (SI01_adjusted_zero_count / len(udf_UPDATED_LODES['SI01_adjusted']))*100

SI02_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SI02_adjusted']==0])
SI02_no_usage_percent = (SI02_adjusted_zero_count / len(udf_UPDATED_LODES['SI02_adjusted']))*100

SI03_adjusted_zero_count = len(udf_UPDATED_LODES[udf_UPDATED_LODES['SI03_adjusted']==0])
SI03_no_usage_percent = (SI03_adjusted_zero_count / len(udf_UPDATED_LODES['SI03_adjusted']))*100


print('Percent of S000_adjusted == 0: ', round(S000_no_usage_percent, 2), '%')
print('Percent of SA01_adjusted == 0: ', round(SA01_no_usage_percent, 2), '%')
print('Percent of SA02_adjusted == 0: ', round(SA02_no_usage_percent, 2), '%')
print('Percent of SA03_adjusted == 0: ', round(SA03_no_usage_percent, 2), '%')
print('Percent of SE01_adjusted == 0: ', round(SE01_no_usage_percent, 2), '%')
print('Percent of SE02_adjusted == 0: ', round(SE02_no_usage_percent, 2), '%')
print('Percent of SE03_adjusted == 0: ', round(SE03_no_usage_percent, 2), '%')
print('Percent of SI01_adjusted == 0: ', round(SI01_no_usage_percent, 2), '%')
print('Percent of SI02_adjusted == 0: ', round(SI02_no_usage_percent, 2), '%')
print('Percent of SI03_adjusted == 0: ', round(SI03_no_usage_percent, 2), '%')