#!/usr/bin/env python
# coding: utf-8

# In[2]:


#IMPORT NECESSARY BUILT IN LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


#IMPORT ML LIBRARIES - FOR MODULES 4,5,6,7
from scipy.cluster.hierarchy import ward, dendrogram, fcluster, single, complete,average
from sklearn.metrics.pairwise import cosine_similarity,nan_euclidean_distances 
from sklearn.cluster import DBSCAN,AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules


# In[4]:


#IMPORT & INTEGRATE DATA - PROJECT SPECIFIC
#--------------------------------------------
df_offers = pd.read_excel("./WineKMC.xlsx", sheet_name=0)
df_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
df_transactions = pd.read_excel("./WineKMC.xlsx", sheet_name=1)
df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['n'] = 1


# In[5]:


#INTEGRATE DATA 
#------------------------------------------------
df = pd.merge(df_offers, df_transactions)
matrix = df.pivot_table(index=['customer_name'], columns=['offer_id'], aggfunc='size',fill_value=0)
matrix = matrix.fillna(0).reset_index()
x_cols = matrix.columns[1:]

data = df


# In[6]:


#ANALYSE THE DATA DISTRIBUTION AND ATTRIBUTE'S CORRELATION
#----------------------------------------------------------
print('SIZE OF DATA:\n', df.shape)   


# In[7]:


print('ATTRIBUTE LIST:\n', df.columns)


# In[8]:


print('DATA TYPE OF ATTRIBUTES:\n', df.info())


# In[9]:


print('NUMERICAL ATTRIBUTE STATISTIC:\n', df.describe())


# In[10]:


print('SAMPLE DATA:\n',df.head()) 


# In[11]:


subsetForClust=matrix.iloc[0:matrix.shape[0], 1:matrix.shape[1]]


# #DENSITY BASED CLUSTERING--.................................................................

# In[12]:


#K-DISTANCE PLOT/GRAPH
#INPUT EXPECTED : data frame processed sutiable for applying numeric distance & index removed, Kth value of number of first nearest neighbors to be compared against
K_NN_Expected = 20
neighbors = NearestNeighbors(K_NN_Expected)
KNNeighbors = neighbors.fit(subsetForClust)
distances, indices = KNNeighbors.kneighbors(subsetForClust)
distances = np.sort(distances, axis=0)
print(indices)
print("INDEX:", distances[:,0],"K-DIST:",distances[:,1])
plt.plot(distances[:,1])
plt.title("K NEAREST NEIGHBOR \n May be used to set approximate value of Epsilon", size=10)
plt.xlabel("Data points - In no particular order")
plt.ylabel("Distance to Kth Nearest Neighbor")


# In[13]:


print('DENSITY BASED CLUSTERS \n ')
expectedNumber = 20 #Found from trying in above section for various values of K_NN_Expected
EPSILON = 0.75
DISTANCEMETRIC='cosine'
db_default = DBSCAN(eps = EPSILON, min_samples = expectedNumber, metric=DISTANCEMETRIC).fit(subsetForClust) 
labels = db_default.labels_ 
print(labels)
cluster_labels_ = labels


# In[16]:


#MERGE THE CLUSTER RESULTS ALONG WITH DATA FOR POST ANALYSIS
#----------------------------------------------------------------
cluster_map = pd.DataFrame()
cluster_map['data_index'] = matrix.index.values
cluster_map['cluster'] = cluster_labels_
    
    
matrix1=matrix.merge(cluster_map, left_on=matrix.index, right_on=cluster_map.index,suffixes=('_left', '_right'))
for customers in matrix1.index:
    for trans in df_transactions.index:
        if df_transactions['customer_name'][trans]==matrix1['customer_name'][customers]:
            #df_transactions['n'][trans]=matrix1['cluster'][customers]
            df_transactions['n'][trans]=matrix1['cluster'][customers]
df_transactions  
df_clustersByOffer = df_transactions.groupby(['offer_id','n']).count()

df_clustersByOfferToMerge=df_clustersByOffer.pivot_table(index=['offer_id'], columns=['n'],values=['customer_name'], aggfunc='sum',fill_value=0)

finalClusterAnalysis=df_offers.merge(df_clustersByOfferToMerge, left_on=df_offers.offer_id, right_on=df_clustersByOfferToMerge.index,suffixes=('_left', '_right'))
print("Offer Wise Customer Count per Cluster: For post analysis ")    
finalClusterAnalysis


# In[1]:


#FOR AUTO ASSOCIATION INTERPRETATION - APPLICATION SPECIFIC POST PROCESSING WILL BE REVISITED UNDER ASSOCIATION RULE MINING MODULE
#------------------------------------------------------------------------------------------------------------------------------------

