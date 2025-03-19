#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Importing necessary libraries for clustering and plotting
get_ipython().system('pip install numpy')
import numpy as np
import pandas as pd
import sklearn as skl
from skl.cluster import KMeans
import matplotlib.pyplot as plt


# In[12]:


# Step 1: Generate 10 students with random Dex, Str, Int
np.random.seed(42)  # Setting seed for reproducibility
students = []

for _ in range(10):
    points = [0, 0, 0] # D S I
    remaining_points = 15
    for i in range(3):  # Looping through 3 attributes: Dex, Str, Int
        if i == 2:
            points[i] = remaining_points  # Assign remaining points to the last attribute
        else:
            points[i] = np.random.randint(0, remaining_points + 1)  # Randomly assign points
            remaining_points -= points[i]  # Decrease remaining points 
    students.append(points)

students


# In[13]:


df = pd.DataFrame([[6, 3, 6],
 [12, 2, 1],
 [10, 4, 1],
 [4, 6, 5],
 [9, 2, 4],
 [6, 7, 2],
 [4, 3, 8],
 [7, 7, 1],
 [2, 5, 8],
 [4, 1, 10]], columns=['Dex', 'Str', 'Int'])


# In[14]:


# Step 2: Create a DataFrame to hold the students' data
df = pd.DataFrame(students, columns=['Dex', 'Str', 'Int'])
df


# In[15]:


# Step 3: Apply K-means clustering
# Initialize K-means with the desired number of clusters (e.g., 3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans


# In[ ]:


# Step 4: Fit the K-means model on the data (Dex, Str, Int) and get the cluster labels
# 'fit' means it tries to identify the clusters, like what we did in group
kmeans.fit(df[['Dex', 'Str', 'Int']])


# In[ ]:


# Step 5: Retrieve cluster labels
# 'labels_' is an attribute that holds the cluster label for each data point (student)
df['Cluster'] = kmeans.labels_ # We create a new column so we can keep track
df


# In[ ]:


# Step 6: Plot the results (3D plot, projecting on Dex vs Str vs Int)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # 111 means you want a single subplot, aka, not subplots

# Scatter plot for 3D data (Dex, Str, Int)
sc = ax.scatter(df['Dex'], df['Str'], df['Int'], c=df['Cluster'], cmap='viridis', s=100)
ax.set_xlabel('Dex')
ax.set_ylabel('Str')
ax.set_zlabel('Int')

# Title and showing plot
plt.title('3D Clustering of Students Based on Attributes')
plt.show()


# In[10]:


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# dendrogram for yourself
linked = linkage(df[['Dex', 'Str', 'Int']], method='ward')
# Ward’s linkage method
# smallest possible increase in the total within-cluster variance
# Like K-mean
# Favor spherical patterns
plt.figure(figsize=(10, 5))

dendrogram(linked, labels=range(1, 11))  
# Labels 1 to 10 for students
plt.title('Simple Dendrogram')
plt.xlabel('Student Liking Index')
plt.ylabel('Distance')
plt.show()


# In[ ]:


# dendrogram for other people
plt.figure(figsize=(12, 6))
dendrogram(
    linked,
    labels=range(1, 11), # Student Awkardness indices
    leaf_rotation=45,  # Rotate labels because I can 
    leaf_font_size=12,  # Increase font for seniors
    distance_sort='descending',  # Sort distances for better visual clarity
    show_leaf_counts=True  # Show count of samples in each cluster
)

plt.title('Pretty Dendrogram', fontsize=16)
plt.xlabel('Student Awkardness Index', fontsize=14)
plt.ylabel('Distance', fontsize=14)
plt.grid(True)  # Add grid for better readability, some people hate it
plt.tight_layout()  # Adjust layout for better spacing, usually useless
plt.show()


# In[ ]:


df['Cluster'] = fcluster(linked, t=3, criterion='maxclust')
df


# In[ ]:


# Step 2: Perform K-Means clustering on the normalized data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_normalized[['y2', 'y3', 'x4']])




# In[ ]:


def recursive_cluster(df, c ,columns)
    kmeans = KMeans(n_clusters = c, random_state = 42)
    kmeans.fit(df[columns])
    df['Normalized_Cluster'] = kmeans.labels_
    return df_clustered

