import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def distance(x1, y1, x2, y2):
    """
    Parameters:
        x1, y1, x2, y2 - (float): Are coordinates of two points.
    Return:
        dist - (float): zDistance of two points.
    """
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def core_points(df, processing_point, radious):
    """
    Parameters:
        df (ndarray): Input data to cluster.
        processing_point (ndarray): one point of df.
        radious (float): Max_distance to be a neighborhood.
    Return:
        neighbors (ndarray): Index of points that is a neighborhood of processing_point
    """
    neighbors = np.array([])
    for item in df:
        if item[0] == processing_point[0]:
            continue
        if distance(item[1], item[2], processing_point[1], processing_point[2]) <= radious:
            neighbors = np.append(neighbors, item[0])
    return neighbors

def DBSCAN(df, radious, min_samples):
    """
    Parameters:
        df (ndarray): Input data to cluster.
        radious (float): Max_distance to be a neighborhood.
        min_samples (int): Minimum number of points required to form a core_points.
    Returns:
        labels (ndarray): Cluster labels of points in data points.
    """
    #Initialize variables
    labels = np.zeros(df.shape[0], dtype=float)
    cluster_id = 1

    #Loop to find neighborhoods
    for item in df:
    
        #Ignore if point is clustere
        if labels[int(item[0]) - 1] != 0:
            continue
        
        #Find neighbors within radious
        neighbors = core_points(df, item, radious)
        
        #If there are not enough neighbors, dotting non-core_point
        if len(neighbors) < min_samples:
            labels[int(item[0]) - 1] = -1
            continue

        # Assign new cluster ID and start expanding cluster
        labels[int(item[0]) - 1] = cluster_id
        
        expand_cluster(df, labels, neighbors, cluster_id, radious, min_samples)
        
        # Move to next cluster ID
        cluster_id += 1  
    return labels
def expand_cluster(df, labels, neighbors, cluster_id, radious, min_samples):
    """
    Parameters:
        df (ndarray): Input data to cluster.
        labels (ndarray): Cluster labels for each point in X.
        neighbors (ndarray): Indices of neighbors of the seed point.
        cluster_id (int): ID of the current cluster being expanded.
        radious (float): Radius of the neighborhood to search for points.
        min_samples (int): Minimum number of points required to form a dense region.
    Return:
        None
    """
    #Loop over all neighbor points
    for i in range(len(neighbors)):
        
        #If point has not been assigned a label, assign to current cluster
        if labels[int(neighbors[i]) - 1] == 0:
            labels[int(neighbors[i]) -1 ] = cluster_id
            
            #Find neighbors of processing neighbor point
            neighbors_of_neighbors = core_points(df, df[int(neighbors[i]) - 1], radious)
            
            #If this new point is also a core point, update neighbors array
            if len(neighbors_of_neighbors) >= min_samples:
                neighbors = np.append(neighbors, neighbors_of_neighbors)
                neighbors = np.unique(neighbors)
        #If point is non-core_point, assign to current cluster
        elif labels[int(neighbors[i]) - 1] == -1:
            labels[int(neighbors[i]) - 1] = cluster_id

#Upload the dataset
df = pd.read_csv("data_clustering.csv")
df = df.to_numpy()

#Cluster data using DBSCAN
labels = DBSCAN(df, radious=10, min_samples=5)

# Plot clusters
plt.scatter(df[:,1], df[:,2], c=labels)
plt.colorbar()
plt.show()
"""
# Define a color map
cmap = plt.cm.get_cmap('cool')
# Create the plot
fig, ax = plt.subplots()
sc = ax.scatter(df[:,1], df[:,2], c=labels, cmap=cmap)

# Add a color bar
cbar = plt.colorbar(sc)

# Add annotations
for i in range(len(df)):
    ax.annotate(df[:,1], df[:,2], color=cmap(df[i]), fontsize=8)

# Set the labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Color Annotation that Represents the Number')

# Show the plot
plt.show()
"""