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

def confirm_core_points(df, radious, min_samples):
    """
    Parameters:
        df (ndarray): Input data to cluster.
        radious (float): Max_distance to be a neighborhood.
        min_samples (int): Minimum number of points required to form a core_points.
    Returns:
        labels (ndarray): labels of core_points and non-core_points.
    """
    #Initialize variables
    labels = np.zeros(df.shape[0], dtype=float)
 
    #Loop to find neighborhoods
    for item in df:
    
        #Ignore if point is clustered
        if labels[int(item[0]) - 1] != 0:
            continue
        
        # Find neighbors within radious
        # **neighbors = np.where(np.linalg.norm(df - df[i], axis=1) < radious)[0]
        neighbors = core_points(df, item, radious)
        
        # If there are not enough neighbors, label as noise
        if len(neighbors) >= min_samples:
            labels[int(item[0]) - 1] = -1
            continue
        "points that are not satified are called non-core_point"
    return labels
"""def expand_clusters()
    # Assign new cluster ID and start expanding cluster
    labels[i] = cluster_id
    expand_cluster(X, labels, neighbors, cluster_id, radious, min_samples)
                
    # Move to next cluster ID
    cluster_id += 1"""


df = pd.read_csv("data_clustering.csv")
df = df.to_numpy()
print(confirm_core_points(df, radious=2, min_samples=5))
'''
plt.plot(df['Age'], df['Income'], 'o')
plt.show()
'''
