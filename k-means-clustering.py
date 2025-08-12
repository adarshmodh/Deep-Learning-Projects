import random
import math
import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100):
        """
        Initialize the KMeans instance.
        
        Args:
            k (int): Number of clusters.
            max_iters (int): Maximum number of iterations for convergence.
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    def _distance(self, point1, point2):
        """
        Compute Euclidean distance between two 2D points.
        
        Args:
            point1 (tuple): First point as (x, y).
            point2 (tuple): Second point as (x, y).
            
        Returns:
            float: Euclidean distance.
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    
    def train(self, points):
        """
        Train the model using the KMeans algorithm on the given points.
        
        Args:
            points (list): A list of tuples representing the 2D points [(x1, y1), (x2, y2), ...].
        
        Returns:
            list: The list of centroids found.
        
        Pseudocode - 
        1. initialize the centroids randomly (np function)
        2. Loop through point in points - 
        	2.1 - calculate the distance between each point and centroids (array of distances for all points)
        	2.2 - Do argmin - to assign each point to the closest centroid 
        	2.3 - compute new centroids - calculate new clusters from argmin, and then centroids from those clusters 
        	2.4 - new centroids = older centroids
        
        """
        points_np = np.array(points) # shape is [N,2]
        random_idx = np.random.choice(len(points), self.k, replace=False)
        centroids = points_np[random_idx] # shape is [k,2] 
        new_labels = np.zeros(len(points), dtype=int)
        
        for i in range(self.max_iters):
          labels = np.zeros(len(points), dtype = int)
          for j in range(len(points)):
            distances = []
            for centroid in centroids:
              distances.append(self.distance(points_np[j], centroid))
            
            labels[j] = np.argmin(np.array(distances))  # 0,1,2,0,1,2
          
          new_centroids = []
          for l in range(self.k):
            cluster_points = points_np[labels == l]
            temp_centroid = cluster_points.mean(axis=0)
            new_centroids.append(temp_centroid)
          
            
        	# diff = points_np[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        	# distances = np.linalg.norm(diff, axis=2)
        	# labels = np.argmin(distances, axis = 1)
          
          
        return self.centroids
        
              
      
    def infer(self, points):
        """
        Infer the cluster for each point using the trained centroids.
        
        Args:
            points (list): A list of tuples representing the 2D points to assign clusters.
        
        Returns:
            list: A list of cluster indices corresponding to each input point.
        """
        pass
            

# Example usage:
if __name__ == '__main__':
    # Sample dataset: list of 2D points.
    data = [(1, 2), (2, 1), (1, 0), (10, 10), (11, 11), (12, 12), (5, 5), (6, 4)]
    
    # Create a KMeans instance for 3 clusters.
    kmeans = KMeans(k=3)
    
    # Train the model with the data.
    centroids = kmeans.train(data)
    print("Centroids:", centroids)
    
    # Infer the cluster for new points.
    new_points = [(2, 3), (8, 8)]
    assigned_clusters = kmeans.infer(new_points)
    print("Assigned clusters for new points:", assigned_clusters)
