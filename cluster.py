class cluster:

    def __init__(self):
        pass

    def fit(self, X):
        pass

class KMeansClustering:
    
    def __init__(self, k = 5, max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        
    def fit(self, X):
        # Randomly initialize centroids, in range(rows) and k number of centroids
        self.centroids = X[np.random.choice(range(len(X)), self.k, replace = False)]
        # Loop runs max number of times = max_iterations
        for iteration in range(self.max_iterations):
            # Will contain the cluster index that a data point belongs to - example for 8 data points with k = 2 {1, 0, 1, 0, 0, 0, 0, 1}
            clusters = np.zeros(len(X))
            # For every data point in the dataset
            for i in range(len(X)):
                # Find the eucledian distance of that point to all centroids
                distance = np.sqrt(np.sum((X[i] - self.centroids) ** 2, axis = 1))
                # Pick the min distance out of all the distances
                cluster = np.argmin(distance);
                # For the datapoint (denoted as index i in clusters) assign the index
                clusters[i] = cluster
            new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(self.k)])
            if(np.all(new_centroids == self.centroids)):
                break
            self.centroids = new_centroids
        return self.centroids, clusters