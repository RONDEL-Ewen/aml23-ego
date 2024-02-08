import torch

class KMeans:
    def __init__(
        self,
        n_clusters = 3,
        max_iter = 300,
        tol = 1e-4
    ):
        """
        Initialize K-Means algorithm.

        :param n_clusters: Number of clusters
        :param max_iter: Maximum number of iterations.
        :param tol: Tolerance for convergence.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(
        self,
        X
    ):
        """
        Train K-Means algorithm on data X.

        :param X: Input data, 2D tensor where every line is a data point.
        """
        # Random initialization of the centroids within the data points
        random_idx = torch.randperm(X.size(0))[:self.n_clusters]
        self.centroids = X[random_idx, :]

        for _ in range(self.max_iter):
            # Assign each point to the nearest cluster
            dists = torch.cdist(X, self.centroids)  # Compute the distances between points and centroids
            labels = torch.argmin(dists, dim=1)  # Find the nearest centroid for each point

            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                # Update centroids to reflect the average of points in each cluster
                members = X[labels == i]
                if len(members) > 0:
                    new_centroids[i] = torch.mean(members, dim=0)

            # Check if centroids have converged (change less than tolerance)
            if torch.norm(self.centroids - new_centroids, dim = 1).max() < self.tol:
                break  # Stop if convergence is reached

            self.centroids = new_centroids  # Update centroids for next iteration

    def predict(
        self,
        X
    ):
        """
        Predict the cluster for each point of data X.

        :param X: Data to cluster, 2D tensor.
        :return: Clusters labels for each point of data X.
        """
        dists = torch.cdist(X, self.centroids)  # Computation of distances to centroids
        labels = torch.argmin(dists, dim = 1)  # Assignment to the nearest cluster
        return labels