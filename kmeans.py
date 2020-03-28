import cv2
import numpy as np

class KMeans:

    def __init__(self, k=3, magnification=500):
        # Set magnification factor of the display
        self.magnification = magnification
        # Set width and height of the environment
        self.width = 1.0
        self.height = 1.0
        # Create image of the environment for display
        self.image = np.zeros([int(self.magnification * self.height), int(self.magnification * self.width), 3], dtype=np.uint8)
        self.k = k
        # Generate points
        self.points = np.concatenate((
            np.random.multivariate_normal((0.2, 0.2), [[0.01, 0], [0, 0.05]], 200),
            np.random.multivariate_normal((0.7, 0.3), [[0.05, 0], [0, 0.01]], 200),
            np.random.multivariate_normal((0.5, 0.7), [[0.01, 0], [0, 0.01]], 200)
        ))
        # Generate centers
        self.centroids = np.random.rand(self.k, 2)
        # Generate list of random colors
        self.colors = np.random.randint(255, size=(k, 3))
        self.assign()

    @staticmethod
    def _distance(point, centroid):
        return np.linalg.norm(point - centroid)

    def assign(self):
        distances = np.array([]).reshape(self.points.shape[0], 0)
        # For each centroid ...
        for centroid in self.centroids:
            # ... compute distance from each point to the centroid ...
            distance_k = np.apply_along_axis(KMeans._distance, 1, self.points, centroid)
            # ... store computed distances
            distances = np.hstack([distances, distance_k[:, np.newaxis]])
        # Assign point to closest centroid
        labels = np.argmin(distances, axis=1)
        return labels

    def update(self, labels):
        # For each class ...
        for k in range(self.k):
            # ... find cluster with same label
            cluster_k = self.points[labels == k]
            # ... update centroid to mean of cluster
            self.centroids[k] = np.mean(cluster_k, axis=0)

    def _draw_point(self, point, color, centroid=False):
        # Point center
        center = (int(point[0] * self.magnification), int((self.height - point[1]) * self.magnification))
        # Draw a square for centroid
        if centroid:
            width = int(0.03 * self.magnification)
            top_left = (center[0] - width // 2, center[1] - width // 2)
            bottom_right = (center[0] + width // 2, center[1] + width // 2)
            cv2.rectangle(self.image, top_left, bottom_right, color, cv2.FILLED)
        # Draw a circle for point
        else:
            radius = int(0.005 * self.magnification)
            cv2.circle(self.image, center, radius, color, cv2.FILLED)

    def show(self, labels):
        # Create black background
        self.image.fill(0)
        # Draw points
        for i, point in enumerate(self.points):
            # Point class label
            color = self.colors[labels[i]].tolist()
            self._draw_point(point, color)
        # Draw centroids
        for i, centroid in enumerate(self.centroids):
            # Check centroid is not nan (No points assigned to centroid)
            if not np.isnan(np.sum(centroid)):
                color = self.colors[i].tolist()
                self._draw_point(centroid, color, True)
        # Show image
        cv2.imshow(f"K Means Clustering: K={self.k}", self.image)
        # Give time for image to be rendered on screen
        cv2.waitKey(100)

def main():
    kmeans = KMeans(k=4)
    while True:
        labels = kmeans.assign()
        kmeans.show(labels)
        kmeans.update(labels)

if __name__ == '__main__':
    main()
