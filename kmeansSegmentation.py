import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_mask(image, labels, label):
    # Create black image
    mask = np.zeros(image.shape, np.uint8)
    # Flatten mask
    mask = mask.reshape((-1, 3))
    # Set color of label to white
    mask[~(labels.flatten() == label)] = [255, 255, 255]
    # Show mask
    mask = mask.reshape(image.shape)
    plt.imshow(mask)
    plt.show()

def main():
    # Read image
    image = cv2.imread("image.jpg")
    # OpenCV uses BGR color space by default. Convert to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flatten image from shape (H, W, C) to (H x W, C)
    pixels = image.reshape((-1, 3))
    # K Means should by applied to np.float32 data type
    pixels = np.float32(pixels)
    # Define K Means Algorithm
    # Termination criteria (Type, Max Iteration, Epsilon)
    #   TERM_CRITERIA_EPS: Stops algorithm if clusters move less than epsilon
    #   TERM_CRITERIA_MAX_ITER: Stops algorithm after specified iterations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    # Define flag specifies how to initial centers
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Define number of clusters
    k = 3
    # Perform K means (data, cluster, labels, criteria, attempts, flags)
    #   compactness: Sum of squared distance of each point to its center
    #   labels: Array of labels
    #   Centers: Array of centers
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    # Convert center color to 8-bit pixel values
    centers = np.uint8(centers)
    # Convert pixels to the color of centers
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    # Show segmented image
    plt.imshow(segmented_image)
    plt.show()
    # Show masks
    for i in range(k):
        show_mask(image, labels, i)

if __name__ == '__main__':
    main()
