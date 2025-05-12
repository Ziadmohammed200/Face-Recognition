import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

main_folder = 'D:\CV\\faces'
test_image = cv2.imread("C:\\Users\DELL\Downloads\Telegram Desktop\WhatsApp Image 2025-05-09 at 8.34.31 PM.jpeg",cv2.IMREAD_GRAYSCALE)

images = []
labels = []

def generate_dataset(main_folder):
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(('.pgm', '.jpg', '.png')):
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    images.append(image)
                    labels.append(subfolder)

    return np.array(images), np.array(labels)



def generate_covariance_matrix(images):
    flattened_images = [image.flatten() for image in images]
    images_vectors = np.array(flattened_images, dtype=np.float32).T

    images_mean = np.mean(images_vectors, axis=1, keepdims=True)

    centered_data = images_vectors - images_mean
    small_covariance_matrix = centered_data.T @ centered_data

    return small_covariance_matrix, centered_data, images_mean

def get_lambdas_and_eigenvectors(covariance_matrix,centered_data):
    eigvalues, eigvectors_small = np.linalg.eigh(covariance_matrix)

    eigvectors_large = centered_data @ eigvectors_small

    eigvectors_large = eigvectors_large / np.linalg.norm(eigvectors_large, axis=0)

    return eigvalues, eigvectors_large


def get_k_vectors(eigen_values, eigenvectors, threshold=0.95):
    # Sort eigenvalues in descending order and reorder eigenvectors accordingly
    sorted_eigen_indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_values = eigen_values[sorted_eigen_indices]
    sorted_eigen_vectors = eigenvectors[:, sorted_eigen_indices]

    # Compute cumulative sum of eigenvalues
    cumulative_eigen_values = np.cumsum(sorted_eigen_values)

    total_variance = cumulative_eigen_values[-1]
    threshold_variance = total_variance * threshold

    k = np.argmax(cumulative_eigen_values >= threshold_variance) + 1

    top_k_eigenvectors = sorted_eigen_vectors[:, :k]

    return k, top_k_eigenvectors

def get_a_coefficients_dataset(eigenvectors, images, images_mean):
    flattened_images = [image.flatten() for image in images]
    image_matrix = np.array(flattened_images, dtype=np.float32).T  ##10304*400
    centered_images = image_matrix - images_mean                  ## 10304*400 - 10304*1
    a_coefficients = eigenvectors.T @ centered_images             ##(190*10304)(10304*400) ==> (190*400)

    return a_coefficients

def get_a_coefficients_image(eigenvectors, image, images_mean):
    flattened_image = image.flatten().astype(np.float32).reshape(-1, 1)  #(pixels, 1)
    centered_image = flattened_image - images_mean
    a_coeff = eigenvectors.T @ centered_image  #(190, 1)
    return a_coeff


def k_nearest_neighbour(images_a_coefficients, test_image_a_coefficients, labels, k=3):
    # Flatten test vector for distance computation if needed
    if test_image_a_coefficients.ndim == 2:
        test_image_a_coefficients = test_image_a_coefficients.flatten()

    # Compute Euclidean distances to all training images
    distances = np.linalg.norm(images_a_coefficients.T - test_image_a_coefficients, axis=1)  # shape: (num_images,)

    # Get the indices of the k smallest distances
    nearest_indices = np.argsort(distances)[:k]

    # Retrieve the labels of these k nearest neighbors
    nearest_labels = labels[nearest_indices]

    # Optional: Majority vote
    from collections import Counter
    most_common = Counter(nearest_labels).most_common(1)[0][0]

    return most_common, nearest_indices, distances[nearest_indices]


def plot_nearest_neighbors(test_image, images, labels, neighbor_indices, predicted_label):
    k = len(neighbor_indices)
    plt.figure(figsize=(12, 3))

    # Plot test image
    plt.subplot(1, k + 1, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title(f"Test Image\nPredicted: {predicted_label}")
    plt.axis('off')

    # Plot k nearest neighbors
    for i, idx in enumerate(neighbor_indices):
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"Neighbor {i+1}\nLabel: {labels[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()








images,labels = generate_dataset(main_folder)
covarianceMatrix,center, mean = generate_covariance_matrix(images)
l,v=get_lambdas_and_eigenvectors(covarianceMatrix,center)
k,v_k = get_k_vectors(l,v,0.95)
a = get_a_coefficients_dataset(v_k,images,mean)
test_a = get_a_coefficients_image(v_k, test_image, mean)

predicted_label, neighbor_indices, distances = k_nearest_neighbour(a, test_a, labels, k=5)
images_detected = images[neighbor_indices[4]]


print("Predicted label:", predicted_label)
print("Neighbor indices:", neighbor_indices)
plot_nearest_neighbors(test_image, images, labels, neighbor_indices, predicted_label)
cv2.imshow('Image Window', images_detected)

# Wait for a key press indefinitely or for a specific amount of time (in milliseconds)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()




