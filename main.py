from PyQt5.QtWidgets import QApplication, QFileDialog
from GUI import ImageKNNClassifier
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from eigen_face import generate_dataset, generate_covariance_matrix, get_lambdas_and_eigenvectors, get_k_vectors
from eigen_face import get_a_coefficients_dataset, get_a_coefficients_image , k_nearest_neighbour
import matplotlib.pyplot as plt

def handle_dataset_selection(window):
    folder = QFileDialog.getExistingDirectory(window, "Select Dataset Folder")
    if folder:
        window.dataset_path = folder
        print("Dataset path selected:", window.dataset_path)

def handle_test_image_selection(window):
    # Open file dialog to select an image (now includes .pgm format)
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Select Test Image",
        "",
        "Image Files (*.png *.jpg *.jpeg *.bmp *.pgm)"
    )
    
    if file_path:
        # Read image with OpenCV
        if file_path.lower().endswith('.pgm'):
            # Special handling for PGM format
            window.test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert grayscale to RGB for display (since QLabel expects 3 channels)
            rgb_image = cv2.cvtColor(window.test_image, cv2.COLOR_GRAY2RGB)
        else:
            # Normal handling for color images
            window.test_image = cv2.imread(file_path)
            rgb_image = cv2.cvtColor(window.test_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display in label
        pixmap = QPixmap.fromImage(q_img)
        # Scale pixmap to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            window.test_image_label.width(),
            window.test_image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        window.test_image_label.setPixmap(pixmap)
        window.test_image_label.setAlignment(Qt.AlignCenter)
        
        print(f"Test image loaded and displayed (Format: {'PGM' if file_path.lower().endswith('.pgm') else 'Color'})")

def plot_nearest_neighbors(window, test_image, images, labels, neighbor_indices, predicted_label):
    k = len(neighbor_indices)
    fig, axes = plt.subplots(1, k + 1, figsize=(12, 3))

    # Plot test image
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title(f"Test Image\nPredicted: {predicted_label}")
    axes[0].axis('off')

    # Plot k nearest neighbors
    for i, idx in enumerate(neighbor_indices):
        axes[i + 1].imshow(images[idx], cmap='gray')
        axes[i + 1].set_title(f"Neighbor {i + 1}\nLabel: {labels[idx]}")
        axes[i + 1].axis('off')

    # Tight layout for better spacing
    plt.tight_layout()

    # Convert Matplotlib plot to QImage
    fig.canvas.draw()

    # Convert the plot to an image
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Convert to (height, width, 3)
    
    # Convert the image to QImage
    h, w, ch = img_arr.shape
    q_img = QImage(img_arr.data, w, h, w * ch, QImage.Format_RGB888)
    
    # Convert QImage to QPixmap
    pixmap = QPixmap.fromImage(q_img)

    # Display the pixmap in the widget
    if not hasattr(window, "knn_image_label"):
        window.knn_image_label = QLabel(window.knn_widget)
        window.knn_image_label.setGeometry(10, 10, 800, 300)  # Adjust size to fit the plot
    window.knn_image_label.setPixmap(pixmap.scaled(800, 300, Qt.KeepAspectRatio))

    # Clean up the plot (close the figure)
    plt.close(fig)

def apply_knn_classification(window):
    k_value = window.k_spinbox.value()
    # Generate dataset
    images, labels = generate_dataset(window.dataset_path)
    
    # Generate covariance matrix
    covariance_matrix, centered_data, images_mean = generate_covariance_matrix(images)
    
    # Get eigenvalues and eigenvectors
    eigen_values, eigenvectors = get_lambdas_and_eigenvectors(covariance_matrix, centered_data)
    
    # Get k eigenvectors based on threshold
    k_eigenvectors = get_k_vectors(eigen_values, eigenvectors)
    grayscale_image = cv2.cvtColor(window.test_image, cv2.COLOR_BGR2GRAY)

    # Get coefficients for dataset and test image
    a_coefficients_dataset = get_a_coefficients_dataset(k_eigenvectors, images_mean, images)
    a_coefficients_image = get_a_coefficients_image(k_eigenvectors, images_mean, grayscale_image)
    predicted_label, neighbor_indices, distances = k_nearest_neighbour(a_coefficients_dataset, a_coefficients_image, labels, k_value)
    print(f"Predicted label: {predicted_label}")
    print(f"Neighbor indices: {neighbor_indices}")
    print(f"Distances: {distances}")

    # Display nearest neighbors plot in widget
    plot_nearest_neighbors(window, window.test_image, images, labels, neighbor_indices, predicted_label)

    print("KNN classification applied.")

if __name__ == "__main__":
    app = QApplication([])
    window = ImageKNNClassifier()
    
    # Connect buttons to their handlers
    window.dataset_button.clicked.connect(lambda: handle_dataset_selection(window))
    window.apply_button.clicked.connect(lambda: apply_knn_classification(window))
    window.test_image_button.clicked.connect(lambda: handle_test_image_selection(window))
    
    window.show()
    app.exec_()
