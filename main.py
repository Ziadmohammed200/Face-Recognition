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
    if len(window.test_image.shape) == 3:
        # It's a color image
        grayscale_image = cv2.cvtColor(window.test_image, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        grayscale_image = window.test_image

    # Get coefficients for dataset and test image
    a_coefficients_dataset = get_a_coefficients_dataset(k_eigenvectors, images_mean, images)
    a_coefficients_image = get_a_coefficients_image(k_eigenvectors, images_mean, grayscale_image)
    predicted_label, neighbor_indices, distances = k_nearest_neighbour(a_coefficients_dataset, a_coefficients_image, labels, k_value)
    images_detected = images[neighbor_indices[k_value-1]]
    print(f"Predicted label: {predicted_label}")
    print(f"Neighbor indices: {neighbor_indices}")
    print(f"Distances: {distances}")
    # Display predicted label in the GUI

    
    # KNN classification logic (to be implemented)
    print("KNN classification applied.")


    # Normalize and convert image to uint8
    image = images_detected
    if image is None or len(image.shape) != 2:
        print("Error: Detected image is not 2D grayscale.")
        return

    if image.dtype != np.uint8:
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)

    # Convert NumPy image to QPixmap and show in QLabel
    height, width = image.shape
    q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(q_image)

    # Display in QLabel inside knn_widget
    if not hasattr(window, "knn_image_label"):
        window.knn_image_label = QLabel(window.knn_widget)
        window.knn_image_label.setGeometry(10, 10, 150, 150)

    window.knn_image_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))




if __name__ == "__main__":
    app = QApplication([])
    window = ImageKNNClassifier()
    
    # Connect buttons to their handlers
    window.dataset_button.clicked.connect(lambda: handle_dataset_selection(window))
    window.apply_button.clicked.connect(lambda: apply_knn_classification(window))
    window.test_image_button.clicked.connect(lambda: handle_test_image_selection(window))
    
    window.show()
    app.exec_()