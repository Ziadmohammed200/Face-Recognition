from PyQt5.QtWidgets import QApplication, QFileDialog
from GUI import ImageKNNClassifier
import math
from sklearn.preprocessing import label_binarize

import cv2
from PyQt5.QtGui import QImage, QPixmap
from sklearn.neighbors import KNeighborsClassifier
from ROC_UI import ROCGraphWindow
from PyQt5.QtWidgets import QLabel
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from eigen_face import generate_dataset, generate_covariance_matrix, get_lambdas_and_eigenvectors, get_k_vectors
from eigen_face import get_a_coefficients_dataset, get_a_coefficients_image , k_nearest_neighbour
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import cv2
import math
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from sklearn.decomposition import PCA


from PyQt5.QtWidgets import QApplication, QFileDialog, QVBoxLayout, QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from eigen_face import generate_dataset, generate_covariance_matrix, get_lambdas_and_eigenvectors, get_k_vectors
from eigen_face import get_a_coefficients_dataset, get_a_coefficients_image , k_nearest_neighbour

def handle_dataset_selection(window):
    folder = QFileDialog.getExistingDirectory(window, "Select Dataset Folder")
    if folder:
        window.dataset_path = folder
        print("Dataset path selected:", window.dataset_path)

def handle_test_image_selection(window):
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Select Test Image",
        "",
        "Image Files (*.png *.jpg *.jpeg *.bmp *.pgm)"
    )
    
    if file_path:
        # Read image with OpenCV
        if file_path.lower().endswith('.pgm'):
            window.test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            rgb_image = cv2.cvtColor(window.test_image, cv2.COLOR_GRAY2RGB)
        else:
            window.test_image = cv2.imread(file_path)
            rgb_image = cv2.cvtColor(window.test_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display in label
        pixmap = QPixmap.fromImage(q_img)
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
    # 1. Validate inputs
    if not hasattr(window, 'dataset_path') or not window.dataset_path:
        print("Error: No dataset selected!")
        return
    if not hasattr(window, 'test_image') or window.test_image is None:
        print("Error: No test image loaded!")
        return

    # 2. Get parameters from GUI
    k_value = window.k_spinbox.value()
    
    try:
        # 3. Execute pipeline (matches standalone version)
        images, labels = generate_dataset(window.dataset_path)
        cov_matrix, centered_data, mean = generate_covariance_matrix(images)
        eigvals, eigvecs = get_lambdas_and_eigenvectors(cov_matrix, centered_data)
        k_eig, top_eigenvectors = get_k_vectors(eigvals, eigvecs, 0.95)  # Added threshold
        
        a_coeffs = get_a_coefficients_dataset(top_eigenvectors, images, mean)
        test_coeffs = get_a_coefficients_image(top_eigenvectors, window.test_image, mean)
        
        # 4. Get results
        predicted_label, neighbors, distances = k_nearest_neighbour(a_coeffs, test_coeffs, labels, k_value)
        
        # Collect detected images
        detected_images = [images[neighbors[i]] for i in range(k_value)]  # Get the top k images
        
        # Get the test image dimensions
        test_h, test_w = window.test_image.shape[:2]

        # Resize detected images to test image size
        resized_images = [cv2.resize(images[neighbors[i]], (test_w, test_h)) for i in range(k_value)]

        # Calculate square grid layout
        grid_cols = math.ceil(math.sqrt(k_value))
        grid_rows = math.ceil(k_value / grid_cols)

        # Pad with black images if necessary
        while len(resized_images) < grid_rows * grid_cols:
            black_img = np.zeros_like(resized_images[0])
            resized_images.append(black_img)

        # Build rows
        grid_rows_images = []
        for r in range(grid_rows):
            row_images = resized_images[r * grid_cols:(r + 1) * grid_cols]
            row_combined = np.hstack(row_images)
            grid_rows_images.append(row_combined)

        # Stack rows to form final grid image
        combined_image = np.vstack(grid_rows_images)

        # Convert to RGB for display
        if len(combined_image.shape) == 2:
            combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2RGB)
        else:
            combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        # Convert to QImage and QPixmap
        h, w, ch = combined_image_rgb.shape
        bytes_per_line = ch * w
        q_combined_img = QImage(combined_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap_combined = QPixmap.fromImage(q_combined_img)

        # Display in QLabel
        window.knn_images_result.setPixmap(pixmap_combined)
        window.knn_images_result.setAlignment(Qt.AlignCenter)

        print("Detected images displayed in knn_widget.")

    except Exception as e:
        print(f"Classification failed: {str(e)}")
#####################################################################################################################
def load_faces_from_directory(dataset_path):
    """
    Loads face images from the dataset directory and returns the images and labels.
    
    dataset_path: str
        Path to the dataset directory containing subdirectories with images for each person.
    
    Returns:
    - faces: numpy array of shape (n_samples, n_features), where n_samples is the number of images and n_features is the flattened image size.
    - labels: numpy array of shape (n_samples,), where each entry is the label for the corresponding image.
    """
    faces = []
    labels = []
    label_map = {}  # This map will store the name or ID of each person

    # Walk through the dataset directory
    label = 0
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        # Only process directories
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                # Read the image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                if image is None:
                    continue  # Skip if the image is invalid

                # Resize to a fixed size (e.g., 64x64)
                image_resized = cv2.resize(image, (64, 64))

                # Flatten the image and append it to faces list
                faces.append(image_resized.flatten())  # Flatten the 2D image to 1D

                # Append the label for this image
                labels.append(label)

            # Increment the label for the next person
            label_map[label] = person_name
            label += 1

    # Convert faces and labels to numpy arrays
    faces = np.array(faces)
    labels = np.array(labels)

    return faces, labels


def apply_pca(faces):
    pca = PCA(n_components=50)
    pca_faces = pca.fit_transform(faces)
    return pca, pca_faces

def show_roc_curve(self):
    # Validate dataset path
    if not hasattr(self, 'dataset_path') or not self.dataset_path:
        print("Error: No dataset selected!")
        return

    # Load the face dataset
    faces, labels = load_faces_from_directory(self.dataset_path)

    # Apply PCA to the faces for dimensionality reduction
    pca, pca_faces = apply_pca(faces)

    # For this example, we are using a KNN classifier to generate predicted labels
    k_value = 3  # Or get from GUI, e.g., self.k_spinbox.value()
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(pca_faces, labels)

    # Predict labels using the KNN classifier
    predicted_labels = knn.predict(pca_faces)

    # Binarize the labels (one-vs-rest approach for multiclass)
    binary_labels = label_binarize(labels, classes=np.unique(labels))

    # Predict the probabilities for each class (for ROC)
    predicted_probs = knn.predict_proba(pca_faces)

    # Calculate ROC curve for each class
    fpr = {}
    tpr = {}
    thresholds = {}
    for i in range(binary_labels.shape[1]):
        fpr[i], tpr[i], thresholds[i] = roc_curve(binary_labels[:, i], predicted_probs[:, i])

    # Create and show ROCGraphWindow for each class
    for i in range(binary_labels.shape[1]):
        # Initialize the ROC window and pass fpr and tpr for the current class
        roc_window = ROCGraphWindow(fpr[i], tpr[i])
        roc_window.show()


if __name__ == "__main__":
    app = QApplication([])
    window = ImageKNNClassifier()

    # Create a layout for the window to hold the widgets
    layout = QVBoxLayout()
    
    # Add the knn_widget where the results will be displayed
    window.knn_widget = QLabel("KNN Results")
    layout.addWidget(window.knn_widget)

    # Add other widgets such as dataset_button, test_image_button, etc.
    window.dataset_button.clicked.connect(lambda: handle_dataset_selection(window))
    window.apply_button.clicked.connect(lambda: apply_knn_classification(window))
    window.test_image_button.clicked.connect(lambda: handle_test_image_selection(window))
    window.roc_button.clicked.connect(lambda: show_roc_curve(window))
    
    window.setLayout(layout)  # Set the layout of the window
    
    window.show()
    app.exec_()
