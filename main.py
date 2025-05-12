from PyQt5.QtWidgets import QApplication, QFileDialog
from GUI import ImageKNNClassifier
from sklearn.preprocessing import label_binarize
from PyQt5.QtGui import QImage, QPixmap
from sklearn.neighbors import KNeighborsClassifier
from PyQt5.QtWidgets import QLabel, QFrame
from PyQt5.QtCore import Qt, QSize
from eigen_face import generate_dataset, generate_covariance_matrix, get_lambdas_and_eigenvectors, get_k_vectors
from eigen_face import get_a_coefficients_dataset, get_a_coefficients_image , k_nearest_neighbour
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import math
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
        
        # Clear existing KNN display in the grid
        window.clear_knn_display()

        # Determine grid layout (e.g., as square as possible)
        grid_cols = max(1, math.ceil(math.sqrt(k_value))) # Ensure at least 1 column
        grid_rows = math.ceil(k_value / grid_cols)

        # Define a fixed size for each neighbor image display for consistency
        # You can adjust this size as needed to fit your layout
        neighbor_display_size = QSize(150, 150) # Width, Height for each neighbor image label

        for i in range(k_value):
            neighbor_index = neighbors[i]
            neighbor_image_data = images[neighbor_index] # Get the actual image data
            neighbor_label_text = f"Neighbor {i+1}\nClass: {labels[neighbor_index]}\nDist: {distances[i]:.2f}"

            # Convert image data to QPixmap
            # Ensure the image is in a format that can be displayed by QImage
            if len(neighbor_image_data.shape) == 2: # Grayscale
                q_img = QImage(neighbor_image_data.data, neighbor_image_data.shape[1], neighbor_image_data.shape[0],
                               neighbor_image_data.shape[1], QImage.Format_Grayscale8)
            else: # Assuming BGR if 3 channels (from OpenCV)
                # Convert BGR to RGB for QImage
                rgb_image = cv2.cvtColor(neighbor_image_data, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                               rgb_image.shape[1] * rgb_image.shape[2], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            # Scale pixmap to fit the label's fixed size
            scaled_pixmap = pixmap.scaled(
                neighbor_display_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Create a QLabel for the image
            image_label = QLabel()
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(neighbor_display_size) # Set fixed size for consistent layout

            # Apply styling similar to test_image_label
            image_label.setFrameShape(QFrame.StyledPanel)
            image_label.setStyleSheet("""
                background-color: white;
                border-radius: 10px;
                padding: 5px; /* Adjust padding for smaller labels */
                border: 2px dashed #bdc3c7;
            """)

            # Create a QLabel for the text info
            info_label = QLabel(neighbor_label_text)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setWordWrap(True) # Allow text to wrap if too long
            info_label.setFont(window.font()) # Inherit font from main window for consistency

            # Calculate row and column for the grid layout
            row = i // grid_cols
            col = i % grid_cols

            # Create a vertical layout for each image and its info
            item_layout = QVBoxLayout()
            item_layout.addWidget(image_label)
            item_layout.addWidget(info_label)
            item_layout.setAlignment(Qt.AlignCenter) # Center the content within the item layout

            # Create a QWidget to hold the item_layout for consistent spacing in the grid
            item_widget = QWidget()
            item_widget.setLayout(item_layout)

            # Add the widget to the grid layout
            window.knn_grid.addWidget(item_widget, row, col)

        print(f"Displayed {k_value} K-Nearest Neighbors in grid.")

        # --- END OF MODIFICATIONS FOR KNN DISPLAY ---

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


def show_roc_curve(dataset_path):
    # Validate dataset path
    if not dataset_path:
        print("Error: No dataset selected!")
        return

    # Load the face dataset
    faces, labels = load_faces_from_directory(window.dataset_path)

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

    # Create subplots to show the ROC curves for each class
    num_classes = binary_labels.shape[1]
    fig, axs = plt.subplots(1, num_classes, figsize=(12, 6))

    if num_classes == 1:
        axs = [axs]  # Ensure axs is iterable if there's only one subplot

    # Plot each ROC curve in its corresponding subplot
    for i in range(num_classes):
        axs[i].plot(fpr[i], tpr[i], color=(52/255, 152/255, 219/255), lw=3, label=f"ROC Curve {i+1}")
        axs[i].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label="Random Classifier")
        axs[i].set_title(f"ROC Curve {i+1}")
        axs[i].set_xlabel("False Positive Rate")
        axs[i].set_ylabel("True Positive Rate")
        axs[i].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app = QApplication([])
    window = ImageKNNClassifier()

    

    # Add other widgets such as dataset_button, test_image_button, etc.
    window.dataset_button.clicked.connect(lambda: handle_dataset_selection(window))
    window.apply_button.clicked.connect(lambda: apply_knn_classification(window))
    window.test_image_button.clicked.connect(lambda: handle_test_image_selection(window))
    window.roc_button.clicked.connect(lambda: show_roc_curve(window))
        
    window.show()
    app.exec_()
