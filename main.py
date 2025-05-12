from PyQt5.QtWidgets import QApplication, QFileDialog
from GUI import ImageKNNClassifier
from sklearn.preprocessing import label_binarize
from PyQt5.QtGui import QImage, QPixmap
from sklearn.neighbors import KNeighborsClassifier
from PyQt5.QtWidgets import QLabel, QFrame
from PyQt5.QtCore import Qt, QSize
from eigen_face import generate_dataset, generate_covariance_matrix, get_lambdas_and_eigenvectors, get_k_vectors
from eigen_face import get_a_coefficients_dataset, get_a_coefficients_image, k_nearest_neighbour
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
from PyQt5.QtWidgets import QVBoxLayout, QWidget

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
        if file_path.lower().endswith('.pgm'):
            window.test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            rgb_image = cv2.cvtColor(window.test_image, cv2.COLOR_GRAY2RGB)
        else:
            window.test_image = cv2.imread(file_path)
            rgb_image = cv2.cvtColor(window.test_image, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
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
    if not hasattr(window, 'dataset_path') or not window.dataset_path:
        print("Error: No dataset selected!")
        return
    if not hasattr(window, 'test_image') or window.test_image is None:
        print("Error: No test image loaded!")
        return

    k_value = window.k_spinbox.value()
    
    try:
        images, labels = generate_dataset(window.dataset_path)
        cov_matrix, centered_data, mean = generate_covariance_matrix(images)
        eigvals, eigvecs = get_lambdas_and_eigenvectors(cov_matrix, centered_data)
        k_eig, top_eigenvectors = get_k_vectors(eigvals, eigvecs, 0.95)
        
        a_coeffs = get_a_coefficients_dataset(top_eigenvectors, images, mean)
        test_coeffs = get_a_coefficients_image(top_eigenvectors, window.test_image, mean)
        
        predicted_label, neighbors, distances = k_nearest_neighbour(a_coeffs, test_coeffs, labels, k_value)
        
        window.clear_knn_display()
        grid_cols = max(1, math.ceil(math.sqrt(k_value)))
        grid_rows = math.ceil(k_value / grid_cols)
        neighbor_display_size = QSize(150, 150)

        for i in range(k_value):
            neighbor_index = neighbors[i]
            neighbor_image_data = images[neighbor_index]
            neighbor_label_text = f"Neighbor {i+1}\nClass: {labels[neighbor_index]}\nDist: {distances[i]:.2f}"

            if len(neighbor_image_data.shape) == 2:
                q_img = QImage(neighbor_image_data.data, neighbor_image_data.shape[1], neighbor_image_data.shape[0],
                               neighbor_image_data.shape[1], QImage.Format_Grayscale8)
            else:
                rgb_image = cv2.cvtColor(neighbor_image_data, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                               rgb_image.shape[1] * rgb_image.shape[2], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                neighbor_display_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            image_label = QLabel()
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(neighbor_display_size)

            image_label.setFrameShape(QFrame.StyledPanel)
            image_label.setStyleSheet("""
                background-color: white;
                border-radius: 10px;
                padding: 5px;
                border: 2px dashed #bdc3c7;
            """)

            info_label = QLabel(neighbor_label_text)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setWordWrap(True)
            info_label.setFont(window.font())

            row = i // grid_cols
            col = i % grid_cols

            item_layout = QVBoxLayout()
            item_layout.addWidget(image_label)
            item_layout.addWidget(info_label)
            item_layout.setAlignment(Qt.AlignCenter)

            item_widget = QWidget()
            item_widget.setLayout(item_layout)

            window.knn_grid.addWidget(item_widget, row, col)

        print(f"Displayed {k_value} K-Nearest Neighbors in grid.")

    except Exception as e:
        print(f"Classification failed: {str(e)}")

def load_faces_from_directory(dataset_path, target_size=(64, 64)):
    faces = []
    labels = []
    label_names = []
    label_dict = {}
    
    for label, person_name in enumerate(sorted(os.listdir(dataset_path))):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        label_dict[label] = person_name
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                image_resized = cv2.resize(image, target_size)
                faces.append(image_resized.flatten())
                labels.append(label)
                label_names.append(person_name)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    return np.array(faces), np.array(labels), label_dict

def plot_roc_curves( roc_curves, label_dict):
    """Plot ROC curves for multi-class classification.
    
    Args:
        roc_curves (dict): Dictionary containing ROC curve data for each class
        label_dict (dict): Mapping from class labels to human-readable names
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    # Plot each class's ROC curve
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for class_label, color in zip(roc_curves.keys(), colors):
        fpr = roc_curves[class_label]['fpr']
        tpr = roc_curves[class_label]['tpr']
        auc_score = roc_curves[class_label]['auc']
        plt.plot(fpr, tpr, color=color, 
                label=f'{label_dict.get(class_label, class_label)} (AUC = {auc_score:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Face Recognition')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_roc_curve(window):
    if not hasattr(window, 'dataset_path') or not window.dataset_path:
        print("Error: No dataset selected!")
        return

    try:
        # Load and preprocess data
        faces, labels, label_dict = load_faces_from_directory(window.dataset_path)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            faces, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Apply PCA
        pca = PCA(n_components=min(50, X_train.shape[0], X_train.shape[1]), svd_solver='auto')
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Train KNN classifier
        k_value = window.k_spinbox.value()
        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train_pca, y_train)
        
        # Get predicted probabilities for ROC curve
        y_score = knn.predict_proba(X_test_pca)
        
        # Binarize the labels for multiclass ROC
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        
        # Compute ROC curve and ROC area for each class
        roc_curves = {}
        for class_label in np.unique(labels):
            class_idx = np.where(lb.classes_ == class_label)[0][0]
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
            roc_auc = auc(fpr, tpr)
            roc_curves[class_label] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        # Plot the ROC curves
        plot_roc_curves(roc_curves, label_dict)
        
    except Exception as e:
        print(f"Error generating ROC curve: {str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    window = ImageKNNClassifier()
    
    window.dataset_button.clicked.connect(lambda: handle_dataset_selection(window))
    window.apply_button.clicked.connect(lambda: apply_knn_classification(window))
    window.test_image_button.clicked.connect(lambda: handle_test_image_selection(window))
    window.roc_button.clicked.connect(lambda: show_roc_curve(window))
        
    window.show()
    app.exec_()