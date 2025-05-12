import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from sklearn.metrics import roc_curve
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class ROCGraphWindow(QMainWindow):
    def __init__(self, fpr_list=None, tpr_list=None):
        super().__init__()
        self.setWindowTitle("ROC Curve Viewer")
        self.setGeometry(300, 150, 800, 600)

        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        title = QLabel("Receiver Operating Characteristic (ROC) Curve")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create matplotlib subplot to plot multiple ROC curves
        self.fig, self.axs = plt.subplots(1, len(fpr_list), figsize=(12, 6))
        if len(fpr_list) == 1:  # Make axs iterable
            self.axs = [self.axs]
        
        # Plot each ROC curve on the subplots
        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            self.axs[i].plot(fpr, tpr, color=(52/255, 152/255, 219/255), lw=3, label=f"ROC Curve {i+1}")
            self.axs[i].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label="Random Classifier")
            self.axs[i].set_title(f"ROC Curve {i+1}")
            self.axs[i].set_xlabel("False Positive Rate")
            self.axs[i].set_ylabel("True Positive Rate")
            self.axs[i].legend(loc="lower right")

        self.fig.tight_layout()
        self.show()

    def plot_roc(self, fpr_list, tpr_list):
        # Simply call the matplotlib plotting function when plotting all ROC curves
        self.__init__(fpr_list, tpr_list)

# Example usage
def show_roc_curve(window):
    # Dummy example: Generate multiple ROC curves for demonstration
    # Normally, you would use your classifier and the actual fpr, tpr values here
    fpr_list = [np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
    tpr_list = [np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0])]

    # Additional ROC curve data for demonstration
    fpr_list.append(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    tpr_list.append(np.array([0.0, 0.1, 0.3, 0.6, 0.8, 1.0]))

    # Create and display the ROC window with multiple curves
    roc_window = ROCGraphWindow(fpr_list, tpr_list)
    roc_window.show()

