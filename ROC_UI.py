from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, QLabel, 
                            QScrollArea, QHBoxLayout, QSizePolicy)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from itertools import cycle

class ROCGraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROC Curve Viewer - Face Recognition")
        self.setGeometry(300, 150, 1000, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Title
        title = QLabel("Face Recognition Performance - ROC Curves")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # Set size policy to make it expand
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Initialize axes
        self.ax = self.fig.add_subplot(111)
        
        # Store the window reference
        self.setAttribute(Qt.WA_DeleteOnClose, False)

    def plot_roc_curves(self, roc_curves, label_dict):
        """Plot all ROC curves on a single graph"""
        self.ax.clear()
        
        # Colors for different curves
        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'])
        
        # Plot each curve
        for (class_label, color) in zip(roc_curves.keys(), colors):
            curve_data = roc_curves[class_label]
            label = f"{label_dict.get(class_label, class_label)} (AUC = {curve_data['auc']:.2f})"
            self.ax.plot(curve_data['fpr'], curve_data['tpr'], 
                        color=color, lw=2, label=label)
        
        # Plot random classifier line
        self.ax.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Formatting
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')
        self.ax.set_title('Combined ROC Curves')
        self.ax.legend(loc="lower right", bbox_to_anchor=(1.04, 0))
        self.ax.grid(True)
        
        # Adjust layout
        self.fig.tight_layout()
        self.canvas.draw()