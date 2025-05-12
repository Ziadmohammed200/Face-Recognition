import sys
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget,
                             QSpinBox, QSizePolicy, QScrollArea, QGridLayout,
                             QFrame, QGroupBox, QSlider, QProgressBar)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize


class ImageKNNClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image K-Nearest Neighbors Classifier")
        self.setGeometry(100, 100, 1200, 900)

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QGroupBox {
                font: bold 12pt "Segoe UI";
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 20px;
                padding: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font: bold 11pt "Segoe UI";
                transition: all 0.3s;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
            QLabel {
                font: 10pt "Segoe UI";
                color: #2c3e50;
            }
            QSpinBox, QSlider {
                font: 10pt "Segoe UI";
            }
            QSlider::groove:horizontal {
                height: 10px;
                background: #d0d0d0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 10px;
                text-align: center;
                height: 20px;
                font: 9pt "Segoe UI";
                background: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 10px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        # Initialize variables
        self.dataset_path = None
        self.test_image_path = None
        self.dataset_features = None
        self.dataset_filenames = []
        self.image_size = (100, 100)  # Size for processing images

        # Create the main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setCentralWidget(main_widget)

        # Application title
        title_label = QLabel("Image K-Nearest Neighbors Classifier")
        title_font = QFont("Arial", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        # Top controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)

        # Dataset upload button with icon
        self.dataset_button = QPushButton("  Upload Images Dataset")
        self.dataset_button.setIcon(self.style().standardIcon(self.style().SP_DialogOpenButton))
        self.dataset_button.setIconSize(QSize(24, 24))
        self.dataset_button.setMinimumHeight(40)
        buttons_layout.addWidget(self.dataset_button)

        # Test image upload button with icon
        self.test_image_button = QPushButton("  Upload Test Image")
        self.test_image_button.setIcon(self.style().standardIcon(self.style().SP_FileDialogStart))
        self.test_image_button.setIconSize(QSize(24, 24))
        self.test_image_button.setMinimumHeight(40)
        buttons_layout.addWidget(self.test_image_button)

        controls_layout.addLayout(buttons_layout)

        # K value control
        k_layout = QHBoxLayout()
        k_label = QLabel("Number of Neighbors (K):")
        k_label.setFont(QFont("Arial", 10))
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(1)
        self.k_spinbox.setMaximum(20)
        self.k_spinbox.setValue(5)
        self.k_spinbox.valueChanged.connect(self.k_value_changed)
        self.k_spinbox.setMinimumHeight(30)

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(20)
        self.k_slider.setValue(5)
        self.k_slider.setTickPosition(QSlider.TicksBelow)
        self.k_slider.setTickInterval(1)
        self.k_slider.valueChanged.connect(self.k_spinbox.setValue)
        self.k_spinbox.valueChanged.connect(self.k_slider.setValue)

        k_layout.addWidget(k_label)
        k_layout.addWidget(self.k_slider)
        k_layout.addWidget(self.k_spinbox)
        controls_layout.addLayout(k_layout)

        # Progress bar for loading datasets
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)  # Hidden by default
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        controls_layout.addLayout(progress_layout)

        main_layout.addWidget(controls_group)

        # Status label
        # Replace the status label with an "Apply" button
        self.apply_button = QPushButton("Apply")
        self.apply_button.setFont(QFont("Arial", 10))
        self.apply_button.setStyleSheet(
            """
            QPushButton {
                color: #ffffff;
                font-weight: bold;
                padding: 10px;
                background-color: #27ae60;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1e8449;
            }
            QPushButton:pressed {
                background-color: #145a32;
            }
            """
        )
        self.apply_button.setMinimumHeight(40)
        main_layout.addWidget(self.apply_button)

        # Create display area
        display_layout = QHBoxLayout()
        display_layout.setSpacing(20)

        # Test image display
        test_group = QGroupBox("Test Image")
        test_layout = QVBoxLayout(test_group)

        self.test_image_label = QLabel()
        self.test_image_label.setAlignment(Qt.AlignCenter)
        self.test_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.test_image_label.setMinimumSize(350, 350)
        self.test_image_label.setFrameShape(QFrame.StyledPanel)
        self.test_image_label.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            border: 2px dashed #bdc3c7;
        """)

        test_layout.addWidget(self.test_image_label)

        # Image information label
        self.test_info_label = QLabel("No image loaded")
        self.test_info_label.setAlignment(Qt.AlignCenter)
        test_layout.addWidget(self.test_info_label)

        display_layout.addWidget(test_group)

        # K-nearest neighbors display
        knn_group = QGroupBox("K Nearest Neighbors")
        knn_layout = QVBoxLayout(knn_group)

        # Scroll area for KNN images
        self.knn_scroll = QScrollArea()
        self.knn_scroll.setWidgetResizable(True)
        self.knn_widget = QWidget()
        self.knn_widget.setStyleSheet("background-color: white; border-radius: 10px;")
        self.knn_grid = QGridLayout(self.knn_widget)
        self.knn_grid.setSpacing(15)
        self.knn_grid.setContentsMargins(20, 20, 20, 20)
        self.knn_scroll.setWidget(self.knn_widget)
        knn_layout.addWidget(self.knn_scroll)
        self.knn_images_result = QLabel("No neighbors displayed")
        self.knn_images_result.setAlignment(Qt.AlignCenter)
        self.knn_images_result.setFont(QFont("Segoe UI", 10))
        self.knn_images_result.setStyleSheet("color: #2c3e50; margin-top: 10px;")
        knn_layout.addWidget(self.knn_images_result)

        display_layout.addWidget(knn_group)
        main_layout.addLayout(display_layout, 1)


    def clear_knn_display(self):
        """Clear all widgets from the KNN display grid"""
        while self.knn_grid.count():
            item = self.knn_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def k_value_changed(self):
        """Called when the K value is changed"""
        if self.test_image_path and self.dataset_features is not None:
            self.find_knn()


def apply_dark_mode_style(app):
    """Apply dark mode styling to the entire application"""
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, QColor(230, 230, 230))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    dark_palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))

    app.setPalette(dark_palette)

    # Additional stylesheet for dark mode
    app.setStyleSheet("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a2a2a; 
            border: 1px solid #3498db; 
            border-radius: 3px; 
        }
        QGroupBox {
            border: 2px solid #3498db;
            border-radius: 5px;
            margin-top: 1em;
            color: #3498db;
        }
        QPushButton {
            background-color: #2c3e50;
            color: white;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3498db;
        }
        QPushButton:pressed {
            background-color: #1f618d;
        }
        QSpinBox {
            background-color: #353535;
            color: white;
            border: 2px solid #3498db;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999;
            background: #2a2a2a;
        }
        QSlider::handle:horizontal {
            background: #3498db;
        }
        QProgressBar {
            border: 2px solid #3498db;
            background-color: #353535;
            color: white;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #3498db;
        }
    """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Uncomment the line below to enable dark mode
    # apply_dark_mode_style(app)

    window = ImageKNNClassifier()
    window.show()
    sys.exit(app.exec_())