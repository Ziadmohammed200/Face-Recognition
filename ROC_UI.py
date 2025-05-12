import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt
import pyqtgraph as pg


class ROCGraphWindow(QMainWindow):
    def __init__(self, fpr=None, tpr=None):
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

        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'True Positive Rate')
        self.plot_widget.setLabel('bottom', 'False Positive Rate')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()

        layout.addWidget(self.plot_widget)

        if fpr and tpr:
            self.plot_roc(fpr, tpr)

    def plot_roc(self, fpr, tpr):
        self.plot_widget.plot(fpr, tpr, pen=pg.mkPen(color=(52, 152, 219), width=3), name="ROC Curve")
        self.plot_widget.plot([0, 1], [0, 1], pen=pg.mkPen(style=Qt.DashLine, color='gray'), name="Random Classifier")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example fake ROC values
    fpr = [0.0, 0.1, 0.2, 0.4, 1.0]
    tpr = [0.0, 0.4, 0.7, 0.9, 1.0]

    window = ROCGraphWindow(fpr, tpr)
    window.show()

    sys.exit(app.exec_())
