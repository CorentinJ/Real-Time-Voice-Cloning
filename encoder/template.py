from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt4.QtGui import *
import numpy as np
import sys


class TemplateUI(QDialog):
    def __init__(self):
        # Create the ui
        app = QApplication(sys.argv)
        super().__init__(None)
        
        # Display it and stay in mainloop until the window is closed
        self.setup_ui()
        self.show()
        app.exec_()
    
    def setup_ui(self):
        root_layout = QHBoxLayout()
        
        # A menu
        menu_layout = QVBoxLayout()
        root_layout.addLayout(menu_layout)
        a_first_button = QPushButton('Plot again')
        a_first_button.clicked.connect(lambda: self.plot_stuff())
        a_first_button.setMinimumHeight(200)
        menu_layout.addWidget(a_first_button)
        menu_layout.addStretch()
        a_second_button = QPushButton('Clear plot')
        a_second_button.clicked.connect(lambda: self.clear_plot())
        a_second_button.setMinimumHeight(200)
        menu_layout.addWidget(a_second_button)
        
        # A plot
        umap_canvas = FigureCanvas(Figure())
        root_layout.addWidget(umap_canvas)
        self.ax = umap_canvas.figure.subplots()
        self.plot_stuff()
        
        self.setLayout(root_layout)
        max_size = QDesktopWidget().availableGeometry(self).size()
        self.resize(max_size * 0.8)
    
    def plot_stuff(self):
        self.clear_plot()
        x = np.linspace(-3, 8, 200)
        y1 = np.sin(x) + (1 / 3) * x ** 3 - 2 * x ** 2
        y2 = y1 + np.random.normal(size=200) * 3
        self.ax.plot(x, y1, label="potatinuity")
        self.ax.scatter(x, y2, marker="x", color="c", label="noisy potatinuity")
        self.ax.set_xlabel("Potatoes")
        self.ax.set_ylabel("Tomatoes")
        self.ax.set_title("Potatoes vs. Tomatoes")
        self.ax.legend()
        self.ax.figure.canvas.draw()
    
    def clear_plot(self):
        self.ax.clear()
        self.ax.figure.canvas.draw()

if __name__ == '__main__':
    TemplateUI()
