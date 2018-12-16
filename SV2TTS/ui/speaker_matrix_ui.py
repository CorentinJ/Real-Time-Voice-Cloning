from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4
from vlibs import nowarnings
from functools import partial
from ui import gui
import numpy as np


class SpeakerMatrixUI(gui.UI):
    def __init__(self, speakers, partial_utterances, title="Hey"):
        self.speakers = speakers
        self.partial_utterances = partial_utterances
        self.title = title
    
    def draw_images(self, redraw_source=True):
        self.edit_frame.setPixmap(gui.standard_to_pixmap(self.edited_image))
    
    def _on_click(self, event):      
        # Edit the image
        coords = (event.y() // self.scale, event.x() // self.scale)
        button = event.button()
        self.on_click(self.index, self.edited_image, coords, button)
    
    def setup_ui(self, window):
        print('hey')
        pass
        # # Find the layout parameters
        # dims = np.array(self.source_image.shape) * self.scale
        # 
        # # Create a frame for each image
        # self.source_frame = QLabel(window)
        # self.source_frame.setGeometry(10, 10, dims[1], dims[0])
        # self.source_frame.setScaledContents(True)
        # 
        # self.edit_frame = QLabel(window)
        # self.edit_frame.setGeometry(20 + dims[1], 10, dims[1], dims[0])
        # self.edit_frame.setScaledContents(True)
        # GUI.make_clickable(self.edit_frame, self._on_click)
        # 
        # # Create the buttons
        # x_min = dims[1] * 2 + 30
        # y_min = dims[0] + 20
        # 
        # button = QPushButton(window, text="Next")
        # button.setGeometry(x_min, 10, 140, 60)
        # GUI.make_clickable(button, key=Qt.Key_A).connect(self.next_images)
        # 
        # button = QPushButton(window, text="Undo")
        # button.setGeometry(x_min, 80, 140, 60)
        # GUI.make_clickable(button, key=Qt.Key_Z).connect(self.undo)
        # 
        # # button = QPushButton(window, text="All black")
        # # button.setGeometry(x_min, 150, 140, 60)
        # # GUI.make_clickable(button, key=Qt.Key_E).connect(self.all_black)
        # 
        # button = QPushButton(window, text="Skip")
        # button.setGeometry(x_min, 220, 140, 60)
        # GUI.make_clickable(button, key=Qt.Key_R).connect(partial(self.next_images, True))
        # 
        # # Fit the window to the contents
        # window.resize(x_min + 150, y_min)
        # self.draw_images()

