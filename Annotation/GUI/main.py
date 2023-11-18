from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap

import sys
import os

# fix relative imports
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, path)

import annotool.constants as const

# This App is an annotation Tool, which loads in an Image, allows the user to draw a bounding box around the object of interest, as well as select one of four traffic light states and then saves the image with the bounding box coordinates in a text file.



class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Light Annotation Tool")
        self.next = QPushButton("Next Image")
        self.last = QPushButton("Previous Image")
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.next)
        self.layout.addWidget(self.last)

        
        # remove pixmap from layout
        # remove label from layout

        #self.layout.removeWidget(label)
        #label.deleteLater()
        #del label
        #del pixmap
        #del path
        #self.update()

        

    # listen for key press events
    def keyPressEvent(self, event):
        # if hit backspace, start previous_image
        # allowed, if image is not first image
        # current input is not saved
        if event.key() == 16777219:
            self.previous_image()

        # if hit enter, start next_image
        # only allowed if bounding box and traffic light state are set
        elif event.key() == 16777220:
            self.next_image()

        else:
            pass

    def next_image(self):
        # get filepath of next image
        # load in next image
        # get user input for bounding box and traffic light state
        # save image with bounding box coordinates and traffic light state in the directory "annotations"
        pass

    def previous_image(self):
        # get filepath of previous image
        # load in previous image
        # get user input for bounding box and traffic light state
        # update json file with bounding box coordinates and traffic light state
        pass
    
    # get path of next image in directory "frames"
    def get_path(self):
        return "frames/django.jpg"
    
    # load image into GUI
    def load_image(self, path):
        label = QLabel(self)
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(const.IMAGE_WIDTH, const.IMAGE_HEIGHT)
        label.setPixmap(pixmap)
        self.layout.addWidget(label)
        self.show()
        return label, pixmap
    
    # upon mouse drag, draw bounding box and save coordinates
    def get_bounding_box(self):
        pass

    # upon click of 1, 2, 3, or 4, save traffic light state
    def get_traffic_light_state(self):
        pass
    
    # copy image into "annotations" directory, and save bounding box coordinates and traffic light state in json file
    def save_image_with_annotations(self):
        pass

    # delete image from "frames" directory
    def delete_image(self):
        pass

    



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())

