from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter
from PyQt6.QtCore import Qt, QPoint

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
        
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.next, 2, 1)
        self.layout.addWidget(self.last, 2, 0)
        self.next.clicked.connect(self.next_image)
        self.last.clicked.connect(self.previous_image)
        self.last.setEnabled(False)
        self.origin = None
        self.end = None
        self.state = 1

        self.next_image()

        

    # listen for key press events
    def keyPressEvent(self, event):
        # if hit backspace, start previous_image
        # allowed, if image is not first image
        # current input is not saved
        if event.key() == 16777219:
            if self.last:
                self.previous_image()
            else:
                print("This is the first image")

        # if hit enter, start next_image
        # only allowed if bounding box and traffic light state are set
        elif event.key() == 16777220:
            if self.origin and self.end and self.state:
                self.next_image()
            else:
                print("Please set bounding box and traffic light state")

        elif event.key() == 49: # 1
            self.state = 1
            print("State: Red")
        elif event.key() == 50: # 2
            self.state = 2
            print("State: Red-Yellow")
        elif event.key() == 51: # 3
            self.state = 3
            print("State: Yellow")
        elif event.key() == 52: # 4
            self.state = 4
            print("State: Green")
        elif event.key() == 16777216: # ESC
            self.close()
            print("Close")
        if self.origin and self.end:
                self.next.setEnabled(True)

    def next_image(self):
        self.next.setEnabled(False)
        self.state = None
        self.origin = None
        self.end = None
        self.load_image(self.get_path())
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
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.pixmap = QPixmap(path)
        self.pixmap = self.pixmap.scaled(const.IMAGE_WIDTH, const.IMAGE_HEIGHT)

        self.label.setPixmap(self.pixmap)
        self.layout.addWidget(self.label, 0, 0, 1, 2)
        self.show()

    def mousePressEvent(self, event):
        # check if it was click on image (self.label)
        # if yes, get coordinates of click
        # relative to the pixmap

        if event.button() != Qt.MouseButton.LeftButton or not self.label.underMouse():
            return
        
        self.origin = event.pos()
        self.origin = self.label.mapFrom(self, self.origin)

        if self.origin.x() > self.pixmap.width() or self.origin.y() > self.pixmap.height():
            self.origin = None
            return


    def mouseMoveEvent(self, event):
        if not self.origin:
            return
        
        self.end = event.pos()
        self.end = self.label.mapFrom(self, self.end)

        if self.end.x() > self.pixmap.width() or self.end.y() > self.pixmap.height():
            self.end = None
            return
        
        self.update_bounding_box()

    def mouseReleaseEvent(self, event):
        if not self.origin and not self.end:
            return
        
        self.end = event.pos()
        self.end = self.label.mapFrom(self, self.end)

        if self.end.x() > self.pixmap.width() or self.end.y() > self.pixmap.height():
            self.end = None
            return
        

        self.update_bounding_box()
        if self.origin and self.end and self.state:
                self.next.setEnabled(True)

        print(self.origin, self.end)


        #self.origin = self.label.mapFrom(self, self.origin)
        #self.origin = self.pixmap.mapFrom(self.label, self.origin)
        
    

    def update_bounding_box(self):
        x1 = min(self.origin.x(), self.end.x())
        y1 = min(self.origin.y(), self.end.y())
        x2 = max(self.origin.x(), self.end.x())
        y2 = max(self.origin.y(), self.end.y())
        
        canvas = self.pixmap.copy()
        painter = QPainter(canvas)
        pen2 = QPen(QColor(255, 255, 255))
        pen2.setStyle(Qt.PenStyle.DotLine)
        pen2.setWidth(2)
        painter.setPen(pen2)
        painter.drawRect(x1, y1, x2-x1, y2-y1)
        self.label.setPixmap(canvas)
        del painter
        del canvas
        self.update()
    
    # upon mouse drag, draw bounding box and save coordinates
    def get_bounding_box(self, pixmap):
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

