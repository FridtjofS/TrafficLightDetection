from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter
from PyQt6.QtCore import Qt, QPoint
import shutil
import json
from pathlib import Path

import sys
import os

# fix relative imports
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, path)

import annotool.constants as const

# This App is an annotation Tool, which loads in an Image, allows the user to draw a bounding box around the object of interest, as well as select one of four traffic light states and then saves the image with the bounding box coordinates in a text file.



class AnnotationWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Light Annotation Tool")
        self.next = QPushButton("Next Image")
        self.last = QPushButton("Previous Image")
        self.next.clicked.connect(self.next_image)
        self.last.clicked.connect(self.previous_image)
        
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.next, 2, 1)
        self.layout.addWidget(self.last, 2, 0)
        
        self.last.setEnabled(False)

        # bounding box coordinates
        self.origin = None
        self.end = None
        # traffic light state
        self.state = None
        # move bounding box origin with mouse movement
        self.move_origin = False

        # load the first image into the GUI
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
            return

        # if hit enter, start next_image
        # only allowed if bounding box and traffic light state are set
        elif event.key() == 16777220:
            if self.origin and self.end and self.state:
                self.next_image()
            else:
                print("Please set bounding box and traffic light state")
            return

        # if press down space, move origin
        elif event.key() == 32:
            if self.origin and self.end:
                # (handle move origin with mouse movement in mouseMoveEvent)
                self.move_origin = True
            else:
                print("Please set origin")

        # traffic light states
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

        # if press ESC, close window
        elif event.key() == 16777216: # ESC
            self.close()
            print("Close")

        # if all conditions are met, enable next button
        if self.origin and self.end and self.state and self.origin != self.end:
                self.next.setEnabled(True)

    def keyReleaseEvent(self, event):
        # if release space, stop moving origin
        if event.key() == 32:
            self.move_origin = False

    def next_image(self):
        self.save_image_with_annotations(self.get_path())
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

        # align top, so that coordinates can be mapped correctly
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)

        # load image and scale it to fit into GUI
        self.pixmap = QPixmap(path)
        self.pixmap = self.pixmap.scaled(const.IMAGE_WIDTH, const.IMAGE_HEIGHT)

        # set image as label
        self.label.setPixmap(self.pixmap)
        self.layout.addWidget(self.label, 0, 0, 1, 2)
        self.show()

    def mousePressEvent(self, event):
        # check if it was click on image (self.label)
        # if yes, get coordinates of click
        # relative to the pixmap

        # only relevant if left mouse button was clicked on image
        if event.button() != Qt.MouseButton.LeftButton or not self.label.underMouse():
            return
        
        # get coordinates of click relative to the pixmap and set as bbox origin
        self.origin = event.pos()
        self.origin = self.label.mapFrom(self, self.origin)

        # check if origin is outside of image
        if self.origin.x() > self.pixmap.width() or self.origin.y() > self.pixmap.height():
            self.origin = None
            return


    def mouseMoveEvent(self, event):
        if not self.origin:
            return

        # if space is pressed, move origin with mouse movement
        if self.move_origin and self.end:
            # and keep bounding box size
            difference = event.pos() - self.end
            self.origin += difference
            self.origin = self.label.mapFrom(self, self.origin)
            self.end = event.pos()
            self.end = self.label.mapFrom(self, self.end)
        else:
            # else, set end of bounding box to mouse position
            self.end = event.pos()
            self.end = self.label.mapFrom(self, self.end)

            # make sure end is not outside of image
            self.end = QPoint(min(self.pixmap.width(), self.end.x()), min(self.pixmap.height(), self.end.y()))
        
        self.update_bounding_box()


    def mouseReleaseEvent(self, event):
        if not self.origin or not self.end:
            return
        
        # set end of bounding box to mouse position
        self.end = event.pos()
        self.end = self.label.mapFrom(self, self.end)

        # make sure end is not outside of image
        self.end = QPoint(min(self.pixmap.width(), self.end.x()), min(self.pixmap.height(), self.end.y()))
        
        # draw bounding box
        self.update_bounding_box()
        # enable next button if all conditions are met
        if self.origin and self.end and self.state and self.origin != self.end:
                self.next.setEnabled(True)
        
    
    # draw bounding box on image
    def update_bounding_box(self):
        x1 = min(self.origin.x(), self.end.x())
        y1 = min(self.origin.y(), self.end.y())
        x2 = max(self.origin.x(), self.end.x())
        y2 = max(self.origin.y(), self.end.y())
        
        canvas = self.pixmap.copy()
        painter = QPainter(canvas)

        pen = QPen(QColor(255, 255, 255))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(1)

        painter.setPen(pen)
        painter.drawRect(x1, y1, x2-x1, y2-y1)
        self.label.setPixmap(canvas)
        del painter
        del canvas
        self.update()
    
    
    # copy image into "annotations" directory, and save bounding box coordinates and traffic light state in json file
    def save_image_with_annotations(self, path):
        if not self.origin or not self.end or not self.state or self.origin == self.end:
            return
        
        # save image in "annotations" directory
        shutil.copy(path, os.path.join("annotations", os.path.basename(path)))

        # save bounding box coordinates and traffic light state in json file with same name as image
        with open(os.path.join("annotations", Path(path).stem + ".json" ), 'w') as f:
            json.dump({
                'bounding_box': {
                    'x1': min(self.origin.x(), self.end.x()),
                    'y1': min(self.origin.y(), self.end.y()),
                    'x2': max(self.origin.x(), self.end.x()),
                    'y2': max(self.origin.y(), self.end.y())
                },
                'state': self.state
            }, f)


    # delete image from "frames" directory
    def delete_image(self):
        pass

