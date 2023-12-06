from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QGridLayout, QLineEdit, QCheckBox, QScrollArea
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QIntValidator
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



class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Light Annotation Tool")
        self.next = QPushButton("Next Image")
        self.last = QPushButton("Previous Image")
        self.next.clicked.connect(self.next_image)
        self.last.clicked.connect(self.previous_image)
        
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.next, 1, 2)
        self.layout.addWidget(self.last, 1, 1)
        
        self.last.setEnabled(False)

        self.annotated = []

        # bounding box coordinates
        self.origin = None
        self.end = None
        # traffic light state
        self.state = None
        # move bounding box origin with mouse movement
        self.move_origin = False

        # load the first image into the GUI
        self.next_image()

        self.annotation_window = AnnotationWindow(self)
        self.layout.addWidget(self.annotation_window, 0, 0, 2, 1)

        

    # listen for key press events
    def keyPressEvent(self, event):
        # if hit backspace, start previous_image
        # allowed, if image is not first image
        # current input is not saved
        if event.key() == 16777219:
            self.last_annotation()

            #if self.last:
            #    self.previous_image()
            #else:
            #    print("This is the first image")
            #return

        # if hit enter, start next_image
        # only allowed if bounding box and traffic light state are set
        elif event.key() == 16777220:
            self.lock_in_bbox()

            #if self.origin and self.end and self.state:
            #    self.next_image()
            #else:
            #    print("Please set bounding box and traffic light state")
            #return

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
        elif event.key() == 53: # 5
            self.state = 5
            print("State: Off")

        # if press ESC, close window
        elif event.key() == 16777216: # ESC
            self.close()
            print("Close")
        
        self.update_bounding_box()
        self.update_annotation_window()

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
        self.annotated = []
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
        # get the first image in the directory const.IMAGE_DIR

        # check if directory is empty
        if len(os.listdir(const.IMAGE_DIR)) == 0:
            print("Directory is empty")
            return
        
        images = os.listdir(const.IMAGE_DIR)
        return os.path.join(const.IMAGE_DIR, images[0])
        


        #return "frames/django.jpg"
    
    # load image into GUI
    def load_image(self, path):
        self.label = QLabel(self)

        # align top, so that coordinates can be mapped correctly
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)

        # load image and scale it to fit into GUI
        if path == None:
            self.pixmap = QPixmap()
        else:
            self.pixmap = QPixmap(path)
        self.pixmap = self.pixmap.scaled(const.IMAGE_WIDTH, const.IMAGE_HEIGHT)

        # set focus to label, so that key press events are registered
        self.label.setFocus()

        # set image as label
        self.label.setPixmap(self.pixmap)
        self.layout.addWidget(self.label, 0, 1, 1, 2)
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
        self.update_annotation_window()


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
        self.update_annotation_window()

        # enable next button if all conditions are met
        if self.origin and self.end and self.state and self.origin != self.end:
                self.next.setEnabled(True)
        
    
    # draw bounding box on image
    def update_bounding_box(self):
        if not self.origin or not self.end:
            return

        x1 = min(self.origin.x(), self.end.x())
        y1 = min(self.origin.y(), self.end.y())
        x2 = max(self.origin.x(), self.end.x())
        y2 = max(self.origin.y(), self.end.y())
        
        canvas = self.pixmap.copy()
        painter = QPainter(canvas)

        pen = QPen(self.get_color(self.state))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(1)

        painter.setPen(pen)
        painter.drawRect(x1, y1, x2-x1, y2-y1)

        # fill bounding box with color (10% opacity)
        painter.setOpacity(0.1)
        painter.fillRect(x1, y1, x2-x1, y2-y1, self.get_color(self.state))
        painter.setOpacity(1)

        for annotation in self.annotated:
            color = self.get_color(annotation['state'])
            pen = QPen(color)
            pen.setStyle(Qt.PenStyle.SolidLine)
            pen.setWidth(2)

            painter.setPen(pen)
            painter.drawRect(annotation['x1'], annotation['y1'], annotation['x2']-annotation['x1'], annotation['y2']-annotation['y1'])
            
            # fill bounding box with color (10% opacity)
            painter.setOpacity(0.1)
            painter.fillRect(annotation['x1'], annotation['y1'], annotation['x2']-annotation['x1'], annotation['y2']-annotation['y1'], color)
            painter.setOpacity(1)

        self.label.setPixmap(canvas)
        del painter
        del canvas
        self.update()
        self.show()

    def update_annotation_window(self):
        if self.origin and self.end:
            # clear current annotation
            for i in reversed(range(self.annotation_window.current_annotation.count())):
                self.annotation_window.current_annotation.itemAt(i).widget().setParent(None)
            self.annotation_window.draw_annotation({'x1': min(self.origin.x(), self.end.x()), 'y1': min(self.origin.y(), self.end.y()), 'x2': max(self.origin.x(), self.end.x()), 'y2': max(self.origin.y(), self.end.y()), 'state': self.state}, self.annotation_window.current_annotation)
        # clear previous annotations
        for i in reversed(range(self.annotation_window.previous_annotations.count())): 
            self.annotation_window.previous_annotations.itemAt(i).widget().setParent(None)
        for annotation in self.annotated:
            self.annotation_window.draw_annotation(annotation, self.annotation_window.previous_annotations)

        self.annotation_window.show()

    def get_color(self, state):
        if state == None:
            return const.COLORS['none']
        elif state == 1:
            return const.COLORS['red']
        elif state == 2:
            return const.COLORS['red-yellow']
        elif state == 3:
            return const.COLORS['yellow']
        elif state == 4:
            return const.COLORS['green']
        elif state == 5:
            return const.COLORS['off']
        return const.COLORS['none']
    
    def lock_in_bbox(self):
        # if bounding box is not set, return
        if not self.origin or not self.end or self.origin == self.end:
            return

        # if traffic light state is not set, return
        if not self.state:
            return
        
        bbox = {
            'x1': min(self.origin.x(), self.end.x()),
            'y1': min(self.origin.y(), self.end.y()),
            'x2': max(self.origin.x(), self.end.x()),
            'y2': max(self.origin.y(), self.end.y()),
            'state': self.state
        }

        self.annotated.append(bbox)

        self.update_bounding_box()

        # clear bounding box
        self.origin = None
        self.end = None
        self.state = None

    def last_annotation(self):
        if len(self.annotated) == 0:
            print("No annotations to delete")
            return
        
        bbox = self.annotated.pop()
        self.origin = QPoint(bbox['x1'], bbox['y1'])
        self.end = QPoint(bbox['x2'], bbox['y2'])
        self.state = bbox['state']

        self.update_bounding_box()


    
    
    # copy image into "annotations" directory, and save bounding box coordinates and traffic light state in json file
    def save_image_with_annotations(self, path):
        if self.annotated == [] or not path:
            return
            #TODO open extra window to ask if user wants to save image without annotations

        
        # save image in "annotations" directory
        shutil.copy(path, os.path.join("annotations", os.path.basename(path)))

        # save bounding box coordinates and traffic light state in json file with same name as image
        json_data = {}
        for i in range(len(self.annotated)):
            json_data['traffic_light_' + str(i)] = {
                'x1': self.annotated[i]['x1'],
                'y1': self.annotated[i]['y1'],
                'x2': self.annotated[i]['x2'],
                'y2': self.annotated[i]['y2'],
                'state' : self.annotated[i]['state']
            }

        with open(os.path.join("annotations", Path(path).stem + ".json" ), 'w') as f:
            json.dump(json_data, f)

                #'bounding_box': {
                #    'x1': min(self.origin.x(), self.end.x()),
                #    'y1': min(self.origin.y(), self.end.y()),
                #    'x2': max(self.origin.x(), self.end.x()),
                #    'y2': max(self.origin.y(), self.end.y())
                #},
                #'state': self.state
            


    # delete image from "frames" directory
    def delete_image(self):
        pass


class AnnotationWindow(QWidget):
    '''
    This is where the annotations of the current image are displayed.
    One on the top of the one to be locked in,
    and a history of the ones already locked in below.
    '''
    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("Annotations")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_content.setLayout(scroll_layout)

        self.current_annotation = QVBoxLayout()
        scroll_layout.addLayout(self.current_annotation)

        self.previous_annotations = QVBoxLayout()
        scroll_layout.addLayout(self.previous_annotations)

        scroll_area.setWidget(scroll_content)

        # set width of window to fit the contents
        self.setFixedWidth(int(const.IMAGE_WIDTH / 3))

        self.draw_annotation({'x1': 100, 'y1': 200, 'x2': 300, 'y2': 400, 'state': None}, self.previous_annotations)

        self.show()

        # self.draw_annotation(self.parent().bbox, self.current_annotation)
        for annotation in self.parent().annotated:
            self.draw_annotation(annotation, self.previous_annotations)


    def draw_annotation(self, bbox, parent):
        '''
        An annotation has 4 text inputs for x1, y1, x2, y2 coordinates
        and one dropdown menu for the traffic light state.
        '''


        # create new annotation
        annotation = QWidget()
        parent.addWidget(annotation)
        annotation_layout = QGridLayout()
        annotation.setLayout(annotation_layout)

        # set Title
        title = QLabel("Traffic Light " + str(parent.count()-1) + ":")
        annotation_layout.addWidget(title, 0, 0, 1, 2)

        # x1
        x1_input = QLineEdit(str(bbox['x1']))
        x1_input.setValidator(QIntValidator())
        annotation_layout.addWidget(x1_input, 1, 0)
        x1_input.returnPressed.connect(lambda: x1_input.clearFocus())

        # y1
        y1_input = QLineEdit(str(bbox['y1']))
        y1_input.setValidator(QIntValidator())
        annotation_layout.addWidget(y1_input, 1, 1)
        y1_input.returnPressed.connect(lambda: y1_input.clearFocus())

        # x2
        x2_input = QLineEdit(str(bbox['x2']))
        x2_input.setValidator(QIntValidator())
        annotation_layout.addWidget(x2_input, 2, 0)
        x2_input.returnPressed.connect(lambda: x2_input.clearFocus())

        # y2
        y2_input = QLineEdit(str(bbox['y2']))
        y2_input.setValidator(QIntValidator())
        annotation_layout.addWidget(y2_input, 2, 1)
        y2_input.returnPressed.connect(lambda: y2_input.clearFocus())

        # create 3 buttons inside vertical layout spanning 2 rows
        button_layout = QVBoxLayout()
        annotation_layout.addLayout(button_layout, 1, 3, 2, 1)

        # add 3 buttons to vertical layout (red, yellow, green)
        red_button = QCheckBox()
        red_button.setChecked(True) if bbox['state'] == 1 or bbox['state'] == 2 else red_button.setChecked(False)
        button_layout.addWidget(red_button)

        yellow_button = QCheckBox()
        yellow_button.setChecked(True) if bbox['state'] == 2 or bbox['state'] == 3 else yellow_button.setChecked(False)
        button_layout.addWidget(yellow_button)

        green_button = QCheckBox()
        green_button.setChecked(True) if bbox['state'] == 4 else green_button.setChecked(False)
        button_layout.addWidget(green_button)

        # if current state is yellow, check red and yellow button
        # else check only red button
        def red_button_clicked():
            if  bbox['state'] == 3:
                bbox['state'] = 2
                red_button.setChecked(True)
                yellow_button.setChecked(True)
                green_button.setChecked(False)
            else:
                bbox['state'] = 1
                red_button.setChecked(True)
                yellow_button.setChecked(False)
                green_button.setChecked(False)
            self.parent().update_bounding_box()
            self.parent().show()

        red_button.clicked.connect(red_button_clicked)

        # if current state is red, check yellow button
        # else check only yellow button
        def yellow_button_clicked():
            if  bbox['state'] == 1:
                bbox['state'] = 2
                red_button.setChecked(True)
                green_button.setChecked(False)
            else:
                bbox['state'] = 3
                red_button.setChecked(False)
                yellow_button.setChecked(True)
                green_button.setChecked(False)
            self.parent().update_bounding_box()
            self.parent().show()

        yellow_button.clicked.connect(yellow_button_clicked)

        # check only green button
        def green_button_clicked():
            bbox['state'] = 4
            red_button.setChecked(False)
            yellow_button.setChecked(False)
            green_button.setChecked(True)
            self.parent().update_bounding_box()
            self.parent().show()

        green_button.clicked.connect(green_button_clicked)

        








app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())

