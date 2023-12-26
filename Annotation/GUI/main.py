from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QLineEdit, QCheckBox, QScrollArea
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QIntValidator, QPainterPath, QPolygonF, QIcon
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent, QPropertyAnimation, QSize
import shutil
import json
from pathlib import Path
import time
import sys
import os

# fix relative imports
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, path)

import annotool.constants as const
from annotool.annotationWindow import AnnotationWindow
from annotool.settingsWindow import SettingsWindow
from annotool.statsWindow import StatsWindow

# This App is an annotation Tool, which loads in an Image, allows the user to draw a bounding box around the object of interest, as well as select one of four traffic light states and then saves the image with the bounding box coordinates in a text file.

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Light Annotation Tool")
        # set a window icon
        self.setWindowIcon(QIcon("annotool/img/icon.svg"))

        self.next_last_layout = QHBoxLayout()
        self.next_last_layout.setContentsMargins(0, 10, 0, 10)
        self.next_last_layout.setSpacing(0)
        # set next and last buttons
        self.next = QPushButton()
        self.next.setStyleSheet("""
        QPushButton {
            image: url(annotool/img/next.svg);
            background-color: transparent;
            border: none;
        }
        QPushButton:disabled {
            image: url(annotool/img/next_disabled.svg);
        }
        """)
        self.last = QPushButton()
        self.last.setStyleSheet("""
        QPushButton {
            image: url(annotool/img/last.svg);
            background-color: transparent;
            border: none;
        }
        QPushButton:disabled {
            image: url(annotool/img/last_disabled.svg);
        }
        """)
        self.next.setFixedSize(40, 40)
        self.last.setFixedSize(40, 40)
        self.next.clicked.connect(self.next_image)
        self.last.clicked.connect(self.previous_image)
        self.next_last_layout.addWidget(self.last)
        self.next_last_layout.addWidget(self.next)
        
        self.layout = QGridLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.layout.addLayout(self.next_last_layout, 2, 1, 1, 2)

        self.first_run = True
        self.wrong_path = False
        
        self.last.setEnabled(False)

        self.annotated = []
        self.undo_stack = []

        # bounding box coordinates
        self.origin = None
        self.end = None
        # traffic light state
        self.state = None
        self.bbox = {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None,
            'state': None
        }
        

        
        self.label = QLabel(self)
        # align top, so that coordinates can be mapped correctly
        self.label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(self.label, 0, 1, 2, 2)

        # move bounding box origin with mouse movement
        self.move_origin = False

        # load the first image into the GUI

        self.left_layout = QVBoxLayout()

        self.layout.addLayout(self.left_layout, 0, 0, 3, 1)
        
        label = QLabel("Traffic Light\nAnnotation Tool")
        label.setStyleSheet("font-size: 15px; font-weight: bold; margin: 5px;")
        self.left_layout.addWidget(label)

        self.annotation_window = AnnotationWindow(self)
        self.annotation_window.setContentsMargins(0, 0, 0, 0)
        self.left_layout.addWidget(self.annotation_window)



        self.settingsButton = QPushButton()
        self.settingsButton.setStyleSheet("""
        QPushButton {
            image: url(annotool/img/settings.svg);
            background-color: transparent;
            border: none;
        }
        QPushButton:hover {
            image: url(annotool/img/settings_hover.svg);
        }
        """)
        self.settingsButton.setToolTip("Settings")
        self.settingsButton.setFixedSize(20, 20)

        self.settingsButton.clicked.connect(self.open_settings_window)
        self.lower_left_layout = QHBoxLayout()
        #self.lower_left_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lower_left_layout.setContentsMargins(5, 5, 5, 5)
        self.lower_left_layout.setSpacing(5)
        self.lower_left_layout.addStretch()
        self.lower_left_layout.addWidget(self.settingsButton, alignment=Qt.AlignmentFlag.AlignLeft)
        self.left_layout.addLayout(self.lower_left_layout)

        self.endSessionButton = QPushButton("End Session")
        self.endSessionButton.setStyleSheet("""
        QPushButton {
            background-color: #3A4450;
            border: none;
            font-size: 15px;
            color: #FFFFFF;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #2C333D;
        }
        """)
        self.endSessionButton.clicked.connect(self.open_stats_window)
        self.lower_left_layout.addWidget(self.endSessionButton)

        self.open_settings_window()
        
        
        screen = app.primaryScreen()
        self.size = screen.size()
        #print('Size: %d x %d' % (self.size.width(), self.size.height()))

        

    # listen for key press events
    def keyPressEvent(self, event):
        # if hit backspace, start previous_image
        # allowed, if image is not first image
        # current input is not saved
        if event.key() == 16777219:
            #self.last_annotation()

            # delete last annotation if there is one
            if len(self.annotated) > 0:
                self.annotated.pop()
                self.update_annotation_window()
                self.update_bounding_box()


            else:
                print("No annotations to delete")

            #if self.last:
            #    self.previous_image()
            #else:
            #    print("This is the first image")
            #return

        # if hit enter, start next_image
        # only allowed if bounding box and traffic light state are set
        elif event.key() == 16777220:
            self.lock_in_bbox() 
            self.setFocus()
            self.update_bounding_box()
            self.update_annotation_window()
            
            

            #if self.origin and self.end and self.state:
            #    self.next_image()
            #else:
            #    print("Please set bounding box and traffic light state")
            #return

        # if press down space, move origin
        elif event.key() == 32:
            if self.end:
                # (handle move origin with mouse movement in mouseMoveEvent)
                self.move_origin = True
                self.update_bounding_box()
            else:
                print("Please set origin")

        # traffic light states
        elif event.key() == 49: # 1
            self.state = 1
            self.bbox['state'] = 1
            self.update_current_annotation()
        elif event.key() == 50: # 2
            self.state = 2
            self.bbox['state'] = 2
            self.update_current_annotation()
        elif event.key() == 51: # 3
            self.state = 3
            self.bbox['state'] = 3
            self.update_current_annotation()
        elif event.key() == 52: # 4
            self.state = 4
            self.bbox['state'] = 4
            self.update_current_annotation()
        elif event.key() == 53: # 5
            self.state = 5
            self.bbox['state'] = 5
            self.update_current_annotation()

        # if press ESC, close window
        elif event.key() == 16777216: # ESC
            self.close()
            print("Close")
        
        self.update_bounding_box()

    def keyReleaseEvent(self, event):
        # if release space, stop moving origin
        if event.key() == 32:
            self.move_origin = False

    def next_image(self):
        #if len(os.listdir(const.IMAGE_DIR)) == 0:	
        #    print("Directory is empty")
        #    return

        path = self.get_path()
        if path == None:
            return
        
        if len(self.undo_stack) > 0:
            self.save_image_with_annotations(path)

        self.undo_stack.append(path)
        #self.next.setEnabled(False)
        self.last.setEnabled(True) if len(self.undo_stack) > 0 else self.last.setEnabled(False)
        self.state = None
        self.origin = None
        self.end = None
        self.annotated = []
        self.bbox = {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None,
            'state': None
        }
        self.update_annotation_window()
        self.load_image(self.get_path())
        self.showMaximized()

    def previous_image(self):
        # get filepath of previous image
        # load in previous image
        # get user input for bounding box and traffic light state
        # update json file with bounding box coordinates and traffic light state
        
        # if there is no previous image, return
        if len(self.undo_stack) == 0:
            return
        
        if len(self.undo_stack) == 1:
            self.last.setEnabled(False)
        
        #self.next.setEnabled(True)
        
        input_path = self.settings_window.input_folder.text()
        output_path = self.settings_window.output_folder.text()

        path = os.path.join(output_path, os.path.basename(self.undo_stack.pop()))

        # copy image into IMAGE_DIR
        shutil.copy(path, input_path)

        # get json data and set as self.annotated
        with open(os.path.join(output_path, Path(path).stem + ".json" ), 'r') as f:
            json_data = json.load(f)
            self.annotated = []
            for key in json_data:
                self.annotated.append(self.translate_coordinates(json_data[key], True))

        self.update_stats(True)

        # load image into GUI
        self.load_image(path)
        self.update_annotation_window()
        self.update_bounding_box()

        # delete image from const.ANNOTATED_PATH directory
        os.remove(path)

        # delete the json file with the same name as the image
        os.remove(os.path.join(output_path, Path(path).stem + ".json"))

        
    def settings_error(self, message):    
            self.wrong_path = True
            # open warning window
            widget = QWidget()
            widget.setStyleSheet(os.path.join("annotool", "style.qss"))
            widget.setStyleSheet("""
            background-color: #3A4450;
            font-family: 'Montserrat', sans-serif;
            color: #FFFFFF;
            """)
            widget.setWindowTitle("Warning")
            layout = QVBoxLayout()
            widget.setLayout(layout)
            label = QLabel(message)
            layout.addWidget(label)
            ok = QPushButton("OK")
            def okay():
                widget.close()
                self.open_settings_window()

            ok.clicked.connect(okay)
            layout.addWidget(ok)
            widget.show()
            # set on top
            widget.raise_()
    
    # get path of next image in directory
    def get_path(self):

        # get the first image in the directory image_dir
        
        image_dir = self.settings_window.input_folder.text()

        # check if directory is empty
        if len(os.listdir(image_dir)) == 0:
            self.settings_error("The input directory you selected is empty,\nplease select a different directory")
            return None
        else:
            images = os.listdir(image_dir)
            image = os.path.join(image_dir, images[0])
            ending = Path(image).suffix
            if ending == ".jpg" or ending == ".png":
                return image
            else:
                print("Directory does not contain only images")
                self.settings_error("The input directory you selected does not contain only images,\nplease select a different directory")
                return None
        


        #return "frames/django.jpg"
    
    # load image into GUI
    def load_image(self, path):

        # load image
        if path == None:
            self.pixmap = QPixmap()
        else:
            self.pixmap = QPixmap(path)

        # scale image to fit the maximum available size
        max_size_x = self.size.width() - self.left_layout.sizeHint().width()
        max_size_y = self.size.height() - self.next_last_layout.sizeHint().height() - 40
        max_size = QSize(max_size_x, max_size_y)
        
        self.pixmap = self.pixmap.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio)

        # set focus to label, so that key press events are registered
        self.label.setFocus()

        # set image as label
        self.label.setPixmap(self.pixmap)

        # start timer
        self.timer = time.time()
    
    #def resizeEvent(self, event):
    #    # scale image to fit the maximum available size
    #    print("resize")
    #    super().resizeEvent(event)
    #    
    #    # get the root widget
    #    root = self.window()
    #    # get the size of the root widget
    #    #self.size = root.size()
    #    print(self.window().size)
    #
    #
    #    max_size_x = self.size.width() - self.left_layout.sizeHint().width()
    #    max_size_y = self.size.height() - self.next_last_layout.sizeHint().height() - 40
    #    max_size = QSize(max_size_x, max_size_y)
    #    #print(max_size)
    #    self.pixmap = self.pixmap.scaled(max_size)#, Qt.AspectRatioMode.KeepAspectRatio)


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
        # update self.bbox
        self.bbox['x1'] = self.origin.x()
        self.bbox['y1'] = self.origin.y()
        self.bbox['x2'] = self.origin.x()
        self.bbox['y2'] = self.origin.y()


    def mouseMoveEvent(self, event):
        if not self.origin:
            return
        
        if event.buttons() != Qt.MouseButton.LeftButton or not self.label.underMouse():
            return

        # if space is pressed, move origin with mouse movement
        if self.move_origin and self.end:
            # and keep bounding box size
            difference = event.pos() - self.end

            self.origin += difference
            self.origin = self.label.mapFrom(self, self.origin)
            self.end = event.pos()
            self.end = self.label.mapFrom(self, self.end)
            # make sure origin is not outside of image
            self.origin = QPoint(max(1, min(self.pixmap.width(), self.origin.x())), max(1, min(self.pixmap.height(), self.origin.y())))
            # make sure end is not outside of image
            self.end = QPoint(max(1, min(self.pixmap.width(), self.end.x())), max(1, min(self.pixmap.height(), self.end.y())))
            
        else:
            # else, set end of bounding box to mouse position
            self.end = event.pos()
            self.end = self.label.mapFrom(self, self.end)

            # make sure end is not outside of image
            self.end = QPoint(max(0, min(self.pixmap.width(), self.end.x())), max(0, min(self.pixmap.height(), self.end.y())))
        
        # update self.bbox
        self.bbox['x1'] = min(self.origin.x(), self.end.x())
        self.bbox['y1'] = min(self.origin.y(), self.end.y())
        self.bbox['x2'] = max(self.origin.x(), self.end.x())
        self.bbox['y2'] = max(self.origin.y(), self.end.y())
        self.update_bounding_box()
        self.update_current_annotation()


    def mouseReleaseEvent(self, event):
        if not self.origin or not self.end:
            return
        
        if event.button() != Qt.MouseButton.LeftButton or not self.label.underMouse():
            return
        
        # set end of bounding box to mouse position
        self.end = event.pos()
        self.end = self.label.mapFrom(self, self.end)

        # make sure end is not outside of image
        self.end = QPoint(max(0,min(self.pixmap.width(), self.end.x())), max(0, min(self.pixmap.height(), self.end.y())))
        
        # update self.bbox
        self.bbox['x1'] = min(self.origin.x(), self.end.x())
        self.bbox['y1'] = min(self.origin.y(), self.end.y())
        self.bbox['x2'] = max(self.origin.x(), self.end.x())
        self.bbox['y2'] = max(self.origin.y(), self.end.y())
        # draw bounding box
        self.update_bounding_box()
        self.update_current_annotation()

        # enable next button if all conditions are met
        #if self.bbox['x1'] and self.bbox['y1'] and self.bbox['x2'] and self.bbox['y2'] and self.bbox['state'] and self.bbox['x1'] != self.bbox['x2'] and self.bbox['y1'] != self.bbox['y2']:
        #        self.next.setEnabled(True)
        
    
    # draw bounding box on image
    def update_bounding_box(self):
        
        canvas = self.pixmap.copy()
        painter = QPainter(canvas)

        

        

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

            # label bounding box with number + 1 with background color
            #painter.fillRect(annotation['x1']-1, annotation['y1'] - 12, 12, 12, color)
            y_poly = annotation['y1'] - 12 if annotation['y1'] - 12 > 0 else annotation['y1'] + 12
            y_text = annotation['y1'] - 1 if annotation['y1'] - 12 > 0 else annotation['y1'] + 10
            x_poly = annotation['x1'] + 12 if self.annotated.index(annotation) < 9 else annotation['x1'] + 18

            polygon = QPolygonF([QPointF(annotation['x1']-1, y_poly), QPointF(annotation['x1']-1, annotation['y1']), QPointF(x_poly + 12, annotation['y1']), QPointF(x_poly, y_poly)])
            path = QPainterPath()
            path.addPolygon(polygon)
            painter.fillPath(path, color)
            #path.addPolygon(QPolygonF([QPointF(annotation['x1']-1, annotation['y1'] - 12), QPointF(annotation['x1']-1, annotation['y1'] - 1), QPointF(annotation['x1'] + 12, annotation['y1'] - 1)]))
            #painter.fillPath(path, color)
            if annotation['state'] == 5:
                painter.setPen(QPen(Qt.GlobalColor.white))
            else:
                painter.setPen(QPen(Qt.GlobalColor.black))
            
            painter.drawText(annotation['x1'] + 2, y_text, str(self.annotated.index(annotation) + 1))


        if self.bbox['x1'] and self.bbox['y1'] and self.bbox['x2'] and self.bbox['y2']:
            pen = QPen(self.get_color(self.bbox['state']))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(1)

            x1 = self.bbox["x1"]
            y1 = self.bbox["y1"]
            x2 = self.bbox["x2"]
            y2 = self.bbox["y2"]

            painter.setPen(pen)
            painter.drawRect(x1, y1, x2-x1, y2-y1)

            # fill bounding box with color (10% opacity)
            painter.setOpacity(0.1)
            painter.fillRect(x1, y1, x2-x1, y2-y1, self.get_color(self.bbox['state']))
            painter.setOpacity(1)
        
        self.label.setPixmap(canvas)
        del painter
        del canvas
        self.label.update()

    def update_current_annotation(self):
        # clear current annotation
        for i in reversed(range(self.annotation_window.current_annotation.count())):
            self.annotation_window.current_annotation.itemAt(i).widget().setParent(None)
        self.annotation_window.draw_annotation(self.bbox, self.annotation_window.current_annotation, "currently editing...", True)
        self.annotation_window.update()


    def update_annotation_window(self):
        for i in reversed(range(self.annotation_window.current_annotation.count())):
            self.annotation_window.current_annotation.itemAt(i).widget().setParent(None)
        self.annotation_window.draw_annotation(self.window().bbox, self.annotation_window.current_annotation, "currently editing...", True)
        # clear previous annotations
        for i in reversed(range(self.annotation_window.previous_annotations.count())): 
            self.annotation_window.previous_annotations.itemAt(i).widget().setParent(None)
        for i in reversed(range(len(self.annotated))):
            self.annotation_window.draw_annotation(self.annotated[i], self.annotation_window.previous_annotations,("Traffic Light " + str(i + 1) + ":"))

        self.annotation_window.update()

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
        if not self.bbox['x1'] or not self.bbox['y1'] or not self.bbox['x2'] or not self.bbox['y2']:
            print("Please set all coordinates")
            return
        
        # if bounding box is a point or a line, return
        if self.bbox['x1'] == self.bbox['x2'] or self.bbox['y1'] == self.bbox['y2']:
            print("The bounding box has to be a rectangle")
            return

        # if traffic light state is not set, return
        if not self.bbox['state']:
            print("Please set traffic light state")
            return
        
        #bbox = {
        #    'x1': min(self.origin.x(), self.end.x()),
        #    'y1': min(self.origin.y(), self.end.y()),
        #    'x2': max(self.origin.x(), self.end.x()),
        #    'y2': max(self.origin.y(), self.end.y()),
        #    'state': self.bbox['state']
        #}

        self.annotated.append(self.bbox)
        #self.next.setEnabled(True)

        

        # clear bounding box
        self.origin = None
        self.end = None
        self.state = None

        self.bbox = {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None,
            'state': None
        }
        self.update_bounding_box()
        self.update_annotation_window()
        


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
        if self.annotated == []:
            print("No annotations to save")
            #TODO open extra window to ask if user wants to save image without annotations

        output_path = self.settings_window.output_folder.text()

        # check if output directory exists
        if not os.path.exists(output_path):
            self.settings_error("The output directory you selected does not exist,\nplease select a different directory")

        # save image in "annotations" directory
        shutil.copy(path, os.path.join(output_path, os.path.basename(path)))

        # save bounding box coordinates and traffic light state in json file with same name as image
        json_data = {}
        for i in range(len(self.annotated)):
            json_data['traffic_light_' + str(i)] = self.translate_coordinates(self.annotated[i])

        with open(os.path.join(output_path, Path(path).stem + ".json" ), 'w') as f:
            json.dump(json_data, f)

        self.update_stats()
        

    def open_settings_window(self):
        self.settings_window = SettingsWindow(self)
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()
        self.settings_window.setFocus()

    def translate_coordinates(self, bbox, inverse=False):
        resolution = self.settings_window.output_size.currentText()
        res_x = int(resolution.split("x")[0])
        res_y = int(resolution.split("x")[1])

        if inverse:
            x1 = bbox['x1'] * self.label.pixmap().width() / res_x
            y1 = bbox['y1'] * self.label.pixmap().height() / res_y
            x2 = bbox['x2'] * self.label.pixmap().width() / res_x
            y2 = bbox['y2'] * self.label.pixmap().height() / res_y
        else:
            x1 = bbox['x1'] * res_x / self.label.pixmap().width()
            y1 = bbox['y1'] * res_y / self.label.pixmap().height()
            x2 = bbox['x2'] * res_x / self.label.pixmap().width()
            y2 = bbox['y2'] * res_y / self.label.pixmap().height()
            
        return {
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'state': bbox['state']
        }
    
    def open_stats_window(self):
        self.stats_window = StatsWindow(self)
        self.stats_window.show()
        self.stats_window.raise_()
        self.stats_window.activateWindow()
        self.stats_window.setFocus()
        #self.hide()

    def update_stats(self, undo = False):
        # update stats in user.json file
        # get user settings
        user = self.settings_window.login.currentText()

        # get path to user settings file
        path = os.path.join("annotool", "users", user + ".json")
        # open file and read contents
        with open(path, "r") as f:
            settings = json.load(f)


        if undo:
            # update stats
            settings["stats"]["total_images"] -= 1
            settings["stats"]["total_annotations"] -= len(self.annotated)
            settings["stats"]["total_red"] -= len([x for x in self.annotated if x['state'] == 1])
            settings["stats"]["total_red_yellow"] -= len([x for x in self.annotated if x['state'] == 2])
            settings["stats"]["total_yellow"] -= len([x for x in self.annotated if x['state'] == 3])
            settings["stats"]["total_green"] -= len([x for x in self.annotated if x['state'] == 4])
            settings["stats"]["total_off"] -= len([x for x in self.annotated if x['state'] == 5])
        else:
            # update stats
            settings["stats"]["total_images"] += 1
            settings["stats"]["total_annotations"] += len(self.annotated)
            settings["stats"]["total_red"] += len([x for x in self.annotated if x['state'] == 1])
            settings["stats"]["total_red_yellow"] += len([x for x in self.annotated if x['state'] == 2])
            settings["stats"]["total_yellow"] += len([x for x in self.annotated if x['state'] == 3])
            settings["stats"]["total_green"] += len([x for x in self.annotated if x['state'] == 4])
            settings["stats"]["total_off"] += len([x for x in self.annotated if x['state'] == 5])
            settings["stats"]["most_traffic_lights_in_one_image"] = max(settings["stats"]["most_traffic_lights_in_one_image"], len(self.annotated))

            # end timer
            end = time.time()
            settings["stats"]["total_time"] += end - self.timer

        # save updated stats
        with open(path, "w") as f:
            json.dump(settings, f, indent=4)



app = QApplication(sys.argv)
app.setStyleSheet(Path("annotool/style.qss").read_text())
window = MainWindow()
sys.exit(app.exec())

