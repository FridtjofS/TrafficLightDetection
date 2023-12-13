from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout, QLineEdit, QCheckBox, QScrollArea
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QIntValidator, QPainterPath, QPolygonF, QIcon
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent, QPropertyAnimation, QSize
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
        # set a window icon
        self.setWindowIcon(QIcon("annotool/img/icon.svg"))

        self.next_last_layout = QHBoxLayout()
        self.next_last_layout.setContentsMargins(0, 0, 0, 0)
        self.next_last_layout.setSpacing(0)
        # set next and last buttons
        self.next = QPushButton()
        self.next.setStyleSheet("image: url(annotool/img/next.svg); background-color: transparent; border: none;")
        self.last = QPushButton()
        self.last.setStyleSheet("image: url(annotool/img/last.svg); background-color: transparent; border: none;")
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
        self.layout.addLayout(self.next_last_layout, 1, 1, 1, 2)
        
        self.last.setEnabled(False)

        self.annotated = []

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


        # move bounding box origin with mouse movement
        self.move_origin = False

        # load the first image into the GUI
        

        self.annotation_window = AnnotationWindow(self)
        self.annotation_window.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.annotation_window, 0, 0, 2, 1)

        #self.update_current_annotation()

        self.next_image()
        
        #self.showMaximized()
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
                self.show()


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
            print("State: Red")
        elif event.key() == 50: # 2
            self.state = 2
            self.bbox['state'] = 2
            self.update_current_annotation()
            print("State: Red-Yellow")
        elif event.key() == 51: # 3
            self.state = 3
            self.bbox['state'] = 3
            self.update_current_annotation()
            print("State: Yellow")
        elif event.key() == 52: # 4
            self.state = 4
            self.bbox['state'] = 4
            self.update_current_annotation()
            print("State: Green")
        elif event.key() == 53: # 5
            self.state = 5
            self.bbox['state'] = 5
            self.update_current_annotation()
            print("State: Off")

        # if press ESC, close window
        elif event.key() == 16777216: # ESC
            self.close()
            print("Close")
        
        self.update_bounding_box()
        

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
        self.bbox = {
            'x1': None,
            'y1': None,
            'x2': None,
            'y2': None,
            'state': None
        }
        self.update_annotation_window()
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
        # update self.bbox
        self.bbox['x1'] = self.origin.x()
        self.bbox['y1'] = self.origin.y()
        self.bbox['x2'] = self.origin.x()
        self.bbox['y2'] = self.origin.y()


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
        if self.origin and self.end and self.state and self.origin != self.end:
                self.next.setEnabled(True)
        
    
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
            print("lol1")
            return
        
        # if bounding box is a point or a line, return
        if self.bbox['x1'] == self.bbox['x2'] or self.bbox['y1'] == self.bbox['y2']:
            print("lol2")
            return

        # if traffic light state is not set, return
        if not self.bbox['state']:
            print("lol3")
            return
        
        #bbox = {
        #    'x1': min(self.origin.x(), self.end.x()),
        #    'y1': min(self.origin.y(), self.end.y()),
        #    'x2': max(self.origin.x(), self.end.x()),
        #    'y2': max(self.origin.y(), self.end.y()),
        #    'state': self.bbox['state']
        #}

        self.annotated.append(self.bbox)

        

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
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.layout)
        
        self.current_annotation_widget = QWidget()
        self.current_annotation_widget.setContentsMargins(0, 0, 0, 0)
        self.current_annotation_widget.setFixedHeight(120)
        self.current_annotation = QVBoxLayout()
        self.current_annotation.setContentsMargins(0, 0, 0, 0)

        
        self.current_annotation_widget.show()
        #self.current_annotation_widget.setFocus()
        #self.show()


        self.current_annotation_widget.setLayout(self.current_annotation)
        self.layout.addWidget(self.current_annotation_widget)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # disable scroll bar
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.layout.addWidget(scroll_area)

        scroll_content = QWidget()
        scroll_content.setAutoFillBackground(True)
        p = scroll_content.palette()
        p.setColor(scroll_content.backgroundRole(), QColor(58, 68, 80))
        scroll_content.setPalette(p)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_content.setLayout(scroll_layout)

        

        self.previous_annotations = QVBoxLayout()
        scroll_layout.addLayout(self.previous_annotations)

        scroll_area.setWidget(scroll_content)

        # set width of window to fit the contents
        self.setFixedWidth(200)

        #self.draw_annotation({} , self.previous_annotations, "currently editing...")

        self.show()

        # self.add_new_annotation is a button which is the default if there is no current annotation
        add_new_annotation = QPushButton("+ Add New Annotation")
        #self.add_new_annotation.clicked.connect(self.parent().update_current_annotation)
        add_new_annotation.setFixedSize(120, 120)
        add_new_annotation.setStyleSheet("""
        background-color: rgb(0,0,255);
        """)
        self.current_annotation.addWidget(add_new_annotation)
        # background color
        #self.add_new_annotation.setAutoFillBackground(True)
        #p = self.add_new_annotation.palette()
        #p.setColor(self.add_new_annotation.backgroundRole(), QColor(0,255, 109))
        #self.add_new_annotation.setPalette(p)
        add_new_annotation.show()
        self.current_annotation_widget.show()
        self.show()
        self.update()
        self.repaint()
        self.show()

        self.draw_annotation(self.parent().bbox, self.current_annotation, "currently editing...", True)

        # self.draw_annotation(self.parent().bbox, self.current_annotation)
        #for annotation in self.parent().annotated:
        #    self.draw_annotation(annotation, self.previous_annotations)


    def draw_annotation(self, bbox, parent, title="", current=False):
        '''
        An annotation has 4 text inputs for x1, y1, x2, y2 coordinates
        and one dropdown menu for the traffic light state.
        '''


        # create new annotation
        annotation = QWidget()
        parent.addWidget(annotation)
        # background color
        annotation.setAutoFillBackground(True)
        p = annotation.palette()
        p.setColor(annotation.backgroundRole(), QColor(85, 96, 109))
        annotation.setPalette(p)
        # set size to fit contents
        annotation.setFixedHeight(110)
        # set layout
        annotation_layout = QGridLayout()
        annotation.setLayout(annotation_layout)
        

        # set Title
        title = QLabel(title)
        annotation_layout.addWidget(title, 0, 0, 1, 2)

        # set delete or accept button
        if current:
            accept_button = QPushButton("")
            accept_button.setToolTip("Press Enter to lock in")
            accept_button.setFixedSize(13, 13)
            accept_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                image: url(annotool/img/accept.svg);
            }
            QPushButton:hover {
                image: url(annotool/img/accept_hover.svg);
            }
            """)
            accept_button.clicked.connect(self.parent().lock_in_bbox)
            annotation_layout.addWidget(accept_button, 0, 3)
        else:
            delete_button = QPushButton("")
            if len(parent) == 1:
                delete_button.setToolTip("Press Backspace to delete last annotation")
            delete_button.setFixedSize(13, 13)
            delete_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                image: url(annotool/img/delete.svg);
            }
            QPushButton:hover {
                image: url(annotool/img/delete_hover.svg);
            }
            """)
            def delete():
                self.parent().annotated.remove(bbox)
                self.parent().update_annotation_window()
                self.parent().update_bounding_box()
                self.parent().show()
            delete_button.clicked.connect(delete)
            annotation_layout.addWidget(delete_button, 0, 3)


        def change_input():
            bbox['x1'] = max(min(int(x1_input.text()), self.parent().pixmap.width()), 0) if x1_input.text() else None
            bbox['y1'] = max(min(int(y1_input.text()), self.parent().pixmap.height()), 0) if y1_input.text() else None
            bbox['x2'] = max(min(int(x2_input.text()), self.parent().pixmap.width()), 0) if x2_input.text() else None
            bbox['y2'] = max(min(int(y2_input.text()), self.parent().pixmap.height()), 0) if y2_input.text() else None

            if bbox['x1'] and bbox['x2'] and bbox['x1'] > bbox['x2']:
                bbox['x1'], bbox['x2'] = bbox['x2'], bbox['x1']
            if bbox['y1'] and bbox['y2'] and bbox['y1'] > bbox['y2']:
                bbox['y1'], bbox['y2'] = bbox['y2'], bbox['y1']

            x1_input.setText(str(bbox['x1'])) if bbox['x1'] else x1_input.setText("")
            y1_input.setText(str(bbox['y1'])) if bbox['y1'] else y1_input.setText("")
            x2_input.setText(str(bbox['x2'])) if bbox['x2'] else x2_input.setText("")
            y2_input.setText(str(bbox['y2'])) if bbox['y2'] else y2_input.setText("")

            self.parent().annotation_window.setFocus()
            self.parent().update_bounding_box()
            self.parent().show()
            
        widget_style = """
        QWidget {
            font-size: 8px;
            color: #A0AEC0;
            background-color: #2D3748;
            border-radius: 5px;
            border: 1px solid #A0AEC0;
        }
        QWidget:hover {
            color: #ffffff;
            border-color: #ffffff;
        }
        """
        label_style = """
        QLabel {
            color: #A0AEC0;
            background-color: transparent;
            border: none;
        }
        """
        input_style = """
        QLineEdit {
            background-color: transparent;
            border: none;
            color: #A0AEC0;
            font-size: 13px;
        }
        QLineEdit:hover{
            color: #ffffff;
        }
        """


        # x1
        x1_label = QLabel(" x min")

        x1_input = QLineEdit(str(bbox['x1'])) if bbox['x1'] else QLineEdit()
        x1_input.setValidator(QIntValidator())
        x1_input.editingFinished.connect(change_input)
        
        x1_layout = QVBoxLayout()
        x1_layout.setContentsMargins(5, 1, 5, 2)
        x1_layout.setSpacing(0)
        x1_layout.addWidget(x1_label)
        x1_layout.addWidget(x1_input)

        x1_widget = QWidget()
        x1_widget.setFixedHeight(30)
        x1_widget.setLayout(x1_layout)
        x1_widget.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(x1_widget, 1, 0)

        x1_widget.setStyleSheet(widget_style)
        x1_label.setStyleSheet(label_style)
        x1_input.setStyleSheet(input_style)

        # y1
        y1_label = QLabel(" y min")

        y1_input = QLineEdit(str(bbox['y1'])) if bbox['y1'] else QLineEdit()
        y1_input.setValidator(QIntValidator())
        y1_input.editingFinished.connect(change_input)
        
        y1_layout = QVBoxLayout()
        y1_layout.setContentsMargins(5, 1, 5, 2)
        y1_layout.setSpacing(0)
        y1_layout.addWidget(y1_label)
        y1_layout.addWidget(y1_input)

        y1_widget = QWidget()
        y1_widget.setFixedHeight(30)
        y1_widget.setLayout(y1_layout)
        y1_widget.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(y1_widget, 1, 1)

        y1_widget.setStyleSheet(widget_style)
        y1_label.setStyleSheet(label_style)
        y1_input.setStyleSheet(input_style)

        # x2
        x2_label = QLabel(" x max")

        x2_input = QLineEdit(str(bbox['x2'])) if bbox['x2'] else QLineEdit()
        x2_input.setValidator(QIntValidator())
        x2_input.editingFinished.connect(change_input)

        x2_layout = QVBoxLayout()
        x2_layout.setContentsMargins(5, 1, 5, 2)
        x2_layout.setSpacing(0)
        x2_layout.addWidget(x2_label)
        x2_layout.addWidget(x2_input)

        x2_widget = QWidget()
        x2_widget.setFixedHeight(30)
        x2_widget.setLayout(x2_layout)
        x2_widget.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(x2_widget, 2, 0)

        x2_widget.setStyleSheet(widget_style)
        x2_label.setStyleSheet(label_style)
        x2_input.setStyleSheet(input_style)

        # y2
        y2_label = QLabel(" y max")

        y2_input = QLineEdit(str(bbox['y2'])) if bbox['y2'] else QLineEdit()
        y2_input.setValidator(QIntValidator())
        y2_input.editingFinished.connect(change_input)

        y2_layout = QVBoxLayout()
        y2_layout.setContentsMargins(5, 1, 5, 2)
        y2_layout.setSpacing(0)
        y2_layout.addWidget(y2_label)
        y2_layout.addWidget(y2_input)

        y2_widget = QWidget()
        y2_widget.setFixedHeight(30)
        y2_widget.setLayout(y2_layout)
        y2_widget.setContentsMargins(0, 0, 0, 0)
        annotation_layout.addWidget(y2_widget, 2, 1)

        y2_widget.setStyleSheet(widget_style)
        y2_label.setStyleSheet(label_style)
        y2_input.setStyleSheet(input_style)

        # create 3 buttons inside vertical layout spanning 2 rows
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        annotation_layout.addLayout(button_layout, 1, 3, 2, 1)

        # add 3 buttons to vertical layout (red, yellow, green)
        red_button = QCheckBox()
        red_button.setStyleSheet("""
        QCheckBox::indicator::unchecked { image: url(annotool/img/red_unchecked.svg); }
        QCheckBox::indicator::checked { image: url(annotool/img/red.svg); }
        QCheckBox::indicator::unchecked:hover { image: url(annotool/img/red_hover.svg); }
        """)
        red_button.setChecked(True) if bbox['state'] == 1 or bbox['state'] == 2 else red_button.setChecked(False)
        button_layout.addWidget(red_button)

        yellow_button = QCheckBox()
        yellow_button.setStyleSheet("""
        QCheckBox::indicator::unchecked { image: url(annotool/img/yellow_unchecked.svg); }
        QCheckBox::indicator::checked { image: url(annotool/img/yellow.svg); }
        QCheckBox::indicator::unchecked:hover { image: url(annotool/img/yellow_hover.svg); }
        """)
        yellow_button.setChecked(True) if bbox['state'] == 2 or bbox['state'] == 3 else yellow_button.setChecked(False)
        button_layout.addWidget(yellow_button)

        green_button = QCheckBox()
        if parent == self.current_annotation:
            green_button.setToolTip("""Press 1 to set state to red
Press 2 to set state to red-yellow
Press 3 to set state to yellow
Press 4 to set state to green
Press 5 to set state to off""")
        green_button.setStyleSheet("""
        QCheckBox::indicator::unchecked { image: url(annotool/img/green_unchecked.svg); }
        QCheckBox::indicator::checked { image: url(annotool/img/green.svg); }
        QCheckBox::indicator::unchecked:hover { image: url(annotool/img/green_hover.svg); }
        """)
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
            elif bbox['state'] == 1:
                bbox['state'] = 5
                red_button.setChecked(False)
                yellow_button.setChecked(False)
                green_button.setChecked(False)
            else:
                bbox['state'] = 1
                red_button.setChecked(True)
                yellow_button.setChecked(False)
                green_button.setChecked(False)
            self.parent().label.setFocus()
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
            elif bbox['state'] == 3:
                bbox['state'] = 5
                red_button.setChecked(False)
                yellow_button.setChecked(False)
                green_button.setChecked(False)
            else:
                bbox['state'] = 3
                red_button.setChecked(False)
                yellow_button.setChecked(True)
                green_button.setChecked(False)
            self.parent().annotation_window.setFocus()
            self.parent().update_bounding_box()
            self.parent().show()

        yellow_button.clicked.connect(yellow_button_clicked)

        # check only green button
        def green_button_clicked():
            if bbox['state'] == 4:
                bbox['state'] = 5
                red_button.setChecked(False)
                yellow_button.setChecked(False)
                green_button.setChecked(False)
            else:
                bbox['state'] = 4
                red_button.setChecked(False)
                yellow_button.setChecked(False)
                green_button.setChecked(True)
            self.parent().annotation_window.setFocus()
            self.parent().update_bounding_box()
            self.parent().show()

        green_button.clicked.connect(green_button_clicked)

        return annotation



app = QApplication(sys.argv)
app.setStyleSheet(Path("annotool/style.qss").read_text())
window = MainWindow()
window.show()
sys.exit(app.exec())

