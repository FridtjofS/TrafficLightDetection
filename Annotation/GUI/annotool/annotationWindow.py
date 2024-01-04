from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt6.QtGui import QIntValidator, QColor
from PyQt6.QtCore import Qt


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
        self.current_annotation_widget.setStyleSheet("background-color: rgb(85, 96, 109);")
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
        
        self.layout.addWidget(scroll_area)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: rgb(85, 96, 109);")
        #scroll_content.setPalette(p)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 2, 0, 0)
        scroll_layout.setSpacing(2)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_content.setLayout(scroll_layout)

        

        self.previous_annotations = QVBoxLayout()
        scroll_layout.addLayout(self.previous_annotations)

        scroll_area.setWidget(scroll_content)

        # set width of window to fit the contents
        self.setFixedWidth(200)

        #self.draw_annotation({} , self.previous_annotations, "currently editing...")

        self.show()

        # background color
        #self.add_new_annotation.setAutoFillBackground(True)
        #p = self.add_new_annotation.palette()
        #p.setColor(self.add_new_annotation.backgroundRole(), QColor(0,255, 109))
        #self.add_new_annotation.setPalette(p)
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

        # add a line on the bottom of the annotation
        line = QWidget()
        line.setStyleSheet("background-color: rgb(55, 65, 81);")
        line.setFixedSize(200, 2) if current else line.setFixedSize(190, 2)
        parent.addWidget(line)

        # set size to fit contents
        annotation.setFixedHeight(110)
        annotation.setFixedWidth(190)
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
            accept_button.setFixedSize(17, 17)
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
            delete_button.setFixedSize(17, 17)
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
                if len(self.parent().annotated) == 0:
                    self.parent().next.setEnabled(False)
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
        QCheckBox::indicator { width: 18px; height: 18px; }
        QCheckBox::indicator::unchecked { image: url(annotool/img/red_unchecked.svg); }
        QCheckBox::indicator::checked { image: url(annotool/img/red.svg); }
        QCheckBox::indicator::unchecked:hover { image: url(annotool/img/red_hover.svg); }
        """)
        #red_button.setFixedSize(30, 30)
        red_button.setChecked(True) if bbox['state'] == 1 or bbox['state'] == 2 else red_button.setChecked(False)
        button_layout.addWidget(red_button)

        yellow_button = QCheckBox()
        yellow_button.setStyleSheet("""
        QCheckBox::indicator { width: 18px; height: 18px; }
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
        QCheckBox::indicator { width: 18px; height: 18px; }
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
