import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QRadioButton, QSpinBox, QLineEdit, QPushButton, QCheckBox
from PyQt6.QtGui import QIcon



class VizGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.input_type = None
        self.input_path = None
        self.save = False
        self.save_path = None

        self.specify_input()

    def specify_input(self):
        layout = QVBoxLayout()

        self.live_cap = QRadioButton('Live Video Capture')
        self.live_cap.clicked.connect(self.enable_live_video_capture)
        layout.addWidget(self.live_cap)

        self.live_cap_input = QSpinBox()
        self.live_cap_input.setEnabled(False)
        layout.addWidget(self.live_cap_input)

        self.file_cap = QRadioButton('Use Video from File')
        self.file_cap.clicked.connect(self.enable_video_from_file)
        layout.addWidget(self.file_cap)

        self.file_cap_input = QLineEdit()
        self.file_cap_input.setEnabled(False)
        layout.addWidget(self.file_cap_input)

        self.save_check = QCheckBox('Save Video')
        self.save_check.stateChanged.connect(self.enable_video_save)
        layout.addWidget(self.save_check)

        self.save_path_input = QLineEdit()
        self.save_path_input.setEnabled(False)
        layout.addWidget(self.save_path_input)

        self.start_button = QPushButton('Process Video')
        self.start_button.clicked.connect(self.start)
        layout.addWidget(self.start_button)

        self.setLayout(layout)
        self.setWindowTitle('Traffic Light Detection Visualization')

    def enable_live_video_capture(self):
        self.input_type = 0
        self.live_cap_input.setEnabled(True)
        self.file_cap_input.setEnabled(False) 

    def enable_video_from_file(self):
        self.input_type = 1
        self.live_cap_input.setEnabled(False)
        self.file_cap_input.setEnabled(True)
    
    def enable_video_save(self, state):
        if state == 2:  # 2 represents checked state
            self.save = True
            self.save_path_input.setEnabled(True)
        else:
            self.save = False

    def start(self):
        global input_type, input_path, save, save_path

        input_type = self.input_type

        if self.input_type == 0:
            self.input_path = self.live_cap_input.value()
        elif self.input_type == 1:
            self.input_path = self.file_cap_input.text()
        else:
            print('Input Type has to be specified')
        
        input_path = self.input_path

        save = self.save

        save_path = self.save_path_input.text()

       
        self.close()



def get_input():

    global input_type, input_path, save, save_path

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join('Visualization', 'TL_icon.png')))
    window = VizGUI()
    window.show()
    app.exec()

    return input_type, input_path, save, save_path
