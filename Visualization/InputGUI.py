import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFrame, QCheckBox, QInputDialog, QMessageBox, QFileDialog
from PyQt6.QtGui import QIcon


class InputGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Traffic Light Detection - Visualizer")
        self.setGeometry(100, 100, 400, 400)

        self.input_type = None
        self.file_path = None
        self.cam_num = 0
        self.show_status = False
        self.save_status = True
        self.save_dir = None
        
        self.initUI()

        
    def initUI(self):
        self.live_button = QPushButton('Capture Live Video', self)
        self.live_button.setGeometry(10, 10, 380, 50) 
        self.live_button.clicked.connect(self.live_button_clicked)

        self.text_or = QLabel("OR", self)
        self.text_or.setGeometry(190, 67, 20, 14)

        self.file_button = QPushButton('Load Video from File', self)
        self.file_button.setGeometry(10, 90, 380, 50)
        self.file_button.clicked.connect(self.file_button_clicked)

        self.hor_line = QFrame(self)
        self.hor_line.setFrameShape(QFrame.Shape.HLine)
        self.hor_line.setFrameShadow(QFrame.Shadow.Sunken)
        self.hor_line.setGeometry(10, 149, 380, 2)

        self.text_processing = QLabel('The Visualization Tool will process the following input:', self)
        self.text_processing.setGeometry(30, 165, 380, 15)

        self.input_dir = QLabel('', self)
        self.input_dir.setGeometry(30, 185, 380, 15)

        self.show_check = QCheckBox('Show annotated frames', self)
        self.show_check.setGeometry(10, 230, 380, 15)
        self.show_check.stateChanged.connect(self.show_checkbox_changed)

        self.save_check = QCheckBox('Save annotated frames to directory:', self)
        self.save_check.setGeometry(10, 275, 380, 15)
        self.save_check.setChecked(self.save_status)
        self.save_check.stateChanged.connect(self.save_checkbox_changed)

        self.save_button = QPushButton('', self)
        self.save_button.setGeometry(10, 295, 380, 30)
        self.save_button.setStyleSheet("text-align:left;")
        self.save_button.clicked.connect(self.save_button_clicked)

        self.start_button = QPushButton('Process Video', self)
        self.start_button.setGeometry(10, 340, 380, 50)
        self.start_button.clicked.connect(self.start_processing)

    
    def live_button_clicked(self):
        self.input_type = 0
        cam_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
        chosen_cam, ok = QInputDialog.getItem(self, "Select Camera", "Choose a camera", [str(cam) for cam in cam_list], 0, False)
        self.cam_num = int(chosen_cam)
        self.input_dir.setText(f"Camera input from camera {self.cam_num}.")


    def file_button_clicked(self):
        self.input_type = 1
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Choose File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.file_path = selected_files[0]
                file_path = str(self.file_path)
                truncated_file_path = ('...' + file_path[-50:]) if len(file_path) > 50 else file_path
                self.input_dir.setText(truncated_file_path)
                save_dir = os.path.dirname(self.file_path)
                self.save_dir = save_dir
                truncated_save_dir = ('...' + save_dir[-50:]) if len(save_dir) > 50 else save_dir
                self.save_button.setText(truncated_save_dir)


    def show_checkbox_changed(self, state):
        if state == 2:
            self.show_status = True
        else:
            self.show_status = False

    def save_checkbox_changed(self, state):
        if state == 2:
            self.save_status = True
        else:
            self.save_status = False

    def save_button_clicked(self):
        self.save_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.save_dir:
            save_dir = str(self.save_dir)
            truncated_save_dir = ('...' + save_dir[-50:]) if len(save_dir) > 50 else save_dir
            self.save_button.setText(truncated_save_dir)


    def start_processing(self):

        global input_type, file_path, cam_num, show_status, save_status, save_dir

        input_type = None
        if self.input_type == 0 or 1:
            input_type = self.input_type
        else:
            QMessageBox.warning(self, "Warning", "No input selected. Please choose input to proceed.")

        file_path = self.file_path
        cam_num = self.cam_num
        show_status = self.show_status
        save_status = self.save_status

        save_dir = None
        
        if self.save_status:
            if self.save_dir:
                save_dir = self.save_dir
            else:
                QMessageBox.warning(self, "Warning", "Please specify a directory to save processed video.")

        self.close()

        

def get_input():

    global input_type, file_path, cam_num, show_status, save_status, save_dir

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join('Visualization', 'TL_icon.png')))
    window = InputGUI()
    window.show()
    app.exec()

    #if save_dir == None:
    #    save_dir = os.getcwd()

    return input_type, file_path, cam_num, show_status, save_status, save_dir


