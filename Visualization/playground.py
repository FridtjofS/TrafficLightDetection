import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox, QLabel, QFrame, QCheckBox, QInputDialog
from PyQt6.QtCore import Qt

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.input_type = None
        self.input_path = None
        self.cam_num = None
        self.save_status = False
        self.save_path = None

        self.setWindowTitle("GUI with PyQt6")
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.live_button = QPushButton("Live Button")
        self.live_button.clicked.connect(self.live_button_clicked)
        self.layout.addWidget(self.live_button)

        self.or_label = QLabel("OR")
        self.or_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layout.addWidget(self.or_label)

        self.file_button = QPushButton("File Button")
        self.file_button.clicked.connect(self.file_button_clicked)
        self.layout.addWidget(self.file_button)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.line)

        self.status_label = QLabel()
        self.layout.addWidget(self.status_label)

        self.save_check = QCheckBox("Save")
        self.layout.addWidget(self.save_check)

        self.save_check.stateChanged.connect(self.save_status_changed)

    def live_button_clicked(self):
        self.input_type = 0
        cam_list = [1, 2, 3]  # Example list of integers
        chosen_cam, ok = QInputDialog.getItem(self, "Select Camera", "Choose a camera", [str(cam) for cam in cam_list], 0, False)
        if ok:
            self.cam_num = int(chosen_cam)
            self.status_label.setText(f"Processing camera input from camera {self.cam_num}")
        else:
            QMessageBox.warning(self, "Warning", "No camera selected!")
            return
        self.file_button.setEnabled(False)

    def file_button_clicked(self):
        self.input_type = 1
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Choose File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.input_path = selected_files[0]
                self.status_label.setText(f"Processing video specified in the selected file: {self.input_path}")
            else:
                QMessageBox.warning(self, "Warning", "No file selected!")
                return
        else:
            QMessageBox.warning(self, "Warning", "File dialog canceled!")
            return
        self.live_button.setEnabled(False)

    def save_status_changed(self, state):
        if state == 2:
            self.save_status = True
            directory = QFileDialog.getExistingDirectory(self, "Select Directory", "/")
            if directory:
                self.save_path = directory
            else:
                self.save_check.setChecked(False)
        else:
            self.save_status = False

def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
