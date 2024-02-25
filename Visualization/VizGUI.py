import sys
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

        live_cap = QRadioButton('Live Video Capture')
        live_cap.clicked.connect(self.enable_live_video_capture)
        layout.addWidget(live_cap)

        self.live_cap_input = QSpinBox()
        self.live_cap_input.setEnabled(False)
        layout.addWidget(self.live_cap_input)

        file_cap = QRadioButton('Use Video from File')
        file_cap.clicked.connect(self.enable_video_from_file)
        layout.addWidget(file_cap)

        self.file_cap_input = QLineEdit()
        self.file_cap_input.setEnabled(False)
        layout.addWidget(self.file_cap_input)

        save_check = QCheckBox('Save Video')
        save_check.stateChanged.connect(self.enable_video_save)
        layout.addWidget(save_check)

        self.save_path = QLineEdit()
        self.save_path.setEnabled(False)
        layout.addWidget(self.save_path)

        start_button = QPushButton('Process Video')
        start_button.clicked.connect(self.start)
        layout.addWidget(start_button)

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
        self.save = state == 2
        self.save_path.setEnabled(state == 2)

    def start(self):
        if self.input_type == 0:
            self.input_path = self.live_cap_input.value()
        elif self.input_type == 1:
            self.input_path = self.file_cap_input.text()

        if self.save:
            self.save_path = self.save_path.text()



def main():

    global input_type, input_path, save, save_path

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("TL_icon.png"))
    window = VizGUI()
    window.show()
    sys.exit(app.exec())

    print("Input Type:", input_type)
    print("Input Path:", input_path)
    print("Save Video:", self.save)
    if self.save:
        print("Save Path:", self.save_path)


if __name__ == '__main__':

    input_type = None
    input_path = None
    save = None
    save_path = None

    main()
