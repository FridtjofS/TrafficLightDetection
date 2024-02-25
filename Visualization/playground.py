import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QRadioButton, QSpinBox, QLineEdit, QPushButton, QCheckBox


class VideoProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.input_type = None
        self.input_path = None
        self.save = False
        self.save_path = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        live_video_radio = QRadioButton('Live Video Capture')
        live_video_radio.clicked.connect(self.enable_live_video_capture)
        layout.addWidget(live_video_radio)

        self.live_video_spinbox = QSpinBox()
        self.live_video_spinbox.setEnabled(False)
        layout.addWidget(self.live_video_spinbox)

        file_video_radio = QRadioButton('Use Video from File')
        file_video_radio.clicked.connect(self.enable_video_from_file)
        layout.addWidget(file_video_radio)

        self.file_video_lineedit = QLineEdit()
        self.file_video_lineedit.setEnabled(False)
        layout.addWidget(self.file_video_lineedit)

        save_checkbox = QCheckBox('Save Video')
        save_checkbox.stateChanged.connect(self.enable_video_save)
        layout.addWidget(save_checkbox)

        self.save_path_lineedit = QLineEdit()
        self.save_path_lineedit.setEnabled(False)
        layout.addWidget(self.save_path_lineedit)

        process_button = QPushButton('Process Video')
        process_button.clicked.connect(self.process_video)
        layout.addWidget(process_button)

        self.setLayout(layout)
        self.setWindowTitle('Video Processor')

    def enable_live_video_capture(self):
        self.input_type = 0
        self.live_video_spinbox.setEnabled(True)
        self.file_video_lineedit.setEnabled(False)

    def enable_video_from_file(self):
        self.input_type = 1
        self.live_video_spinbox.setEnabled(False)
        self.file_video_lineedit.setEnabled(True)

    def enable_video_save(self, state):
        self.save = state == 2
        self.save_path_lineedit.setEnabled(state == 2)

    def process_video(self):
        if self.input_type == 0:
            self.input_path = self.live_video_spinbox.value()
        elif self.input_type == 1:
            self.input_path = self.file_video_lineedit.text()

        if self.save:
            self.save_path = self.save_path_lineedit.text()

        print("Input Type:", self.input_type)
        print("Input Path:", self.input_path)
        print("Save Video:", self.save)
        if self.save:
            print("Save Path:", self.save_path)


def main():
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
