import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QRadioButton, QSpinBox, QLineEdit, QPushButton, QMessageBox

class InputCollector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Processing Input')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.radio_live = QRadioButton('Live Video Capture')
        self.radio_file = QRadioButton('Use Video from File')
        self.radio_live.setChecked(True)  # Default selection

        self.spinBox = QSpinBox()
        self.spinBox.setRange(0, 9999)
        self.spinBox.setEnabled(True)

        self.lineEdit = QLineEdit()
        self.lineEdit.setEnabled(False)

        self.radio_live.toggled.connect(self.toggleSpinBox)
        self.radio_file.toggled.connect(self.toggleLineEdit)

        self.btn_submit = QPushButton('Submit')
        self.btn_submit.clicked.connect(self.submit)

        layout.addWidget(self.radio_live)
        layout.addWidget(self.radio_file)
        layout.addWidget(self.spinBox)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.btn_submit)

        self.setLayout(layout)

    def toggleSpinBox(self, checked):
        self.spinBox.setEnabled(checked)

    def toggleLineEdit(self, checked):
        self.lineEdit.setEnabled(checked)

    def submit(self):
        global input_type, input_path

        if self.radio_live.isChecked():
            input_type = 0
            input_path = self.spinBox.value()
        elif self.radio_file.isChecked():
            input_type = 1
            input_path = self.lineEdit.text()

        # Show a message box indicating successful submission
        QMessageBox.information(self, 'Submission', f'Input Type: {input_type}\nInput Path: {input_path}')
        self.close()


def main():
    global input_type, input_path

    app = QApplication(sys.argv)
    window = InputCollector()
    window.show()
    app.exec()

    # Do further processing using input_type and input_path
    print("Input Type:", input_type)
    print("Input Path:", input_path)
    # Example: process_video(input_type, input_path)

if __name__ == '__main__':
    input_type = None
    input_path = None
    main()


# Parse input -> Will be done with GUI in the end

def input_parser():
    
    parser = argparse.ArgumentParser(description='Process input to VizTool')
    parser.add_argument("--input_type", help="Specify input type (0: live input; 1: input from file). ", choices=[0, 1], default=1, required=True)
    parser.add_argument("--input_path", help="Path to video or image that serves as input for VizTool. In case of live video capture, leave blank", type=str, required=False)

    return parser.parse_args()

input_type, input_path = input_parser()
