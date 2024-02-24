import sys

from PyQt6.QtWidgets import QWidget, QApplication, QRadioButton, QLineEdit, QSpinBox, QCheckBox, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt6.QtGui import QIcon




class VizWindow(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        self.liveCap_widget = QRadioButton("Live video capture")
        layout.addWidget(self.liveCap_widget)

        self.liveCapInput_widget = QSpinBox()
        layout.addWidget(self.liveCapInput_widget)

        self.fileCap_widget = QRadioButton("Load video from file")
        layout.addWidget(self.fileCap_widget)

        self.fileCapInput_widget = QLineEdit()
        layout.addWidget(self.fileCapInput_widget)

        self.saveAnno_widget = QCheckBox("I would like to save the annotations to the following directory")
        layout.addWidget(self.saveAnno_widget)

        self.saveAnnoPath_widget = QLineEdit()
        layout.addWidget(self.saveAnnoPath_widget)

        self.startViz_widget = QPushButton("Start visualization")
        layout.addWidget(self.startViz_widget)

        self.setLayout(layout)
        self.setWindowTitle("Traffic Light Visualization Tool")




app = QApplication(sys.argv)
app.setWindowIcon(QIcon("TL_icon.png"))
#app.setStyleSheet(Path("annotool/style.qss").read_text())
window = VizWindow()
window.show()
sys.exit(app.exec())
