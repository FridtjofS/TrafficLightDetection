import sys

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtGui import QIcon, QPixmap




class VizWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Light Visualization Tool")








app = QApplication(sys.argv)
app.setWindowIcon(QIcon("TL_icon.png"))
#app.setStyleSheet(Path("annotool/style.qss").read_text())
window = VizWindow()
window.show()
sys.exit(app.exec())
