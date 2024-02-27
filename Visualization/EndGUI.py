import os
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt6.QtGui import QIcon


class EndGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Traffic Light Detection - Visualizer")
        self.setGeometry(100, 100, 370, 110)
        
        self.initUI()
        
    def initUI(self):

        text_or = QLabel("Processing finished. How would you like to proceed?", self)
        text_or.setGeometry(30, 20, 330, 15)

        playvid_button = QPushButton('Play Video', self)
        playvid_button.setGeometry(10, 50, 110, 50)
    
        newvid_button = QPushButton('New Video', self)
        newvid_button.setGeometry(130, 50, 110, 50) 

        close_button = QPushButton('Close Tool', self)
        close_button.setGeometry(250, 50, 110, 50)

        

        

def main():

    os.chdir("/Users/nadia/TrafficLightDetection")

    app = QApplication(sys.argv)
    window = EndGUI()
    app.setWindowIcon(QIcon(os.path.join('Visualization', 'TL_icon.png')))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()