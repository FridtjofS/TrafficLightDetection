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

        self.text = QLabel("Processing finished. How would you like to proceed?", self)
        self.text.setGeometry(30, 20, 330, 15)

        self.playvid_button = QPushButton('Play Video', self)
        self.playvid_button.setGeometry(10, 50, 110, 50)
        self.playvid_button.clicked.connect(self.play_video)
    
        self.newvid_button = QPushButton('New Video', self)
        self.newvid_button.setGeometry(130, 50, 110, 50) 
        self.newvid_button.clicked.connect(self.new_video)

        self.close_button = QPushButton('Close Tool', self)
        self.close_button.setGeometry(250, 50, 110, 50)
        self.close_button.clicked.connect(self.close_tool)
    

    def play_video(self):

        global task
        task = 'play'

        self.close()

    def new_video(self):

        global task
        task = 'new'

        self.close()

    def close_tool(self):

        global task
        task = 'close'

        self.close()


        

        

def todo_next():

    global task

    app = QApplication(sys.argv)
    window = EndGUI()
    app.setWindowIcon(QIcon(os.path.join('Visualization', 'TL_icon.png')))
    window.show()
    app.exec()

    return task

