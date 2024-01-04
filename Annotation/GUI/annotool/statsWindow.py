from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QFileDialog, QSizePolicy
from PyQt6.QtGui import QIntValidator, QColor, QPixmap, QPainter
from PyQt6.QtCore import QMetaMethod, Qt, QPropertyAnimation, QSize, QEasingCurve, QTimer

from pathlib import Path
import json

import sys
import os
import time


class StatsWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Statistics")
        self.setWindowIcon(parent.windowIcon())

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        title = QLabel("Traffic Light Annotation\nStatistics")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.layout.addWidget(title, 0,0,1,2)

        continue_button = QPushButton("Continue Session")
        continue_button.clicked.connect(self.continue_session)
        self.layout.addWidget(continue_button, 2,0)

        end_button = QPushButton("End Session")
        end_button.clicked.connect(lambda: sys.exit())
        self.layout.addWidget(end_button, 2,1)





        self.stats = [("Total Images", self.get_statistics("total_images")), ("Total Annotations", self.get_statistics("total_annotations"))]

        #self.stats.append(("Time per Image", (self.get_statistics("total_time") / self.get_statistics("total_images"))))
        #self.stats.append(("Time per Traffic Light", (self.get_statistics("total_time") / self.get_statistics("total_annotations"))))

        time_per_image = self.get_statistics("total_time")
        for user, stat in self.get_statistics("total_time").items():
            time_per_image[user] = stat / self.get_statistics("total_images")[user]
        self.stats.append(("Time per Image", time_per_image))

        time_per_annotation = self.get_statistics("total_time")
        for user, stat in self.get_statistics("total_time").items():
            time_per_annotation[user] = stat / self.get_statistics("total_annotations")[user]
        self.stats.append(("Time per Traffic Light", time_per_annotation))


        self.stats.append(("Most Traffic Lights in one Image", self.get_statistics("most_traffic_lights_in_one_image")))

        
        #self.showMaximized()
        self.next_stats()


        #self.draw_stats_bars("Total Time", self.get_statistics("total_time"), self.layout)
        #self.layout.addWidget(test, 3,0,1,2)


    def get_statistics(self, key):
        directory = Path(os.path.join("annotool", "users"))
        users = (os.listdir(directory))
        stats = {}
        for user in users:
            # get path to user settings file
            path = os.path.join("annotool", "users", user)
            user = Path(user).stem
            # open file and read contents
            with open(path, "r") as f:
                user_stats = json.load(f)["stats"]
                stats.update({user : user_stats[key]})

        return stats
    
    def draw_stats_bars(self, title, stats, parent):
        # reorder so the current user is first
        stats = {k: stats[k] for k in sorted(stats, key=lambda k: k == self.parent.settings_window.login.currentText(), reverse=True)}


        max_width = 200

        # create widget
        widget = QWidget()
        if parent.itemAtPosition(1, 0) != None:
            current_widget = parent.itemAtPosition(1, 0).widget()
            parent.removeWidget(current_widget)
            current_widget.deleteLater()
        parent.addWidget(widget, 1,0,1,2)
        layout = QGridLayout()
        widget.setLayout(layout)

        # create title
        title = QLabel(title)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title, 0,0,1,2)

        # get max value
        max_val = max(stats.values())

        bars = []
        
        layout.setColumnMinimumWidth(1, max_width) 

        #widget.show()

        best_animation = None

        # create bars
        for i, (user, stat) in enumerate(stats.items()):
            # create label
            label = QLabel(user)
            layout.addWidget(label, i+1, 0)
            
            # round stat to 2 decimals
            stat = round(stat, 2) if type(stat) == float else stat
            # create bar
            bar = QLabel()
            layout.addWidget(bar, i+1, 1)
            width = int((stat/max_val) * max_width)
            color = "#66f556" if stat == round(max_val, 2) else "#f5ea56"
            bar.setFixedHeight(17)
            bar.setStyleSheet("background-color: '#f5ea56';")

            #bar.show()
            best = False if stat != round(max_val, 2) else True

            
        
            # animate bar from 0 to width
            animation = QPropertyAnimation(bar, b"size", widget)
            animation.setDuration(int(4000 * (stat/max_val)))
            animation.setStartValue(QSize(0, 20))
            animation.setEndValue(QSize(width, 20))
            animation.setEasingCurve(QEasingCurve.Type.OutSine)

            if best:
                best_animation = animation

            animation.start()
            bars.append((bar, stat, color, width))

        def write_stats(bars):
            for bar, stat, color, width in bars:
                bar.setText(" " + str(stat))
                bar.setStyleSheet(f"color: #000; background-color: {color};")
                bar.setFixedWidth(width)

            QTimer.singleShot(2000, self.next_stats)  # 1000 milliseconds = 1 second

        best_animation.finished.connect(lambda: write_stats(bars))


    def next_stats(self):
        if len(self.stats) == 0:
            return
        
        # remove old stats
        #if self.layout.itemAt(1,0) != None:
        #    self.layout.itemAt(1,0).widget().deleteLater()

        #self.layout.itemAt(2).widget().deleteLater()
        
        (title, stats) = self.stats.pop(0)
        self.draw_stats_bars(title, stats, self.layout)
    

    def continue_session(self):
        self.parent.show()
        self.close()
        self.parent.timer = time.time()