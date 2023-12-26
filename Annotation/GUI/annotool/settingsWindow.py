from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QFileDialog
from PyQt6.QtGui import QIntValidator, QColor
from PyQt6.QtCore import QMetaMethod, Qt

from pathlib import Path
import json

import sys
import os


class SettingsWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Settings")
        self.setWindowIcon(parent.windowIcon())
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint)
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint)
        self.setWindowFlag(Qt.WindowType.WindowTitleHint)
        self.setWindowFlag(Qt.WindowType.WindowSystemMenuHint)
        self.setWindowFlag(Qt.WindowType.Window)

        self.setStyleSheet("style.qss")

        #self.setFixedSize(400, 400)

        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)

        
        label = QLabel("Settings")
        label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(label, 0, 0, 1, 2)

        self.layout.addWidget(QLabel("Login as:"), 1, 0)
        self.login = QComboBox()
        self.login.activated.connect(self.update_settings)
        self.add_users(Path(os.path.join("annotool", "users")))
        self.layout.addWidget(self.login, 1, 1)

        self.layout.addWidget(QLabel("Path to input folder:"), 2, 0)
        self.input_folder = QPushButton(self.get_user_settings()["input_folder"])
        self.input_folder.clicked.connect(self.open_folder_dialog_input)
        self.input_folder.setStyleSheet("text-align: left; border-radius: 0;")
        self.layout.addWidget(self.input_folder, 2, 1)


        self.layout.addWidget(QLabel("Path to output folder:"), 3, 0)
        self.output_folder = QPushButton(self.get_user_settings()["output_folder"])
        self.output_folder.clicked.connect(self.open_folder_dialog_output)
        self.output_folder.setStyleSheet("text-align: left; border-radius: 0;")
        self.layout.addWidget(self.output_folder, 3, 1)

        self.layout.addWidget(QLabel("Output Image Size:"), 4, 0)
        self.output_size = QComboBox()
        self.output_size.addItem("1920x1080")
        self.output_size.addItem("1280x720")
        self.output_size.addItem("640x480")
        self.output_size.addItem("320x240")
        self.output_size.setCurrentText(self.get_user_settings()["output_size"])



        self.layout.addWidget(self.output_size, 4, 1)

        self.save = QPushButton("Save Settings")
        self.save.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save, 5, 1)

        self.discard = QPushButton("Discard Changes")
        self.discard.clicked.connect(self.discard_changes)
        self.layout.addWidget(self.discard, 5, 0)




    def add_users(self, directory):
        # get filenames in directory and parse them to get usernames
        users = (os.listdir(directory))
        for user in users:
            self.login.addItem(Path(user).stem)

    def update_settings(self):
        # get user settings
        settings = self.get_user_settings()
        # update input folder
        self.input_folder.setText(settings["input_folder"])
        # update output folder
        self.output_folder.setText(settings["output_folder"])
        # update output size
        self.output_size.setCurrentText(settings["output_size"])


    def get_user_settings(self):
        user = self.login.currentText()
        # get path to user settings file
        path = os.path.join("annotool", "users", user + ".json")
        # open file and read contents
        with open(path, "r") as f:
            settings = json.load(f)
        return settings
        

    def open_folder_dialog_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        # get relative path to annotool folder 
        relative_path = os.path.relpath(folder, os.getcwd())
        # update button text
        self.input_folder.setText(relative_path)

    def open_folder_dialog_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        # get relative path to annotool folder 
        relative_path = os.path.relpath(folder, os.getcwd())
        # update button text
        self.output_folder.setText(relative_path)

    def save_settings(self):
        # get user settings
        settings = self.get_user_settings()
        # update input folder
        settings["input_folder"] = self.input_folder.text()
        # update output folder
        settings["output_folder"] = self.output_folder.text()
        # update output size
        settings["output_size"] = self.output_size.currentText()
        # get path to user settings file
        path = os.path.join("annotool", "users", self.login.currentText() + ".json")
        # open file and read contents
        with open(path, "w") as f:
            json.dump(settings, f, indent=4)

        if self.parent.first_run and self.parent.wrong_path:
            self.parent.first_run = False
            self.parent.wrong_path = False
            self.parent.next_image()
        elif self.parent.first_run:
            self.parent.first_run = False
            self.parent.next_image()
        elif self.parent.wrong_path:
            self.parent.wrong_path = False
            self.parent.next_image()

        self.close()

    def discard_changes(self):
        if self.parent.first_run:
            self.parent.first_run = False
            self.parent.next_image()
            self.parent.show()
        elif self.parent.wrong_path:
            self.parent.wrong_path = False
            self.parent.next_image()
            self.parent.show()

        self.close()
