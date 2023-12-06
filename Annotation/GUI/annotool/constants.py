IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

ANNOTATED_PATH = 'annotations'
IMAGE_DIR = 'frames'

from PyQt6.QtGui import QColor
COLORS = {'none': QColor(255, 255, 255),
          'red': QColor(255, 0, 0),
          'red-yellow': QColor(255, 128, 0),
          'yellow': QColor(255, 255, 0),
          'green': QColor(0, 255, 0),
          'off': QColor(0, 0, 100)
          }