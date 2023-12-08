IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

ANNOTATED_PATH = 'annotations'
IMAGE_DIR = 'frames'

from PyQt6.QtGui import QColor
COLORS = {'none': QColor(255, 255, 255),
          'red': QColor(245, 86, 86),
          'red-yellow': QColor(245, 150, 86),
          'yellow': QColor(245, 245, 86),
          'green': QColor(100, 245, 86),
          'off': QColor(20, 20, 86)
          }