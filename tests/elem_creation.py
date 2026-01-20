import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Assuming these imports exist in your project structure
# from my_library import SingletLens, Rays, collimatedSource

from RayTraceTorch.gui import MainWindow

from PyQt6.QtWidgets import QApplication



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())