import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import RayTraceTorch as rtt

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark Mode Styling for Professional Feel
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.black)
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)

    window = rtt.gui.UnifiedWorkbench()
    window.show()
    sys.exit(app.exec())