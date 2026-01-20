import sys
import torch
import torch.nn.functional as F
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QSplitter)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor

# Imports from your codebase structure
# Ensure your PYTHONPATH is set or these files are in the package
from .camera import Renderer, Camera
from ..rays.ray import Rays


class InteractiveCamera(Camera):
    """
    Extends the static Camera with mutation methods for navigation.
    """

    def rotate_yaw_pitch(self, d_yaw, d_pitch):
        """
        Rotates the camera direction based on mouse drag.
        """

        # 1. Pitch (Rotate around Right vector)
        # Create rotation matrix for Pitch
        # Axis-angle rotation formula or simple quaternion could be used.
        # Here we use Rodrigues' rotation formula logic simplified for vectors.

        # Rotations using PyTorch for consistency
        def rotate_vector(vec, axis, angle):
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            return vec * cos_a + torch.cross(axis, vec) * sin_a + axis * torch.dot(axis, vec) * (1 - cos_a)

        # Apply Pitch
        self.forward = rotate_vector(self.forward, self.right, d_pitch)
        self.up_cam = rotate_vector(self.up_cam, self.right, d_pitch)

        # Apply Yaw (Rotate around World Up, usually (0,1,0), or Local Up?)
        # Standard FPS controls use World Up for Yaw to prevent rolling.
        # Flight controls use Local Up. Let's use Local Up for orbital feel
        # or World Up (0,1,0) for stable horizon.
        # Using Camera Up (Local) allows full 6DOF rotation (like holding an object).
        self.forward = rotate_vector(self.forward, self.up_cam, d_yaw)
        self.right = rotate_vector(self.right, self.up_cam, d_yaw)

        # Re-normalize and orthogonalize to prevent drift
        self.forward = F.normalize(self.forward, dim=0)
        self.right = F.normalize(torch.cross(self.forward, self.up_cam), dim=0)
        self.up_cam = torch.cross(self.right, self.forward)

    def roll(self, angle):
        """Rotates around the Forward axis."""
        c = torch.cos(angle)
        s = torch.sin(angle)

        # Standard 2D rotation in the Right-Up plane
        new_right = c * self.right - s * self.up_cam
        new_up = s * self.right + c * self.up_cam

        self.right = new_right
        self.up_cam = new_up

    def move_local(self, dx, dy, dz):
        """
        Moves the camera origin relative to its orientation.
        dx: Right, dy: Up, dz: Forward
        """
        move_vec = (self.right * dx) + (self.up_cam * dy) + (self.forward * dz)
        self.origin += move_vec


class RenderWidget(QLabel):
    """
    The 3D Viewport. Intercepts mouse events to control the camera.
    """

    def __init__(self, renderer, camera, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self.camera = camera
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.last_pos = QPoint(0, 0)

        # Settings
        self.mouse_sensitivity = 0.005
        self.move_speed = 0.5

        # Initial Render
        self.refresh_render()

    def refresh_render(self):
        """Calls the renderer and updates the widget image."""
        # Get tensor image: [H, W, 3] (0.0 - 1.0)
        img_tensor = self.renderer.render_3d(self.camera)

        # Convert to uint8 numpy for QImage
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        h, w, c = img_np.shape

        # Create QImage
        q_img = QImage(img_np.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img))

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        self.last_pos = event.pos()

        # Input Handling Logic

        # 1. Alt + Drag (Left or Right) -> Roll
        if event.modifiers() & Qt.AltModifier:
            # Roll based on horizontal drag
            self.camera.roll(torch.tensor(dx * self.mouse_sensitivity))
            self.refresh_render()
            return

        # 2. Middle Mouse (Button 3) -> Pan/Truck (XY Translation)
        if event.buttons() & Qt.MiddleButton:
            # -dx moves right (so we subtract to drag scene), +dy moves down (subtract to drag up)
            # Actually, dragging scene means moving camera opposite.
            self.camera.move_local(
                torch.tensor(-dx * self.move_speed * 0.1),
                torch.tensor(dy * self.move_speed * 0.1),
                torch.tensor(0.0)
            )
            self.refresh_render()
            return

        # 3. Standard Left Drag -> Look (Yaw/Pitch)
        if event.buttons() & Qt.LeftButton:
            # Invert dy for natural "head" movement (push up to look up)
            self.camera.rotate_yaw_pitch(
                torch.tensor(-dx * self.mouse_sensitivity),
                torch.tensor(-dy * self.mouse_sensitivity)
            )
            self.refresh_render()

    def wheelEvent(self, event):
        """Scroll to Dolly (Move Z)."""
        # angleDelta().y() is usually +/- 120
        delta = event.angleDelta().y() / 120.0
        self.camera.move_local(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(delta * self.move_speed * 5.0)
        )
        self.refresh_render()


class ProfileWidget(QWidget):
    """
    Displays the 2D cross-section using Matplotlib backend or simple QPainter.
    For speed/simplicity here, we use QPainter to draw the points directly.
    """

    def __init__(self, renderer, scene):
        super().__init__()
        self.renderer = renderer
        self.scene = scene
        self.profiles = []
        self.setMinimumWidth(200)
        self.setStyleSheet("background-color: #222;")

    def update_data(self):
        """Re-scans the scene."""
        self.profiles = []
        # Scan all elements
        for el in self.scene.elements:
            # Heuristic scan width
            scan_data = self.renderer.scan_profile(el, axis='x', num_points=200)
            self.profiles.extend(scan_data)
        self.update()  # Trigger paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # Coordinate System:
        # Center of widget is (0,0) in world space.
        # Scale: 10 pixels per mm?
        scale = 10.0
        cx, cy = w // 2, h // 2

        # Draw Axis
        painter.setPen(QColor(50, 50, 50))
        painter.drawLine(0, cy, w, cy)  # Z axis (horizontal)
        painter.drawLine(cx, 0, cx, h)  # Y/X axis (vertical)

        # Draw Points
        for surf in self.profiles:
            # Color by surface index to distinguish
            hue = (surf['surf_idx'] * 50) % 255
            painter.setPen(QColor.fromHsv(hue, 200, 255))

            h_vals = surf['h']
            z_vals = surf['z']

            # Draw individual points
            for i in range(len(h_vals)):
                # Map World (z, h) -> Screen (x, y)
                # World Z is horizontal, World H is vertical
                sx = cx + int(z_vals[i] * scale)
                sy = cy - int(h_vals[i] * scale)  # Invert Y for screen coords

                painter.drawPoint(sx, sy)


class CamWindow(QMainWindow):
    def __init__(self, scene):
        super().__init__()
        self.setWindowTitle("Differentiable Ray Tracer")
        self.resize(1200, 800)

        # Initialize Logic
        # Position camera back 50 units
        self.cam = InteractiveCamera(
            position=(0, 0, -50), look_at=(0, 0, 0), up_vector=(0, 1, 0),
            fov_deg=40, width=800, height=800, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.renderer = Renderer(scene, background_color=(0.1, 0.1, 0.15))

        # Widgets
        self.render_widget = RenderWidget(self.renderer, self.cam)
        self.profile_widget = ProfileWidget(self.renderer, scene)
        self.profile_widget.update_data()  # Initial scan

        # Layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.render_widget)
        splitter.addWidget(self.profile_widget)
        splitter.setStretchFactor(0, 3)  # 3D view is larger
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # Instructions Overlay
        overlay = QLabel(
            "Controls:\nLeft Drag: Rotate\nAlt+Drag: Roll\nMiddle Drag: Pan XY\nScroll: Pan Z",
            self.render_widget
        )
        overlay.setStyleSheet("color: white; background: rgba(0,0,0,0.5); padding: 5px;")
        overlay.move(10, 10)


def run_gui(scene):
    app = QApplication(sys.argv)
    window = CamWindow(scene)
    window.show()
    sys.exit(app.exec_())