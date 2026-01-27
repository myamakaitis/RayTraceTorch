from ..render.camera import Renderer, Camera, OrbitCamera
# from ..rays.ray import Rays

import sys
import torch
import torch.nn.functional as F
import numpy as np

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QSplitter)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor


class InteractiveCamera(Camera):
    """Extends Camera with navigation logic."""

    def rotate_yaw_pitch(self, d_yaw, d_pitch):
        def rotate_vector(vec, axis, angle):
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            return vec * cos_a + torch.linalg.cross(axis, vec) * sin_a + axis * torch.dot(axis, vec) * (1 - cos_a)

        self.forward = rotate_vector(self.forward, self.right, d_pitch)
        self.up_cam = rotate_vector(self.up_cam, self.right, d_pitch)

        self.forward = rotate_vector(self.forward, self.up_cam, d_yaw)
        self.right = rotate_vector(self.right, self.up_cam, d_yaw)

        self.forward = F.normalize(self.forward, dim=0)
        self.right = F.normalize(torch.cross(self.forward, self.up_cam), dim=0)
        self.up_cam = torch.cross(self.right, self.forward)

    def roll(self, angle):
        c = torch.cos(angle)
        s = torch.sin(angle)
        new_right = c * self.right - s * self.up_cam
        new_up = s * self.right + c * self.up_cam
        self.right = new_right
        self.up_cam = new_up

    def move_local(self, dx, dy, dz):
        move_vec = (self.right * dx) + (self.up_cam * dy) + (self.forward * dz)
        self.origin += move_vec


class RenderWidget(QLabel):
    def __init__(self, renderer, camera, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self.camera = camera
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.last_pos = QPoint(0, 0)
        self._img_buffer = None
        self.refresh_render()

    def refresh_render(self):
        try:
            img_tensor = self.renderer.render_3d(self.camera)
            h, w, c = img_tensor.shape

            # 2. Force RGBA (4 channels)
            if c == 3:
                alpha = torch.ones((h, w, 1), dtype=img_tensor.dtype)
                img_tensor = torch.cat((img_tensor, alpha), dim=2)

            # 3. Convert to Bytes (The "Nuclear Option" for stability)
            # We create a distinct bytes object. QImage copies this internally
            # when using loadFromData, OR we pass it to constructor.
            # Using tobytes() creates a copy, ensuring no shared memory stride issues.
            img_data = (torch.clamp(img_tensor, 0, 1) * 255).byte().numpy()

            # 4. Create QImage
            q_img = QImage(
                img_data,
                w,
                h,
                w * 4,
                QImage.Format.Format_RGBA8888
            )

            # 5. Set Pixmap (Deep Copy)
            self.setPixmap(QPixmap.fromImage(q_img.copy()))

        except Exception as e:
            print(f"Render Error: {e}")

    def mousePressEvent(self, event):
        self.last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        dx = pos.x() - self.last_pos.x()
        dy = pos.y() - self.last_pos.y()
        self.last_pos = pos

        orbit_sens = 0.01
        pan_sens = 0.05

        # 1. Alt + Drag: ROLL (Restored functionality)
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            # Roll uses dx (horizontal drag rotates screen)
            self.camera.roll(torch.tensor(dx * orbit_sens))
            self.refresh_render()
            return

        # 2. Left Drag: ORBIT
        if event.buttons() & Qt.MouseButton.LeftButton:
            # Pass d_yaw (dx) and d_pitch (dy)
            self.camera.orbit(
                torch.tensor(dx * orbit_sens),
                torch.tensor(dy * orbit_sens)
            )
            self.refresh_render()

        # 3. Middle Drag: PAN
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self.camera.pan(
                torch.tensor(dx * pan_sens),
                torch.tensor(dy * pan_sens)
            )
            self.refresh_render()

    def wheelEvent(self, event):
        # ZOOM
        delta = event.angleDelta().y() / 120.0
        self.camera.zoom(torch.tensor(delta))
        self.refresh_render()


class ProfileWidget(QWidget):
    def __init__(self, renderer, scene, axis='x', parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self.scene = scene
        self.axis = axis  # 'x' (XZ plane) or 'y' (YZ plane)
        self.profiles = []

        # Ensure the widget has a minimum size so it doesn't collapse in the layout
        self.setMinimumHeight(250)
        self.setMinimumWidth(250)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")

    def update_data(self):
        self.profiles = []
        # Guard against missing scene/elements
        if not self.scene or not hasattr(self.scene, 'elements'):
            return

        try:
            for el in self.scene.elements:
                # Ensure the renderer has the scanning method
                if hasattr(self.renderer, 'scan_profile'):
                    # scan_profile returns a list of dicts: [{'h': np.array, 'z': np.array, ...}]
                    # We assume these arrays are already on CPU/Numpy via the Renderer implementation
                    scan_data = self.renderer.scan_profile(el, axis=self.axis, num_points=200)
                    self.profiles.extend(scan_data)
            self.update()  # Trigger paintEvent
        except Exception as e:
            print(f"Profile Scan Error ({self.axis}): {e}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        # Scale: Pixels per Unit (Adjust this if your elements are very small/large)
        scale = 10.0

        # 1. Draw Axis Lines
        painter.setPen(QColor(60, 60, 60))
        painter.drawLine(0, cy, w, cy)  # Optical Axis (Z)
        painter.drawLine(cx, 0, cx, h)  # Transverse Axis (X or Y)

        # 2. Draw Labels
        painter.setPen(QColor(180, 180, 180))
        label = "Top View (XZ)" if self.axis == 'x' else "Side View (YZ)"
        painter.drawText(10, 20, label)

        # 3. Draw Profiles
        for surf in self.profiles:
            # Generate a consistent color based on surface index
            hue = int((surf['surf_idx'] * 50) % 255)
            painter.setPen(QColor.fromHsv(hue, 200, 255))

            h_vals = surf['h']  # Transverse position
            z_vals = surf['z']  # Optical axis position

            # We iterate using standard python range to avoid tensor issues
            for i in range(len(h_vals)):
                try:
                    raw_z = z_vals[i]
                    raw_h = h_vals[i]

                    # Robust Finite Check (using numpy, matching working standalone)
                    if not (np.isfinite(raw_z) and np.isfinite(raw_h)):
                        continue

                    # Mapping:
                    # Z -> Screen X (Horizontal)
                    # H -> Screen Y (Vertical)
                    sx = int(cx + raw_z * scale)
                    sy = int(cy - raw_h * scale)

                    # Simple bounds check to prevent Qt rendering artifacts
                    if -5000 < sx < 5000 and -5000 < sy < 5000:
                        painter.drawPoint(sx, sy)
                except Exception:
                    continue


class CamWindow(QMainWindow):
    def __init__(self, scene):
        super().__init__()
        self.setWindowTitle("Differentiable Ray Tracer (PyQt6)")
        self.resize(1200, 800)

        device = next(scene.parameters()).device

        self.cam = OrbitCamera(
            pivot=(0, 0, 0),
            position=(0, 0, -60),
            look_at=(0, 0, 0),
            up_vector=(0, 1, 0),
            fov_deg=40, width=512, height=512,
            device=device
        )

        # Renderer with visibility
        self.renderer = Renderer(scene, background_color=(0.9, 0.9, 0.9))

        self.render_widget = RenderWidget(self.renderer, self.cam)
        self.profile_widget = ProfileWidget(self.renderer, scene)

        # Trigger initial data
        self.profile_widget.update_data()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.render_widget)
        splitter.addWidget(self.profile_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

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
    sys.exit(app.exec())