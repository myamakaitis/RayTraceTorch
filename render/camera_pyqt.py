from .camera import Renderer, Camera


# Note: Ensure these imports match your folder structure
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
            return vec * cos_a + torch.cross(axis, vec) * sin_a + axis * torch.dot(axis, vec) * (1 - cos_a)

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
        self.mouse_sensitivity = 0.005
        self.move_speed = 0.5

        # Initial Render
        self.refresh_render()

    def refresh_render(self):
        try:
            img_tensor = self.renderer.render_3d(self.camera).cpu()
            h, w, c = img_tensor.shape

            # 2. Force RGBA (4 channels)
            if c == 3:
                alpha = torch.ones((h, w, 1), dtype=img_tensor.dtype)
                img_tensor = torch.cat((img_tensor, alpha), dim=2)

            # 3. Convert to Bytes (The "Nuclear Option" for stability)
            # We create a distinct bytes object. QImage copies this internally
            # when using loadFromData, OR we pass it to constructor.
            # Using tobytes() creates a copy, ensuring no shared memory stride issues.
            img_data = (torch.clamp(img_tensor, 0, 1) * 255).byte().numpy().tobytes()

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

        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self.camera.roll(torch.tensor(dx * self.mouse_sensitivity))
            self.refresh_render()
            return

        if event.buttons() & Qt.MouseButton.MiddleButton:
            self.camera.move_local(
                torch.tensor(-dx * self.move_speed * 0.1),
                torch.tensor(dy * self.move_speed * 0.1),
                torch.tensor(0.0)
            )
            self.refresh_render()
            return

        if event.buttons() & Qt.MouseButton.LeftButton:
            self.camera.rotate_yaw_pitch(
                torch.tensor(-dx * self.mouse_sensitivity),
                torch.tensor(-dy * self.mouse_sensitivity)
            )
            self.refresh_render()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.camera.move_local(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(delta * self.move_speed * 5.0))
        self.refresh_render()


class ProfileWidget(QWidget):
    def __init__(self, renderer, scene):
        super().__init__()
        self.renderer = renderer
        self.scene = scene
        self.profiles = []
        self.setMinimumWidth(200)
        self.setStyleSheet("background-color: #222;")

    def update_data(self):
        self.profiles = []
        # Wrap in try-except to prevent startup crashes if physics fails
        try:
            for el in self.scene.elements:
                if hasattr(self.renderer, 'scan_profile'):
                    scan_data = self.renderer.scan_profile(el, axis='x', num_points=200)
                    self.profiles.extend(scan_data)
            self.update()
        except Exception as e:
            print(f"Profile Scan Error: {e}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        scale = 10.0

        painter.setPen(QColor(50, 50, 50))
        painter.drawLine(0, cy, w, cy)
        painter.drawLine(cx, 0, cx, h)

        for surf in self.profiles:
            # Safe Color Generation
            hue = int((surf['surf_idx'] * 50) % 255)
            painter.setPen(QColor.fromHsv(hue, 200, 255))

            h_vals = surf['h']
            z_vals = surf['z']

            for i in range(len(h_vals)):
                # Coordinate Clamping to prevent drawing at Infinity (Crash Risk)
                try:
                    raw_x = z_vals[i]
                    raw_y = h_vals[i]

                    # Skip NaNs or Infs
                    if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
                        continue

                    sx = int(cx + raw_x * scale)
                    sy = int(cy - raw_y * scale)

                    # Draw only if within reasonable bounds (Qt crashes on extreme coords)
                    if -5000 < sx < 5000 and -5000 < sy < 5000:
                        painter.drawPoint(sx, sy)
                except ValueError:
                    continue


class CamWindow(QMainWindow):
    def __init__(self, scene):
        super().__init__()
        self.setWindowTitle("Differentiable Ray Tracer (PyQt6)")
        self.resize(1200, 800)

        device = next(model.parameters()).device

        self.cam = InteractiveCamera(
            position=(0, 0, -50), look_at=(0, 0, 0), up_vector=(0, 1, 0),
            fov_deg=40, width=512, height=512,
            device=device
        )

        # Renderer with visibility
        self.renderer = Renderer(scene, background_color=(0.1, 0.1, 0.15))

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