import sys
import torch
import torch.nn as nn
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QPushButton, QSplitter,
                             QTabWidget, QMessageBox, QLabel, QListWidgetItem)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from ..scene import Scene
from ..rays.ray import Rays
from ..elements import Element
from ..render.camera import Renderer, Camera, OrbitCamera
from .camera_pyqt import RenderWidget, ProfileWidget
from .elementWindow import RecursiveElementDialog

# ==========================================
# 3. GENERIC MANAGER WIDGET
# ==========================================
class ManagerWidget(QWidget):
    def __init__(self, title, base_cls, on_update_callback):
        super().__init__()
        self.base_cls = base_cls
        self.on_update_callback = on_update_callback
        self.configs = []

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.edit_item)
        layout.addWidget(QLabel(f"Active {title}:"))
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_item)
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self.edit_item)
        self.del_btn = QPushButton("Del")
        self.del_btn.clicked.connect(self.remove_item)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.del_btn)
        layout.addLayout(btn_layout)

        self.build_btn = QPushButton(f"UPDATE SCENE")
        self.build_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #d0f0c0; color: black;")
        self.build_btn.clicked.connect(self.trigger_build)
        layout.addWidget(self.build_btn)

    def refresh_list(self):
        self.list_widget.clear()
        for idx, item in enumerate(self.configs):
            name = item['config'].get('name') or f"Item_{idx}"
            cls = item['config']['class']
            list_item = QListWidgetItem(f"{name} [{cls}]")
            list_item.setData(Qt.ItemDataRole.UserRole, idx)
            self.list_widget.addItem(list_item)

    def add_item(self):
        dlg = RecursiveElementDialog(self, element_base_cls=self.base_cls)
        dlg.setWindowTitle(f"Configure {self.base_cls.__name__}")
        if dlg.exec():
            config = dlg.get_configuration()
            if not config['name']:
                config['name'] = f"{config['class']}_{len(self.configs)}"
            self.configs.append({'config': config})
            self.refresh_list()

    def edit_item(self):
        items = self.list_widget.selectedItems()
        if not items: return
        idx = items[0].data(Qt.ItemDataRole.UserRole)
        data = self.configs[idx]

        dlg = RecursiveElementDialog(self, element_base_cls=self.base_cls)
        dlg.set_configuration(data['config'])

        if dlg.exec():
            self.configs[idx]['config'] = dlg.get_configuration()
            self.refresh_list()

    def remove_item(self):
        items = self.list_widget.selectedItems()
        if not items: return
        idx = items[0].data(Qt.ItemDataRole.UserRole)
        self.configs.pop(idx)
        self.refresh_list()

    def trigger_build(self):
        built_objects = []
        try:
            for item in self.configs:
                dlg = RecursiveElementDialog(self, element_base_cls=self.base_cls)
                dlg.set_configuration(item['config'])
                obj = dlg.instantiate_current_state()
                built_objects.append(obj)

            self.on_update_callback(built_objects)

        except Exception as e:
            QMessageBox.critical(self, "Build Error", f"Failed to instantiate: {str(e)}")
            import traceback
            traceback.print_exc()


# ==========================================
# 4. UNIFIED WORKBENCH WINDOW (New Layout)
# ==========================================
class UnifiedWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Design Workbench")
        self.resize(1600, 900)

        # 1. Initialize Scene & Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Workbench Initialized on Device: {self.device}")

        self.scene = Scene()
        self.scene.to(self.device)  # Initial move

        # 2. Setup Camera & Renderer
        self.cam = OrbitCamera(
            pivot=(0, 0, 0),
            position=(0, 0, -60),
            look_at=(0, 0, 0),
            up_vector=(0, 1, 0),
            fov_deg=40, width=800, height=800,
            device=self.device
        )
        self.renderer = Renderer(self.scene, background_color=(0.8, 0.8, 0.8))

        # 3. Main Layout Construction
        central = QWidget()
        self.setCentralWidget(central)

        # We use a Splitter for resizable columns
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.main_splitter)

        # --- COL 1: MANAGERS (Left) ---
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(300)

        self.element_manager = ManagerWidget(
            title="Optical Elements",
            base_cls=Element,
            on_update_callback=self.update_scene_elements
        )
        self.tabs.addTab(self.element_manager, "Elements")

        self.bundle_manager = ManagerWidget(
            title="Ray Bundles",
            base_cls=Rays,
            on_update_callback=self.update_scene_rays
        )
        self.tabs.addTab(self.bundle_manager, "Rays")

        self.main_splitter.addWidget(self.tabs)

        # --- COL 2: 3D RENDER (Center) ---
        self.render_container = QWidget()
        rc_layout = QVBoxLayout(self.render_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.addWidget(QLabel("<b>3D Interactive View</b>"))

        self.render_widget = RenderWidget(self.renderer, self.cam)
        rc_layout.addWidget(self.render_widget)

        self.main_splitter.addWidget(self.render_container)

        # --- COL 3: CROSS SECTIONS (Right) ---
        self.profiles_container = QWidget()
        pc_layout = QVBoxLayout(self.profiles_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(10)

        # XZ View (Top)
        self.profile_xz = ProfileWidget(self.renderer, self.scene, axis='x')
        pc_layout.addWidget(self.profile_xz)

        # YZ View (Bottom)
        self.profile_yz = ProfileWidget(self.renderer, self.scene, axis='y')
        pc_layout.addWidget(self.profile_yz)

        self.main_splitter.addWidget(self.profiles_container)

        # Set Stretch Factors to ensure visibility
        # Manager : 3D View : Profiles
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setStretchFactor(2, 2)
    # --- Callbacks ---

    def update_scene_elements(self, new_elements):
        print(f"Updating Scene with {len(new_elements)} elements on {self.device}...")

        self.scene.clear_elements()
        self.scene.elements.extend(nn.ModuleList(new_elements))

        # CRITICAL: Move scene to GPU *before* rebuilding maps
        self.scene.to(self.device)

        # Rebuild maps (now that elements are on GPU, maps will be created on GPU)
        self.scene._build_index_maps()

        self.refresh_all_views()

    def update_scene_rays(self, new_bundles):
        print(f"Updating Scene with {len(new_bundles)} ray bundles...")

        self.scene.clear_rays()
        if len(new_bundles) > 0:
            # Assuming single bundle for now
            self.scene.rays = new_bundles[0]

            # Ensure rays are on correct device
            if hasattr(self.scene.rays, 'to'):
                self.scene.rays = self.scene.rays.to(self.device)

        # Scene is already on device, but ensure consistency
        self.scene.to(self.device)

        self.refresh_all_views()

    def refresh_all_views(self):
        """Forces a repaint of 3D and Profile widgets."""
        self.render_widget.refresh_render()
        self.profile_xz.update_data()
        self.profile_yz.update_data()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.Base, QColor(15, 15, 15))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(palette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(palette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(palette.ColorRole.Highlight, QColor(100, 100, 225))
    palette.setColor(palette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    window = UnifiedWorkbench()
    window.show()
    sys.exit(app.exec())