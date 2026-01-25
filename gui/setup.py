import sys
import torch
import torch.nn as nn
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QPushButton, QSplitter,
                             QTabWidget, QMessageBox, QLabel, QListWidgetItem)
from PyQt6.QtCore import Qt
from ..scene import Scene
from ..rays.ray import Rays
from ..elements import Element
from ..render.camera import Renderer, Camera, OrbitCamera
from .camera_pyqt import RenderWidget, ProfileWidget
from .elementWindow import RecursiveElementDialog
# ==========================================
# 2. GENERIC MANAGER WIDGET (Elements & Bundles)
# ==========================================
class ManagerWidget(QWidget):
    """
    A generic version of ElementManagerWindow that can handle
    either Elements or Rays based on the `base_cls` passed to it.
    """

    def __init__(self, title, base_cls, on_update_callback):
        super().__init__()
        self.base_cls = base_cls
        self.on_update_callback = on_update_callback
        self.configs = []  # Stores dicts of {'config': {...}}

        # Layout
        layout = QVBoxLayout(self)

        # List
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.edit_item)
        layout.addWidget(QLabel(f"Active {title}:"))
        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add New")
        self.add_btn.clicked.connect(self.add_item)
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self.edit_item)
        self.del_btn = QPushButton("Remove")
        self.del_btn.clicked.connect(self.remove_item)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.del_btn)
        layout.addLayout(btn_layout)

        # The "Push to Scene" button
        self.build_btn = QPushButton(f"UPDATE SCENE ({title})")
        self.build_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #d0f0c0;")
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
        # We reuse the powerful RecursiveElementDialog
        dlg = RecursiveElementDialog(self, element_base_cls=self.base_cls)
        dlg.setWindowTitle(f"Configure {self.base_cls.__name__}")
        if dlg.exec():
            config = dlg.get_configuration()
            if not config['name']:
                config['name'] = f"{config['class']}_{len(self.configs)}"
            self.configs.append({'config': config})
            self.refresh_list()
            # Optional: Auto-build on add?
            # self.trigger_build()

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
        """
        Instantiates objects from configs and sends them to the main window
        via the callback.
        """
        built_objects = []
        try:
            for item in self.configs:
                # We use a dummy dialog just to access the instantiation logic
                # logic is contained in the class, not the instance window really
                dlg = RecursiveElementDialog(self, element_base_cls=self.base_cls)
                dlg.set_configuration(item['config'])
                obj = dlg.instantiate_current_state()
                built_objects.append(obj)

            # Fire callback with the list of new PyTorch objects
            self.on_update_callback(built_objects)

        except Exception as e:
            QMessageBox.critical(self, "Build Error", f"Failed to instantiate: {str(e)}")
            import traceback
            traceback.print_exc()


# ==========================================
# 3. UNIFIED WORKBENCH WINDOW
# ==========================================
class UnifiedWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Design Workbench")
        self.resize(1400, 900)

        # 1. Initialize Scene
        self.scene = Scene()

        # 2. Setup Camera & Renderer
        # (Assuming CUDA if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cam = OrbitCamera(
            pivot=(0, 0, 0),
            position=(0, 0, -60),
            look_at=(0, 0, 0),
            up_vector=(0, 1, 0),
            fov_deg=40, width=800, height=800,
            device=device
        )
        self.renderer = Renderer(self.scene, background_color=(0.1, 0.1, 0.15))

        # 3. Main Layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- LEFT PANEL (Managers) ---
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(350)

        # Element Manager
        self.element_manager = ManagerWidget(
            title="Optical Elements",
            base_cls=Element,
            on_update_callback=self.update_scene_elements
        )
        self.tabs.addTab(self.element_manager, "Elements")

        # Bundle Manager (Requested Feature)
        self.bundle_manager = ManagerWidget(
            title="Ray Bundles",
            base_cls=Rays,
            on_update_callback=self.update_scene_rays
        )
        self.tabs.addTab(self.bundle_manager, "Rays")

        main_layout.addWidget(self.tabs)

        # --- RIGHT PANEL (Visualizer) ---
        self.viz_splitter = QSplitter(Qt.Orientation.Vertical)

        # 3D View
        self.render_widget = RenderWidget(self.renderer, self.cam)
        self.viz_splitter.addWidget(self.render_widget)

        # Profile View (Bottom)
        self.profile_widget = ProfileWidget(self.renderer, self.scene)
        self.viz_splitter.addWidget(self.profile_widget)
        self.viz_splitter.setStretchFactor(0, 4)
        self.viz_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.viz_splitter)

    # --- Callbacks ---

    def update_scene_elements(self, new_elements):
        """Called when 'UPDATE SCENE' is clicked in the Elements tab."""
        print(f"Updating Scene with {len(new_elements)} elements...")

        # Clear and Replace
        self.scene.clear_elements()
        self.scene.elements.extend(nn.ModuleList(new_elements))

        # Move to device
        device = self.cam.device
        self.scene.to(device)

        # Refresh Views
        self.refresh_all_views()

    def update_scene_rays(self, new_bundles):
        """Called when 'UPDATE SCENE' is clicked in the Rays tab."""
        print(f"Updating Scene with {len(new_bundles)} ray bundles...")

        self.scene.clear_rays()
        self.scene.rays.extend(nn.ModuleList(new_bundles))

        device = self.cam.device
        self.scene.to(device)

        self.refresh_all_views()

    def refresh_all_views(self):
        """Forces a repaint of 3D and Profile widgets."""
        self.render_widget.refresh_render()
        self.profile_widget.update_data()


# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dark Mode Styling for Professional Feel
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.black)
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)

    window = UnifiedWorkbench()
    window.show()
    sys.exit(app.exec())