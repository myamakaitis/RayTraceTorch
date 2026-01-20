import sys
import inspect
import torch
import torch.nn as nn
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QPushButton, QComboBox,
                             QFormLayout, QDialog, QLineEdit, QDialogButtonBox,
                             QLabel, QCheckBox, QMessageBox, QScrollArea, QGroupBox)
import ast
import typing


from ..elements import *
from ..geom import Vector3, Bool3
from ..geom import RayTransform
# ==========================================
# 2. INTROSPECTION LOGIC
# ==========================================

def get_subclasses(cls):
    """Iterative subclass finder."""
    if cls is None: return set()
    subclasses = set()
    queue = [cls]
    while queue:
        parent = queue.pop(0)
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                queue.append(child)
    return subclasses

def get_actual_class(p_type):
    """Unwraps Optional[Class] or Union[Class, None] to return Class."""
    if inspect.isclass(p_type):
        return p_type

    # Handle Unions/Optionals
    origin = typing.get_origin(p_type)
    if origin is typing.Union:
        args = typing.get_args(p_type)
        for arg in args:
            if inspect.isclass(arg) and arg is not type(None):
                return arg
    return None

def get_constructor_params(cls):
    """Safely extract __init__ signature."""
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}
    params = {}
    for name, p in sig.parameters.items():
        if name == 'self': continue
        p_type = p.annotation if p.annotation != inspect.Parameter.empty else str
        default = p.default if p.default != inspect.Parameter.empty else None
        params[name] = (p_type, default)
    return params


# ==========================================
# 3. GUI IMPLEMENTATION
# ==========================================

class Vector3Input(QWidget):
    def __init__(self, default=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.inputs = []

        # FIX: Ensure 'defaults' is always a list to prevent len() crash
        if isinstance(default, (list, tuple)):
            vals = default
        elif isinstance(default, torch.Tensor):
            vals = default.tolist()
        elif isinstance(default, (int, float)):
            vals = [default] * 3
        else:
            vals = [0.0, 0.0, 0.0]

        for i in range(3):
            val = vals[i] if i < len(vals) else 0.0
            le = QLineEdit(str(val))
            layout.addWidget(le)
            self.inputs.append(le)

    def get_value(self):
        try:
            return [float(x.text()) for x in self.inputs]
        except ValueError:
            return [0.0, 0.0, 0.0]


class Bool3Input(QWidget):
    def __init__(self, default=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.checks = []

        # FIX: Ensure list to prevent crash
        if isinstance(default, (list, tuple)):
            vals = default
        elif isinstance(default, torch.Tensor):
            vals = default.tolist()
        elif isinstance(default, bool):
            vals = [default] * 3
        else:
            vals = [False, False, False]

        for i in range(3):
            chk = QCheckBox()
            if i < len(vals) and vals[i]: chk.setChecked(True)
            layout.addWidget(chk)
            self.checks.append(chk)

    def get_value(self):
        return [chk.isChecked() for chk in self.checks]


class RecursiveElementDialog(QDialog):
    def __init__(self, parent=None, element_base_cls=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Element")
        self.resize(600, 750)

        self.main_layout = QVBoxLayout(self)

        # 1. Class Selector
        self.class_selector = QComboBox()
        self.known_classes = {cls.__name__: cls for cls in get_subclasses(element_base_cls)}
        self.class_selector.addItems(list(self.known_classes.keys()))
        self.class_selector.currentTextChanged.connect(self.refresh_form)

        self.main_layout.addWidget(QLabel("Select Element Type:"))
        self.main_layout.addWidget(self.class_selector)

        # 2. Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.form_container = None
        self.main_layout.addWidget(self.scroll)

        # 3. Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.validate_and_accept)
        btns.rejected.connect(self.reject)
        self.main_layout.addWidget(btns)

        if self.known_classes:
            self.refresh_form(self.class_selector.currentText())

    def refresh_form(self, class_name):
        if not class_name: return
        if self.form_container: self.form_container.deleteLater()

        self.form_container = QWidget()
        self.scroll.setWidget(self.form_container)
        self.current_layout = QFormLayout(self.form_container)

        cls = self.known_classes[class_name]
        try:
            self.build_recursive_form(self.current_layout, cls, depth=0)
        except Exception as e:
            # This catches bugs gracefully instead of crashing hard
            self.current_layout.addRow(QLabel(f"Error building form: {e}"))
            print(f"CRITICAL: {e}")

    def build_recursive_form(self, parent_layout, cls, depth):
        if depth > 5:
            parent_layout.addRow(QLabel("Max recursion depth"))
            return

        params = get_constructor_params(cls)
        parent_layout.target_class = cls
        parent_layout.widgets = {}
        parent_layout.sub_forms = {}

        # Identify "Gradient" pairs (e.g., 'c1' and 'c1_grad')
        grad_keys = {k for k in params.keys() if k.endswith("_grad")}

        for name, (p_type, default) in params.items():
            # SKIP if this is a _grad param (we handle it attached to the main param)
            if name in grad_keys:
                continue

            type_str = str(p_type)
            widget = None
            w_type_tag = 'PRIMITIVE'

            # --- 1. DETERMINE WIDGET TYPE ---

            if p_type == bool:
                widget = QCheckBox()
                if default is True: widget.setChecked(True)
                w_type_tag = 'BOOL'

            elif p_type == torch.dtype:
                widget = QComboBox()
                widget.addItems(["float32", "float64"])
                if default == torch.float64: widget.setCurrentText("float64")
                w_type_tag = 'DTYPE'

            elif "Vector3" in type_str or p_type == Vector3:
                widget = Vector3Input(default)
                w_type_tag = 'VEC3'

            elif "Bool3" in type_str or p_type == Bool3:
                widget = Bool3Input(default)
                w_type_tag = 'BOOL3'

            # --- 2. CHECK FOR RECURSION ---
            else:
                target_sub_class = get_actual_class(p_type)
                # CRITICAL FIX: Add 'bool' to the exclusion list
                if target_sub_class and target_sub_class not in (str, int, float, bool, list, tuple, dict,
                                                                 torch.Tensor):

                    group = QGroupBox(f"{name} ({target_sub_class.__name__})")
                    group_layout = QFormLayout()
                    group.setLayout(group_layout)

                    # Handle Optional Groups (defaults to None)
                    if default is None:
                        group.setCheckable(True)
                        group.setChecked(False)
                        group.setTitle(f"{name} (Optional)")

                    self.build_recursive_form(group_layout, target_sub_class, depth + 1)
                    parent_layout.addRow(group)
                    parent_layout.sub_forms[name] = (group_layout, group)
                    continue

                # Default Primitive
                widget = QLineEdit()
                if default is not None: widget.setText(str(default))
                w_type_tag = 'PRIMITIVE'

            # --- 3. ADD WIDGET TO LAYOUT (With optional Gradient Checkbox) ---

            grad_name = f"{name}_grad"
            if grad_name in grad_keys:
                # Create Side-by-Side Layout
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(0, 0, 0, 0)

                # Add Main Widget
                h_layout.addWidget(widget)

                # Add Gradient Checkbox
                grad_chk = QCheckBox("Grad?")
                # Check default for the grad param
                grad_default = params[grad_name][1]
                if grad_default is True: grad_chk.setChecked(True)

                h_layout.addWidget(grad_chk)

                parent_layout.addRow(name, container)

                # Register BOTH widgets
                parent_layout.widgets[name] = (widget, w_type_tag)
                parent_layout.widgets[grad_name] = (grad_chk, 'BOOL')  # Treat as standard bool
            else:
                # Standard Row
                parent_layout.addRow(name, widget)
                parent_layout.widgets[name] = (widget, w_type_tag)

    def instantiate_recursive(self, layout):
        kwargs = {}

        # 1. Harvest Widgets
        for name, (widget, w_type) in layout.widgets.items():
            if w_type == 'BOOL':
                kwargs[name] = widget.isChecked()
            elif w_type == 'DTYPE':
                kwargs[name] = torch.float64 if widget.currentText() == "float64" else torch.float32
            elif w_type == 'VEC3':
                kwargs[name] = widget.get_value()
            elif w_type == 'BOOL3':
                kwargs[name] = widget.get_value()
            elif w_type == 'PRIMITIVE':
                val = widget.text()
                try:
                    val = ast.literal_eval(val)
                except:
                    pass
                kwargs[name] = val

        # 2. Harvest Sub-Forms
        for name, (sub_layout, group_box) in layout.sub_forms.items():
            if group_box.isCheckable() and not group_box.isChecked():
                kwargs[name] = None
            else:
                kwargs[name] = self.instantiate_recursive(sub_layout)

        return layout.target_class(**kwargs)

    def validate_and_accept(self):
        try:
            self.new_instance = self.instantiate_recursive(self.current_layout)
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Creation Error", str(e))

    def get_instance(self):
        return self.new_instance


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Element Manager")
        self.resize(600, 400)
        self.elements = []

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.add_btn = QPushButton("Add Element")
        self.add_btn.clicked.connect(self.open_add_dialog)
        layout.addWidget(self.add_btn)

    def open_add_dialog(self):
        # FIX IS HERE: We pass 'Element' as the base class
        dlg = RecursiveElementDialog(self, element_base_cls=Element)

        if dlg.exec():
            new_obj = dlg.get_instance()
            self.elements.append(new_obj)

            # Display result
            cls_name = new_obj.__class__.__name__
            self.list_widget.addItem(f"{cls_name} added.")
            print(f"Created {cls_name}: {new_obj.__dict__}")
            # If it has a transform, print that too
            if hasattr(new_obj, 'transform'):
                print(f" -> Transform: {new_obj.transform.__dict__}")

# ==========================================
# 4. RUNNER
# ==========================================

