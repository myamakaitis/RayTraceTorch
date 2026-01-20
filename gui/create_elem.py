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

import sys
import inspect
import ast
import torch
import torch.nn as nn
import typing
from typing import Union, List, Tuple, Optional, Any
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QPushButton, QComboBox,
                             QFormLayout, QDialog, QLineEdit, QDialogButtonBox,
                             QLabel, QCheckBox, QGroupBox, QScrollArea, QMessageBox)

# ==========================================
# 1. TYPE DEFINITIONS & ANALYZER
# ==========================================


def analyze_type(p_type):
    """
    Robustly determines the 'Intent' of a type hint.
    Returns: 'BOOL', 'VEC3', 'BOOL3', 'DTYPE', 'CLASS', or 'PRIMITIVE'
    """
    # 1. Handle Torch Dtype explicitly
    if p_type == torch.dtype:
        return 'DTYPE'

    # 2. Unwrap Optional/Union layers to find the "core" types
    # This turns Optional[Vector3] into {Tensor, List[float], Tuple[float]}
    origins = set()
    args = set()

    def unpack(t):
        origin = typing.get_origin(t)
        if origin is typing.Union:
            for arg in typing.get_args(t):
                unpack(arg)
        elif t is not type(None):  # Ignore NoneType
            origins.add(origin)
            args.add(t)

    unpack(p_type)

    # 3. Match against known signatures

    # Check for Vector3 (Looks for Tensor AND float lists)
    # We check if torch.Tensor is one of the allowed types
    if torch.Tensor in args:
        # Distinguish Vector3 from Bool3 based on list contents if possible
        # Vector3 usually implies List[float], Bool3 implies List[bool]
        if List[bool] in args or Tuple[bool, ...] in args:
            return 'BOOL3'
        return 'VEC3'  # Default to Vec3 if Tensor is present

    # Check for simple Bool (excluding the list variants)
    if bool in args and len(args) == 1:
        return 'BOOL'

    # Check for Class (Recursion)
    # If we have a custom class in the mix
    for a in args:
        if inspect.isclass(a) and a not in (int, float, str, bool, list, tuple, dict, torch.Tensor):
            return 'CLASS'

    return 'PRIMITIVE'


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


def get_actual_class(p_type):
    """Helper to extract the class from Optional[Class]."""
    if inspect.isclass(p_type): return p_type
    origin = typing.get_origin(p_type)
    if origin is typing.Union:
        for arg in typing.get_args(p_type):
            if inspect.isclass(arg) and arg is not type(None):
                return arg
    return None


# ==========================================
# 2. CUSTOM WIDGETS
# ==========================================

class Vector3Input(QWidget):
    def __init__(self, default=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.inputs = []

        # Safe default handling
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

        if isinstance(default, (list, tuple)):
            vals = default
        elif isinstance(default, torch.Tensor):
            vals = default.tolist()
        elif isinstance(default, bool):
            vals = [default] * 3
        else:
            vals = [False, False, False]

        labels = ["", "", ""]
        for i in range(3):
            chk = QCheckBox(labels[i])
            if i < len(vals) and vals[i]: chk.setChecked(True)
            layout.addWidget(chk)
            self.checks.append(chk)

    def get_value(self):
        return [chk.isChecked() for chk in self.checks]


# ==========================================
# 3. MAIN DIALOG LOGIC
# ==========================================

class RecursiveElementDialog(QDialog):
    def __init__(self, parent=None, element_base_cls=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Element")
        self.resize(650, 800)

        layout = QVBoxLayout(self)

        # Class Selector
        self.class_selector = QComboBox()
        self.known_classes = {cls.__name__: cls for cls in get_subclasses(element_base_cls)}
        self.class_selector.addItems(list(self.known_classes.keys()))
        self.class_selector.currentTextChanged.connect(self.refresh_form)

        layout.addWidget(QLabel("Select Element Type:"))
        layout.addWidget(self.class_selector)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.form_container = None
        layout.addWidget(self.scroll)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        if self.known_classes:
            self.refresh_form(self.class_selector.currentText())

    def refresh_form(self, class_name):
        if not class_name: return
        if self.form_container: self.form_container.deleteLater()

        self.form_container = QWidget()
        self.scroll.setWidget(self.form_container)
        self.current_layout = QFormLayout(self.form_container)

        try:
            cls = self.known_classes[class_name]
            self.build_recursive_form(self.current_layout, cls, depth=0)
        except Exception as e:
            self.current_layout.addRow(QLabel(f"Error: {e}"))
            print(f"Error: {e}")

    def build_recursive_form(self, parent_layout, cls, depth):
        if depth > 5:
            parent_layout.addRow(QLabel("Max recursion depth"))
            return

        params = get_constructor_params(cls)
        parent_layout.target_class = cls
        parent_layout.widgets = {}
        parent_layout.sub_forms = {}

        # Identify gradient pairs
        grad_keys = {k for k in params.keys() if k.endswith("_grad")}

        for name, (p_type, default) in params.items():
            if name in grad_keys: continue

            # --- 1. ANALYZE TYPE INTENT ---
            intent = analyze_type(p_type)

            widget = None

            # --- 2. CREATE WIDGET BASED ON INTENT ---
            if intent == 'BOOL':
                widget = QCheckBox()
                if default is True: widget.setChecked(True)

            elif intent == 'DTYPE':
                widget = QComboBox()
                widget.addItems(["float32", "float64"])
                if default == torch.float64: widget.setCurrentText("float64")

            elif intent == 'VEC3':
                widget = Vector3Input(default)

            elif intent == 'BOOL3':
                widget = Bool3Input(default)

            elif intent == 'CLASS':
                # Handle Recursion
                target_cls = get_actual_class(p_type)
                if target_cls:
                    group = QGroupBox(f"{name} ({target_cls.__name__})")
                    group_layout = QFormLayout()
                    group.setLayout(group_layout)

                    if default is None:
                        group.setCheckable(True)
                        group.setChecked(False)
                        group.setTitle(f"{name} (Optional)")

                    self.build_recursive_form(group_layout, target_cls, depth + 1)
                    parent_layout.addRow(group)
                    parent_layout.sub_forms[name] = (group_layout, group)
                    continue

            # Default / Primitive fallback
            if widget is None:
                widget = QLineEdit()
                if default is not None: widget.setText(str(default))
                intent = 'PRIMITIVE'  # Force tag for instantiation

            # --- 3. HANDLE GRADIENT CHECKBOX PAIRING ---
            grad_name = f"{name}_grad"
            if grad_name in grad_keys:
                # We need a container to hold [MainWidget] + [GradCheckbox]
                container = QWidget()
                h_layout = QHBoxLayout(container)
                h_layout.setContentsMargins(0, 0, 0, 0)

                h_layout.addWidget(widget)  # Add the main widget (e.g., Vector3Input)

                # Create the gradient checkbox
                grad_chk = QCheckBox("Grad?")
                # Check default for the gradient param
                g_default = params[grad_name][1]
                if g_default is True: grad_chk.setChecked(True)

                h_layout.addWidget(grad_chk)

                parent_layout.addRow(name, container)

                # Register BOTH
                parent_layout.widgets[name] = (widget, intent)
                parent_layout.widgets[grad_name] = (grad_chk, 'BOOL')
            else:
                # Standard single row
                parent_layout.addRow(name, widget)
                parent_layout.widgets[name] = (widget, intent)

    def instantiate_recursive(self, layout):
        kwargs = {}
        for name, (widget, tag) in layout.widgets.items():
            if tag == 'BOOL':
                kwargs[name] = widget.isChecked()
            elif tag == 'DTYPE':
                kwargs[name] = torch.float64 if widget.currentText() == "float64" else torch.float32
            elif tag == 'VEC3':
                kwargs[name] = widget.get_value()
            elif tag == 'BOOL3':
                kwargs[name] = widget.get_value()
            elif tag == 'PRIMITIVE':
                val = widget.text()
                try:
                    val = ast.literal_eval(val)
                except:
                    pass
                kwargs[name] = val

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