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
                             QLabel, QCheckBox, QGroupBox, QScrollArea, QMessageBox,
                             QListWidgetItem, QTextEdit)
from PyQt6.QtCore import Qt

# ==========================================
# IMPORTANT: RESTORED IMPORTS
# ==========================================
# These must be active for the code to find your Element subclasses

from ..elements import *
from ..geom import *


# ==========================================
# 1. TYPE DEFINITIONS & ANALYZER
# ==========================================

def analyze_type(p_type):
    """
    Robustly determines the 'Intent' of a type hint.
    """
    if p_type == torch.dtype:
        return 'DTYPE'

    origins = set()
    args = set()

    def unpack(t):
        origin = typing.get_origin(t)
        if origin is typing.Union:
            for arg in typing.get_args(t):
                unpack(arg)
        elif t is not type(None):
            origins.add(origin)
            args.add(t)

    unpack(p_type)

    if torch.Tensor in args:
        if List[bool] in args or Tuple[bool, ...] in args:
            return 'BOOL3'
        return 'VEC3'

    if bool in args and len(args) == 1:
        return 'BOOL'

    for a in args:
        if inspect.isclass(a) and a not in (int, float, str, bool, list, tuple, dict, torch.Tensor):
            return 'CLASS'

    return 'PRIMITIVE'


def get_subclasses(cls):
    """Recursively find all subclasses."""
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

        for i in range(3):
            le = QLineEdit("0.0")
            layout.addWidget(le)
            self.inputs.append(le)

        self.set_value(default)

    def set_value(self, val):
        if isinstance(val, (list, tuple)):
            vals = val
        elif isinstance(val, torch.Tensor):
            vals = val.tolist()
        elif isinstance(val, (int, float)):
            vals = [val] * 3
        else:
            vals = [0.0, 0.0, 0.0]

        for i, le in enumerate(self.inputs):
            v = vals[i] if i < len(vals) else 0.0
            le.setText(str(v))

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

        labels = ["X", "Y", "Z"]
        for i in range(3):
            chk = QCheckBox(labels[i])
            layout.addWidget(chk)
            self.checks.append(chk)

        self.set_value(default)

    def set_value(self, val):
        if isinstance(val, (list, tuple)):
            vals = val
        elif isinstance(val, torch.Tensor):
            vals = val.tolist()
        elif isinstance(val, bool):
            vals = [val] * 3
        else:
            vals = [False, False, False]

        for i, chk in enumerate(self.checks):
            checked = vals[i] if i < len(vals) else False
            chk.setChecked(bool(checked))

    def get_value(self):
        return [chk.isChecked() for chk in self.checks]


class PolymorphicElementWidget(QWidget):
    def __init__(self, name, base_roots, dialog_ref, depth, default=None):
        super().__init__()
        self.dialog = dialog_ref
        self.depth = depth
        self.base_roots = base_roots if isinstance(base_roots, list) else [base_roots]

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Master Checkbox
        self.enable_chk = None
        self.is_optional = (default is None)
        if self.is_optional:
            self.enable_chk = QCheckBox(f"Enable {name}?")
            self.enable_chk.setChecked(False)
            self.enable_chk.toggled.connect(self.toggle_content)
            self.layout.addWidget(self.enable_chk)

        # 2. Content Wrapper
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.content_widget)

        # 3. Class Selector
        self.selector = QComboBox()
        self.subclasses = {}
        for root in self.base_roots:
            for cls in get_subclasses(root):
                self.subclasses[cls.__name__] = cls

        self.selector.addItems(sorted(list(self.subclasses.keys())))
        self.selector.currentTextChanged.connect(self.refresh_subform)
        self.content_layout.addWidget(self.selector)

        # 4. Form Container
        self.form_container = QGroupBox(f"Parameters")
        self.form_layout = QFormLayout(self.form_container)
        self.form_container.setLayout(self.form_layout)
        self.content_layout.addWidget(self.form_container)

        # Initial State
        if self.subclasses:
            self.refresh_subform(self.selector.currentText())

        if self.is_optional:
            self.toggle_content(False)

    def toggle_content(self, checked):
        self.content_widget.setVisible(checked)

    def refresh_subform(self, class_name):
        if not class_name: return

        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        cls = self.subclasses[class_name]
        self.form_layout.target_class = cls
        self.form_layout.widgets = {}
        self.form_layout.sub_forms = {}
        self.dialog.build_recursive_form(self.form_layout, cls, self.depth + 1)

    def get_data(self):
        if self.is_optional and not self.enable_chk.isChecked():
            return None
        current_class = self.selector.currentText()
        if not current_class: return None
        params = self.dialog.get_form_data_recursive(self.form_layout)
        return {'class': current_class, 'params': params}

    def set_data(self, data):
        if data is None:
            if self.is_optional: self.enable_chk.setChecked(False)
            return

        if self.is_optional: self.enable_chk.setChecked(True)

        class_name = data.get('class')
        if class_name and class_name in self.subclasses:
            if self.selector.currentText() != class_name:
                self.selector.setCurrentText(class_name)
            self.refresh_subform(class_name)

            params = data.get('params', {})
            self.dialog.set_form_data_recursive(self.form_layout, params)


# ==========================================
# 3. RECURSIVE DIALOG (Logic Engine)
# ==========================================

class RecursiveElementDialog(QDialog):
    def __init__(self, parent=None, element_base_cls=None):
        super().__init__(parent)
        self.setWindowTitle("Element Configuration")
        self.resize(600, 750)

        self.element_base_cls = element_base_cls
        self.layout = QVBoxLayout(self)

        # --- 1. Element Name Input ---
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Element Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. MyLens_01")
        name_layout.addWidget(self.name_input)
        self.layout.addLayout(name_layout)

        # --- 2. Element Type Selector ---
        self.class_selector = QComboBox()

        # DEBUG: Verify we found classes
        subclasses = get_subclasses(element_base_cls)
        if not subclasses:
            print(f"DEBUG: No subclasses found for {element_base_cls}. Check imports!")
            self.class_selector.addItem("No Elements Found (Check Imports)")

        self.known_classes = {cls.__name__: cls for cls in subclasses}
        self.class_selector.addItems(sorted(list(self.known_classes.keys())))
        self.class_selector.currentTextChanged.connect(self.refresh_form)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Element Type:"))
        type_layout.addWidget(self.class_selector)
        self.layout.addLayout(type_layout)

        # --- 3. Scroll Area for Form ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.form_container = QWidget()
        self.current_layout = QFormLayout(self.form_container)
        self.scroll.setWidget(self.form_container)

        self.layout.addWidget(self.scroll)

        # --- 4. Dialog Buttons ---
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.layout.addWidget(btns)

        # Initial Build
        if self.known_classes:
            self.refresh_form(self.class_selector.currentText())

    def refresh_form(self, class_name):
        """Rebuilds the entire form container to prevent ghost widgets."""
        if not class_name or class_name not in self.known_classes:
            return

        # Destroy old container
        if self.form_container:
            self.form_container.deleteLater()

        # Create new container
        self.form_container = QWidget()
        self.current_layout = QFormLayout(self.form_container)
        self.current_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll.setWidget(self.form_container)

        try:
            cls = self.known_classes[class_name]
            self.build_recursive_form(self.current_layout, cls, depth=0)
        except Exception as e:
            self.current_layout.addRow(QLabel(f"Error building form: {e}"))
            print(f"Error: {e}")

    # --- Builder Logic ---
    def find_gradient_partner(self, name, all_params):
        candidate = f"{name}_grad"
        if candidate in all_params: return candidate
        abbrevs = {'translation': 'trans_grad', 'rotation': 'rot_grad', 'ior_glass': 'ior_grad'}
        if name in abbrevs and abbrevs[name] in all_params: return abbrevs[name]
        return None

    def build_recursive_form(self, parent_layout, cls, depth):
        if depth > 5:
            parent_layout.addRow(QLabel("Max recursion depth reached"))
            return

        params = get_constructor_params(cls)
        parent_layout.target_class = cls
        parent_layout.widgets = {}
        parent_layout.sub_forms = {}

        consumed_params = set()

        for name, (p_type, default) in params.items():
            if name in consumed_params: continue

            grad_partner = self.find_gradient_partner(name, params)
            intent = analyze_type(p_type)
            widget = None

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
                target_cls = get_actual_class(p_type)
                if target_cls:
                    if target_cls.__name__ in ['Shape', 'Surface']:
                        search_roots = [target_cls]
                        if target_cls.__name__ == 'Shape': search_roots = [Shape, Surface]
                        poly_widget = PolymorphicElementWidget(name, search_roots, self, depth, default)
                        parent_layout.addRow(name, poly_widget)
                        parent_layout.widgets[name] = (poly_widget, 'POLY_CLASS')
                        consumed_params.add(name)
                        continue

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
                    consumed_params.add(name)
                    continue

            if widget is None:
                widget = QLineEdit()
                if default is not None: widget.setText(str(default))
                intent = 'PRIMITIVE'

            if grad_partner and grad_partner not in consumed_params:
                container = QWidget()
                h = QHBoxLayout(container)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(widget)

                grad_chk = QCheckBox("Grad?")
                g_default = params[grad_partner][1]
                is_checked = (g_default is True) or (isinstance(g_default, (list, tuple)) and any(g_default))
                grad_chk.setChecked(is_checked)
                h.addWidget(grad_chk)

                parent_layout.addRow(name, container)
                parent_layout.widgets[name] = (widget, intent)
                parent_layout.widgets[grad_partner] = (grad_chk, 'BOOL')
                consumed_params.add(name)
                consumed_params.add(grad_partner)
            else:
                parent_layout.addRow(name, widget)
                parent_layout.widgets[name] = (widget, intent)
                consumed_params.add(name)

    # --- Data Extraction ---
    def get_form_data_recursive(self, layout):
        data = {}
        for name, (widget, tag) in layout.widgets.items():
            if tag == 'BOOL':
                data[name] = widget.isChecked()
            elif tag == 'DTYPE':
                data[name] = "float64" if widget.currentText() == "float64" else "float32"
            elif tag == 'VEC3':
                data[name] = widget.get_value()
            elif tag == 'BOOL3':
                data[name] = widget.get_value()
            elif tag == 'POLY_CLASS':
                data[name] = widget.get_data()
            elif tag == 'PRIMITIVE':
                val = widget.text()
                try:
                    val = ast.literal_eval(val)
                except:
                    pass
                data[name] = val

        for name, (sub_layout, group_box) in layout.sub_forms.items():
            if group_box.isCheckable() and not group_box.isChecked():
                data[name] = None
            else:
                data[name] = self.get_form_data_recursive(sub_layout)
        return data

    def get_configuration(self):
        return {
            'name': self.name_input.text(),
            'class': self.class_selector.currentText(),
            'params': self.get_form_data_recursive(self.current_layout)
        }

    # --- Data Injection ---
    def set_form_data_recursive(self, layout, params):
        if not params: return

        for name, (widget, tag) in layout.widgets.items():
            if name not in params: continue
            val = params[name]

            if tag == 'BOOL':
                widget.setChecked(val)
            elif tag == 'DTYPE':
                widget.setCurrentText(val if isinstance(val, str) else "float32")
            elif tag == 'VEC3':
                widget.set_value(val)
            elif tag == 'BOOL3':
                widget.set_value(val)
            elif tag == 'POLY_CLASS':
                widget.set_data(val)
            elif tag == 'PRIMITIVE':
                widget.setText(str(val))

        for name, (sub_layout, group_box) in layout.sub_forms.items():
            if name not in params: continue
            val = params[name]

            if val is None:
                if group_box.isCheckable(): group_box.setChecked(False)
            else:
                if group_box.isCheckable(): group_box.setChecked(True)
                self.set_form_data_recursive(sub_layout, val)

    def set_configuration(self, config):
        self.name_input.setText(config.get('name', ''))

        cls_name = config.get('class')
        if cls_name and cls_name in self.known_classes:
            self.class_selector.blockSignals(True)
            self.class_selector.setCurrentText(cls_name)
            self.class_selector.blockSignals(False)

            # Force Rebuild
            self.refresh_form(cls_name)
            self.set_form_data_recursive(self.current_layout, config.get('params', {}))

    # --- Instantiation ---
    def instantiate_current_state(self):
        return self.instantiate_recursive(self.current_layout)

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
            elif tag == 'POLY_CLASS':
                poly_data = widget.get_data()
                if poly_data is None:
                    kwargs[name] = None
                else:
                    kwargs[name] = self.instantiate_recursive(widget.form_layout)
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


# ==========================================
# 4. MAIN WINDOW
# ==========================================

class ElementManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Element Manager")
        self.resize(800, 600)

        self.element_configs = []

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left Panel
        left_layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.edit_element)
        left_layout.addWidget(QLabel("Active Elements:"))
        left_layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add New")
        self.add_btn.clicked.connect(self.add_element)
        self.edit_btn = QPushButton("Edit Selected")
        self.edit_btn.clicked.connect(self.edit_element)
        self.del_btn = QPushButton("Remove")
        self.del_btn.clicked.connect(self.remove_element)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.edit_btn)
        btn_layout.addWidget(self.del_btn)
        left_layout.addLayout(btn_layout)

        self.build_btn = QPushButton("BUILD / RUN")
        self.build_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #d0f0c0;")
        self.build_btn.clicked.connect(self.build_all)
        left_layout.addWidget(self.build_btn)

        main_layout.addLayout(left_layout, 1)

        # Right Panel
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Build Log / Output:"))
        right_layout.addWidget(self.log_area)
        main_layout.addLayout(right_layout, 2)

    def refresh_list(self):
        self.list_widget.clear()
        for idx, item in enumerate(self.element_configs):
            name = item['config'].get('name') or f"Element_{idx}"
            cls = item['config']['class']
            summary = f"{name} [{cls}]"

            list_item = QListWidgetItem(summary)
            list_item.setData(Qt.ItemDataRole.UserRole, idx)
            self.list_widget.addItem(list_item)

    def add_element(self):
        dlg = RecursiveElementDialog(self, element_base_cls=Element)
        if dlg.exec():
            config = dlg.get_configuration()
            if not config['name']:
                config['name'] = f"{config['class']}_{len(self.element_configs)}"

            self.element_configs.append({'config': config})
            self.refresh_list()
            self.log(f"Added configuration for {config['name']}")

    def edit_element(self):
        items = self.list_widget.selectedItems()
        if not items: return

        idx = items[0].data(Qt.ItemDataRole.UserRole)
        data = self.element_configs[idx]

        dlg = RecursiveElementDialog(self, element_base_cls=Element)
        dlg.set_configuration(data['config'])

        if dlg.exec():
            new_config = dlg.get_configuration()
            if not new_config['name']:
                new_config['name'] = data['config']['name']

            self.element_configs[idx]['config'] = new_config
            self.refresh_list()
            self.log(f"Updated configuration for {new_config['name']}")

    def remove_element(self):
        items = self.list_widget.selectedItems()
        if not items: return
        idx = items[0].data(Qt.ItemDataRole.UserRole)
        removed = self.element_configs.pop(idx)
        self.refresh_list()
        self.log(f"Removed {removed['config']['name']}")

    def log(self, msg):
        self.log_area.append(msg)

    def build_all(self):
        self.log("-" * 30)
        self.log("STARTING BUILD PROCESS...")
        built_objects = []

        for item in self.element_configs:
            try:
                dummy_dlg = RecursiveElementDialog(self, element_base_cls=Element)
                dummy_dlg.set_configuration(item['config'])
                obj = dummy_dlg.instantiate_current_state()
                built_objects.append(obj)
                name = item['config'].get('name', 'Unknown')
                self.log(f"SUCCESS: Built {name} -> {obj}")
            except Exception as e:
                self.log(f"ERROR building {item['config'].get('name')}: {e}")
                import traceback
                traceback.print_exc()

        self.log(f"Build Complete. {len(built_objects)} objects created.")
        self.log("-" * 30)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ElementManagerWindow()
    window.show()
    sys.exit(app.exec())