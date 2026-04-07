"""
gui/forms.py
------------
Dear PyGui form builder for optical element / bundle configuration.

Replaces the PyQt6 elementWindow.py.  All widget tags are namespaced under
a caller-supplied prefix so multiple forms can coexist in the same viewport.
"""

import ast
import inspect
import typing
from typing import Union, List, Tuple, Optional

import torch
import dearpygui.dearpygui as dpg

# Wildcard imports make the recursive class-finder work for all subclasses.
from ..elements import *        # noqa: F401,F403
from ..geom import *            # noqa: F401,F403
from ..rays.bundle import Bundle


# ============================================================
# Type utilities  (logic ported verbatim from elementWindow.py)
# ============================================================

def analyze_type(p_type) -> str:
    if p_type == torch.dtype:
        return 'DTYPE'

    origins: set = set()
    args: set = set()

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


def get_subclasses(cls) -> set:
    if cls is None:
        return set()
    subclasses: set = set()
    queue = [cls]
    while queue:
        parent = queue.pop(0)
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                queue.append(child)
    return subclasses


def get_constructor_params(cls) -> dict:
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return {}
    params = {}
    for name, p in sig.parameters.items():
        if name == 'self':
            continue
        p_type = p.annotation if p.annotation != inspect.Parameter.empty else str
        default = p.default if p.default != inspect.Parameter.empty else None
        params[name] = (p_type, default)
    return params


def get_actual_class(p_type):
    if inspect.isclass(p_type):
        return p_type
    origin = typing.get_origin(p_type)
    if origin is typing.Union:
        for arg in typing.get_args(p_type):
            if inspect.isclass(arg) and arg is not type(None):
                return arg
    return None


# ============================================================
# Pure-Python instantiation (no dpg dependency)
# ============================================================

def _find_class_by_name(name: str):
    """Search all subclasses of known base types for a matching name."""
    import sys
    for base in _KNOWN_BASES:
        for sub in get_subclasses(base):
            if sub.__name__ == name:
                return sub
    # Fallback: scan loaded modules
    for mod in sys.modules.values():
        obj = getattr(mod, name, None)
        if obj and inspect.isclass(obj):
            return obj
    return None


try:
    from ..elements.parent import Element
    from ..geom.shape import Shape
    from ..geom.primitives import Surface
    from ..geom.transform import RayTransform, RayTransformBundle
    _KNOWN_BASES = [Element, Bundle, Shape, Surface, RayTransform, RayTransformBundle]
except Exception:
    _KNOWN_BASES = [Bundle]


def instantiate_from_config(config: dict):
    """
    Build a Python object from a {'name', 'class', 'params'} config dict.
    Does not require any dpg widgets — safe to call outside the dpg context.
    """
    cls_name = config.get('class', '')
    params = config.get('params', {})
    cls = _find_class_by_name(cls_name)
    if cls is None:
        raise ValueError(f"Cannot find class '{cls_name}'. Check imports.")
    return _instantiate_recursive(cls, params)


def _instantiate_recursive(cls, params: dict):
    kwargs = {}
    ctor_params = get_constructor_params(cls)
    for name, (p_type, default) in ctor_params.items():
        if name not in params:
            continue
        val = params[name]
        intent = analyze_type(p_type)

        if intent == 'BOOL':
            kwargs[name] = bool(val)
        elif intent == 'DTYPE':
            kwargs[name] = torch.float64 if val == 'float64' else torch.float32
        elif intent in ('VEC3', 'BOOL3'):
            kwargs[name] = list(val) if val is not None else None
        elif intent == 'PRIMITIVE':
            if isinstance(val, str):
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    pass
            kwargs[name] = val
        elif intent == 'CLASS':
            target_cls = get_actual_class(p_type)
            if val is None:
                kwargs[name] = None
            elif isinstance(val, dict) and target_cls:
                kwargs[name] = _instantiate_recursive(target_cls, val)
        elif intent == 'POLY_CLASS':
            if val is None:
                kwargs[name] = None
            elif isinstance(val, dict):
                sub_name = val.get('class', '')
                sub_cls = _find_class_by_name(sub_name)
                if sub_cls:
                    kwargs[name] = _instantiate_recursive(sub_cls, val.get('params', {}))
                else:
                    kwargs[name] = None

    return cls(**kwargs)


# ============================================================
# FormBuilder
# ============================================================

class FormBuilder:
    """
    Builds a live Dear PyGui form for any class, driven by its __init__
    type hints.  Call build(parent_tag) once, then use get_values() /
    set_values() / instantiate() at any time while the widgets are alive.
    """

    def __init__(self, base_cls, tag_prefix: str):
        self._base_cls = base_cls
        self._pfx = tag_prefix
        self._known_classes: dict = {}   # cls_name -> cls
        self._registry: dict = {}        # param_name -> record
        self._class_sel_tag = f"{tag_prefix}__cls"
        self._name_tag = f"{tag_prefix}__name"
        self._form_tag = f"{tag_prefix}__form"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, parent_tag: str):
        subs = get_subclasses(self._base_cls)
        self._known_classes = {c.__name__: c for c in subs}
        sorted_names = sorted(self._known_classes.keys())

        dpg.add_input_text(label="Name", tag=self._name_tag,
                           parent=parent_tag, width=-1)
        dpg.add_combo(sorted_names, label="Type",
                      tag=self._class_sel_tag, parent=parent_tag, width=-1,
                      callback=lambda s, a: self._rebuild_form(a))
        with dpg.child_window(tag=self._form_tag, parent=parent_tag,
                               auto_resize_y=True, border=False):
            pass

        if sorted_names:
            dpg.set_value(self._class_sel_tag, sorted_names[0])
            self._rebuild_form(sorted_names[0])

    def get_values(self) -> dict:
        return {
            'name': dpg.get_value(self._name_tag) or '',
            'class': dpg.get_value(self._class_sel_tag) or '',
            'params': self._read_registry(self._registry),
        }

    def set_values(self, config: dict):
        if config.get('name'):
            dpg.set_value(self._name_tag, config['name'])
        cls_name = config.get('class', '')
        if cls_name and cls_name in self._known_classes:
            dpg.set_value(self._class_sel_tag, cls_name)
            self._rebuild_form(cls_name)
        self._write_registry(self._registry, config.get('params', {}))

    def instantiate(self):
        """Read current widget state and construct the Python object."""
        cls_name = dpg.get_value(self._class_sel_tag)
        if not cls_name or cls_name not in self._known_classes:
            raise ValueError(f"No valid class selected: {cls_name!r}")
        return self._instantiate_registry(self._known_classes[cls_name], self._registry)

    # ------------------------------------------------------------------
    # Internal: form construction
    # ------------------------------------------------------------------

    def _rebuild_form(self, class_name: str):
        dpg.delete_item(self._form_tag, children_only=True)
        self._registry = {}
        if class_name in self._known_classes:
            self._build_params(self._known_classes[class_name],
                               self._form_tag, self._pfx, self._registry, depth=0)

    # Params that are managed globally (device/dtype) and hidden from the form
    _HIDDEN_PARAMS = frozenset({'device', 'dtype'})

    def _build_params(self, cls, parent: str, pfx: str, registry: dict, depth: int):
        if depth > 5:
            dpg.add_text("(max depth reached)", parent=parent)
            return

        params = get_constructor_params(cls)
        consumed: set = set()

        for name, (p_type, default) in params.items():
            if name in consumed or name in self._HIDDEN_PARAMS:
                continue

            intent = analyze_type(p_type)
            tag = f"{pfx}__{name}"
            grad_partner = self._find_grad_partner(name, params)
            has_grad = bool(grad_partner and grad_partner not in consumed)
            g_tag = f"{pfx}__{grad_partner}" if has_grad else None

            if intent == 'BOOL':
                dpg.add_checkbox(label=name, tag=tag, parent=parent,
                                 default_value=bool(default) if default is not None else False)
                registry[name] = {'tag': tag, 'intent': 'BOOL'}
                consumed.add(name)

            elif intent == 'DTYPE':
                dpg.add_text(name, parent=parent)
                dpg.add_combo(["float32", "float64"], label=f"##{tag}", tag=tag,
                              parent=parent, width=-1,
                              default_value="float64" if default == torch.float64 else "float32")
                registry[name] = {'tag': tag, 'intent': 'DTYPE'}
                consumed.add(name)

            elif intent == 'VEC3':
                dv = self._unpack_vec3(default)
                tags = [f"{tag}__0", f"{tag}__1", f"{tag}__2"]
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_text(f"{name}:")
                    for i, ax in enumerate('XYZ'):
                        dpg.add_input_float(label=ax, tag=tags[i],
                                            default_value=float(dv[i]), width=72)
                    if has_grad:
                        dpg.add_checkbox(label="grad", tag=g_tag,
                                         default_value=self._is_grad_on(params[grad_partner][1]))
                        registry[grad_partner] = {'tag': g_tag, 'intent': 'BOOL'}
                        consumed.add(grad_partner)
                registry[name] = {'tags': tags, 'intent': 'VEC3'}
                consumed.add(name)

            elif intent == 'BOOL3':
                dv = self._unpack_bool3(default)
                tags = [f"{tag}__0", f"{tag}__1", f"{tag}__2"]
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_text(f"{name}:")
                    for i, ax in enumerate('XYZ'):
                        dpg.add_checkbox(label=ax, tag=tags[i], default_value=bool(dv[i]))
                    if has_grad:
                        dpg.add_checkbox(label="grad", tag=g_tag,
                                         default_value=self._is_grad_on(params[grad_partner][1]))
                        registry[grad_partner] = {'tag': g_tag, 'intent': 'BOOL'}
                        consumed.add(grad_partner)
                registry[name] = {'tags': tags, 'intent': 'BOOL3'}
                consumed.add(name)

            elif intent == 'CLASS':
                target_cls = get_actual_class(p_type)
                if target_cls:
                    sub_subs = get_subclasses(target_cls)
                    if sub_subs or target_cls.__name__ in ('Shape', 'Surface'):
                        self._build_poly(name, target_cls, sub_subs, default,
                                         parent, pfx, registry, depth)
                    else:
                        sub_reg: dict = {}
                        with dpg.tree_node(label=name, parent=parent, default_open=True):
                            self._build_params(target_cls, dpg.last_item(),
                                               f"{pfx}__{name}", sub_reg, depth + 1)
                        registry[name] = {'intent': 'CLASS', 'cls': target_cls, 'sub_reg': sub_reg}
                    consumed.add(name)

            else:  # PRIMITIVE
                val_str = "" if default is None else str(default)
                if has_grad:
                    dpg.add_text(name, parent=parent)
                    with dpg.group(horizontal=True, parent=parent):
                        dpg.add_input_text(label=f"##{tag}", tag=tag,
                                           default_value=val_str, width=120)
                        dpg.add_checkbox(label="grad", tag=g_tag,
                                         default_value=self._is_grad_on(params[grad_partner][1]))
                    registry[grad_partner] = {'tag': g_tag, 'intent': 'BOOL'}
                    consumed.add(grad_partner)
                else:
                    dpg.add_text(name, parent=parent)
                    dpg.add_input_text(label=f"##{tag}", tag=tag, parent=parent,
                                       default_value=val_str, width=-1)
                registry[name] = {'tag': tag, 'intent': 'PRIMITIVE'}
                consumed.add(name)

    def _build_poly(self, name, target_cls, sub_subs, default,
                    parent: str, pfx: str, registry: dict, depth: int):
        """Polymorphic class selector: combo box + rebuilding child window."""
        is_optional = (default is None)
        poly_subs = {c.__name__: c for c in sub_subs}
        # Also include the base class itself if it's concrete (not abstract)
        if target_cls.__name__ not in poly_subs and not inspect.isabstract(target_cls):
            try:
                inspect.signature(target_cls.__init__)
                poly_subs[target_cls.__name__] = target_cls
            except Exception:
                pass
        sel_tag = f"{pfx}__{name}__sel"
        win_tag = f"{pfx}__{name}__win"
        grp_tag = f"{pfx}__{name}__grp"
        enable_tag = f"{pfx}__{name}__en" if is_optional else None

        poly_entry = {
            'intent': 'POLY_CLASS',
            'sel_tag': sel_tag,
            'win_tag': win_tag,
            'grp_tag': grp_tag,
            'poly_subs': poly_subs,
            'active_reg': {},
            'pfx': f"{pfx}__{name}",
            'is_optional': is_optional,
            'enable_tag': enable_tag,
            'depth': depth,
        }
        registry[name] = poly_entry

        if is_optional:
            dpg.add_checkbox(
                label=f"Enable {name}", tag=enable_tag, parent=parent,
                default_value=False,
                callback=lambda s, a, u: dpg.configure_item(u, show=a),
                user_data=grp_tag,
            )

        with dpg.group(tag=grp_tag, parent=parent, show=not is_optional):
            dpg.add_combo(sorted(poly_subs.keys()), label=name, tag=sel_tag, width=-1,
                          callback=lambda s, a, u: self._rebuild_poly(u, a),
                          user_data=poly_entry)
            with dpg.child_window(tag=win_tag, auto_resize_y=True, border=True):
                pass

        if poly_subs:
            first = sorted(poly_subs.keys())[0]
            dpg.set_value(sel_tag, first)
            self._rebuild_poly(poly_entry, first)

    def _rebuild_poly(self, poly_entry: dict, class_name: str):
        win_tag = poly_entry['win_tag']
        dpg.delete_item(win_tag, children_only=True)
        poly_entry['active_reg'] = {}
        if class_name in poly_entry['poly_subs']:
            cls = poly_entry['poly_subs'][class_name]
            self._build_params(cls, win_tag, poly_entry['pfx'],
                               poly_entry['active_reg'], poly_entry['depth'] + 1)

    # ------------------------------------------------------------------
    # Internal: read / write / instantiate
    # ------------------------------------------------------------------

    def _read_registry(self, registry: dict) -> dict:
        data = {}
        for name, rec in registry.items():
            intent = rec['intent']
            if intent == 'BOOL':
                data[name] = dpg.get_value(rec['tag'])
            elif intent == 'DTYPE':
                data[name] = dpg.get_value(rec['tag'])
            elif intent in ('VEC3', 'BOOL3'):
                data[name] = [dpg.get_value(t) for t in rec['tags']]
            elif intent == 'PRIMITIVE':
                raw = dpg.get_value(rec['tag'])
                try:
                    raw = ast.literal_eval(raw)
                except Exception:
                    pass
                data[name] = raw
            elif intent == 'CLASS':
                data[name] = self._read_registry(rec['sub_reg'])
            elif intent == 'POLY_CLASS':
                if rec['is_optional'] and not dpg.get_value(rec['enable_tag']):
                    data[name] = None
                else:
                    cls_name = dpg.get_value(rec['sel_tag'])
                    data[name] = {
                        'class': cls_name,
                        'params': self._read_registry(rec['active_reg']),
                    }
        return data

    def _write_registry(self, registry: dict, params: dict):
        for name, rec in registry.items():
            if name not in params:
                continue
            val = params[name]
            intent = rec['intent']
            if intent == 'BOOL':
                dpg.set_value(rec['tag'], bool(val))
            elif intent == 'DTYPE':
                dpg.set_value(rec['tag'], val if isinstance(val, str) else 'float32')
            elif intent in ('VEC3', 'BOOL3'):
                vals = list(val) if val is not None else [0, 0, 0]
                for t, v in zip(rec['tags'], vals):
                    dpg.set_value(t, v)
            elif intent == 'PRIMITIVE':
                dpg.set_value(rec['tag'], str(val))
            elif intent == 'CLASS':
                self._write_registry(rec['sub_reg'], val or {})
            elif intent == 'POLY_CLASS':
                if val is None:
                    if rec['is_optional']:
                        dpg.set_value(rec['enable_tag'], False)
                        dpg.configure_item(rec['grp_tag'], show=False)
                elif isinstance(val, dict):
                    if rec['is_optional']:
                        dpg.set_value(rec['enable_tag'], True)
                        dpg.configure_item(rec['grp_tag'], show=True)
                    cls_name = val.get('class', '')
                    if cls_name in rec['poly_subs']:
                        dpg.set_value(rec['sel_tag'], cls_name)
                        self._rebuild_poly(rec, cls_name)
                        self._write_registry(rec['active_reg'], val.get('params', {}))

    def _instantiate_registry(self, cls, registry: dict):
        kwargs = {}
        for name, rec in registry.items():
            intent = rec['intent']
            if intent == 'BOOL':
                kwargs[name] = dpg.get_value(rec['tag'])
            elif intent == 'DTYPE':
                kwargs[name] = (torch.float64 if dpg.get_value(rec['tag']) == 'float64'
                                else torch.float32)
            elif intent == 'VEC3':
                kwargs[name] = [dpg.get_value(t) for t in rec['tags']]
            elif intent == 'BOOL3':
                kwargs[name] = [dpg.get_value(t) for t in rec['tags']]
            elif intent == 'PRIMITIVE':
                raw = dpg.get_value(rec['tag'])
                try:
                    raw = ast.literal_eval(raw)
                except Exception:
                    pass
                kwargs[name] = raw
            elif intent == 'CLASS':
                kwargs[name] = self._instantiate_registry(rec['cls'], rec['sub_reg'])
            elif intent == 'POLY_CLASS':
                if rec['is_optional'] and not dpg.get_value(rec['enable_tag']):
                    kwargs[name] = None
                else:
                    cls_name = dpg.get_value(rec['sel_tag'])
                    sub_cls = rec['poly_subs'].get(cls_name)
                    kwargs[name] = (self._instantiate_registry(sub_cls, rec['active_reg'])
                                    if sub_cls else None)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_grad_partner(name: str, params: dict):
        candidate = f"{name}_grad"
        if candidate in params:
            return candidate
        abbrevs = {'translation': 'trans_grad', 'rotation': 'rot_grad', 'ior_glass': 'ior_grad'}
        mapped = abbrevs.get(name)
        if mapped and mapped in params:
            return mapped
        return None

    @staticmethod
    def _unpack_vec3(default) -> list:
        if isinstance(default, (list, tuple)):
            return list(default)
        if isinstance(default, torch.Tensor):
            return default.tolist()
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _unpack_bool3(default) -> list:
        if isinstance(default, (list, tuple)):
            return list(default)
        return [False, False, False]

    @staticmethod
    def _is_grad_on(g_default) -> bool:
        return (g_default is True or
                (isinstance(g_default, (list, tuple)) and any(g_default)))


# ============================================================
# ItemManager
# ============================================================

class ItemManager:
    """
    Manages a list of named items (elements or bundles).
    Provides a listbox + Add/Edit/Delete/Update Scene buttons,
    and a modal popup containing a FormBuilder.
    """

    _counter = 0  # class-level counter for unique tag namespacing

    def __init__(self, title: str, base_cls, on_update, on_data_changed=None):
        ItemManager._counter += 1
        self._pfx = f"imgr{ItemManager._counter}"
        self._title = title
        self._base_cls = base_cls
        self._on_update = on_update
        self._on_data_changed = on_data_changed

        self.configs: list = []          # [{'config': {...}}, ...]
        self._editing_idx: int = -1
        self._active_form: FormBuilder | None = None

        self._listbox_tag = f"{self._pfx}__list"
        self._popup_tag = f"{self._pfx}__popup"
        self._form_area_tag = f"{self._pfx}__formarea"

    # ------------------------------------------------------------------
    # Build the persistent UI
    # ------------------------------------------------------------------

    def setup_popup(self):
        """
        Create the modal popup window at the top level of the DPG tree.
        Must be called BEFORE _build_ui() and outside any container context,
        otherwise DPG parents the window to the active container and segfaults.
        """
        with dpg.window(label=f"Configure {self._title}", modal=True, show=False,
                        tag=self._popup_tag, width=640, height=710, no_resize=True):
            with dpg.child_window(tag=self._form_area_tag, height=618, border=False):
                pass
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="OK",     width=90, callback=self._on_ok)
                dpg.add_button(label="Cancel", width=90,
                               callback=lambda: dpg.hide_item(self._popup_tag))

    def build(self, parent_tag: str):
        dpg.add_listbox([], label="", tag=self._listbox_tag,
                        parent=parent_tag, num_items=8, width=-1)
        with dpg.group(horizontal=True, parent=parent_tag):
            dpg.add_button(label="Add",    width=65, callback=self._open_add)
            dpg.add_button(label="Edit",   width=65, callback=self._open_edit)
            dpg.add_button(label="Delete", width=65, callback=self._do_delete)
        dpg.add_button(label="UPDATE SCENE", parent=parent_tag, width=-1,
                       callback=self._do_build)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _open_add(self):
        try:
            self._editing_idx = -1
            self._rebuild_popup_form()
            dpg.show_item(self._popup_tag)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ItemManager] Error opening Add dialog: {e}")

    def _open_edit(self):
        try:
            sel = dpg.get_value(self._listbox_tag)
            labels = self._get_labels()
            if sel not in labels:
                return
            idx = labels.index(sel)
            self._editing_idx = idx
            self._rebuild_popup_form()
            self._active_form.set_values(self.configs[idx]['config'])
            dpg.show_item(self._popup_tag)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ItemManager] Error opening Edit dialog: {e}")

    def _do_delete(self):
        sel = dpg.get_value(self._listbox_tag)
        labels = self._get_labels()
        if sel not in labels:
            return
        self.configs.pop(labels.index(sel))
        self._refresh_list()
        if self._on_data_changed:
            self._on_data_changed()

    def _do_build(self):
        objects = []
        errors = []
        for item in self.configs:
            try:
                obj = instantiate_from_config(item['config'])
                objects.append(obj)
            except Exception as e:
                errors.append(str(e))
                import traceback
                traceback.print_exc()
        if errors:
            print(f"[ItemManager] Build errors: {errors}")
        self._on_update(objects)

    def _on_ok(self):
        config = self._active_form.get_values()
        if not config.get('name'):
            config['name'] = f"{config.get('class', 'Item')}_{len(self.configs)}"
        if self._editing_idx >= 0:
            self.configs[self._editing_idx]['config'] = config
        else:
            self.configs.append({'config': config})
        self._refresh_list()
        dpg.hide_item(self._popup_tag)
        if self._on_data_changed:
            self._on_data_changed()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_popup_form(self):
        dpg.delete_item(self._form_area_tag, children_only=True)
        self._active_form = FormBuilder(self._base_cls, f"{self._pfx}__form")
        self._active_form.build(self._form_area_tag)

    def _refresh_list(self):
        dpg.configure_item(self._listbox_tag, items=self._get_labels())

    def _get_labels(self) -> list:
        return [
            f"{c['config'].get('name', f'Item_{i}')} [{c['config'].get('class', '?')}]"
            for i, c in enumerate(self.configs)
        ]


# ============================================================
# BundleItemManager
# ============================================================

class BundleItemManager(ItemManager):
    """
    Extends ItemManager with an N_rays field per bundle.
    Stores {'N_rays': int, 'config': {...}} entries.
    On build, instantiates Bundle subclasses and calls .sample(N_rays).
    Also maintains self.bundle_instances for the optimizer.
    """

    def __init__(self, on_update, on_data_changed=None):
        super().__init__("Ray Bundle", Bundle, on_update, on_data_changed)
        self.configs: list = []           # [{'N_rays': int, 'config': {...}}, ...]
        self.bundle_instances: list = []  # [(Bundle, N_rays), ...] — set by _do_build
        self._nrays_tag = f"{self._pfx}__nrays"

    def build(self, parent_tag: str):
        # Override to inject N_rays popup field — handled in _rebuild_popup_form
        super().build(parent_tag)

    def _rebuild_popup_form(self):
        dpg.delete_item(self._form_area_tag, children_only=True)
        # N_rays spinner at the top of the popup
        dpg.add_input_int(label="N rays", tag=self._nrays_tag,
                          default_value=200, min_value=1, max_value=100_000,
                          width=120, parent=self._form_area_tag)
        dpg.add_separator(parent=self._form_area_tag)
        self._active_form = FormBuilder(Bundle, f"{self._pfx}__form")
        self._active_form.build(self._form_area_tag)

    def _open_edit(self):
        sel = dpg.get_value(self._listbox_tag)
        labels = self._get_labels()
        if sel not in labels:
            return
        idx = labels.index(sel)
        self._editing_idx = idx
        self._rebuild_popup_form()
        item = self.configs[idx]
        dpg.set_value(self._nrays_tag, item.get('N_rays', 200))
        self._active_form.set_values(item['config'])
        dpg.show_item(self._popup_tag)

    def _on_ok(self):
        config = self._active_form.get_values()
        if not config.get('name'):
            config['name'] = f"{config.get('class', 'Bundle')}_{len(self.configs)}"
        n_rays = dpg.get_value(self._nrays_tag)
        entry = {'N_rays': n_rays, 'config': config}
        if self._editing_idx >= 0:
            self.configs[self._editing_idx] = entry
        else:
            self.configs.append(entry)
        self._refresh_list()
        dpg.hide_item(self._popup_tag)
        if self._on_data_changed:
            self._on_data_changed()

    def _do_build(self):
        rays_list = []
        self.bundle_instances = []
        errors = []
        for item in self.configs:
            try:
                bundle = instantiate_from_config(item['config'])
                n = item.get('N_rays', 200)
                self.bundle_instances.append((bundle, n))
                rays_list.append(bundle.sample(n))
            except Exception as e:
                errors.append(str(e))
                import traceback
                traceback.print_exc()
        if errors:
            print(f"[BundleItemManager] Build errors: {errors}")
        self._on_update(rays_list)

    def _get_labels(self) -> list:
        return [
            f"{c['config'].get('name', f'Bundle_{i}')} [{c['config'].get('class', '?')}]  N={c.get('N_rays', '?')}"
            for i, c in enumerate(self.configs)
        ]
