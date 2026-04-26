"""
gui/forms.py
------------
Dear PyGui form builder for optical element / bundle configuration.

All widget tags are namespaced under a caller-supplied prefix so multiple
forms can coexist in the same viewport.

Config ↔ object conversion and type-introspection helpers live in
``RayTraceTorch.config`` — re-exported here for backward compatibility.
"""

import ast
import inspect

import torch
import torch.nn as nn
import dearpygui.dearpygui as dpg

from ..rays.bundle import Bundle
from ..config import (
    analyze_type,
    get_subclasses,
    get_constructor_params,
    get_actual_class,
    get_all_actual_classes,
    instantiate_from_config,
)


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

        with dpg.group(horizontal=True, parent=parent_tag):
            dpg.add_text("Name:")
            dpg.add_input_text(label=f"##{self._name_tag}", tag=self._name_tag, width=-1)
        dpg.add_spacer(height=4, parent=parent_tag)
        with dpg.group(horizontal=True, parent=parent_tag):
            dpg.add_text("Type:")
            dpg.add_combo(sorted_names, label=f"##{self._class_sel_tag}",
                          tag=self._class_sel_tag, width=-1,
                          callback=lambda s, a: self._rebuild_form(a))
        dpg.add_spacer(height=6, parent=parent_tag)
        dpg.add_separator(parent=parent_tag)
        dpg.add_spacer(height=4, parent=parent_tag)
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

    # Width (px) of the left-hand label column in inline label/input rows
    _LABEL_W = 110

    def _build_params(self, cls, parent: str, pfx: str, registry: dict, depth: int):
        if depth > 5:
            dpg.add_text("(max depth reached)", parent=parent)
            return

        params = get_constructor_params(cls)
        consumed: set = set()
        first = True

        for name, (p_type, default) in params.items():
            if name in consumed or name in self._HIDDEN_PARAMS:
                continue

            intent = analyze_type(p_type)
            tag = f"{pfx}__{name}"
            grad_partner = self._find_grad_partner(name, params)
            has_grad = bool(grad_partner and grad_partner not in consumed)
            g_tag = f"{pfx}__{grad_partner}" if has_grad else None

            # Breathing room between consecutive fields
            if not first:
                dpg.add_spacer(height=5, parent=parent)
            first = False

            if intent == 'BOOL':
                # Checkbox: DPG puts label to the right — natural for booleans
                dpg.add_checkbox(label=name, tag=tag, parent=parent,
                                 default_value=bool(default) if default is not None else False)
                registry[name] = {'tag': tag, 'intent': 'BOOL'}
                consumed.add(name)

            elif intent == 'DTYPE':
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_text(f"{name}:", indent=-1)
                    dpg.add_combo(["float32", "float64"], label=f"##{tag}", tag=tag,
                                  width=-1,
                                  default_value="float64" if default == torch.float64 else "float32")
                registry[name] = {'tag': tag, 'intent': 'DTYPE'}
                consumed.add(name)

            elif intent == 'VEC3':
                dv = self._unpack_vec3(default)
                tags = [f"{tag}__0", f"{tag}__1", f"{tag}__2"]
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_text(f"{name}:", indent=-1)
                    for i, ax in enumerate('XYZ'):
                        dpg.add_input_float(label=ax, tag=tags[i],
                                            default_value=float(dv[i]), width=95)
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
                    dpg.add_text(f"{name}:", indent=-1)
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
                # Collect every class type in the annotation (handles Union[A, B, ...])
                target_classes = get_all_actual_classes(p_type)
                if target_classes:
                    # Merge subclasses from every target type into one set,
                    # and include any concrete base classes themselves.
                    combined: set = set()
                    for tc in target_classes:
                        combined |= get_subclasses(tc)
                        if not inspect.isabstract(tc):
                            combined.add(tc)

                    primary_cls = target_classes[0]   # used as nominal base in _build_poly
                    needs_poly  = (
                        len(combined) > 0
                        or any(tc.__name__ in ('Shape', 'Surface') for tc in target_classes)
                    )
                    if needs_poly:
                        self._build_poly(name, primary_cls, combined, default,
                                         parent, pfx, registry, depth)
                    else:
                        sub_reg: dict = {}
                        with dpg.tree_node(label=name, parent=parent, default_open=True):
                            self._build_params(primary_cls, dpg.last_item(),
                                               f"{pfx}__{name}", sub_reg, depth + 1)
                        registry[name] = {'intent': 'CLASS', 'cls': primary_cls, 'sub_reg': sub_reg}
                    consumed.add(name)

            else:  # PRIMITIVE
                val_str = "" if default is None else str(default)
                if has_grad:
                    with dpg.group(horizontal=True, parent=parent):
                        dpg.add_text(f"{name}:", indent=-1)
                        dpg.add_input_text(label=f"##{tag}", tag=tag,
                                           default_value=val_str, width=100)
                        dpg.add_checkbox(label="grad", tag=g_tag,
                                         default_value=self._is_grad_on(params[grad_partner][1]))
                    registry[grad_partner] = {'tag': g_tag, 'intent': 'BOOL'}
                    consumed.add(grad_partner)
                else:
                    with dpg.group(horizontal=True, parent=parent):
                        dpg.add_text(f"{name}:", indent=-1)
                        dpg.add_input_text(label=f"##{tag}", tag=tag,
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
            with dpg.group(horizontal=True):
                dpg.add_text(f"{name}:", indent=-1)
                dpg.add_combo(sorted(poly_subs.keys()), label=f"##{sel_tag}", tag=sel_tag,
                              width=-1,
                              callback=lambda s, a, u: self._rebuild_poly(u, a),
                              user_data=poly_entry)
            dpg.add_spacer(height=2)
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
    _clipboard: dict | None = None  # class-level clipboard for copy/paste

    def __init__(self, title: str, base_cls, on_update, on_data_changed=None,
                 device=None, dtype=None):
        ItemManager._counter += 1
        self._pfx = f"imgr{ItemManager._counter}"
        self._title = title
        self._base_cls = base_cls
        self._on_update = on_update
        self._on_data_changed = on_data_changed
        self._device = device   # injected into every instantiate_from_config call
        self._dtype  = dtype

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
            dpg.add_button(label="Add",    width=55, callback=self._open_add)
            dpg.add_button(label="Edit",   width=55, callback=self._open_edit)
            dpg.add_button(label="Delete", width=55, callback=self._do_delete)
            dpg.add_button(label="Copy",   width=45, callback=self._do_copy)
            dpg.add_button(label="Paste",  width=45, callback=self._do_paste)
        with dpg.group(horizontal=True, parent=parent_tag):
            dpg.add_button(label="Up", width=30, callback=self._move_up)
            dpg.add_button(label="Dn", width=30, callback=self._move_down)
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

    def _move_up(self):
        sel = dpg.get_value(self._listbox_tag)
        labels = self._get_labels()
        if sel not in labels:
            return
        idx = labels.index(sel)
        if idx <= 0:
            return
        self.configs[idx], self.configs[idx - 1] = self.configs[idx - 1], self.configs[idx]
        self._refresh_list()
        # Re-select the moved item at its new position
        new_labels = self._get_labels()
        if idx - 1 < len(new_labels):
            dpg.set_value(self._listbox_tag, new_labels[idx - 1])
        if self._on_data_changed:
            self._on_data_changed()

    def _move_down(self):
        sel = dpg.get_value(self._listbox_tag)
        labels = self._get_labels()
        if sel not in labels:
            return
        idx = labels.index(sel)
        if idx >= len(self.configs) - 1:
            return
        self.configs[idx], self.configs[idx + 1] = self.configs[idx + 1], self.configs[idx]
        self._refresh_list()
        # Re-select the moved item at its new position
        new_labels = self._get_labels()
        if idx + 1 < len(new_labels):
            dpg.set_value(self._listbox_tag, new_labels[idx + 1])
        if self._on_data_changed:
            self._on_data_changed()

    def _do_copy(self):
        import copy
        sel = dpg.get_value(self._listbox_tag)
        labels = self._get_labels()
        if sel not in labels:
            return
        idx = labels.index(sel)
        ItemManager._clipboard = copy.deepcopy(self.configs[idx])

    def _do_paste(self):
        import copy
        if ItemManager._clipboard is None:
            return
        entry = copy.deepcopy(ItemManager._clipboard)
        # Append "_copy" to the name to avoid duplicates
        cfg = entry.get('config', entry)
        name = cfg.get('name', 'Item')
        cfg['name'] = f"{name}_copy"
        self.configs.append(entry)
        self._refresh_list()
        if self._on_data_changed:
            self._on_data_changed()

    def _do_build(self):
        objects = []
        errors = []
        for item in self.configs:
            try:
                obj = instantiate_from_config(
                    item['config'], device=self._device, dtype=self._dtype)
                # Sub-objects such as RayTransform store nn.Parameters created on
                # CPU because RayTransform.__init__ has no device arg.  Move the
                # whole element to device here; workbench also calls _scene.to()
                # but this ensures the object is correct before it enters the scene.
                if self._device is not None and isinstance(obj, nn.Module):
                    obj.to(self._device)
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

    def __init__(self, on_update, on_data_changed=None, device=None, dtype=None):
        super().__init__("Ray Bundle", Bundle, on_update, on_data_changed,
                         device=device, dtype=dtype)
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
                bundle = instantiate_from_config(
                    item['config'], device=self._device, dtype=self._dtype)
                # RayTransform stores trans/rot_vec as nn.Parameters created on CPU
                # (no device arg in its __init__). Move the whole module to device
                # before sampling so all tensors are on the same device.
                if self._device is not None and isinstance(bundle, nn.Module):
                    bundle.to(self._device)
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
