"""
RayTraceTorch/config.py
-----------------------
Pure-Python config → object instantiation.

Turns a ``{'name', 'class', 'params'}`` dict (as produced by the GUI form
builder and stored inside .rtt project files) into a live Python object.

No GUI dependency — safe to import without Dear PyGui installed.
"""

import ast
import inspect
import sys
import typing
from typing import List, Tuple

import torch

# Wildcard imports so get_subclasses() discovers every concrete class.
from .elements import *        # noqa: F401,F403
from .geom import *            # noqa: F401,F403
from .rays.bundle import Bundle


# ============================================================
# Type introspection
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
    """Return the first concrete class in a type annotation (Union-aware)."""
    if inspect.isclass(p_type):
        return p_type
    origin = typing.get_origin(p_type)
    if origin is typing.Union:
        for arg in typing.get_args(p_type):
            if inspect.isclass(arg) and arg is not type(None):
                return arg
    return None


def get_all_actual_classes(p_type) -> list:
    """
    Return ALL class types present in a type annotation.

    For ``Union[Shape, Surface]`` this returns ``[Shape, Surface]`` rather
    than just the first match, which is needed when a parameter accepts
    instances of several different class hierarchies.
    """
    if inspect.isclass(p_type):
        return [p_type]
    origin = typing.get_origin(p_type)
    if origin is typing.Union:
        return [
            arg for arg in typing.get_args(p_type)
            if inspect.isclass(arg) and arg is not type(None)
        ]
    return []


# ============================================================
# Class lookup
# ============================================================

try:
    from .elements.parent import Element
    from .geom.shape import Shape
    from .geom.primitives import Surface
    from .geom.transform import RayTransform, RayTransformBundle
    from .phys.std import SurfaceFunction
    _KNOWN_BASES = [Element, Bundle, Shape, Surface, RayTransform, RayTransformBundle, SurfaceFunction]
except Exception:
    _KNOWN_BASES = [Bundle]


def _find_class_by_name(name: str):
    """Search all subclasses of known base types for a matching name."""
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


# ============================================================
# Instantiation
# ============================================================

def instantiate_from_config(config: dict,
                            device=None,
                            dtype=None):
    """
    Build a Python object from a ``{'name', 'class', 'params'}`` config dict.

    ``device`` and ``dtype`` are injected into every constructor that accepts
    them (the GUI hides these from the form via ``_HIDDEN_PARAMS``). Pass
    the desired device/dtype so all sub-objects land on the same hardware
    from the moment they are created.
    """
    cls_name = config.get('class', '')
    params = config.get('params', {})
    cls = _find_class_by_name(cls_name)
    if cls is None:
        raise ValueError(f"Cannot find class '{cls_name}'. Check imports.")
    return _instantiate_recursive(cls, params, device=device, dtype=dtype)


def _instantiate_recursive(cls, params: dict, device=None, dtype=None):
    kwargs = {}
    ctor_params = get_constructor_params(cls)

    # Inject device / dtype for any constructor that accepts them but whose
    # widgets are suppressed in the GUI. Guarantees every sub-object
    # (transforms, shapes, bundles …) is created on the correct device
    # without the caller needing to set it manually.
    if device is not None and 'device' in ctor_params and 'device' not in params:
        kwargs['device'] = device
    if dtype is not None and 'dtype' in ctor_params and 'dtype' not in params:
        kwargs['dtype'] = dtype

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
                # POLY_CLASS format: {'class': X, 'params': {...}} —
                # unwrap and use the selected subclass.
                if 'class' in val and 'params' in val:
                    sub_name = val['class']
                    sub_cls = _find_class_by_name(sub_name) or target_cls
                    kwargs[name] = _instantiate_recursive(
                        sub_cls, val['params'], device=device, dtype=dtype)
                else:
                    # Plain flat dict (no subclass selection)
                    kwargs[name] = _instantiate_recursive(
                        target_cls, val, device=device, dtype=dtype)
        elif intent == 'POLY_CLASS':
            if val is None:
                kwargs[name] = None
            elif isinstance(val, dict):
                sub_name = val.get('class', '')
                sub_cls = _find_class_by_name(sub_name)
                if sub_cls:
                    kwargs[name] = _instantiate_recursive(
                        sub_cls, val.get('params', {}), device=device, dtype=dtype)
                else:
                    kwargs[name] = None

    return cls(**kwargs)
