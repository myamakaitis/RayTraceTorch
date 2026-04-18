"""
RayTraceTorch/project.py
------------------------
Project file I/O for .rtt scene files.

Low-level (configs ↔ JSON):
    save_project(path, element_configs, bundle_configs, settings)
    load_project(path) -> dict

High-level (file ↔ live Scene):
    load_scene(path, device=..., dtype=..., sample_rays=True) -> Scene

Typical programmatic workflow — set up a scene in the GUI, save as .rtt,
then sweep parameters in code::

    import torch
    from RayTraceTorch import load_scene

    for r in [10.0, 12.0, 15.0]:
        scene = load_scene("singlet.rtt")
        lens = scene.find_element("front_lens")
        lens.shape.radius.data.fill_(r)
        scene.simulate()
        # ... save / post-process scene.rays or sensor hits ...
"""

import json
import os
from typing import Optional, Union

import torch
import torch.nn as nn

from .scene import Scene
from .config import instantiate_from_config


PROJECT_VERSION = "1.0"


# ============================================================
# Low-level config ↔ JSON
# ============================================================

def save_project(path: str,
                 element_configs: list,
                 bundle_configs: list,
                 settings: Optional[dict] = None) -> None:
    """
    Serialise scene configs to a .rtt JSON file.

    element_configs : list of {'config': {'name', 'class', 'params'}} dicts
    bundle_configs  : list of {'N_rays': int, 'config': {...}} dicts
    settings        : {'device': str, 'Nbounces': int} (optional)
    """
    data = {
        "version": PROJECT_VERSION,
        "settings": settings or {},
        "elements": element_configs,
        "bundles": bundle_configs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_project(path: str) -> dict:
    """
    Deserialise a .rtt JSON file.
    Returns {'version', 'settings', 'elements', 'bundles'}.
    Raises ValueError if the file is missing the version key.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "version" not in data:
        raise ValueError(f"Not a valid .rtt project file: {os.path.basename(path)}")
    return migrate_project(data)


def migrate_project(data: dict) -> dict:
    """
    Upgrade older project schemas to the current format.
    Add elif branches here when the schema version changes.
    """
    version = data.get("version", "0.0")

    if version == "1.0":
        return data

    raise ValueError(f"Unsupported project version: {version}")


# ============================================================
# High-level: file → live Scene
# ============================================================

def load_scene(path: str,
               *,
               device: Optional[Union[str, torch.device]] = None,
               dtype: torch.dtype = torch.float32,
               sample_rays: bool = True,
               nbounces: Optional[int] = None) -> Scene:
    """
    Load a .rtt project and return a ready-to-simulate :class:`Scene`.

    All elements and bundles are instantiated, placed on ``device``, and
    registered on a fresh ``Scene``. The scene's ``Nbounces`` is set from
    the file (overridable via ``nbounces``), and initial rays are sampled
    so the returned scene can be fed straight into ``.simulate()``.

    Parameters
    ----------
    path : str
        Path to the .rtt file.
    device : str | torch.device, optional
        Target device. Defaults to CUDA if available, else CPU. The device
        string stored inside the file is ignored so projects are portable.
    dtype : torch.dtype, default float32
        Tensor dtype used wherever constructors accept a ``dtype=`` arg.
    sample_rays : bool, default True
        If True, call ``scene._build_rays()`` so ``scene.rays`` is populated.
        Disable if you plan to swap bundles or call simulate manually.
    nbounces : int, optional
        Override the ``Nbounces`` setting stored in the file.

    The returned scene carries two extra attributes for introspection:
        ``scene.element_names`` — names assigned in the GUI, parallel to
        ``scene.elements``.
        ``scene.bundle_names`` — names parallel to ``scene.bundles``.

    And two lookup helpers:
        ``scene.find_element(name)`` — return the Element with matching name.
        ``scene.find_bundle(name)`` — return the Bundle with matching name.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    data = load_project(path)
    settings = data.get('settings', {})

    scene = Scene()

    element_names: list[str] = []
    for item in data.get('elements', []):
        cfg = item.get('config', item)
        element = instantiate_from_config(cfg, device=device, dtype=dtype)
        if isinstance(element, nn.Module):
            element.to(device)
        scene.add_element(element)
        element_names.append(cfg.get('name', ''))

    bundle_names: list[str] = []
    for item in data.get('bundles', []):
        cfg = item.get('config', item)
        n_rays = int(item.get('N_rays', 200))
        bundle = instantiate_from_config(cfg, device=device, dtype=dtype)
        if isinstance(bundle, nn.Module):
            bundle.to(device)
        scene.add_bundle(bundle, n_rays)
        bundle_names.append(cfg.get('name', ''))

    scene.Nbounces = int(nbounces if nbounces is not None
                         else settings.get('Nbounces', 100))
    scene.to(device)
    scene._build_index_maps()

    if sample_rays:
        scene._build_rays()
        if scene.rays is not None and hasattr(scene.rays, 'to'):
            scene.rays = scene.rays.to(device)

    # Attach introspection metadata and lookup helpers.
    scene.element_names = element_names
    scene.bundle_names = bundle_names
    scene.find_element = lambda name, _s=scene: _find_by_name(_s.elements, _s.element_names, name, "element")
    scene.find_bundle  = lambda name, _s=scene: _find_by_name(_s.bundles,  _s.bundle_names,  name, "bundle")

    return scene


def _find_by_name(module_list, names: list, target: str, kind: str):
    for i, n in enumerate(names):
        if n == target:
            return module_list[i]
    raise KeyError(f"No {kind} named {target!r}. Available: {names}")
