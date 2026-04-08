"""
gui/workbench.py
----------------
Main Dear PyGui application for the Optical Design Workbench.

Entry point::

    from RayTraceTorch.gui.workbench import run
    run()

or directly::

    python -m RayTraceTorch.gui.workbench
"""

import sys
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import dearpygui.dearpygui as dpg

from ..scene import Scene
from ..render.camera import Renderer, OrbitCamera
from ..rays.ray import Rays, Paths
from ..elements.parent import Element

from .forms import ItemManager, BundleItemManager, instantiate_from_config
from .viewport import RenderViewport, ProfilePlot
from .project import save_project, load_project


# ============================================================
# Module-level state  (replaces class instance variables)
# ============================================================

_scene: Optional[Scene] = None
_device: Optional[torch.device] = None
_cam: Optional[OrbitCamera] = None
_renderer: Optional[Renderer] = None
_viewport: Optional[RenderViewport] = None
_profile_xz: Optional[ProfilePlot] = None
_profile_yz: Optional[ProfilePlot] = None
_element_manager: Optional[ItemManager] = None
_bundle_manager: Optional[BundleItemManager] = None

_last_sim_rays = None   # Rays after the most recent simulation

_is_dirty: bool = False
_current_path: Optional[str] = None

# Tracks which plot tags currently have equal_aspects=True so the toggle
# button can flip the state without querying DPG (no getter for this flag).
# spot_plot is built with equal_aspects=True; cross-section plots start False.
_equal_aspects_plots: set = {"spot_plot"}

_SPOT_THEME_TAG = "spot_scatter_theme"   # DPG theme item for scatter markers


# ============================================================
# Scene / Ray callbacks  (called by manager UPDATE SCENE)
# ============================================================

def _update_scene_elements(new_elements: list):
    global _scene
    _scene.clear_elements()
    _scene.elements.extend(nn.ModuleList(new_elements))
    _scene.to(_device)
    _scene._build_index_maps()
    _refresh_all_views()


def _update_scene_rays(_unused_rays_list: list):
    """
    Called by BundleItemManager after UPDATE SCENE.
    Re-registers all bundles on the scene and rebuilds self.rays via
    Scene._build_rays() so the scene owns the canonical bundle list.
    """
    global _scene
    _scene.clear_bundles()
    for bundle, n in _bundle_manager.bundle_instances:
        _scene.add_bundle(bundle, n)
    _scene.to(_device)          # move bundle nn.Parameters (transform weights) to device
    _scene._build_rays()
    if _scene.rays is not None and hasattr(_scene.rays, 'to'):
        _scene.rays = _scene.rays.to(_device)
    _refresh_spot_id_combo()    # keep ray-ID filter in sync with registered bundles
    _refresh_all_views()


# ============================================================
# View refresh
# ============================================================

def _refresh_all_views():
    _viewport.refresh()
    _profile_xz.update()
    _profile_yz.update()


# ============================================================
# Dirty tracking
# ============================================================

def _mark_dirty():
    global _is_dirty
    _is_dirty = True
    _update_viewport_title()


def _update_viewport_title():
    name = os.path.basename(_current_path) if _current_path else "Untitled"
    suffix = " *" if _is_dirty else ""
    dpg.set_viewport_title(f"Optical Design Workbench — {name}{suffix}")


# ============================================================
# File operations
# ============================================================

def _action_new():
    global _is_dirty, _current_path
    if _is_dirty:
        if not _confirm_discard():
            return
    _element_manager.configs.clear()
    _element_manager._refresh_list()
    _bundle_manager.configs.clear()
    _bundle_manager.bundle_instances.clear()
    _bundle_manager._refresh_list()
    _scene.clear_elements()
    _scene.clear_bundles()
    _scene._build_index_maps()
    _current_path = None
    _is_dirty = False
    _update_viewport_title()
    _refresh_all_views()


def _setup_confirm_discard_popup():
    """Create the confirm-discard modal at the top level before _build_ui()."""
    with dpg.window(label="Unsaved Changes", modal=True, show=False,
                    tag="confirm_discard_popup", width=320, height=110,
                    no_resize=True):
        dpg.add_text("Discard unsaved changes?")
        dpg.add_spacer(height=8)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Discard", width=90,
                           callback=lambda: dpg.set_value("_discard_confirmed", True) or
                                            dpg.hide_item("confirm_discard_popup"))
            dpg.add_button(label="Cancel",  width=90,
                           callback=lambda: dpg.hide_item("confirm_discard_popup"))
    # Value registry for the confirm result
    with dpg.value_registry():
        dpg.add_bool_value(tag="_discard_confirmed", default_value=False)


def _confirm_discard() -> bool:
    """Show the pre-built confirm-discard modal and pump frames until dismissed."""
    dpg.set_value("_discard_confirmed", False)
    dpg.show_item("confirm_discard_popup")
    while dpg.is_item_shown("confirm_discard_popup"):
        dpg.render_dearpygui_frame()
    return dpg.get_value("_discard_confirmed")


def _on_open_selected(sender, app_data):
    global _current_path, _is_dirty
    path = app_data.get('file_path_name', '')
    if not path or not os.path.isfile(path):
        return
    try:
        data = load_project(path)
    except Exception as e:
        _show_error(f"Could not open project:\n{e}")
        return

    settings = data.get('settings', {})
    _element_manager.configs = data.get('elements', [])
    _element_manager._refresh_list()
    _bundle_manager.configs = data.get('bundles', [])
    _bundle_manager._refresh_list()

    # Apply settings
    global _scene
    _scene.Nbounces = settings.get('Nbounces', 100)
    if dpg.does_item_exist("nbounces_input"):
        dpg.set_value("nbounces_input", _scene.Nbounces)

    # Trigger rebuilds so the live scene matches the loaded config
    _element_manager._do_build()
    _bundle_manager._do_build()

    _current_path = path
    _is_dirty = False
    _update_viewport_title()


def _on_save_selected(sender, app_data):
    global _current_path, _is_dirty
    path = app_data.get('file_path_name', '')
    if not path:
        return
    if not path.endswith('.rtt'):
        path += '.rtt'
    _do_save(path)


def _action_save():
    if _current_path:
        _do_save(_current_path)
    else:
        dpg.show_item("dlg_save")


def _do_save(path: str):
    global _current_path, _is_dirty
    settings = {
        'device': str(_device),
        'Nbounces': int(dpg.get_value("nbounces_input")) if dpg.does_item_exist("nbounces_input") else 100,
    }
    try:
        save_project(path, _element_manager.configs, _bundle_manager.configs, settings)
        _current_path = path
        _is_dirty = False
        _update_viewport_title()
    except Exception as e:
        _show_error(f"Could not save project:\n{e}")


def _show_error(msg: str):
    err_tag = "error_popup"
    if not dpg.does_item_exist(err_tag):
        with dpg.window(label="Error", modal=True, show=False, tag=err_tag,
                        width=400, height=150, no_resize=True):
            dpg.add_text("", tag="error_popup_text", wrap=380)
            dpg.add_button(label="OK", width=80,
                           callback=lambda: dpg.hide_item(err_tag))
    dpg.set_value("error_popup_text", msg)
    dpg.show_item(err_tag)


# ============================================================
# Simulation
# ============================================================

def _compile_scene():
    """Wrap all current scene elements with torch.compile."""
    if not _scene.elements:
        _show_error("No elements in scene.\nAdd elements and click UPDATE SCENE first.")
        return
    try:
        dpg.configure_item("compile_btn", enabled=False)
        dpg.set_value("sim_status", "Compiling (first run will trace)...")
        dpg.render_dearpygui_frame()
        _scene.compile_elements()
        dpg.set_value("sim_status", "Compiled ✓  — first sim traces, then fast")
    except Exception as e:
        import traceback; traceback.print_exc()
        dpg.set_value("sim_status", "Compile failed (see console)")
        _show_error(f"Compile error:\n{e}")
    finally:
        dpg.configure_item("compile_btn", enabled=True)


def _reset_sensors():
    """Call reset() on every sensor in the scene and clear the results display."""
    for el in _scene.elements:
        if hasattr(el, 'reset'):
            el.reset()
    # Clear the spot diagram and metrics
    if dpg.does_item_exist("spot_yax"):
        dpg.delete_item("spot_yax", children_only=True)
    for tag in ("metric_rms", "metric_centroid", "metric_active"):
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, "--")
    dpg.set_value("sim_status", "Sensor reset")


def _run_simulation():
    global _last_sim_rays

    # Always start from a fresh sample so repeated runs are independent
    _scene._build_rays()
    if _scene.rays is None:
        _show_error("No rays found.\nAdd a bundle in the Rays tab and click UPDATE SCENE.")
        return
    if _scene.rays is not None and hasattr(_scene.rays, 'to'):
        _scene.rays = _scene.rays.to(_device)

    nbounces = dpg.get_value("nbounces_input")
    _scene.Nbounces = nbounces

    dpg.configure_item("run_btn", enabled=False)
    dpg.set_value("sim_status", "Running...")
    dpg.render_dearpygui_frame()

    paths = None
    try:
        # Wrap rays in a Paths proxy — positions are snapshotted automatically
        # inside scatter_update() at each bounce, with no extra loop needed.
        paths = Paths(_scene.rays)
        _scene.rays = paths

        _scene._build_index_maps()
        for _ in range(nbounces):
            if not (_scene.rays.intensity > 0).any():
                break
            _scene.step()

        _last_sim_rays = paths.unwrap()
        n_active = int((_last_sim_rays.intensity > 0).sum())
        dpg.set_value("sim_status", f"Done — {n_active} active rays")

    except Exception as e:
        import traceback
        traceback.print_exc()
        dpg.set_value("sim_status", "Error (see console)")
        _show_error(f"Simulation error:\n{e}")
    finally:
        # Always restore a plain Rays object so the optimizer and other code
        # never encounter the Paths wrapper.
        if paths is not None:
            _scene.rays = paths.unwrap()
        dpg.configure_item("run_btn", enabled=True)

    # Hand the recorded history to the viewport for overlay drawing
    if paths is not None:
        _viewport.ray_path_history = paths.get_history()
    _refresh_all_views()
    _update_results_panel()


# ============================================================
# Results / post-processing
# ============================================================

# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def _toggle_plot_equal(plot_tag: str, btn_tag: str):
    """Toggle equal_aspects on a DPG plot and update the button label."""
    global _equal_aspects_plots
    currently_equal = plot_tag in _equal_aspects_plots
    new_val = not currently_equal
    try:
        dpg.configure_item(plot_tag, equal_aspects=new_val)
    except Exception:
        pass
    if new_val:
        _equal_aspects_plots.add(plot_tag)
    else:
        _equal_aspects_plots.discard(plot_tag)
    dpg.configure_item(btn_tag, label="[1:1]" if new_val else "1:1")


def _apply_spot_scatter_theme(series_tag: int, marker_size: float, opacity: int):
    """Create/refresh the DPG theme that controls scatter marker size and opacity."""
    if dpg.does_item_exist(_SPOT_THEME_TAG):
        dpg.delete_item(_SPOT_THEME_TAG)
    with dpg.theme(tag=_SPOT_THEME_TAG):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill,
                                (255, 200, 80, opacity),
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline,
                                (255, 200, 80, opacity),
                                category=dpg.mvThemeCat_Plots)
    dpg.bind_item_theme(series_tag, _SPOT_THEME_TAG)


def _get_available_ray_ids() -> list:
    """Collect unique ray_id values from all registered bundles."""
    ids = ["All"]
    seen: set = set()
    for bundle, _ in _bundle_manager.bundle_instances:
        rid = getattr(bundle, 'ray_id', None)
        if rid is not None and rid not in seen:
            ids.append(str(rid))
            seen.add(rid)
    return ids


def _refresh_spot_id_combo():
    """Update the ray-ID filter combo to reflect the current bundle list."""
    if not dpg.does_item_exist("spot_ray_id"):
        return
    items = _get_available_ray_ids()
    dpg.configure_item("spot_ray_id", items=items)
    cur = dpg.get_value("spot_ray_id")
    if cur not in items:
        dpg.set_value("spot_ray_id", "All")


def _on_spot_mode_changed(sender, app_data):
    is_scatter = (app_data == "Scatter")
    if is_scatter:
        dpg.show_item("scatter_ctrl_grp")
        dpg.hide_item("raster_ctrl_grp")
    else:
        dpg.hide_item("scatter_ctrl_grp")
        dpg.show_item("raster_ctrl_grp")
    _update_results_panel()


def _get_sensor():
    """Return the first Sensor element in the scene, or None."""
    for el in _scene.elements:
        if hasattr(el, 'hitLocs') and el.hitLocs:
            return el
    return None


def _update_results_panel():
    if _last_sim_rays is None:
        return

    sensor = _get_sensor()
    try:
        if sensor is not None:
            locs, intensities, ids_t = sensor.getHitsTensors()
            xy     = locs[:, :2].detach().cpu().numpy()
            w      = intensities.detach().cpu().numpy()
            ids_np = ids_t.detach().cpu().numpy()
        else:
            rays = _last_sim_rays
            with torch.no_grad():
                active = (rays.intensity > 0).cpu()
                if not active.any():
                    return
                xy     = rays.pos[active, :2].cpu().numpy()
                w      = rays.intensity[active].cpu().numpy()
                ids_np = rays.id[active].cpu().numpy()
    except Exception as e:
        print(f"[Results] data error: {e}")
        return

    if len(w) == 0:
        return

    # ---- Ray-ID filter ----
    ray_id_str = dpg.get_value("spot_ray_id") if dpg.does_item_exist("spot_ray_id") else "All"
    if ray_id_str != "All":
        try:
            rid  = int(ray_id_str)
            mask = (ids_np == rid)
            xy     = xy[mask]
            w      = w[mask]
            ids_np = ids_np[mask]
        except (ValueError, TypeError):
            pass

    if len(w) == 0:
        dpg.set_value("metric_active", "0 (no hits for this ID)")
        return

    mode = dpg.get_value("spot_mode") if dpg.does_item_exist("spot_mode") else "Scatter"
    dpg.delete_item("spot_yax", children_only=True)

    if mode == "Scatter":
        size    = float(dpg.get_value("spot_size_sl"))    if dpg.does_item_exist("spot_size_sl")    else 3.0
        opacity = int(dpg.get_value("spot_opacity_sl"))   if dpg.does_item_exist("spot_opacity_sl") else 180
        series_tag = dpg.add_scatter_series(xy[:, 0].tolist(), xy[:, 1].tolist(),
                                            parent="spot_yax")
        _apply_spot_scatter_theme(series_tag, size, opacity)
        dpg.set_axis_limits_auto("spot_xax")
        dpg.set_axis_limits_auto("spot_yax")
        dpg.fit_axis_data("spot_xax")
        dpg.fit_axis_data("spot_yax")
    else:  # Raster / histogram
        n_bins = int(dpg.get_value("spot_bins_sl")) if dpg.does_item_exist("spot_bins_sl") else 64
        xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
        ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
        px = max((xmax - xmin) * 0.05, 1e-6)
        py = max((ymax - ymin) * 0.05, 1e-6)
        xmin -= px;  xmax += px
        ymin -= py;  ymax += py
        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=n_bins,
                                    range=[[xmin, xmax], [ymin, ymax]], weights=w)
        # DPG heat_series is row-major (row=Y, col=X) → transpose numpy's (X,Y) array
        scale_max = float(hist.max()) if hist.max() > 0 else 1.0
        dpg.add_heat_series(hist.T.flatten().tolist(),
                            rows=n_bins, cols=n_bins,
                            scale_min=0.0, scale_max=scale_max,
                            bounds_min=(xmin, ymin), bounds_max=(xmax, ymax),
                            format="",          # suppress per-cell value labels
                            parent="spot_yax", label="Density")
        try:
            dpg.bind_colormap("spot_plot", dpg.mvPlotColormap_Hot)
        except Exception:
            pass
        dpg.set_axis_limits("spot_xax", xmin, xmax)
        dpg.set_axis_limits("spot_yax", ymin, ymax)

    # ---- Metrics ----
    safe_w = np.where(w > 0, w, 1e-12)
    cx  = float(np.average(xy[:, 0], weights=safe_w))
    cy  = float(np.average(xy[:, 1], weights=safe_w))
    rms = float(np.sqrt(np.average(
        (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2, weights=safe_w)))

    dpg.set_value("metric_rms",      f"{rms:.5f}")
    dpg.set_value("metric_centroid", f"({cx:.4f},  {cy:.4f})")
    dpg.set_value("metric_active",   str(len(w)))


# ============================================================
# Optimizer
# ============================================================

def _sample_combined_rays() -> Optional[Rays]:
    """
    Draw a fresh sample from all registered bundles via the scene's
    _build_rays(), which owns the canonical bundle list.
    Returns None if no bundles are registered.
    """
    if len(_scene.bundles) == 0:
        return None
    _scene._build_rays()
    if _scene.rays is not None and hasattr(_scene.rays, 'to'):
        _scene.rays = _scene.rays.to(_device)
    return _scene.rays


def _compute_spot_loss() -> torch.Tensor:
    sensor = _get_sensor()
    if sensor is not None:
        sensor.reset()

    rays = _sample_combined_rays()
    if rays is None:
        return torch.tensor(0.0)

    rays = rays.to(_device)
    _scene.rays = rays
    _scene._build_index_maps()

    for _ in range(_scene.Nbounces):
        if not (_scene.rays.intensity > 0).any():
            break
        _scene.step()

    if sensor is not None and sensor.hitLocs:
        locs, intensities, _ = sensor.getHitsTensors()
        xy = locs[:, :2]
        w  = intensities
    else:
        xy = _scene.rays.pos[:, :2]
        w  = _scene.rays.intensity

    active = w > 0
    if not active.any():
        return torch.tensor(0.0, requires_grad=True)

    xy = xy[active]
    w  = w[active]
    w_sum = w.sum().clamp(min=1e-12)
    cx = (xy[:, 0] * w).sum() / w_sum
    cy = (xy[:, 1] * w).sum() / w_sum
    rms = torch.sqrt(((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) * w / w_sum).sum()
    return rms


def _compute_focal_loss(f_target: float) -> torch.Tensor:
    if not hasattr(_scene, 'getParaxial'):
        _show_error("FocalLengthLoss requires a SequentialScene with getParaxial().")
        return torch.tensor(0.0)
    M = _scene.getParaxial()
    P_actual = -M[1, 0]
    P_target = torch.tensor(1.0 / f_target) if f_target != 0 else torch.tensor(0.0)
    return (P_actual - P_target) ** 2


def _run_optimizer():
    params = [p for p in _scene.parameters() if p.requires_grad]
    if not params:
        _show_error(
            "No optimisable parameters found.\n"
            "Enable 'grad' checkboxes on element parameters, then click UPDATE SCENE."
        )
        return

    loss_type = dpg.get_value("opt_loss_type")
    n_steps   = int(dpg.get_value("opt_steps"))
    lr        = float(dpg.get_value("opt_lr"))
    f_target  = float(dpg.get_value("opt_target"))

    optimizer = torch.optim.Adam(params, lr=lr)
    dpg.configure_item("opt_btn", enabled=False)

    for step in range(n_steps):
        optimizer.zero_grad()

        if loss_type == "SpotSizeLoss":
            loss = _compute_spot_loss()
        else:
            loss = _compute_focal_loss(f_target)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        dpg.set_value("opt_loss_display", f"Loss: {loss_val:.7f}")
        dpg.set_value("opt_progress",     f"Step {step + 1} / {n_steps}")
        dpg.render_dearpygui_frame()

    dpg.configure_item("opt_btn", enabled=True)
    _refresh_all_views()


# ============================================================
# UI construction
# ============================================================

def _build_left_panel():
    with dpg.tab_bar():
        with dpg.tab(label="Elements"):
            _element_manager.build(dpg.last_item())

        with dpg.tab(label="Rays"):
            _bundle_manager.build(dpg.last_item())

        with dpg.tab(label="Optimize"):
            _build_optimizer_panel()


def _build_optimizer_panel():
    dpg.add_combo(["SpotSizeLoss", "FocalLengthLoss"],
                  label="Loss function", tag="opt_loss_type",
                  default_value="SpotSizeLoss", width=-1)
    dpg.add_input_double(label="Target value (f for FocalLength)",
                         tag="opt_target", default_value=50.0, width=-1)
    dpg.add_input_int(label="Steps",        tag="opt_steps",
                      default_value=50, min_value=1, width=100)
    dpg.add_input_float(label="Learn rate",  tag="opt_lr",
                        default_value=0.001, format="%.5f", width=100)
    dpg.add_button(label="Run Optimizer", tag="opt_btn",
                   width=-1, callback=_run_optimizer)
    dpg.add_separator()
    dpg.add_text("--", tag="opt_loss_display")
    dpg.add_text("",   tag="opt_progress")


def _on_ray_vis_changed():
    _viewport.ray_visible  = dpg.get_value("ray_visible_cb")
    _viewport.ray_linewidth = dpg.get_value("ray_width_sl")
    _viewport.ray_opacity  = int(dpg.get_value("ray_opacity_sl"))
    _viewport.redraw_overlay()


def _clear_ray_overlay():
    """Erase the stored ray path history and remove all overlay lines."""
    _viewport.ray_path_history = []
    _viewport.redraw_overlay()


def _build_center_panel():
    dpg.add_text(
        "Left-drag: Orbit  |  Mid-drag: Pan  |  Scroll: Zoom  |  Alt+Left: Roll"
    )
    _viewport.build(dpg.last_container())

    dpg.add_spacer(height=4)
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="Show Rays", tag="ray_visible_cb", default_value=True,
                         callback=lambda s, a: _on_ray_vis_changed())
        dpg.add_text("  Width")
        dpg.add_slider_float(label="##ray_width", tag="ray_width_sl",
                             default_value=1.0, min_value=0.5, max_value=5.0, width=70,
                             callback=lambda s, a: _on_ray_vis_changed())
        dpg.add_text("  Opacity")
        dpg.add_slider_int(label="##ray_opacity", tag="ray_opacity_sl",
                           default_value=160, min_value=0, max_value=255, width=70,
                           callback=lambda s, a: _on_ray_vis_changed())
        dpg.add_button(label="Clear", callback=_clear_ray_overlay)

    dpg.add_spacer(height=4)
    with dpg.group(horizontal=True):
        dpg.add_input_int(label="Bounces", tag="nbounces_input",
                          default_value=100, min_value=1, max_value=10_000, width=80)
        dpg.add_button(label="Run Simulation", tag="run_btn",
                       callback=_run_simulation)
        dpg.add_button(label="Compile", tag="compile_btn",
                       callback=_compile_scene)
        dpg.add_button(label="Reset Sensor", callback=_reset_sensors)
        dpg.add_text("Ready", tag="sim_status")


def _build_right_panel():
    with dpg.tab_bar():
        with dpg.tab(label="Cross Sections"):
            _tab_cs = dpg.last_item()
            _profile_xz.build(_tab_cs)
            dpg.add_button(label="1:1", tag="xz_equal_btn",
                           callback=lambda: _toggle_plot_equal("profile_plot_x",
                                                               "xz_equal_btn"))
            _profile_yz.build(_tab_cs)
            dpg.add_button(label="1:1", tag="yz_equal_btn",
                           callback=lambda: _toggle_plot_equal("profile_plot_y",
                                                               "yz_equal_btn"))

        with dpg.tab(label="Results"):
            _build_results_panel()


def _build_results_panel():
    # ---- Controls row 1: mode selector, ray-ID filter, 1:1 toggle ----
    with dpg.group(horizontal=True):
        dpg.add_combo(["Scatter", "Raster"], default_value="Scatter",
                      label="##spot_mode", tag="spot_mode", width=80,
                      callback=_on_spot_mode_changed)
        dpg.add_text("  ID:")
        dpg.add_combo(["All"], default_value="All",
                      label="##spot_ray_id", tag="spot_ray_id", width=55,
                      callback=lambda s, a: _update_results_panel())
        dpg.add_button(label="[1:1]", tag="spot_equal_btn",
                       callback=lambda: _toggle_plot_equal("spot_plot", "spot_equal_btn"))

    # ---- Controls row 2a: scatter-specific (size + opacity) ----
    with dpg.group(tag="scatter_ctrl_grp", horizontal=False):
        with dpg.group(horizontal=True):
            dpg.add_text("Size:")
            dpg.add_slider_float(label="##spot_size", tag="spot_size_sl",
                                 default_value=3.0, min_value=0.5, max_value=15.0,
                                 width=70, callback=lambda s, a: _update_results_panel())
            dpg.add_text("  Opacity:")
            dpg.add_slider_int(label="##spot_opacity", tag="spot_opacity_sl",
                               default_value=180, min_value=10, max_value=255,
                               width=70, callback=lambda s, a: _update_results_panel())

    # ---- Controls row 2b: raster-specific (bin count) ----
    with dpg.group(tag="raster_ctrl_grp", horizontal=False, show=False):
        with dpg.group(horizontal=True):
            dpg.add_text("Bins:")
            dpg.add_slider_int(label="##spot_bins", tag="spot_bins_sl",
                               default_value=64, min_value=8, max_value=512,
                               width=120, callback=lambda s, a: _update_results_panel())

    # ---- Spot diagram plot ----
    with dpg.plot(label="Spot Diagram", height=280, width=-1,
                  tag="spot_plot", equal_aspects=True):
        dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="spot_xax")
        dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="spot_yax")

    # ---- Metrics ----
    dpg.add_separator()
    dpg.add_text("Metrics", color=(180, 180, 180, 255))
    with dpg.table(header_row=False, borders_innerV=True):
        dpg.add_table_column(init_width_or_weight=0.55)
        dpg.add_table_column(init_width_or_weight=0.45)
        for label, tag in [
            ("RMS Spot Radius", "metric_rms"),
            ("Centroid (x, y)", "metric_centroid"),
            ("Active Rays",     "metric_active"),
        ]:
            with dpg.table_row():
                dpg.add_text(label)
                dpg.add_text("--", tag=tag)


def _build_menu_bar():
    """Build a window-level menu bar (not viewport_menu_bar) so it doesn't overlap content."""
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="New",        shortcut="Ctrl+N",
                              callback=_action_new)
            dpg.add_menu_item(label="Open...",    shortcut="Ctrl+O",
                              callback=lambda: dpg.show_item("dlg_open"))
            dpg.add_separator()
            dpg.add_menu_item(label="Save",       shortcut="Ctrl+S",
                              callback=_action_save)
            dpg.add_menu_item(label="Save As...", shortcut="Ctrl+Shift+S",
                              callback=lambda: dpg.show_item("dlg_save"))

        # Device indicator — green badge for CUDA, amber for CPU
        is_cuda = _device is not None and _device.type == "cuda"
        label  = f"  ● CUDA:{_device.index if _device.index is not None else 0}  " if is_cuda else "  ● CPU  "
        color  = (80, 220, 100, 255) if is_cuda else (220, 170, 50, 255)
        dpg.add_text(label, color=color, tag="device_indicator")


def _setup_file_dialogs():
    with dpg.file_dialog(show=False, callback=_on_open_selected,
                         tag="dlg_open", width=720, height=460,
                         file_count=1):
        dpg.add_file_extension(".rtt", color=(255, 220, 50, 255),
                               custom_text="[RTT Project]")

    with dpg.file_dialog(show=False, callback=_on_save_selected,
                         tag="dlg_save", width=720, height=460):
        dpg.add_file_extension(".rtt", color=(255, 220, 50, 255),
                               custom_text="[RTT Project]")


def _build_ui():
    with dpg.window(tag="main_window", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True, menubar=True):
        _build_menu_bar()

        with dpg.table(header_row=False, borders_innerV=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(init_width_or_weight=0.22)
            dpg.add_table_column(init_width_or_weight=0.52)
            dpg.add_table_column(init_width_or_weight=0.26)

            with dpg.table_row():
                with dpg.table_cell():
                    _build_left_panel()
                with dpg.table_cell():
                    _build_center_panel()
                with dpg.table_cell():
                    _build_right_panel()


# ============================================================
# Entry point
# ============================================================

def run():
    global _scene, _device, _cam, _renderer, _viewport
    global _profile_xz, _profile_yz
    global _element_manager, _bundle_manager

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Workbench] device: {_device}")

    _scene = Scene()
    _scene.to(_device)

    _cam = OrbitCamera(
        pivot=(0, 0, 0),
        position=(0, 0, -60),
        look_at=(0, 0, 0),
        up_vector=(0, 1, 0),
        fov_deg=40,
        width=800,
        height=800,
        device=_device,
    )
    _renderer = Renderer(_scene, background_color=(0.12, 0.12, 0.12))
    _viewport  = RenderViewport(_renderer, _cam, width=800, height=800)
    _profile_xz = ProfilePlot(_renderer, _scene, axis='x')
    _profile_yz = ProfilePlot(_renderer, _scene, axis='y')

    _element_manager = ItemManager(
        title="Optical Element",
        base_cls=Element,
        on_update=_update_scene_elements,
        on_data_changed=_mark_dirty,
        device=_device,
        dtype=torch.float32,
    )
    _bundle_manager = BundleItemManager(
        on_update=_update_scene_rays,
        on_data_changed=_mark_dirty,
        device=_device,
        dtype=torch.float32,
    )

    dpg.create_context()
    dpg.create_viewport(title="Optical Design Workbench — Untitled",
                        width=1600, height=900)
    dpg.setup_dearpygui()

    _setup_file_dialogs()
    _element_manager.setup_popup()
    _bundle_manager.setup_popup()
    _setup_confirm_discard_popup()
    _build_ui()
    dpg.set_primary_window("main_window", True)   # must be after window is created
    _viewport.register_mouse_handlers()

    dpg.show_viewport()
    _viewport.refresh()          # render background on first frame

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run()
