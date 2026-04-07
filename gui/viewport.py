"""
gui/viewport.py
---------------
Dear PyGui widgets for the interactive 3D render view and 2D cross-section
profile plots.

Replaces the PyQt6 camera_pyqt.py.  The OrbitCamera and Renderer classes in
render/camera.py are unchanged — this module is purely the UI layer that sits
on top of them.
"""

import numpy as np
import torch
import dearpygui.dearpygui as dpg

from ..render.camera import Renderer, OrbitCamera


# ============================================================
# RenderViewport
# ============================================================

class RenderViewport:
    """
    Manages a Dear PyGui dynamic texture + drawlist for the 3D render,
    and wires mouse handlers for orbit / pan / zoom / roll.

    Usage
    -----
    viewport = RenderViewport(renderer, camera, width=800, height=800)
    viewport.build(parent_tag="some_dpg_item")
    viewport.register_mouse_handlers()   # call once after dpg.setup_dearpygui()
    viewport.refresh()                   # re-render and update texture
    """

    # Sensitivity constants
    ORBIT_SENS = 0.008
    PAN_SENS   = 0.04
    ZOOM_SENS  = 0.10
    ROLL_SENS  = 0.008

    def __init__(self, renderer: Renderer, camera: OrbitCamera,
                 width: int = 800, height: int = 800):
        self._renderer = renderer
        self._camera = camera
        self._w = width
        self._h = height

        # Ray path history: list of [N, 3] CPU float tensors (one per sim step)
        self.ray_path_history: list = []

        # DPG tags
        self._tex_tag      = "rtvp__tex"
        self._drawlist_tag = "rtvp__drawlist"
        self._paths_node   = "rtvp__paths"

        # Mouse state
        self._prev_mouse: tuple = (0.0, 0.0)
        self._alt_down: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, parent_tag: str):
        """Create the texture registry entry and the drawlist widget."""
        flat = np.zeros(self._w * self._h * 4, dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_dynamic_texture(self._w, self._h, flat, tag=self._tex_tag)

        with dpg.drawlist(width=self._w, height=self._h,
                          tag=self._drawlist_tag, parent=parent_tag):
            dpg.draw_image(self._tex_tag, (0, 0), (self._w, self._h))
            # Container for ray-path line overlays (children deleted/rebuilt each sim)
            dpg.draw_node(tag=self._paths_node)

    def register_mouse_handlers(self):
        """
        Register global mouse handlers.  Must be called after dpg.setup_dearpygui()
        and before dpg.start_dearpygui().
        """
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_wheel_handler(callback=self._on_scroll)
            dpg.add_key_down_handler(dpg.mvKey_Alt, callback=self._on_alt_down)
            dpg.add_key_release_handler(dpg.mvKey_Alt, callback=self._on_alt_up)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-render the scene and update the texture + path overlay."""
        try:
            img = self._renderer.render_3d(self._camera)   # [H, W, 3] float32 CPU
            alpha = torch.ones(*img.shape[:2], 1)
            rgba = torch.cat([img, alpha], dim=2)           # [H, W, 4]
            flat = rgba.numpy().astype(np.float32).flatten()
            dpg.set_value(self._tex_tag, flat)
        except Exception as e:
            print(f"[RenderViewport] render error: {e}")

        self._draw_path_overlay()

    # ------------------------------------------------------------------
    # Ray-path overlay
    # ------------------------------------------------------------------

    def _draw_path_overlay(self):
        """Project recorded ray paths into screen space and draw as lines."""
        dpg.delete_item(self._paths_node, children_only=True)
        if len(self.ray_path_history) < 2:
            return

        # Subsample rays for performance (draw at most 100)
        n_rays = self.ray_path_history[0].shape[0]
        max_vis = 100
        step = max(1, n_rays // max_vis)
        ray_indices = range(0, n_rays, step)

        cam = self._camera
        w, h = self._w, self._h

        for ri in ray_indices:
            prev_screen = None
            for snap in self.ray_path_history:
                pt3d = snap[ri]                              # [3]
                sc, visible = self._project_point(pt3d, cam, w, h)
                if visible and sc is not None:
                    if prev_screen is not None:
                        dpg.draw_line(
                            prev_screen, sc,
                            color=(255, 220, 60, 160),
                            thickness=1,
                            parent=self._paths_node,
                        )
                    prev_screen = sc
                else:
                    prev_screen = None   # break the line on occluded segments

    @staticmethod
    def _project_point(pt3d: torch.Tensor, camera: OrbitCamera,
                       w: int, h: int) -> tuple:
        """
        Project a 3D world-space point onto the screen.
        Returns ((px, py), True) if visible, (None, False) otherwise.
        """
        delta = pt3d.to(camera.origin.device) - camera.origin
        z = torch.dot(delta, camera.forward).item()
        if z < 0.01:
            return None, False

        x = torch.dot(delta, camera.right).item()
        y = torch.dot(delta, camera.up_cam).item()

        aspect = w / h
        scale_y = float(torch.tan(torch.deg2rad(torch.tensor(camera.fov_deg * 0.5))))
        scale_x = scale_y * aspect

        sx = (x / (z * scale_x)) * 0.5 + 0.5
        sy = -(y / (z * scale_y)) * 0.5 + 0.5

        px = sx * w
        py = sy * h
        return (px, py), True

    # ------------------------------------------------------------------
    # Mouse handlers
    # ------------------------------------------------------------------

    def _is_hovered(self) -> bool:
        return dpg.is_item_hovered(self._drawlist_tag)

    def _on_mouse_move(self, sender, app_data):
        mx, my = dpg.get_mouse_pos(local=False)
        dx = mx - self._prev_mouse[0]
        dy = my - self._prev_mouse[1]
        self._prev_mouse = (mx, my)

        if not self._is_hovered():
            return

        left_down   = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
        middle_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Middle)

        if self._alt_down and left_down:
            self._camera.roll(torch.tensor(dx * self.ROLL_SENS))
            self.refresh()
        elif left_down:
            self._camera.orbit(torch.tensor(dx * self.ORBIT_SENS),
                                torch.tensor(dy * self.ORBIT_SENS))
            self.refresh()
        elif middle_down:
            self._camera.pan(torch.tensor(dx * self.PAN_SENS),
                             torch.tensor(dy * self.PAN_SENS))
            self.refresh()

    def _on_scroll(self, sender, app_data):
        if not self._is_hovered():
            return
        self._camera.zoom(torch.tensor(float(app_data) * self.ZOOM_SENS))
        self.refresh()

    def _on_alt_down(self, sender, app_data):
        self._alt_down = True

    def _on_alt_up(self, sender, app_data):
        self._alt_down = False


# ============================================================
# ProfilePlot
# ============================================================

class ProfilePlot:
    """
    Renders a 2D cross-section (XZ or YZ) of the scene using Dear PyGui's
    native plot system.  Much cleaner than a custom paintEvent.
    """

    def __init__(self, renderer: Renderer, scene, axis: str):
        self._renderer = renderer
        self._scene = scene
        self._axis = axis          # 'x' → XZ plane, 'y' → YZ plane

        label = "Top View (XZ)" if axis == 'x' else "Side View (YZ)"
        self._plot_tag = f"profile_plot_{axis}"
        self._xax_tag  = f"profile_xax_{axis}"
        self._yax_tag  = f"profile_yax_{axis}"
        self._label = label

    def build(self, parent_tag: str):
        with dpg.plot(label=self._label, height=240, width=-1,
                      parent=parent_tag, tag=self._plot_tag):
            dpg.add_plot_axis(dpg.mvXAxis, label="Z (optical)",  tag=self._xax_tag)
            dpg.add_plot_axis(dpg.mvYAxis, label="H (transverse)", tag=self._yax_tag)

    def update(self):
        """Re-scan all elements and refresh the line series."""
        if not self._scene or not hasattr(self._scene, 'elements'):
            return

        dpg.delete_item(self._yax_tag, children_only=True)

        for el in self._scene.elements:
            try:
                profiles = self._renderer.scan_profile(el, axis=self._axis, num_points=200)
                for p in profiles:
                    z_list = p['z'].tolist()
                    h_list = p['h'].tolist()
                    if z_list:
                        dpg.add_line_series(z_list, h_list,
                                            label=f"surf_{p['surf_idx']}",
                                            parent=self._yax_tag)
            except Exception as e:
                print(f"[ProfilePlot({self._axis})] scan error: {e}")

        dpg.fit_axis_data(self._xax_tag)
        dpg.fit_axis_data(self._yax_tag)
