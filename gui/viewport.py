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
        self.ray_ids: np.ndarray = np.array([])
        
        # Color palette for ray IDs
        self._palette = [
            (255, 220, 60),    # yellow
            (60, 200, 255),    # light blue
            (255, 100, 100),   # red/pink
            (100, 255, 100),   # green
            (200, 100, 255),   # purple
            (255, 150, 50),    # orange
            (50, 255, 200),    # cyan
            (255, 255, 255),   # white
        ]

        # Ray overlay display settings
        self.ray_visible: bool = True
        self.ray_linewidth: float = 1.0
        self.ray_opacity: int = 160

        # IDs of draw_line items currently compositing the ray overlay.
        # Tracked explicitly so we can delete them by ID without needing a
        # draw_node container (which has unreliable dynamic-child support).
        self._overlay_items: list = []

        # DPG tags
        self._tex_tag      = "rtvp__tex"
        self._drawlist_tag = "rtvp__drawlist"

        # Mouse state
        self._prev_mouse: tuple = (0.0, 0.0)

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
            # Ray overlay lines are added directly to the drawlist and tracked
            # by ID in self._overlay_items — no draw_node container needed.

    def register_mouse_handlers(self):
        """
        Register global mouse handlers.  Must be called after dpg.setup_dearpygui()
        and before dpg.start_dearpygui().
        """
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            dpg.add_mouse_wheel_handler(callback=self._on_scroll)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def redraw_overlay(self):
        """Redraw just the ray-path overlay without re-rendering the scene."""
        try:
            self._draw_path_overlay()
        except Exception as e:
            print(f"[RenderViewport] overlay error: {e}")

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

        try:
            self._draw_path_overlay()
        except Exception as e:
            print(f"[RenderViewport] overlay error: {e}")

    # ------------------------------------------------------------------
    # Ray-path overlay
    # ------------------------------------------------------------------

    def _draw_path_overlay(self):
        """
        Project recorded ray paths into screen space and draw as lines.

        Flicker-free strategy
        ---------------------
        Build the full list of line specs first (pure Python, no DPG calls),
        then apply one of two strategies:

        * **Same count as last frame** (slider drag, camera orbit — geometry
          unchanged): configure every existing draw_line item in-place.
          Zero create/delete calls → overlay is never blank even for a single
          frame.

        * **Count changed** (new simulation, visibility toggled, ray history
          replaced): draw all new lines *first*, then delete the old ones.
          Old and new overlap for one sub-frame, which is invisible, but the
          overlay is never completely absent.
        """
        cam   = self._camera
        w, h  = self._w, self._h
        thick = float(self.ray_linewidth)

        # --- 1. Compute desired line specs (no DPG calls yet) ---
        new_specs: list = []   # [(p1, p2, color, thickness), ...]

        if self.ray_visible and len(self.ray_path_history) >= 2:
            n_rays = self.ray_path_history[0].shape[0]
            stride = max(1, n_rays // 100)

            for ri in range(0, n_rays, stride):
                ray_id = int(self.ray_ids[ri]) if len(self.ray_ids) > ri else 0
                base_color = self._palette[ray_id % len(self._palette)]
                color = (*base_color, int(self.ray_opacity))
                
                prev_sc = None
                for snap in self.ray_path_history:
                    sc, visible = self._project_point(snap[ri], cam, w, h)
                    if visible:
                        if prev_sc is not None:
                            new_specs.append((prev_sc, sc, color, thick))
                        prev_sc = sc
                    else:
                        prev_sc = None   # break line on occluded segment

        # --- 2. Apply without blanking the overlay ---
        if len(new_specs) == len(self._overlay_items):
            # Configure in-place — no items created or destroyed
            for item_id, (p1, p2, col, th) in zip(self._overlay_items, new_specs):
                if dpg.does_item_exist(item_id):
                    dpg.configure_item(item_id, p1=p1, p2=p2, color=col, thickness=th)
        else:
            # Draw new lines first so the overlay is never blank
            new_ids = [
                dpg.draw_line(p1, p2, color=col, thickness=th,
                              parent=self._drawlist_tag)
                for p1, p2, col, th in new_specs
            ]
            # Now safe to delete the old lines
            for old_id in self._overlay_items:
                if dpg.does_item_exist(old_id):
                    dpg.delete_item(old_id)
            self._overlay_items = new_ids

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
        alt_down    = (dpg.is_key_down(dpg.mvKey_LAlt) or
                       dpg.is_key_down(dpg.mvKey_RAlt))

        if alt_down and left_down:
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
