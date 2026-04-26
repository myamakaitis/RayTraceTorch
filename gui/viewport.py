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
import torch.nn.functional as F
import dearpygui.dearpygui as dpg

from ..render.camera import Renderer, OrbitCamera
from ..rays.ray import Rays
from .gizmo import Gizmo


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

    # Click vs drag threshold (pixels)
    _CLICK_THRESHOLD = 4

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
        self._mouse_down_pos: tuple | None = None  # position at left-button-down

        # Gizmo
        self.gizmo = Gizmo(self)

        # Picking callback: fn(element_idx, surface_idx) or fn(None, None) on miss
        self.on_element_picked = None

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
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left,
                                        callback=self._on_left_click)
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left,
                                          callback=self._on_left_release)
            dpg.add_mouse_down_handler(dpg.mvMouseButton_Left,
                                       callback=self._on_left_down)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def pick(self, px: float, py: float):
        """
        Cast a single ray at screen pixel (px, py) and return the hit
        element and surface index, or None.

        Returns
        -------
        (element_idx, surface_idx) : (int, int)  or  None
        """
        scene = self._renderer.scene
        if not scene.elements:
            return None

        cam = self._camera
        # Generate a single ray for this pixel
        aspect = self._w / self._h
        scale_y = float(torch.tan(torch.deg2rad(
            torch.tensor(cam.fov_deg * 0.5, dtype=torch.float32))))
        scale_x = scale_y * aspect

        # Normalised coordinates [-1, 1]
        nx = (2.0 * px / self._w - 1.0) * scale_x
        ny = (1.0 - 2.0 * py / self._h) * scale_y

        direction = (cam.right * nx + cam.up_cam * ny + cam.forward)
        direction = F.normalize(direction, dim=0)

        origin = cam.origin.unsqueeze(0)       # [1, 3]
        direction = direction.unsqueeze(0)     # [1, 3]

        ray = Rays.initialize(origin, direction, device=cam.origin.device)

        # Filter apertures (same as Renderer.render_3d)
        def _is_aperture(el):
            return any('ApertureFilter' in type(f).__name__ or 'Fuzzy' in type(f).__name__
                       for f in el.surface_functions)

        with torch.no_grad():
            t_blocks = []
            local_elem_ids = []
            local_surf_ids = []
            for k, el in enumerate(scene.elements):
                if _is_aperture(el):
                    continue
                t_block = el.intersectTest(ray)  # [1, n_surfs]
                t_blocks.append(t_block)
                for s in range(t_block.shape[1]):
                    local_elem_ids.append(k)
                    local_surf_ids.append(s)

            if not t_blocks:
                return None

            t_matrix = torch.cat(t_blocks, dim=1)  # [1, total_surfs]
            min_t, best_col = torch.min(t_matrix, dim=1)

            if min_t.item() >= float('inf'):
                return None

            elem_idx = local_elem_ids[best_col.item()]
            surf_idx = local_surf_ids[best_col.item()]
            return (elem_idx, surf_idx)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def redraw_overlay(self):
        """Redraw just the ray-path overlay without re-rendering the scene."""
        try:
            self._draw_path_overlay()
        except Exception as e:
            print(f"[RenderViewport] overlay error: {e}")
        try:
            self.gizmo.draw()
        except Exception as e:
            print(f"[RenderViewport] gizmo error: {e}")

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

        try:
            self.gizmo.draw()
        except Exception as e:
            print(f"[RenderViewport] gizmo error: {e}")

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

    def _get_local_mouse(self) -> tuple:
        """Get mouse position relative to the drawlist top-left."""
        mx, my = dpg.get_mouse_pos(local=False)
        # Get the drawlist screen position
        try:
            rect_min = dpg.get_item_rect_min(self._drawlist_tag)
            return (mx - rect_min[0], my - rect_min[1])
        except Exception:
            return (mx, my)

    def _on_left_down(self, sender, app_data):
        """Track where the left button was pressed for click detection."""
        if not self._is_hovered():
            return

        mx, my = dpg.get_mouse_pos(local=False)

        # Only record the press position once at the start of a press
        if self._mouse_down_pos is None:
            self._mouse_down_pos = (mx, my)

        # Only begin a gizmo drag once (down_handler fires every frame)
        if not self.gizmo.is_dragging and self.gizmo.element is not None:
            local = self._get_local_mouse()
            hit = self.gizmo.hit_test(local[0], local[1])
            if hit is not None:
                mode, axis = hit
                self.gizmo.begin_drag(mode, axis, (mx, my))

    def _on_left_release(self, sender, app_data):
        """End gizmo drag on mouse release — fallback (primary is in _on_mouse_move)."""
        if self.gizmo.is_dragging:
            self.gizmo.end_drag()
            self.refresh()
        self._mouse_down_pos = None

    def _on_left_click(self, sender, app_data):
        """
        Fired on left click. Distinguish click from drag using displacement
        threshold. On a true click, perform element picking.
        """
        # Safety net: if we're still dragging when click fires, end it
        if self.gizmo.is_dragging:
            self.gizmo.end_drag()
            self._mouse_down_pos = None
            self.refresh()
            return

        if not self._is_hovered():
            return

        mx, my = dpg.get_mouse_pos(local=False)
        if self._mouse_down_pos is not None:
            dx = mx - self._mouse_down_pos[0]
            dy = my - self._mouse_down_pos[1]
            if (dx * dx + dy * dy) > self._CLICK_THRESHOLD ** 2:
                return  # was a drag, not a click

        local = self._get_local_mouse()
        if 0 <= local[0] <= self._w and 0 <= local[1] <= self._h:
            result = self.pick(local[0], local[1])
            if self.on_element_picked is not None:
                if result is not None:
                    self.on_element_picked(result[0], result[1])
                else:
                    self.on_element_picked(None, None)

    def _on_mouse_move(self, sender, app_data):
        mx, my = dpg.get_mouse_pos(local=False)
        dx = mx - self._prev_mouse[0]
        dy = my - self._prev_mouse[1]
        self._prev_mouse = (mx, my)

        # ── Gizmo drag lifecycle (poll-based, reliable) ──
        if self.gizmo.is_dragging:
            left_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
            if not left_down:
                # Button was released — end the drag
                self.gizmo.end_drag()
                self._mouse_down_pos = None
                self.refresh()
                return
            # Still dragging — update and refresh
            self.gizmo.update_drag((mx, my))
            self.refresh()
            return

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
