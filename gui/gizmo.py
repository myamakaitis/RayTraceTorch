"""
gui/gizmo.py
------------
Interactive translation arrows and rotation rings drawn as 2D overlays
on the Dear PyGui drawlist.  Handles hit-testing and drag interaction
to modify element transforms in real time.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import dearpygui.dearpygui as dpg


# Axis colours: X=red, Y=green, Z=blue
_AXIS_COLORS = [
    (220, 60, 60),     # X – red
    (60, 200, 60),     # Y – green
    (60, 100, 220),    # Z – blue
]
_HIGHLIGHT_COLOR = (255, 255, 80)   # hovered / active handle

# Number of segments for drawing rotation rings
_RING_SEGMENTS = 48


class Gizmo:
    """
    Draws translation arrows and rotation rings for a selected element,
    projected onto the 2D drawlist, and handles mouse interaction.

    Parameters
    ----------
    viewport : RenderViewport
        The viewport that owns the drawlist and camera.
    arm_length : float
        World-space length of each translation arrow arm.
    ring_radius : float
        World-space radius of each rotation ring.
    hit_radius : float
        Screen-space pixel radius for hit-testing handles.
    """

    def __init__(self, viewport, arm_length: float = 8.0,
                 ring_radius: float = 6.0, hit_radius: float = 28.0):
        self._vp = viewport
        self.arm_length = arm_length
        self.ring_radius = ring_radius
        self.hit_radius = hit_radius

        # Current state
        self._transform = None        # the RayTransform to manipulate
        self._target_kind: str = ''   # 'element' | 'bundle'
        self._target_idx: int = -1    # index in scene.elements or bundle list
        self._draw_ids: list = []     # DPG draw item IDs for cleanup

        # Drag state
        self._dragging: bool = False
        self._drag_mode: str | None = None     # 'translate' | 'rotate'
        self._drag_axis: int = -1              # 0=X, 1=Y, 2=Z
        self._drag_start_mouse: tuple = (0, 0)
        self._drag_start_param: torch.Tensor | None = None

        # Cached geometry for hit-testing (screen coords)
        self._arrow_endpoints: list = []  # [(start_px, end_px), ...] for 3 axes
        self._ring_points: list = []      # [[(px, py), ...], ...] for 3 axes

        # Callback invoked after drag completes: fn(target_kind, target_idx)
        self.on_drag_complete = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_element(self, element, element_idx: int):
        """Select an element for gizmo display."""
        self._transform = element.shape.transform
        self._target_kind = 'element'
        self._target_idx = element_idx

    def set_bundle(self, bundle, bundle_idx: int):
        """Select a bundle for gizmo display."""
        self._transform = bundle.transform
        self._target_kind = 'bundle'
        self._target_idx = bundle_idx

    @property
    def element(self):
        """Backward compat — returns the transform (truthy if set)."""
        return self._transform

    @property
    def element_idx(self):
        return self._target_idx

    def clear(self):
        """Deselect — hide gizmo."""
        self._transform = None
        self._target_kind = ''
        self._target_idx = -1
        self._erase()

    def draw(self):
        """Project gizmo geometry to screen and draw onto the viewport drawlist."""
        self._erase()
        if self._transform is None:
            return

        cam = self._vp._camera
        w, h = self._vp._w, self._vp._h
        drawlist = self._vp._drawlist_tag

        origin_world = self._transform.trans.detach().cpu()
        origin_sc, vis = self._vp._project_point(origin_world, cam, w, h)
        if not vis:
            return

        # Compute local axes in world space (rows of R)
        R = self._transform.rot.detach().cpu()  # [3,3]

        self._arrow_endpoints = []
        self._ring_points = []

        # --- Translation arrows ---
        for axis_i in range(3):
            axis_world = R[axis_i]  # local axis_i direction in world
            tip_world = origin_world + axis_world * self.arm_length
            tip_sc, tip_vis = self._vp._project_point(tip_world, cam, w, h)

            color = _AXIS_COLORS[axis_i]
            if self._dragging and self._drag_mode == 'translate' and self._drag_axis == axis_i:
                color = _HIGHLIGHT_COLOR

            if tip_vis:
                # Arrow line
                line_id = dpg.draw_line(origin_sc, tip_sc,
                                        color=(*color, 220), thickness=5.0,
                                        parent=drawlist)
                self._draw_ids.append(line_id)

                # Arrowhead (small triangle)
                dx = tip_sc[0] - origin_sc[0]
                dy = tip_sc[1] - origin_sc[1]
                length = math.sqrt(dx*dx + dy*dy)
                if length > 1e-3:
                    ux, uy = dx / length, dy / length
                    px, py = -uy, ux  # perpendicular
                    head_size = 16
                    p1 = (tip_sc[0] + ux * head_size, tip_sc[1] + uy * head_size)
                    p2 = (tip_sc[0] + px * head_size * 0.6, tip_sc[1] + py * head_size * 0.6)
                    p3 = (tip_sc[0] - px * head_size * 0.6, tip_sc[1] - py * head_size * 0.6)
                    tri_id = dpg.draw_triangle(p1, p2, p3,
                                               color=(*color, 220),
                                               fill=(*color, 180),
                                               parent=drawlist)
                    self._draw_ids.append(tri_id)

                self._arrow_endpoints.append((origin_sc, tip_sc))
            else:
                self._arrow_endpoints.append(None)

        # --- Rotation rings ---
        for axis_i in range(3):
            ring_pts_sc = []
            axis_world = R[axis_i]
            # Two orthogonal vectors in the plane perpendicular to this axis
            perp1 = R[(axis_i + 1) % 3]
            perp2 = R[(axis_i + 2) % 3]

            color = _AXIS_COLORS[axis_i]
            if self._dragging and self._drag_mode == 'rotate' and self._drag_axis == axis_i:
                color = _HIGHLIGHT_COLOR

            prev_sc = None
            for seg in range(_RING_SEGMENTS + 1):
                angle = 2.0 * math.pi * seg / _RING_SEGMENTS
                pt_world = (origin_world
                            + perp1 * (self.ring_radius * math.cos(angle))
                            + perp2 * (self.ring_radius * math.sin(angle)))
                pt_sc, pt_vis = self._vp._project_point(pt_world, cam, w, h)
                if pt_vis:
                    ring_pts_sc.append(pt_sc)
                    if prev_sc is not None:
                        line_id = dpg.draw_line(prev_sc, pt_sc,
                                                color=(*color, 180), thickness=3.5,
                                                parent=drawlist)
                        self._draw_ids.append(line_id)
                    prev_sc = pt_sc
                else:
                    prev_sc = None

            self._ring_points.append(ring_pts_sc)

    def hit_test(self, px: float, py: float):
        """
        Check if screen point (px, py) is near a gizmo handle.

        Arrows are tested first and get priority — if any arrow is
        within the hit radius, it wins without checking rings.

        Returns
        -------
        (mode, axis_idx) : ('translate'|'rotate', int)  or  None
        """
        # Priority 1: Check arrow shafts
        arrow_best_dist = self.hit_radius
        arrow_result = None
        for axis_i, endpoints in enumerate(self._arrow_endpoints):
            if endpoints is None:
                continue
            start, end = endpoints
            d = _point_to_segment_dist(px, py, start, end)
            if d < arrow_best_dist:
                arrow_best_dist = d
                arrow_result = ('translate', axis_i)

        if arrow_result is not None:
            return arrow_result

        # Priority 2: Check ring segments (only if no arrow was hit)
        ring_best_dist = self.hit_radius
        ring_result = None
        for axis_i, pts in enumerate(self._ring_points):
            for i in range(len(pts) - 1):
                d = _point_to_segment_dist(px, py, pts[i], pts[i + 1])
                if d < ring_best_dist:
                    ring_best_dist = d
                    ring_result = ('rotate', axis_i)

        return ring_result

    # ------------------------------------------------------------------
    # Drag interaction
    # ------------------------------------------------------------------

    def begin_drag(self, mode: str, axis: int, mouse_pos: tuple):
        """Start a gizmo drag operation."""
        self._dragging = True
        self._drag_mode = mode
        self._drag_axis = axis
        self._drag_start_mouse = mouse_pos
        if mode == 'translate':
            self._drag_start_param = self._transform.trans.detach().clone()
        else:
            self._drag_start_param = self._transform.rot_vec.detach().clone()

    def update_drag(self, mouse_pos: tuple):
        """
        Update the element transform based on mouse movement during drag.
        """
        if not self._dragging or self._transform is None:
            return

        dx = mouse_pos[0] - self._drag_start_mouse[0]
        dy = mouse_pos[1] - self._drag_start_mouse[1]

        if self._drag_mode == 'translate':
            self._apply_translate(dx, dy)
        else:
            self._apply_rotate(dx, dy)

    def end_drag(self):
        """Finish gizmo drag and fire completion callback."""
        if not self._dragging:
            return
        self._dragging = False
        self._drag_mode = None
        self._drag_axis = -1
        self._drag_start_param = None
        if self.on_drag_complete is not None:
            self.on_drag_complete(self._target_kind, self._target_idx)

    @property
    def is_dragging(self) -> bool:
        return self._dragging

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_translate(self, dx: float, dy: float):
        """Map screen-space drag delta to world-space translation along the active axis."""
        cam = self._vp._camera
        axis = self._drag_axis

        # Sensitivity: map pixels to world units.
        # Use the distance from camera to element to scale appropriately.
        origin = self._drag_start_param
        cam_dist = torch.norm(origin - cam.origin.cpu()).item()
        fov_rad = math.radians(cam.fov_deg * 0.5)
        px_to_world = (2.0 * cam_dist * math.tan(fov_rad)) / self._vp._h

        # Project the world axis onto screen space to determine which
        # screen direction corresponds to positive axis movement.
        R = self._transform.rot.detach().cpu()
        axis_world = R[axis]  # local axis in world coords

        # Screen-space direction of this axis
        right_dot = torch.dot(axis_world, cam.right.cpu()).item()
        up_dot = torch.dot(axis_world, cam.up_cam.cpu()).item()
        screen_len = math.sqrt(right_dot**2 + up_dot**2)
        if screen_len < 1e-6:
            return  # axis points straight at/away from camera

        # Signed screen displacement along axis direction
        disp = (dx * right_dot - dy * up_dot) / screen_len
        delta_world = disp * px_to_world

        with torch.no_grad():
            new_trans = self._drag_start_param.clone()
            new_trans[axis] = new_trans[axis] + delta_world
            self._transform.trans.copy_(new_trans)

    def _apply_rotate(self, dx: float, dy: float):
        """Map screen-space drag delta to rotation around the active axis."""
        sensitivity = 0.005  # radians per pixel
        cam = self._vp._camera

        # Use the component of screen motion perpendicular to the projected axis
        R = self._transform.rot.detach().cpu()
        axis_world = R[self._drag_axis]

        right_dot = torch.dot(axis_world, cam.right.cpu()).item()
        up_dot = torch.dot(axis_world, cam.up_cam.cpu()).item()

        # Rotation amount from tangential screen motion
        angle = (dx * up_dot + dy * right_dot) * sensitivity

        with torch.no_grad():
            new_rot = self._drag_start_param.clone()
            new_rot[self._drag_axis] = new_rot[self._drag_axis] + angle
            self._transform.rot_vec.copy_(new_rot)
            # Invalidate cached rotation matrix so the new rotation takes effect
            if hasattr(self._transform, '_cached_rot'):
                self._transform._cached_rot = None

    def _erase(self):
        """Remove all current gizmo draw items."""
        for item_id in self._draw_ids:
            if dpg.does_item_exist(item_id):
                dpg.delete_item(item_id)
        self._draw_ids.clear()
        self._arrow_endpoints.clear()
        self._ring_points.clear()


# ============================================================
# Geometry helpers
# ============================================================

def _point_to_segment_dist(px, py, seg_start, seg_end) -> float:
    """Shortest distance from point (px, py) to line segment (seg_start, seg_end)."""
    ax, ay = seg_start
    bx, by = seg_end
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.sqrt(apx * apx + apy * apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    dx = px - closest_x
    dy = py - closest_y
    return math.sqrt(dx * dx + dy * dy)
