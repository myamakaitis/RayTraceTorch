import torch
import torch.nn as nn
from .primitives import Plane
from .transform import RayTransform, eulerToRotMat

class Shape(nn.Module):
    """
    Base class for 3D volumetric shapes.
    Defines the volume 'inside' a set of boundaries.
    """
    def __init__(self, transform = None):

        super().__init__()

        self.surfaces = nn.ModuleList() # List of Surface objects defining the boundary

        if transform is None:
            self.transform = RayTransform()
        else:
            self.transform = transform

    def intersectTest(self, rays):
        """
        Calculates intersection distances (t) for ALL contained surfaces.
        Calls self.inBounds() to filter physically invalid hits (clipping).

        Args:
            rays (Rays): Batch of N rays (Global Coordinates).

        Returns:
            t_matrix (Tensor): [N, K] where K is len(self.surfaces).
        """
        if not self.surfaces:
            return torch.zeros((rays.N, 0), device=self.device)

        # 1. Transform Global -> Local (Element Space)
        local_pos, local_dir = self.transform.transform(rays)
        local_rays = rays.with_coords(local_pos, local_dir)

        t_list = []
        # Iterate over all surfaces in the shape
        for i, surf in enumerate(self.surfaces):
            # A. Raw Intersection (Infinite Surface)
            t = surf.intersectTest(local_rays)

            # B. Validate Intersection with 'inside' check
            # We only check points where t is finite to avoid NaNs
            hit_mask = t < float('inf')

            # Create a safe t tensor for position calculation (replace inf with 0)
            # t_safe = torch.where(hit_mask, t, torch.zeros_like(t))
            hit_point = local_pos + t.unsqueeze(1) * local_dir

            # Check if the hit point is valid for this specific surface
            # We pass the surface index so the shape knows which boundary logic to apply
            is_valid = self.inBounds(hit_point, i)

            # Combine: Must hit the math surface AND be inside the physical bounds
            valid_mask = hit_mask & is_valid

            t_final = torch.where(valid_mask, t, torch.full_like(t, float('inf')))
            t_list.append(t_final)

        # Stack into a matrix [N, Number_of_Surfaces]
        return torch.stack(t_list, dim=1)

    def intersectSurface(self, rays, surf_idx):
        """
        Calculates the full differentiable intersection for a SPECIFIC surface index.
        Used after logic determines which surface was actually hit.

        Args:
            rays (Rays): Batch of N rays (Global Coordinates).
            surf_idx (int): Index of the surface in self.surfaces.

        Returns:
            t, hit_point, normal (Standard Surface.intersect output in Global Frame)
        """
        # 1. Transform Global -> Local (Element Space)
        local_pos, local_dir = self.transform.transform(rays)
        local_rays = rays.with_coords(local_pos, local_dir)

        # 2. Call child intersection (Returns results in Element Frame)
        t, elem_hit, elem_normal = self.surfaces[surf_idx].intersect(local_rays)

        # 3. Transform Results back to Global Frame

        # A. Hit Point: Recompute in global to ensure graph consistency
        hit_point = rays.pos + t.unsqueeze(1) * rays.dir

        # B. Normal: Transform Element -> Global
        # Normals rotate with the element: N_global = N_local @ R.T
        global_normal = elem_normal @ self.transform.rot.T

        return t, hit_point, global_normal

    def inBounds(self, local_pos, surf_idx=None):
        """
        Checks if global points 'pos' [N, 3] are inside the volume.
        Must be implemented by children.
        """
        raise NotImplementedError

class CvxPolyhedron(Shape):

    def __init__(self, planes_list = None, transform=None):

        super().__init__(transform=transform)

        if planes_list is not None:
            planes_list = nn.ModuleList()
        else:
            self.surfaces.append(*planes_list)

    @property
    def PlaneRotMat(self):
        return torch.stack([plane.transform.rot[2, :] for plane in planes_list])

    @property
    def PlaneTrans(self):
        return torch.stack([plane.transform.trans for plane in planes_list])

    def inBounds(self, local_pos, surf_idx = None):

        local_pos_trans = local_pos[None, :, :] - self.PlaneTrans[:, None, :]
        local_z = torch.sum(self.PlaneRotMat[:, None, :] * local_pos_trans, dim=-1)

        in_bounds = local_z < 1e-4

        if surf_idx is not None:
            in_bounds[surf_idx, :] = True

        return torch.all(in_bounds, dim=0)


class Box(CvxPolyhedron):
    """
    A rectangular prism defined by 6 Plane surfaces.
    """

    def __init__(self, length, width, height, transform=None,
                 l_grad = False, w_grad = False, h_grad = False):

        super().__init__(transform=transform)

        planes = self._build_surfaces(length, width, height, l_grad, w_grad, h_grad)

        self.surfaces.append(*planes)


    @property
    def length(self):
        return surfaces[0].transform.trans[2] - surfaces[1].transform.trans[2]

    @property
    def width(self):
        return surfaces[2].transform.trans[0] - surfaces[3].transform.trans[0]

    @property
    def height(self):
        return surfaces[4].transform.trans[1] - surfaces[5].transform.trans[1]

    def _build_surfaces(self, length, width, height, l_grad, w_grad, h_grad):
        """
        Generates 6 infinite planes oriented to form the box faces.
        """

        # Helper to create a plane with specific position and rotation
        def make_plane(pos, rot_vec, optimize_flag):
            # rot_angles: [x_deg, y_deg, z_deg]
            # Convert to radians for RayTransform (assuming it takes radians or has a helper)
            # Here assuming RayTransform takes translation and rotation matrix/Euler
            # We construct a Transform that moves the canonical Plane (Z=0 facing +Z)
            # to the desired face.

            # Simple Euler to Matrix conversion (simplified for 90 deg steps)
            # Or pass angles if RayTransform supports it.
            # Assuming RayTransform(translation=..., rotation=...)

            # For simplicity, we define the 6 faces relative to Global.

            with torch.no_grad():
                trans_mask = (torch.abs(torch.tensor(pos))>1e-5)

            t = RayTransform(translation=torch.tensor(pos),
                             rotation=torch.tensor(rot_vec),
                             trans_grad=optimize_flag,
                             trans_mask=trans_mask)

            return Plane(transform=t)

        # 1. Front (+Z face)
        # Canonical Plane is XY facing +Z. We just translate it to +hz.
        surfaces.append(make_plane([0, 0, 0 + length/2], [0.0, 0.0, 0.0], l_grad))

        # 2. Back (-Z face)
        # Rotate 180 Y so it faces -Z. Translate to -hz.
        surfaces.append(make_plane([0, 0, 0 - length/2], [0.0, torch.pi, 0.0], l_grad))

        # 3. Right (+X face)
        # Rotate +90 Y. Normal (0,0,1) -> (1,0,0). Translate to +hx.
        surfaces.append(make_plane([0 + width/2, 0, 0], [0.0, -torch.pi / 2, 0.0], w_grad))

        # 4. Left (-X face)
        # Rotate -90 Y. Normal (0,0,1) -> (-1,0,0). Translate to -hx.
        surfaces.append(make_plane([0 - width/2, 0, 0], [0.0, torch.pi / 2, 0.0], w_grad))

        # 5. Top (+Y face)
        # Rotate -90 X. Normal (0,0,1) -> (0,1,0). Translate to +hy.
        surfaces.append(make_plane([0, 0 + height/2, 0], [torch.pi / 2, 0.0, 0.0], h_grad))

        # 6. Bottom (-Y face)
        # Rotate +90 X. Normal (0,0,1) -> (0,-1,0). Translate to -hy.
        surfaces.append(make_plane([0, 0 - height/2, 0], [-torch.pi / 2, 0.0, 0.0], h_grad))

        return surfaces

