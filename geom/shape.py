import torch
from .primitives import Plane
from .transform import RayTransform, eulerToRotMat

class Shape:
    """
    Base class for 3D volumetric shapes.
    Defines the volume 'inside' a set of boundaries.
    """
    def __init__(self, transform = None, device='cpu'):
        self.device = device
        self.surfaces = [] # List of Surface objects defining the boundary

        if transform is None:
            self.transform = RayTransform(device=device)
        else:
            self.transform = transform

    def to(self, device):
        self.device = device
        # Move any internal tensors (radius, etc.) in child classes
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))

        for s in self.surfaces:
            s.to(device)

        self.transform.to(device)

        return self

    def intersectTest(self, rays):
        """
        Calculates intersection distances (t) for ALL contained surfaces.
        Calls self.inside() to filter physically invalid hits (clipping).

        Args:
            rays (Rays): Batch of N rays (Global Coordinates).

        Returns:
            t_matrix (Tensor): [N, K] where K is len(self.surfaces).
        """
        if not self.surfaces:
            return torch.zeros((rays.N, 0), device=self.device)

        # 1. Transform Global -> Local (Element Space)
        local_pos, local_dir = self.transform.Transform(rays)
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
            t_safe = torch.where(hit_mask, t, torch.zeros_like(t))
            hit_point = local_pos + t_safe.unsqueeze(1) * local_dir

            # Check if the hit point is valid for this specific surface
            # We pass the surface index so the shape knows which boundary logic to apply
            is_valid = self.inside(hit_point, i)

            # Combine: Must hit the math surface AND be inside the physical bounds
            valid_mask = hit_mask & is_valid

            t_final = torch.where(valid_mask, t, torch.full_like(t, float('inf')))
            t_list.append(t_final)

        # Stack into a matrix [N, Number_of_Surfaces]
        return torch.stack(t_list, dim=1)

    def intersect_surface(self, rays, surf_idx):
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
        local_pos, local_dir = self.transform.Transform(rays)
        local_rays = rays.with_coords(local_pos, local_dir)

        # 2. Call child intersection (Returns results in Element Frame)
        t, elem_hit, elem_normal = self.surfaces[surf_idx].intersect(local_rays)

        # 3. Transform Results back to Global Frame

        # A. Hit Point: Recompute in global to ensure graph consistency
        hit_point = rays.pos + t.unsqueeze(1) * rays.dir

        # B. Normal: Transform Element -> Global
        # Normals rotate with the element: N_global = N_local @ R.T
        global_normal = elem_normal @ self.transform.rot

        return t, hit_point, global_normal

    def inside(self, local_pos, surf_idx=None):
        """
        Checks if global points 'pos' [N, 3] are inside the volume.
        Must be implemented by children.
        """
        raise NotImplementedError



class Box(Shape):
    """
    A rectangular prism defined by 6 Plane surfaces.
    """

    def __init__(self, center, length, width, height, transform=None, device='cpu'):
        super().__init__(transform=transform, device=device)

        self.center = center.to(device)
        self.length = length.to(device)
        self.width = width.to(device)
        self.height = height.to(device)

        self.half_size = torch.tensor([self.width/2, self.height/2, self.length/2], device=self.device)

        # Automatically build the 6 boundary planes
        self._build_surfaces()

    def inside(self, local_pos, surf_idx=None):
        # Check if point is within the box volume
        # (Used to clip the infinite planes)
        # Shift to box center
        rel_pos = local_pos - self.center

        # Allow small epsilon for floating point hits exactly on the face
        eps = 1e-4
        inside_mask = torch.all(torch.abs(rel_pos) <= (self.half_size + eps), dim=1)
        return inside_mask

    def _build_surfaces(self):
        """
        Generates 6 infinite planes oriented to form the box faces.
        Note: The 'Plane' primitive is infinite. The logic to treat this as a
        finite box usually relies on:
        1. intersectTest returns t for all 6 infinite planes.
        2. Logic checks if the hit point on plane i is within the bounds of the other axes.
           (OR we simply use this for constructive solid geometry).
        """
        cx, cy, cz = self.center

        # Helper to create a plane with specific position and rotation
        def make_plane(pos, rot_angles):
            # rot_angles: [x_deg, y_deg, z_deg]
            # Convert to radians for RayTransform (assuming it takes radians or has a helper)
            # Here assuming RayTransform takes translation and rotation matrix/Euler
            # We construct a Transform that moves the canonical Plane (Z=0 facing +Z)
            # to the desired face.

            # Simple Euler to Matrix conversion (simplified for 90 deg steps)
            # Or pass angles if RayTransform supports it.
            # Assuming RayTransform(translation=..., rotation=...)

            # For simplicity, we define the 6 faces relative to Global.
            Rmat = eulerToRotMat(torch.tensor(rot_angles)).squeeze()

            t = RayTransform(translation=torch.tensor(pos, device=self.device),
                             rotation=torch.tensor(Rmat, device=self.device))
            return Plane(transform=t, device=self.device)

        # 1. Front (+Z face)
        # Canonical Plane is XY facing +Z. We just translate it to +hz.
        self.surfaces.append(make_plane([cx, cy, cz + self.length/2], [0.0, 0.0, 0.0]))

        # 2. Back (-Z face)
        # Rotate 180 Y so it faces -Z. Translate to -hz.
        self.surfaces.append(make_plane([cx, cy, cz - self.length/2], [0.0, torch.pi, 0.0]))

        # 3. Right (+X face)
        # Rotate +90 Y. Normal (0,0,1) -> (1,0,0). Translate to +hx.
        self.surfaces.append(make_plane([cx + self.width/2, cy, cz], [0.0, torch.pi / 2, 0.0]))

        # 4. Left (-X face)
        # Rotate -90 Y. Normal (0,0,1) -> (-1,0,0). Translate to -hx.
        self.surfaces.append(make_plane([cx - self.width/2, cy, cz], [0.0, -torch.pi / 2, 0.0]))

        # 5. Top (+Y face)
        # Rotate -90 X. Normal (0,0,1) -> (0,1,0). Translate to +hy.
        self.surfaces.append(make_plane([cx, cy + self.height/2, cz], [-torch.pi / 2, 0.0, 0.0]))

        # 6. Bottom (-Y face)
        # Rotate +90 X. Normal (0,0,1) -> (0,-1,0). Translate to -hy.
        self.surfaces.append(make_plane([cx, cy - self.height/2, cz], [torch.pi / 2, 0.0, 0.0]))