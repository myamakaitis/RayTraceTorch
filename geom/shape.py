import torch
from .primitives import Plane
from .transform import RayTransform

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

        Args:
            rays (Rays): Batch of N rays.

        Returns:
            t_matrix (Tensor): [N, K] where K is len(self.surfaces).
                               Contains 't' distance for each ray-surface pair.
        """
        if not self.surfaces:
            return torch.zeros((rays.N, 0), device=self.device)

        t_list = []
        # Iterate over all surfaces in the shape
        for surf in self.surfaces:
            # Calls the detached/fast solver of the surface
            t_val = surf.intersectTest(rays)
            t_list.append(t_val)

        # Stack into a matrix [N, Number_of_Surfaces]
        return torch.stack(t_list, dim=1)

    def intersect_surface(self, rays, surf_idx):
        """
        Calculates the full differentiable intersection for a SPECIFIC surface index.
        Used after logic determines which surface was actually hit.

        Args:
            rays (Rays): Batch of N rays.
            surf_idx (int): Index of the surface in self.surfaces.

        Returns:
            t, hit_point, normal (Standard Surface.intersect output)
        """
        return self.surfaces[surf_idx].intersect(rays)

    def inside(self, pos):
        """
        Checks if global points 'pos' [N, 3] are inside the volume.
        Must be implemented by children.
        """
        raise NotImplementedError



class Box(Shape):
    """
    A rectangular prism defined by 6 Plane surfaces.
    """

    def __init__(self, center, size, device='cpu'):
        super().__init__(device)
        self.center = torch.as_tensor(center, dtype=torch.float32, device=device)
        self.size = torch.as_tensor(size, dtype=torch.float32, device=device)
        self.half_size = self.size / 2.0

        # Automatically build the 6 boundary planes
        self._build_surfaces()

    def contains(self, pos):
        local_pos = pos - self.center
        inside_mask = torch.all(torch.abs(local_pos) <= self.half_size, dim=1)
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
        hx, hy, hz = self.half_size

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
            t = RayTransform(translation=torch.tensor(pos, device=self.device),
                             rotation=torch.tensor(rot_angles, device=self.device))
            return Plane(transform=t, device=self.device)

        # 1. Front (+Z face)
        # Canonical Plane is XY facing +Z. We just translate it to +hz.
        self.surfaces.append(make_plane([cx, cy, cz + hz], [0.0, 0.0, 0.0]))

        # 2. Back (-Z face)
        # Rotate 180 Y so it faces -Z. Translate to -hz.
        self.surfaces.append(make_plane([cx, cy, cz - hz], [0.0, math.pi, 0.0]))

        # 3. Right (+X face)
        # Rotate +90 Y. Normal (0,0,1) -> (1,0,0). Translate to +hx.
        self.surfaces.append(make_plane([cx + hx, cy, cz], [0.0, math.pi / 2, 0.0]))

        # 4. Left (-X face)
        # Rotate -90 Y. Normal (0,0,1) -> (-1,0,0). Translate to -hx.
        self.surfaces.append(make_plane([cx - hx, cy, cz], [0.0, -math.pi / 2, 0.0]))

        # 5. Top (+Y face)
        # Rotate -90 X. Normal (0,0,1) -> (0,1,0). Translate to +hy.
        self.surfaces.append(make_plane([cx, cy + hy, cz], [-math.pi / 2, 0.0, 0.0]))

        # 6. Bottom (-Y face)
        # Rotate +90 X. Normal (0,0,1) -> (0,-1,0). Translate to -hy.
        self.surfaces.append(make_plane([cx, cy - hy, cz], [math.pi / 2, 0.0, 0.0]))