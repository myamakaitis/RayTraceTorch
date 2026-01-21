import torch
import torch.nn.functional as F
import numpy as np

# Adjust imports based on your actual file structure
from ..rays.ray import Rays

# We assume these exist based on Architecture.md.
# If imports fail during dev, we can fallback to string checking.
try:
    from ..phys.phys_std import Reflect, RefractSnell, Block, Transmit
except ImportError:
    Reflect = RefractSnell = Block = Transmit = None


class Camera:
    """
    Pinhole Camera model for generating primary rays.
    """

    def __init__(self, position, look_at, up_vector, fov_deg, width, height, device='cpu'):
        self.device = device
        self.width = width
        self.height = height
        self.fov_deg = fov_deg

        # Camera Coordinate System
        self.origin = torch.tensor(position, dtype=torch.float32, device=device)
        target = torch.tensor(look_at, dtype=torch.float32, device=device)
        up = torch.tensor(up_vector, dtype=torch.float32, device=device)

        # Forward (Z), Right (X), Up (Y)
        # Note: We look down -Z in standard convention, but implementation varies.
        # Here we align: Forward = normalize(target - origin)
        self.forward = F.normalize(target - self.origin, dim=0)
        self.right = F.normalize(torch.linalg.cross(self.forward, up), dim=0)
        self.up_cam = torch.linalg.cross(self.right, self.forward)

    def generate_rays(self):
        """
        Generates a Rays object containing one ray per pixel.
        """
        # 1. Screen Coordinates
        aspect_ratio = self.width / self.height
        # Scale of the image plane at unit distance
        scale_y = torch.tan(torch.deg2rad(torch.tensor(self.fov_deg * 0.5)))
        scale_x = scale_y * aspect_ratio

        # Create Grid
        # linspace moves from -1 to 1
        y_grid = torch.linspace(scale_y, -scale_y, self.height, device=self.device)
        x_grid = torch.linspace(-scale_x, scale_x, self.width, device=self.device)

        # Meshgrid: Y is rows (0), X is cols (1)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')

        # Flatten
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        # 2. Ray Directions in World Space
        # dir = x * Right + y * Up + 1.0 * Forward
        # (assuming image plane is 1 unit away)

        dirs = (xx_flat.unsqueeze(1) * self.right +
                yy_flat.unsqueeze(1) * self.up_cam +
                self.forward)

        # Normalize handled by Rays __init__
        origins = self.origin.expand_as(dirs)

        return Rays(origins, dirs, device=self.device)


class OrbitCamera(Camera):
    """
    Standard CAD-style camera that orbits around a Pivot point.
    """

    def __init__(self, pivot=(0, 0, 0), **kwargs):
        super().__init__(**kwargs)
        self.pivot = torch.tensor(pivot, dtype=torch.float32, device=kwargs.get('device', 'cpu'))

        # Ensure we are pointing at the pivot initially
        self.update_view_matrix()

    def update_view_matrix(self):
        """Re-calculates Forward/Right/Up based on Orbit logic."""
        direction = self.pivot - self.origin
        dist = torch.norm(direction)
        if dist < 1e-3: return

        self.forward = direction / dist

        # Standard Orbit: Right is Cross(Forward, WorldUp)
        # This keeps the horizon level.
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.right = torch.cross(self.forward, world_up)

        # Handle gimbal lock (looking straight up/down)
        if torch.norm(self.right) < 1e-3:
            self.right = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        self.right = F.normalize(self.right, dim=0)
        self.up_cam = torch.cross(self.right, self.forward)
        self.up_cam = F.normalize(self.up_cam, dim=0)

    def orbit(self, d_yaw, d_pitch):
        """
        Rotates the camera Position around the Pivot point.
        """
        radius_vec = self.origin - self.pivot

        def rotate_vec(vec, axis, angle):
            c = torch.cos(angle)
            s = torch.sin(angle)
            return vec * c + torch.cross(axis, vec) * s + axis * torch.dot(axis, vec) * (1 - c)

        # 1. Yaw (Left/Right Drag) -> Rotate around World Up
        # We always yaw around strict World Up to maintain "Turntable" feel
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        radius_vec = rotate_vec(radius_vec, world_up, -d_yaw)

        # 2. Pitch (Up/Down Drag) -> Rotate around Local Right
        # Calculate temp right for the current frame
        # (This must mimic update_view_matrix logic to stay consistent)
        current_fwd = F.normalize(radius_vec, dim=0)  # Note: radius_vec points TO camera (opposite of forward)
        # Actually radius_vec = Origin - Pivot. Forward = Pivot - Origin.
        # So radius_vec is roughly -Forward.

        # Simple cross to find rotation axis
        if torch.abs(torch.dot(F.normalize(radius_vec, dim=0), world_up)) > 0.95:
            temp_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # Fallback X axis
        else:
            temp_axis = F.normalize(torch.cross(F.normalize(radius_vec, dim=0), world_up), dim=0)

        # Apply Pitch
        radius_vec = rotate_vec(radius_vec, temp_axis, d_pitch)

        self.origin = self.pivot + radius_vec
        self.update_view_matrix()

    def roll(self, angle):
        """
        Rolls the camera around its forward axis (Alt + Drag).
        Note: This breaks the 'steady horizon' of the orbit camera until the next orbit move resets it.
        """
        c = torch.cos(angle)
        s = torch.sin(angle)

        # Rotate Right and Up around Forward
        new_right = c * self.right - s * self.up_cam
        new_up = s * self.right + c * self.up_cam

        self.right = new_right
        self.up_cam = new_up

    def pan(self, dx, dy):
        """Moves both Camera and Pivot."""
        move_vec = (self.right * -dx) + (self.up_cam * dy)
        self.origin += move_vec
        self.pivot += move_vec

    def zoom(self, delta):
        radius_vec = self.origin - self.pivot
        dist = torch.norm(radius_vec)
        scale = 1.0 - delta * 0.1
        if dist * scale < 0.1: scale = 1.0
        self.origin = self.pivot + radius_vec * scale


class Renderer:
    """
    Handles visual interaction with the Scene.
    """

    def __init__(self, scene,
                 background_color=(1.0, 1.0, 1.0),
                 light_dir=(-0.5, 1.0, -1.0)):
        """
        Args:
            scene: The Scene object containing elements.
            background_color: RGB tuple (0-1).
        """
        self.scene = scene
        self.bg_color = torch.tensor(background_color, device=scene.map_to_element.device)

        ld = torch.as_tensor(light_dir, device=scene.map_to_element.device)
        self.light_dir = F.normalize(ld, dim=0)

    def render_3d(self, camera):
        """
        Performs a single-bounce ray cast from the camera and returns an image tensor.

        Returns:
            image: Tensor [H, W, 3] suitable for matplotlib or PyQt.
        """
        rays = camera.generate_rays()

        # 1. Intersect Scene (Find nearest hit)
        # We reuse the logic from Scene.step() manually to get normals/material info
        # without triggering the physics bounce updates.

        with torch.no_grad():
            # Collect 't' from all elements
            t_candidates = []
            for element in self.scene.elements:
                t_candidates.append(element.intersectTest(rays))

            # [N_rays, Total_Surfaces]
            t_matrix = torch.cat(t_candidates, dim=1)

            # Find closest hit
            min_t, global_hit_idx = torch.min(t_matrix, dim=1)

            # Identify hits vs misses
            hit_mask = min_t < float('inf')

            # Initialize Output Image (default to background)
            pixel_colors = self.bg_color.expand(rays.N, 3).clone()

            if not hit_mask.any():
                return pixel_colors.reshape(camera.height, camera.width, 3).cpu()

            # Retrieve ID mappings from Scene buffer
            winner_elem_ids = self.scene.map_to_element[global_hit_idx]
            winner_surf_ids = self.scene.map_to_surface[global_hit_idx]

            # Iterate over unique elements hit to compute Normals and Shading
            # (Vectorized per element to allow batch shape/normal calls)
            unique_elements = torch.unique(winner_elem_ids[hit_mask])

            for k_tensor in unique_elements:
                k = k_tensor.item()
                element = self.scene.elements[k]

                # Mask for Rays hitting Element K
                elem_mask = (winner_elem_ids == k) & hit_mask

                # Further break down by Surface Index on this element
                # (Because Element.forward usually requires specific surf_idx for shape calculation)
                surfs_on_elem = winner_surf_ids[elem_mask]
                unique_surfs = torch.unique(surfs_on_elem)

                for j_tensor in unique_surfs:
                    j = j_tensor.item()

                    # Rays hitting Element K, Surface J
                    specific_mask = elem_mask & (winner_surf_ids == j)

                    # Subset rays for calculation
                    # We need the Ray state to compute the Normal (via shape(rays))
                    subset_rays = rays.subset(specific_mask)

                    # Call Shape to get Normals (Global)
                    # shape.forward signature: returns _, hit_point, normal, local_hit
                    _, _, normal_global, _ = element.shape(subset_rays, j)

                    # Determine Physics Type for Coloring
                    phys_func = element.surface_functions[j]

                    # Compute Color
                    colors = self._compute_color(phys_func, normal_global)

                    # Scatter back to main pixel buffer
                    # We use masked_scatter or indexing
                    # pixel_colors[specific_mask] = colors
                    # Note: indexing with boolean mask in PyTorch flattens target
                    pixel_colors[specific_mask] = colors

        # Reshape to Image
        image = pixel_colors.reshape(camera.height, camera.width, 3)
        # Clamp just in case
        image = torch.clamp(image, 0.0, 1.0)

        return image.cpu()

    def _compute_color(self, phys_func, normals):
        """
        Applies coloring rules and Fixed Directional Shading.
        """
        N = normals.shape[0]
        base_color = torch.zeros(N, 3, device=normals.device)

        # --- 1. Base Color Determination ---
        def is_type(obj, cls_name):
            return cls_name in obj.__class__.__name__

        if is_type(phys_func, 'Reflect'):
            base_color[:] = torch.tensor([1.0, 0.6, 0.0], device=normals.device)
        elif is_type(phys_func, 'Block'):
            base_color[:] = torch.tensor([0.2, 0.2, 0.2], device=normals.device)
        elif is_type(phys_func, 'Transmit'):
            base_color[:] = torch.tensor([0.0, 1.0, 0.0], device=normals.device)
        elif is_type(phys_func, 'Refract'):
            # Index gradient logic
            n_val = getattr(phys_func, 'n2', 1.5)
            if isinstance(n_val, torch.Tensor): n_val = n_val.item()

            c_white = torch.tensor([0.9, 0.9, 0.9], device=normals.device)
            c_cyan = torch.tensor([0.0, 1.0, 1.0], device=normals.device)
            c_blue = torch.tensor([0.3, 0.6, 1.0], device=normals.device)
            c_navy = torch.tensor([0.0, 0.0, 0.5], device=normals.device)
            c_purp = torch.tensor([0.3, 0.0, 0.3], device=normals.device)

            if n_val <= 1.0:
                col = c_white
            elif n_val <= 1.3:
                col = torch.lerp(c_white, c_cyan, (n_val - 1.0) / 0.3)
            elif n_val <= 1.4:
                col = torch.lerp(c_cyan, c_blue, (n_val - 1.3) / 0.1)
            elif n_val <= 1.7:
                col = torch.lerp(c_blue, c_navy, (n_val - 1.4) / 0.3)
            else:
                col = torch.lerp(c_navy, c_purp, min((n_val - 1.7) / 0.3, 1.0))

            base_color[:] = col
        else:
            base_color[:] = torch.tensor([1.0, 0.0, 1.0], device=normals.device)

        # --- 2. Fixed Directional Shading ---
        # Lambertian Diffuse: Dot(Normal, LightVec)
        # We calculate the alignment between the surface normal and the light source.

        diffuse_intensity = torch.sum(normals * self.light_dir, dim=1)

        # Clamp to [0, 1] for purely physical light, or take Abs() for
        # "Two-Sided" lighting (useful to see inside shapes).
        # We'll use a soft mix:

        # 0.3 Ambient + 0.7 Diffuse
        # We use .abs() here so back-facing polygons are still visible (good for debugging lenses)
        shading = 0.3 + 0.7 * diffuse_intensity.abs()

        shading = shading.unsqueeze(1)
        return base_color * shading

    def scan_profile(self, target_element, axis='x', num_points=200, bounds=None):
        """
        Generates 2D cross-section data for a specific element.

        Args:
            target_element: The Element object to scan.
            axis: 'x' (XZ plane) or 'y' (YZ plane).
            bounds: Tuple (min, max) for the scan width. If None, inferred.

        Returns:
            List of dictionaries [{'h': [...], 'z': [...], 'surf_idx': int}, ...]
        """
        device = target_element.shape.transform.trans.device  # Heuristic to find device

        # 1. Determine Scan Extent
        if bounds:
            extent = bounds[1] - bounds[0]
            center = (bounds[0] + bounds[1]) / 2
        else:
            # Fallback heuristic
            width = 10.0
            # If shape has radius, use it
            # This depends on specific Shape impl.
            width = getattr(target_element.shape, 'radius', torch.tensor(10.0)).item() * 2.2
            center = 0.0
            extent = width

        # 2. Create Rays
        coords = torch.linspace(center - extent / 2, center + extent / 2, num_points, device=device)
        zeros = torch.zeros_like(coords)
        z_start = torch.full_like(coords, -100.0)  # Start far back

        if axis == 'x':
            # Vary X, Y=0, Z=-100, Dir=+Z
            origins = torch.stack([coords, zeros, z_start], dim=1)
        else:
            # Vary Y, X=0
            origins = torch.stack([zeros, coords, z_start], dim=1)

        directions = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_points, 3)

        rays = Rays(origins, directions, device=device)

        # 3. Intersect
        # We call intersectTest directly on the element
        with torch.no_grad():
            t_matrix = target_element.intersectTest(rays)

        # 4. Extract Profiles per Surface
        results = []
        num_surfaces = t_matrix.shape[1]

        for i in range(num_surfaces):
            t = t_matrix[:, i]
            mask = t < float('inf')

            if mask.any():
                # Recover global Z: z = origin_z + t (since dir_z=1)
                z_vals = (-100.0 + t[mask]).cpu().numpy()
                h_vals = coords[mask].cpu().numpy()
                results.append({
                    'surf_idx': i,
                    'h': h_vals,
                    'z': z_vals
                })

        return results