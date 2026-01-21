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


class Renderer:
    """
    Handles visual interaction with the Scene.
    """

    def __init__(self, scene, background_color=(1.0, 1.0, 1.0)):
        """
        Args:
            scene: The Scene object containing elements.
            background_color: RGB tuple (0-1).
        """
        self.scene = scene
        self.bg_color = torch.tensor(background_color, device=scene.map_to_element.device)

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
                    colors = self._compute_color(phys_func, subset_rays, normal_global)

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

    def _compute_color(self, phys_func, rays, normals):
        """
        Applies coloring rules and shading based on normal incidence.
        """
        N = rays.pos.shape[0]
        base_color = torch.zeros(N, 3, device=rays.device)

        # --- 1. Base Color Determination ---

        # Helper to check type safely (handling potential string-based fallbacks or classes)
        def is_type(obj, cls_name):
            if hasattr(obj, '__class__'):
                return cls_name in obj.__class__.__name__
            return False

        # Rule: Reflect -> Bright Orange
        if is_type(phys_func, 'Reflect'):
            base_color[:] = torch.tensor([1.0, 0.6, 0.0], device=rays.device)

        # Rule: Block -> Dark Gray
        elif is_type(phys_func, 'Block'):
            base_color[:] = torch.tensor([0.2, 0.2, 0.2], device=rays.device)

        # Rule: Transmit -> Green
        elif is_type(phys_func, 'Transmit'):
            base_color[:] = torch.tensor([0.0, 1.0, 0.0], device=rays.device)

        # Rule: Refract -> Index Gradient
        elif is_type(phys_func, 'RefractSnell') or is_type(phys_func, "RefractFresnel"):
            # Extract index 'n'
            # Assuming RefractSnell stores 'n_out' or similar,
            # OR we use the ray's current n vs the medium.
            # For visualization, let's assume the function has an attribute `n2` (destination index)
            # If not, default to 1.5
            n_val1 = getattr(phys_func, 'ior_in', 1.5)
            n_val2 = getattr(phys_func, 'ior_out', 1.5)

            n_val = torch.max(n_val1, n_val2)

            # If n_val is a tensor, we might need logic, but usually it's scalar per surface
            if isinstance(n_val, torch.Tensor): n_val = n_val.item()

            # Map n 1.0 -> 2.0
            # 1.0 (White) -> 1.3 (Cyan) -> 1.7 (Navy) -> 2.0 (Purple)
            c_white = torch.tensor([0.9, 0.9, 0.9], device=rays.device)
            c_cyan = torch.tensor([0.0, 1.0, 1.0], device=rays.device)
            c_blue = torch.tensor([0.3, 0.6, 1.0], device=rays.device)  # Light Blue
            c_navy = torch.tensor([0.0, 0.0, 0.5], device=rays.device)
            c_purp = torch.tensor([0.3, 0.0, 0.3], device=rays.device)  # Dark Purple

            if n_val <= 1.0:
                col = c_white
            elif n_val <= 1.3:
                t = (n_val - 1.0) / 0.3
                col = torch.lerp(c_white, c_cyan, t)
            elif n_val <= 1.4:
                t = (n_val - 1.3) / 0.1
                col = torch.lerp(c_cyan, c_blue, t)
            elif n_val <= 1.7:
                t = (n_val - 1.4) / 0.3
                col = torch.lerp(c_blue, c_navy, t)
            else:
                t = min((n_val - 1.7) / 0.3, 1.0)
                col = torch.lerp(c_navy, c_purp, t)

            base_color[:] = col

        else:
            # Fallback for unknown
            base_color[:] = torch.tensor([1.0, 0.0, 1.0], device=rays.device)  # Magenta

        # --- 2. Shading ---
        # Dot product of Ray Direction and Normal
        # Since Rays point FROM camera TO object, and Normal points OUT,
        # they are opposed. We want alignment intensity.
        # factor = | dot(D, N) |

        dot = torch.sum(rays.dir * normals, dim=1).abs()

        # Modulate: Brightest when perpendicular (dot=1), Darker when glancing (dot=0)
        # We add an ambient term so it's not pitch black at edges
        # Brightness = 0.3 + 0.7 * dot
        shading = 0.3 + 0.7 * dot
        shading = shading.unsqueeze(1)  # Broadcast to [N, 1]

        final_color = base_color * shading
        return final_color

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