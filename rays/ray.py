import torch
import torch.nn.functional as F


class Rays:
    """
    Vectorized container for a batch of optical rays.

    Attributes:
        pos (Tensor): [N, 3] positions (x, y, z).
        dir (Tensor): [N, 3] normalized direction vectors.
        n (Tensor):   [N] refractive indices of the medium the rays are currently in.
        wavelength (Tensor): [N] wavelengths (optional, for dispersion).
        intensity (Tensor): [N] intensity/energy of the ray.
    """

    def __init__(self,
                 origins,
                 directions,
                 wavelengths=None,
                 intensities=None,
                 n_medium=1.0,
                 ray_id=0,
                 device='cpu'):
        """
        Args:
            origins: Array-like [N, 3] or [3] starting points.
            directions: Array-like [N, 3] or [3] direction vectors.
            wavelengths: Array-like [N] (optional).
            intensities: Array-like [N] (optional).
            n_medium: Initial refractive index (float).
            device: 'cpu' or 'cuda'.
        """
        self.device = device

        # Ensure inputs are Float Tensors [N, 3]
        self.pos = torch.as_tensor(origins, dtype=torch.float32, device=device)
        self.dir = torch.as_tensor(directions, dtype=torch.float32, device=device)

        # Handle single ray input broadcasting
        if self.pos.ndim == 1: self.pos = self.pos.unsqueeze(0)
        if self.dir.ndim == 1: self.dir = self.dir.unsqueeze(0)

        # Validation: Normalize Direction Vectors
        # We use F.normalize to ensure this step is differentiable.
        self.dir = F.normalize(self.dir, p=2, dim=1)

        self.N = self.pos.shape[0]

        # Initialize Metadata
        # Refractive index is tracked per ray
        self.n = torch.full((self.N,), n_medium, dtype=torch.float32, device=device)

        # Optical properties
        if wavelengths is not None:
            self.wavelength = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)

        if intensities is not None:
            self.intensity = torch.as_tensor(intensities, dtype=torch.float32, device=device)
        else:
            self.intensity = torch.ones(self.N, device=device)

        self.id = torch.full((self.N,), ray_id, dtype=torch.long, device=device)

    def __getitem__(self, key):
        """
        Returns a new Rays object containing only the rays specified by 'key'.
        'key' can be an integer, a slice, or a boolean tensor mask.
        """
        # 1. Slice the core geometric data
        # PyTorch handles the indexing logic (int, slice, or mask)
        new_pos = self.pos[key]
        new_dir = self.dir[key]

        # 2. Create a new 'bare' instance
        # We invoke __init__ to handle shape validation (1D vs 2D) and normalization.
        # We do NOT pass auxiliary data (n, intensity) to __init__ because
        # __init__ assumes they are scalar/uniform, but our slice might contain mixed values.
        subset = Rays(new_pos, new_dir, device=self.device)

        # 3. Overwrite auxiliary attributes with the sliced data
        # We must manually slice these to preserve per-ray variations (like mixed refractive indices)
        subset.n = self.n[key]
        subset.active = self.active[key]
        subset.id = self.id[key]

        # Intensity is always initialized in your __init__, so we safe to slice
        subset.intensity = self.intensity[key]

        # Wavelength is optional in your __init__, check existence before slicing
        if hasattr(self, 'wavelength'):
            subset.wavelength = self.wavelength[key]

        return subset

    def to(self, device):
        """Moves all internal tensors to the specified device."""
        self.device = device
        self.pos = self.pos.to(device)
        self.dir = self.dir.to(device)
        self.n = self.n.to(device)
        self.active = self.active.to(device)
        if hasattr(self, 'wavelength'):
            self.wavelength = self.wavelength.to(device)
        self.intensity = self.intensity.to(device)
        return self

    def update(self, new_pos, new_dir):
        """
        Updates ray state.
        Note: STRICT avoidance of in-place operations (+=) to preserve Autograd.
        """
        # Only update active rays, but in a vectorized way using torch.where implies
        # keeping old values for inactive ones.
        # However, for simplicity in batch optics, we usually update everything
        # and just ignore the 'inactive' ones during loss calculation.

        self.pos = new_pos
        self.dir = F.normalize(new_dir, p=2, dim=1)

    def __repr__(self):
        return f"<Rays N={self.N}, device={self.device}>"

    @staticmethod
    def merge(ray_list):
        """
        Combines a list of Rays objects into a single Rays object.
        """
        if not ray_list:
            raise ValueError("Cannot merge empty ray list")

        # Extract attributes from all objects
        all_pos = torch.cat([r.pos for r in ray_list], dim=0)
        all_dir = torch.cat([r.dir for r in ray_list], dim=0)
        all_n = torch.cat([r.n for r in ray_list], dim=0)
        all_wave = torch.cat([r.wavelength for r in ray_list], dim=0)
        all_int = torch.cat([r.intensity for r in ray_list], dim=0)
        all_ids = torch.cat([r.id for r in ray_list], dim=0)

        # Create a new instance (using dummy init)
        merged = Rays(all_pos, all_dir, device=ray_list[0].device)

        # Overwrite with concatenated data
        merged.n = all_n
        merged.wavelength = all_wave
        merged.intensity = all_int
        merged.id = all_ids

        return merged

    def with_coords(self, new_pos, new_dir):
        """
        Returns a new Rays object with updated geometry but shared metadata.
        Uses __new__ to skip initialization overhead.
        """
        # Create bare instance (skips __init__)
        new_rays = Rays.__new__(Rays)

        # Set new geometry
        new_rays.device = self.device
        new_rays.pos = new_pos
        new_rays.dir = new_dir

        new_rays.N = new_pos.shape[0]

        # Copy References to Metadata (No memory allocation)
        new_rays.n = self.n
        new_rays.intensity = self.intensity
        new_rays.id = self.id

        # Handle optional attributes
        if hasattr(self, 'wavelength'):
            new_rays.wavelength = self.wavelength

        return new_rays

    def __repr__(self):
        unique_ids = self.id.unique().tolist()
        return f"<Rays N={self.N}, IDs={unique_ids}, Device={self.device}>"


class Paths(Rays):
    """
    Child of Rays that automatically logs position history for 3D visualization.
    """

    def __init__(self, origins, directions, **kwargs):
        super().__init__(origins, directions, **kwargs)

        # History is a list of tensors to avoid constant concatenation overhead
        # We detach() history for plotting to save memory, as we usually
        # don't need gradients for the visualization path itself.
        self.history = [self.pos.clone().detach()]

    def update(self, new_pos, new_dir):
        """
        Updates position and appends to history.
        """
        # Call the parent update to handle state change
        super().update(new_pos, new_dir)

        # Append new position to history (detached for viz efficiency)
        self.history.append(self.pos.clone().detach())

    def get_history(self):
        """Returns the ray paths as a Tensor of shape [Steps, N, 3]"""
        return torch.stack(self.history, dim=0)