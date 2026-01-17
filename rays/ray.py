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
                 ray_id=0,
                 device='cpu',
                 dtype=torch.float32):
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
        self.dtype=dtype

        # Ensure inputs are Float Tensors [N, 3]
        self.pos = torch.as_tensor(origins, dtype=dtype, device=device)
        self.dir = torch.as_tensor(directions, dtype=dtype, device=device)

        # Handle single ray input broadcasting
        if self.pos.ndim == 1: self.pos = self.pos.unsqueeze(0)
        if self.dir.ndim == 1: self.dir = self.dir.unsqueeze(0)

        # Validation: Normalize Direction Vectors
        # We use F.normalize to ensure this step is differentiable.
        self.dir = F.normalize(self.dir, p=2, dim=1)

        self.N = self.pos.shape[0]

        # Optical properties
        if wavelengths is not None:
            self.wavelength = torch.as_tensor(wavelengths, dtype=dtype, device=device)

        if intensities is not None:
            self.intensity = torch.as_tensor(intensities, dtype=dtype, device=device)
        else:
            self.intensity = torch.ones(self.N, device=device)

        self.id = torch.full((self.N,), ray_id, dtype=torch.long, device=device)

    def __getitem__(self, key):
        """
        Fast Slicing: Bypasses __init__ using __new__ for ~10x speedup.
        """
        # 1. Create a blank instance (skips __init__)
        subset = Rays.__new__(Rays)
        subset.device = self.device
        subset.dtype = self.dtype

        # 3. Slice Tensors
        # PyTorch slicing is very fast; the overhead was in the class instantiation
        subset.pos = self.pos[key]
        subset.dir = self.dir[key]
        subset.id = self.id[key]
        subset.intensity = self.intensity[key]

        # Update N based on the result of the slice
        subset.N = subset.pos.shape[0]

        # Handle Optional Wavelength
        if self.wavelength is not None:
            subset.wavelength = self.wavelength[key]
        else:
            subset.wavelength = None

        return subset

    def to(self, device):
        """Moves all internal tensors to the specified device."""
        self.device = device
        self.pos = self.pos.to(device)
        self.dir = self.dir.to(device)
        if hasattr(self, 'wavelength'):
            self.wavelength = self.wavelength.to(device)
        self.intensity = self.intensity.to(device)
        return self

    def update(self, new_pos, new_dir, intensity_mult):
        """
        Updates ray state.
        Note: STRICT avoidance of in-place operations (+=) to preserve Autograd.
        """
        # Only update active rays, but in a vectorized way using torch.where implies
        # keeping old values for inactive ones.
        # However, for simplicity in batch optics, we usually update everything
        # and just ignore the 'inactive' ones during loss calculation.

        self.pos = new_pos
        self.dir = new_dir

        self.intensity = self.intensity*intensity_mult

    def scatter_update(self, mask, new_pos, new_dir, intensity_mod):
        """
        Differentiable, Graph-Safe Update.

        Args:
            mask: Boolean tensor [N] indicating which rays to update.
            new_pos: Tensor [M, 3] new positions for the active rays.
            new_dir: Tensor [M, 3] new directions for the active rays.
            intensity_mod: Tensor [M] or scalar multiplier for intensity.
        """

        # Format mask for index_put (expects tuple of indices or boolean masks)
        idx = (mask,)

        # Update Position and Direction
        self.pos = self.pos.index_put(idx, new_pos)
        self.dir = self.dir.index_put(idx, new_dir)

        # Update Intensity (Multiplicative)
        # We need to slice the current intensity, multiply, and put it back
        current_subset_intensity = self.intensity[mask]
        new_intensity = current_subset_intensity * intensity_mod
        self.intensity = self.intensity.index_put(idx, new_intensity)

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
        all_wave = torch.cat([r.wavelength for r in ray_list], dim=0)
        all_int = torch.cat([r.intensity for r in ray_list], dim=0)
        all_ids = torch.cat([r.id for r in ray_list], dim=0)

        # Create a new instance (using dummy init)
        merged = Rays(all_pos, all_dir, device=ray_list[0].device)

        # Overwrite with concatenated data
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # History is a list of tensors to avoid constant concatenation overhead
        # We detach() history for plotting to save memory, as we usually
        # don't need gradients for the visualization path itself.
        self.pos_hist = [self.pos.clone().detach()]
        self.dir_hist = [self.dir.clone().detach()]
        self.intensity_hist = [self.intensity.clone().detach()]

    def update(self, new_pos, new_dir, intensity_mult):
        """
        Updates position and appends to history.
        """
        # Call the parent update to handle state change
        super().update(new_pos, new_dir, intensity_mult)

        # Append new position to history (detached for viz efficiency)
        self.pos_hist.append(self.pos.clone().detach())
        self.dir_hist.append(self.dir.clone().detach())
        self.intensity_hist.append(self.intensity.clone().detach())

    def get_pos_hist(self):
        """Returns the ray paths as a Tensor of shape [Steps, N, 3]"""
        return torch.stack(self.pos_hist)

    def get_dir_hist(self):
        return torch.stack(self.dir_hist)

    def get_intensity_hist(self):
        return torch.stack(self.intensity_hist)

    def test_history_consistency(self):

        with torch.no_grad():
            pos_hist = self.get_pos_hist()

            dir_finite_diff = pos_hist[1:, :] - pos_hist[:-1, :]
            dir_finite_diff = F.normalize(dir_finite_diff, p=2, dim=2)

            dir_hist = self.get_dir_hist()[:-1]

        return torch.allclose(dir_finite_diff, dir_hist)



