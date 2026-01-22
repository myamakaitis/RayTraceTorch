import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from typing import Optional


@tensorclass
class Rays:
    # 1. Define fields with types.
    # The backend handles storage, so no self.pos = ... needed in init.
    pos: torch.Tensor
    dir: torch.Tensor
    intensity: torch.Tensor
    id: torch.Tensor
    # Handling optional: It's best to initialize with a default (e.g., NaN or 0)
    # rather than None to keep batching semantics consistent.
    wavelength: torch.Tensor

    # 2. Custom Post-Init Logic
    # @tensorclass generates a __init__ for you, but __post_init__ allows
    # you to run your normalization and validation logic.
    def __post_init__(self):
        # Normalize directions automatically upon creation
        # We modify the tensor in-place within the tensordict container safely
        self.dir = F.normalize(self.dir, p=2, dim=1)

    # 3. Your Custom Methods
    # You can keep your custom methods exactly as they are.
    def scatter_update(self, mask, new_pos, new_dir, intensity_mod):
        """
        Differentiable update using standard PyTorch operations.
        """
        # Tensorclass exposes fields as standard tensors, so your logic works as is.
        idx = (mask,)
        self.pos = self.pos.index_put(idx, new_pos)
        self.dir = self.dir.index_put(idx, new_dir)

        current_subset_intensity = self.intensity[mask]
        new_intensity = current_subset_intensity * intensity_mod
        self.intensity = self.intensity.index_put(idx, new_intensity)

    @classmethod
    def initialize(cls, origins, directions, wavelengths=None, intensities=None, ray_id=0, device='cpu'):
        """
        A factory method to handle your specific initialization logic
        (broadcasting, default values) that acts as your old __init__.
        """
        origins = torch.as_tensor(origins, device=device, dtype=torch.float32)
        directions = torch.as_tensor(directions, device=device, dtype=torch.float32)

        # Handle broadcasting
        if origins.ndim == 1: origins = origins.unsqueeze(0)
        if directions.ndim == 1: directions = directions.unsqueeze(0)

        N = origins.shape[0]

        # Handle defaults
        if intensities is None:
            intensities = torch.ones(N, device=device, dtype=torch.float32)
        else:
            intensities = torch.as_tensor(intensities, device=device, dtype=torch.float32)

        if wavelengths is None:
            # Create dummy wavelengths to satisfy tensor structure (e.g., zeros)
            wavelengths = torch.zeros(N, device=device, dtype=torch.float32)
        else:
            wavelengths = torch.as_tensor(wavelengths, device=device, dtype=torch.float32)

        ids = torch.full((N,), ray_id, dtype=torch.long, device=device)

        # Instantiate the tensorclass
        # batch_size must be provided
        return cls(
            pos=origins,
            dir=directions,
            intensity=intensities,
            id=ids,
            wavelength=wavelengths,
            batch_size=[N]
        )


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



