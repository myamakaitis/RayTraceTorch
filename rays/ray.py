import torch
import torch.nn.functional as F
from tensordict import tensorclass
from typing import Optional, Union


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
    def initialize(cls, origins, directions, wavelengths=None, intensities=None,
                   ray_id: int = 0, device: Union[str, torch.device] = 'cpu',
                   dtype: torch.dtype = torch.float32):
        """
        A factory method to handle your specific initialization logic
        (broadcasting, default values) that acts as your old __init__.
        """
        origins = torch.as_tensor(origins, device=device, dtype=dtype)
        directions = torch.as_tensor(directions, device=device, dtype=dtype)

        # Handle broadcasting
        if origins.ndim == 1: origins = origins.unsqueeze(0)
        if directions.ndim == 1: directions = directions.unsqueeze(0)

        N = origins.shape[0]

        # Handle defaults
        if intensities is None:
            intensities = torch.ones(N, device=device, dtype=dtype)
        else:
            intensities = torch.as_tensor(intensities, device=device, dtype=dtype)

        if wavelengths is None:
            # Create dummy wavelengths to satisfy tensor structure (e.g., zeros)
            wavelengths = torch.zeros(N, device=device, dtype=dtype)
        else:
            wavelengths = torch.as_tensor(wavelengths, device=device, dtype=dtype)

        ids = torch.full((N,), ray_id, dtype=torch.int8, device=device)

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

    def with_coords(self, new_pos, new_dir):
        """
        Returns a new Rays object with updated geometry but shared metadata.
        """
        # We assume the batch size (Number of Rays) hasn't changed.
        # If new_pos has a different N, this will raise a shape error (Safety!).
        return Rays(
            pos=new_pos,
            dir=new_dir,
            intensity=self.intensity,  # Passes reference (Zero Memory Overhead)
            id=self.id,  # Passes reference
            wavelength=self.wavelength,  # Passes reference
            batch_size=self.batch_size  # Required by tensorclass
        )


class Paths:
    """
    A thin proxy around a ``Rays`` tensorclass that records the world-space
    position of every ray after each ``scatter_update()`` call.

    **Use only for interactive visualization** — never during optimisation,
    where the memory and detach overhead is undesirable.  The scene's
    ``ray_cast`` / ``step`` machinery is fully transparent to this wrapper:
    attribute access is forwarded directly to the underlying ``Rays`` object
    and ``__getitem__`` (boolean masking) returns a plain ``Rays`` subset so
    element physics code never touches the wrapper.

    Usage
    -----
    paths = Paths(scene.rays)
    scene.rays = paths
    for _ in range(N):
        scene.step()                # scatter_update records each bounce
    history = paths.get_history()   # list of [N_rays, 3] CPU tensors
    scene.rays = paths.unwrap()     # restore plain Rays when done
    """

    def __init__(self, rays: 'Rays'):
        self._rays = rays
        # Snapshot initial positions before any bounce
        self._history: list[torch.Tensor] = [
            rays.pos.clone().detach().cpu()
        ]

    # ------------------------------------------------------------------
    # Tensor attribute proxy — forwards reads and writes to _rays
    # ------------------------------------------------------------------

    @property
    def pos(self) -> torch.Tensor:
        return self._rays.pos

    @pos.setter
    def pos(self, v: torch.Tensor):
        self._rays.pos = v

    @property
    def dir(self) -> torch.Tensor:
        return self._rays.dir

    @dir.setter
    def dir(self, v: torch.Tensor):
        self._rays.dir = v

    @property
    def intensity(self) -> torch.Tensor:
        return self._rays.intensity

    @intensity.setter
    def intensity(self, v: torch.Tensor):
        self._rays.intensity = v

    @property
    def id(self) -> torch.Tensor:
        return self._rays.id

    @property
    def wavelength(self) -> torch.Tensor:
        return self._rays.wavelength

    @property
    def batch_size(self):
        return self._rays.batch_size

    def __getitem__(self, idx):
        """Boolean / integer masking — returns a plain Rays subset."""
        return self._rays[idx]

    # ------------------------------------------------------------------
    # History-recording scatter_update
    # ------------------------------------------------------------------

    def scatter_update(self, mask: torch.Tensor,
                       new_pos: torch.Tensor,
                       new_dir: torch.Tensor,
                       intensity_mod: torch.Tensor):
        """
        Delegate to ``Rays.scatter_update``, then snapshot the updated
        positions into the history list (detached, on CPU).
        """
        self._rays.scatter_update(mask, new_pos, new_dir, intensity_mod)
        self._history.append(self._rays.pos.clone().detach().cpu())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def unwrap(self) -> 'Rays':
        """Return the underlying ``Rays`` tensorclass."""
        return self._rays

    def get_history(self) -> list:
        """
        Return the recorded position history as a list of ``[N_rays, 3]``
        CPU float tensors — one snapshot per simulation step, including the
        initial positions at index 0.
        """
        return self._history

    def to(self, device):
        """Move the underlying rays to *device* and return self."""
        self._rays = self._rays.to(device)
        return self
