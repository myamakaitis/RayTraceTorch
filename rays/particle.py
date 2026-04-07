import torch
from typing import Optional, Union

from ..geom import RayTransformBundle
from .bundle import Bundle
from .ray import Rays


class LambertianSphere(Bundle):
    """
    Spherical Lambertian emitter.

    Positions are sampled uniformly on the sphere surface; directions are
    sampled from the cosine-weighted hemisphere around the outward surface
    normal at each sample point, so the bundle obeys Lambert's cosine law.

    Because the emission direction depends on the surface position (through
    the outward normal), this class overrides ``sample()`` rather than the
    individual ``sample_pos`` / ``sample_dir`` hooks.

    Args:
        radius: Sphere radius (metres).
    """

    def __init__(self, radius: float,
                 ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)
        self.radius = float(radius)

    def sample(self, N: int) -> Rays:
        # --- Uniform positions on sphere surface ---
        u1 = torch.rand(N, device=self.device, dtype=self.dtype)
        u2 = torch.rand(N, device=self.device, dtype=self.dtype)
        cos_t   = 2.0 * u1 - 1.0                               # uniform in [−1, 1]
        sin_t   = torch.sqrt((1.0 - cos_t ** 2).clamp(min=0.0))
        phi     = 2.0 * torch.pi * u2
        normals = torch.stack([sin_t * torch.cos(phi),
                               sin_t * torch.sin(phi),
                               cos_t], dim=1)                   # unit outward normals [N, 3]
        pos = normals * self.radius

        # --- Lambertian directions around each surface normal ---
        dirs = self._lambertian_around_normals(normals, N)

        # --- Apply bundle transform (position / orientation) ---
        pos_g, dir_g = self.transform.transform_(pos, dirs)
        return Rays.initialize(pos_g, dir_g, ray_id=self.ray_id,
                               device=self.device, dtype=self.dtype)

    def _lambertian_around_normals(self, normals: torch.Tensor, N: int) -> torch.Tensor:
        """
        Build a per-sample orthonormal basis from *normals* and rotate a
        cosine-weighted hemisphere sample into each local frame.

        Uses the Frisvad (2012) method for numerically stable ONB construction.
        """
        u1  = torch.rand(N, device=self.device, dtype=self.dtype)
        u2  = torch.rand(N, device=self.device, dtype=self.dtype)
        r   = torch.sqrt(u1)
        lph = 2.0 * torch.pi * u2
        lx  = r * torch.cos(lph)
        ly  = r * torch.sin(lph)
        lz  = torch.sqrt((1.0 - u1).clamp(min=0.0))

        # Frisvad ONB: build tangent and bitangent from the normal
        nz   = normals[:, 2]
        sign = torch.sign(nz + 1e-10)                          # avoids zero at nz == 0
        a    = -1.0 / (sign + nz)
        b    = normals[:, 0] * normals[:, 1] * a

        tangent   = torch.stack([1.0 + sign * normals[:, 0] ** 2 * a,
                                 sign * b,
                                 -sign * normals[:, 0]], dim=1)
        bitangent = torch.stack([b,
                                 sign + normals[:, 1] ** 2 * a,
                                 -normals[:, 1]], dim=1)

        return (lx.unsqueeze(1) * tangent
                + ly.unsqueeze(1) * bitangent
                + lz.unsqueeze(1) * normals)


class RayleighScatter(Bundle):
    """
    Point source whose angular emission follows the Rayleigh scattering phase
    function: p(θ) ∝ 1 + cos²θ.

    The scatter centre is at the local origin; use *transform* to position
    and orient the scatterer in world space.  The +Z axis of the local frame
    is the forward-scattering axis (θ = 0).

    Scattering angles are drawn from the exact analytic inverse CDF derived
    via Cardano's formula for the cubic:

        μ³ + 3μ − (8u − 4) = 0,    μ = cos θ,  u ~ Uniform(0, 1)

    Args:
        ray_id: Integer tag for ray bookkeeping.
    """

    def __init__(self, ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)

    def sample_dir(self, N: int) -> torch.Tensor:
        u = torch.rand(N, device=self.device, dtype=self.dtype)

        # Cardano solution for the depressed cubic μ³ + 3μ + (4 − 8u) = 0
        # Let half_q = 2 − 4u  →  A = ∛(√(half_q² + 1) − half_q)
        # μ = A − 1/A  (exact, one real root, no sign ambiguity)
        half_q    = 2.0 - 4.0 * u
        A         = (torch.sqrt(half_q ** 2 + 1.0) - half_q).pow(1.0 / 3.0)
        cos_theta = A - 1.0 / A                                # ∈ [−1, 1]
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=0.0))

        phi = 2.0 * torch.pi * torch.rand(N, device=self.device, dtype=self.dtype)
        x   = sin_theta * torch.cos(phi)
        y   = sin_theta * torch.sin(phi)
        return torch.stack([x, y, cos_theta], dim=1)


class MieScatter(Bundle):
    """
    Point source with Mie scattering angular emission.

    Mie theory requires solving the full electromagnetic boundary-value
    problem for a dielectric sphere, yielding the S₁/S₂ amplitude scattering
    functions.  A complete implementation needs a dedicated Mie solver such
    as *miepython* or *pymiecoated*.

    This class stores the scattering parameters so it can be serialised and
    displayed in the GUI.  ``sample_dir`` raises ``NotImplementedError``
    until a solver backend is wired in.

    Args:
        particle_size_nm:  Particle diameter (nm).
        wavelength_nm:     Illumination wavelength (nm).
        particle_ior:      Real part of the particle refractive index.
        environment_ior:   Refractive index of the surrounding medium.
    """

    def __init__(self, particle_size_nm: float, wavelength_nm: float,
                 particle_ior: float, environment_ior: float,
                 ray_id: int,
                 device: Union[str, torch.device] = 'cpu',
                 dtype: torch.dtype = torch.float32,
                 transform: Optional[Union[RayTransformBundle, None]] = None):
        super().__init__(transform=transform, ray_id=ray_id, device=device, dtype=dtype)
        self.particle_size_nm = particle_size_nm
        self.wavelength_nm    = wavelength_nm
        self.particle_ior     = particle_ior
        self.environment_ior  = environment_ior

    def sample_dir(self, N: int) -> torch.Tensor:
        raise NotImplementedError(
            "MieScatter requires a Mie-theory solver to compute the phase function CDF. "
            "Install a solver library (e.g., miepython) and override sample_dir()."
        )
