import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geom.transform import RayTransform

from typing import Optional, Union

class SurfaceFunction(nn.Module):
    """
    Base class for optical surface physics.
    Acts as a functor: instantiated with parameters, then called on ray batches.
    """
    def __init__(self):
        super().__init__()

    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        """
        Apply surface physics to incoming rays.

        Args:
            local_intersect (Tensor): [N, 3] Intersection location in local coordinates.
            ray_dir (Tensor): [N, 3] Normalized incident direction (Local or Global).
            normal (Tensor):  [N, 3] Normalized surface normal at hit point (Local or Global, but consistent with ray_dir).
            **kwargs:         Allow passing extra data (wavelength, etc.) that
                              specific children might need.

        Returns:
            new_dir (Tensor): [N, 3] Outgoing ray direction.
            intensity_modulation (Tensor): [N, 1] Modulation intensity.
            meta (dict):      Optional metadata (e.g., attenuation, validity mask).
        """
        raise NotImplementedError


class Linear(SurfaceFunction):

    def __init__(self,
                Cx: float = 0, Cy: float = 0,
                Dx: float = 1, Dy = 1,
                Cx_grad = False, Cy_grad = False,
                Dx_grad = False, Dy_grad = False,
                transform: Optional[Union[RayTransform, None]] =None):

        super().__init__()
        self.Cx = nn.Parameter(torch.as_tensor(Cx), requires_grad=Cx_grad)
        self.Cy = nn.Parameter(torch.as_tensor(Cy), requires_grad=Cy_grad)

        self.Dx = nn.Parameter(torch.as_tensor(Dx), requires_grad=Dx_grad)
        self.Dy = nn.Parameter(torch.as_tensor(Dy), requires_grad=Dy_grad)

        if transform is None:
            self.transform = RayTransform()
        else:
            self.transform = transform

    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        """
        Apply surface physics to incoming rays.

        Args:
            local_intersect (Tensor): [N, 3] Intersection location in local coordinates.
            ray_dir (Tensor): [N, 3] Normalized incident direction (Global).
            normal (Tensor):  [N, 3] Normalized surface normal at hit point.
            **kwargs:         Allow passing extra data (wavelength, etc.) that
                              specific children might need.

        Returns:
            new_dir (Tensor): [N, 3] Outgoing ray direction.
            intensity_modulation (Tensor): [N, 1] Modulation intensity.
            meta (dict):      Optional metadata (e.g., attenuation, validity mask).
        """

        ray_dir_local = ray_dir @ self.transform.rot

        # assumes valid intersect
        ray_dir_local = ray_dir_local / ray_dir_local[:, 2][:, None]

        new_ray_dir_x_local = self.Cx * local_intersect[:, 0] + self.Dx * ray_dir_local[:, 0]
        new_ray_dir_y_local = self.Cy * local_intersect[:, 1] + self.Dy * ray_dir_local[:, 1]
        new_ray_dir_z_local = torch.ones_like(local_intersect[:, 0])

        new_ray_dir_local = torch.stack([new_ray_dir_x_local, new_ray_dir_y_local, new_ray_dir_z_local], dim=1)
        new_ray_dir_local = F.normalize(new_ray_dir_local, p=2, dim=1)
        out_dir = new_ray_dir_local @ self.transform.rot.T

        intensity_mod = torch.ones_like(ray_dir[:, 0])

        return out_dir, intensity_mod


class Reflect(SurfaceFunction):
    """
    Perfect specular reflection.
    Formula: R = I - 2(I . N)N
    """

    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Cosine of incident angle
        # Dot product: (N,3) * (N,3) -> (N,1)
        cos_theta = torch.sum(ray_dir * normal, dim=1, keepdim=True)

        # 2. Vector Reflection
        # Note: This assumes 'normal' is normalized.
        out_dir = ray_dir - 2 * cos_theta * normal

        intensity_mod = torch.ones_like(ray_dir[:, 0])

        return out_dir, intensity_mod


class RefractSnell(SurfaceFunction):
    """
    Snell's Law refraction with Total Internal Reflection (TIR) handling.
    If the incident angle exceeds the critical angle (going High->Low index),
    the ray is reflected instead of refracted.
    """

    def __init__(self, ior_in, ior_out, ior_in_grad = False, ior_out_grad = False):
        super().__init__()
        self.ior_in = nn.Parameter(torch.as_tensor(ior_in), requires_grad=ior_in_grad)
        self.ior_out = nn.Parameter(torch.as_tensor(ior_out), requires_grad=ior_out_grad)

    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        dot = torch.sum(ray_dir * normal, dim=1, keepdim=True)

        entering = dot < 0
        ior_eff = torch.where(entering, normal, -normal)
        c1 = torch.abs(dot)

        # Flip IOR ratio depending on whether the ray is entering or exiting the medium.
        # TIR can only occur when going high->low IOR (exiting), so this also fixes TIR detection.
        mu = torch.where(entering, self.ior_out / self.ior_in, self.ior_in / self.ior_out)

        term_sq = 1.0 - mu ** 2 * (1.0 - c1 ** 2)
        tir_mask = term_sq < 0

        c2 = torch.sqrt(torch.relu(term_sq))
        v_refract = mu * ray_dir + (mu * c1 - c2) * ior_eff
        v_reflect = ray_dir - 2 * dot * normal

        out_dir = torch.where(tir_mask, v_reflect, v_refract)

        intensity_mod = torch.ones_like(ray_dir[:, 0])

        return out_dir, intensity_mod


class RefractFresnel(SurfaceFunction):
    """
    Probabilistic Refraction using Fresnel Equations.

    Instead of splitting rays (which explodes count), this module stochastically
    chooses between Reflection and Refraction based on the Fresnel coefficient R.

    - If TIR: R = 1.0 (Always Reflect)
    - If Normal Incidence: R = ((n1-n2)/(n1+n2))^2
    - Grazing angles -> Higher probability of reflection.
    """

    def __init__(self, ior_in, ior_out, ior_in_grad=False, ior_out_grad=False):
        super().__init__()
        self.ior_in = nn.Parameter(torch.as_tensor(ior_in), requires_grad=ior_in_grad)
        self.ior_out = nn.Parameter(torch.as_tensor(ior_out), requires_grad=ior_out_grad)

    def _fresnel_reflectance(self, cos_i, cos_t, n1, n2):
        """Fresnel reflectance R for unpolarized light: R = 0.5 * (Rs + Rp)."""
        n1_ci = n1 * cos_i
        n2_ct = n2 * cos_t
        rs = ((n1_ci - n2_ct) / (n1_ci + n2_ct + 1e-8)) ** 2

        n1_ct = n1 * cos_t
        n2_ci = n2 * cos_i
        rp = ((n1_ct - n2_ci) / (n1_ct + n2_ci + 1e-8)) ** 2

        return 0.5 * (rs + rp)

    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        dot = torch.sum(ray_dir * normal, dim=1, keepdim=True)
        entering = dot < 0

        ior_eff = torch.where(entering, normal, -normal)
        cos_i = torch.abs(dot)

        # Swap n1/n2 depending on direction so mu and Fresnel equations are correct on exit.
        n1 = torch.where(entering, self.ior_in, self.ior_out)
        n2 = torch.where(entering, self.ior_out, self.ior_in)
        mu = n2 / n1

        sin2_t = (mu ** 2) * (1.0 - cos_i ** 2)
        is_tir = sin2_t > 1.0
        cos_t = torch.sqrt(torch.relu(1.0 - sin2_t))

        r_full = self._fresnel_reflectance(cos_i, cos_t, n1, n2)
        R = torch.where(is_tir, torch.as_tensor(1.0), r_full)

        # 4. Stochastic Decision (Monte Carlo)
        # Generate random values [0, 1)
        # We detach this because the *choice* of path is not differentiable,
        # only the geometry of the chosen path is.
        rand_val = torch.rand_like(R)

        # Boolean mask: True if we Reflect
        reflect_mask = rand_val < R

        # 5. Compute Vectors

        # A. Reflection Vector: I - 2(I.N)N
        # Uses original normal/dot to handle geometric orientation correctly
        v_reflect = ray_dir - 2 * dot * normal

        # B. Refraction Vector: mu*I + (mu*cos_i - cos_t)*N_eff
        v_refract = mu * ray_dir + (mu * cos_i - cos_t) * ior_eff

        # 6. Select Output
        out_dir = torch.where(reflect_mask, v_reflect, v_refract)

        intensity_mod = torch.ones_like(ray_dir[:, 0])

        return out_dir, intensity_mod

class Transmit(SurfaceFunction):
    """
    Pass-through physics.
    The ray continues with no change in direction or intensity.
    Used for 'dummy' surfaces or internal check planes.
    """
    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Direction: Unchanged
        out_dir = ray_dir

        # 2. Intensity: 1.0 (No loss)
        # Create a tensor of 1s matching the batch size [N]
        intensity_mod = torch.ones_like(ray_dir[:, 0])

        return out_dir, intensity_mod

class Block(SurfaceFunction):
    """
    Perfect absorber / Stop.
    The ray is effectively terminated.
    We represent this by zeroing the intensity.
    """
    def forward(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Direction:
        # We can either keep the old direction (ghost ray) or zero it out.
        # Zeroing it makes it obvious visually if something went wrong (ray vanishes).
        # However, keeping it allows debugging "where it would have gone".
        # Standard practice: Zero it or set to NaN to ensure it doesn't contribute to sums.
        out_dir = torch.zeros_like(ray_dir)

        # 2. Intensity: 0.0 (Fully absorbed)
        intensity_mod = torch.zeros_like(ray_dir[:, 0])

        return out_dir, intensity_mod