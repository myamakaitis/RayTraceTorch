import torch
import torch.nn.functional as F

class SurfaceFunction:
    """
    Base class for optical surface physics.
    Acts as a functor: instantiated with parameters, then called on ray batches.
    """
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, local_intersect, ray_dir, normal, **kwargs):
        """
        Apply surface physics to incoming rays.

        Args:
            ray_dir (Tensor): [N, 3] Normalized incident direction (Local or Global).
            normal (Tensor):  [N, 3] Normalized surface normal at hit point.
            **kwargs:         Allow passing extra data (wavelength, etc.) that
                              specific children might need.

        Returns:
            new_dir (Tensor): [N, 3] Outgoing ray direction.
            intensity_modulation (Tensor): [N, 1] Modulation intensity.
            meta (dict):      Optional metadata (e.g., attenuation, validity mask).
        """
        raise NotImplementedError

    def to(self, device):
        self.device = device
        # Move any tensor attributes to device
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, key, val.to(device))
        return self


class Reflect(SurfaceFunction):
    """
    Perfect specular reflection.
    Formula: R = I - 2(I . N)N
    """

    def __call__(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Cosine of incident angle
        # Dot product: (N,3) * (N,3) -> (N,1)
        cos_theta = torch.sum(ray_dir * normal, dim=1, keepdim=True)

        # 2. Vector Reflection
        # Note: This assumes 'normal' is normalized.
        reflect_dir = ray_dir - 2 * cos_theta * normal

        intensity_mod = torch.ones(ray_dir.shape[0], dtype=torch.float32, device=self.device)

        return reflect_dir, intensity_mod


class RefractSnell(SurfaceFunction):
    """
    Snell's Law refraction with Total Internal Reflection (TIR) handling.
    If the incident angle exceeds the critical angle (going High->Low index),
    the ray is reflected instead of refracted.
    """

    def __init__(self, n_in, n_out, device='cpu'):
        super().__init__(device)
        self.n_in = torch.tensor(n_in, dtype=torch.float32, device=device)
        self.n_out = torch.tensor(n_out, dtype=torch.float32, device=device)
        self.mu = self.n_in / self.n_out

    def __call__(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Orientation Check
        # Calculate dot product (cos theta_in)
        # ray_dir and normal should be normalized.
        dot = torch.sum(ray_dir * normal, dim=1, keepdim=True)

        # Standardize Normal:
        # Refraction math assumes Normal points INTO the material we are entering (against the ray).
        # Reflection math assumes Normal points OUT of the surface.

        # We define 'n_eff' as the normal pointing AGAINST the incoming ray for Snell's calculation.
        # If dot < 0, Ray opposes Normal (standard entry). n_eff = normal.
        # If dot > 0, Ray aligns with Normal (standard exit). n_eff = -normal.
        entering = dot < 0
        n_eff = torch.where(entering, normal, -normal)
        c1 = torch.abs(dot)  # cos(theta_1) must be positive magnitude

        # 2. Snell's Law Discriminant
        # term = 1 - mu^2 * (1 - cos^2(theta_1))
        # If term < 0, we have TIR.
        term_sq = 1.0 - self.mu ** 2 * (1.0 - c1 ** 2)

        # Create TIR Mask
        tir_mask = term_sq < 0

        # --- BRANCH A: REFRACTION (Where term_sq >= 0) ---
        # Safe sqrt for gradients (clamp negative to 0)
        c2 = torch.sqrt(torch.relu(term_sq))

        # Vector Snell's Law: V_out = mu*I + (mu*c1 - c2)*N
        v_refract = self.mu * ray_dir + (self.mu * c1 - c2) * n_eff

        # --- BRANCH B: REFLECTION (Where term_sq < 0) ---
        # Formula: R = I - 2(I . N)N
        # Note: We must use the ORIGINAL normal logic for reflection (Standard reflection uses N).
        # Actually, reflection is symmetric.
        # I - 2(I.N)N works regardless of whether N points up or down,
        # as long as N is the surface normal axis.
        v_reflect = ray_dir - 2 * dot * normal

        # --- COMBINE ---
        # Select result based on TIR mask
        out_dir = torch.where(tir_mask, v_reflect, v_refract)

        # Normalize to prevent drift
        out_dir = F.normalize(out_dir, p=2, dim=1)

        intensity_mod = torch.ones(ray_dir.shape[0], dtype=torch.float32, device=self.device)

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

    def __init__(self, n_in, n_out, device='cpu'):
        super().__init__(device)
        self.n_in = torch.tensor(n_in, dtype=torch.float32, device=device)
        self.n_out = torch.tensor(n_out, dtype=torch.float32, device=device)
        self.mu = self.n_in / self.n_out

    def _fresnel_reflectance(self, cos_i, cos_t):
        """
        Computes Fresnel Reflectance R for unpolarized light.
        R = 0.5 * (Rs + Rp)
        """
        # Rs (s-polarized): Perpendicular
        # (n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t)
        n1_ci = self.n_in * cos_i
        n2_ct = self.n_out * cos_t
        rs_num = n1_ci - n2_ct
        rs_den = n1_ci + n2_ct
        rs = (rs_num / (rs_den + 1e-8)) ** 2

        # Rp (p-polarized): Parallel
        # (n1 cos_t - n2 cos_i) / (n1 cos_t + n2 cos_i)
        n1_ct = self.n_in * cos_t
        n2_ci = self.n_out * cos_i
        rp_num = n1_ct - n2_ci
        rp_den = n1_ct + n2_ci
        rp = (rp_num / (rp_den + 1e-8)) ** 2

        return 0.5 * (rs + rp)

    def __call__(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Geometry Setup
        # dot < 0: Entering (cos_i > 0). dot > 0: Exiting.
        dot = torch.sum(ray_dir * normal, dim=1, keepdim=True)
        entering = dot < 0

        # Effective Normal points AGAINST ray
        n_eff = torch.where(entering, normal, -normal)
        cos_i = torch.abs(dot)  # Enforce positive cosine

        # 2. Compute Refraction Angle (Snell's Law)
        # sin^2(t) = mu^2 * sin^2(i)
        # sin^2(i) = 1 - cos^2(i)
        sin2_t = (self.mu ** 2) * (1.0 - cos_i ** 2)

        # Check TIR
        is_tir = sin2_t > 1.0

        # Calculate cos_t (safe sqrt)
        # If TIR, term is negative, relu makes it 0.
        cos_t = torch.sqrt(torch.relu(1.0 - sin2_t))

        # 3. Calculate Probability of Reflection (R)
        # Initialize R with 1.0 (TIR case)
        R = torch.ones_like(cos_i)

        # Full tensor computation (safe because we use where later)
        # The calculation might produce garbage for TIR indices, but we filter them.
        r_full = self._fresnel_reflectance(cos_i, cos_t)
        R = torch.where(is_tir, torch.tensor(1.0, device=self.device), r_full)

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
        v_refract = self.mu * ray_dir + (self.mu * cos_i - cos_t) * n_eff

        # 6. Select Output
        out_dir = torch.where(reflect_mask, v_reflect, v_refract)
        out_dir = F.normalize(out_dir, p=2, dim=1)

        intensity_mod = torch.ones(ray_dir.shape[0], dtype=torch.float32, device=self.device)

        return out_dir, intensity_mod

class Transmit(SurfaceFunction):
    """
    Pass-through physics.
    The ray continues with no change in direction or intensity.
    Used for 'dummy' surfaces or internal check planes.
    """
    def __call__(self, local_intersect, ray_dir, normal, **kwargs):
        # 1. Direction: Unchanged
        out_dir = ray_dir

        # 2. Intensity: 1.0 (No loss)
        # Create a tensor of 1s matching the batch size [N]
        intensity_mod = torch.ones(ray_dir.shape[0], dtype=torch.float32, device=self.device)

        return out_dir, intensity_mod

class Block(SurfaceFunction):
    """
    Perfect absorber / Stop.
    The ray is effectively terminated.
    We represent this by zeroing the intensity.
    """
    def __call__(self, intersect, ray_dir, normal, **kwargs):
        # 1. Direction:
        # We can either keep the old direction (ghost ray) or zero it out.
        # Zeroing it makes it obvious visually if something went wrong (ray vanishes).
        # However, keeping it allows debugging "where it would have gone".
        # Standard practice: Zero it or set to NaN to ensure it doesn't contribute to sums.
        out_dir = torch.zeros_like(ray_dir)

        # 2. Intensity: 0.0 (Fully absorbed)
        intensity_mod = torch.zeros(ray_dir.shape[0], dtype=torch.float32, device=self.device)

        return out_dir, intensity_mod