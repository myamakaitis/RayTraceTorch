import torch
import torch.nn as nn

from typing import Optional, Union, List, Tuple
from torch.distributions import Normal

Vector3 = Union[torch.Tensor, List[float], Tuple[float, ...]]
Bool3 = Union[torch.Tensor, List[bool], Tuple[bool, ...]]

class RayTransform(nn.Module):

    def __init__(self, rotation: Optional[Vector3] = None, translation: Optional[Vector3] = None,
                 dtype: torch.dtype =torch.float32,
                 trans_grad: bool = False, trans_mask: Bool3 = None,
                 rot_grad: bool = False, rot_mask: Bool3 = None):
        """
        Args:
            rotation: [3, 3] Rotation matrix (Local -> Global).
            translation: [3] Translation vector (Object position).
        """
        super().__init__()

        # Initialize Translation
        if translation is not None:
            self.trans = torch.nn.Parameter(torch.as_tensor(translation), requires_grad=trans_grad)
        else:
            self.trans = torch.nn.Parameter(torch.zeros(3, dtype=dtype), requires_grad=trans_grad)

        if trans_grad and (trans_mask is not None):
            # Register buffer so it moves to GPU/CPU automatically with the model
            self.register_buffer('trans_mask', torch.as_tensor(trans_mask, dtype=dtype))

            # The Hook: Multiplies gradient by mask before optimizer sees it
            # We use self.trans_mask (the buffer) to ensure correct device
            self.trans.register_hook(lambda grad: grad * self.trans_mask)

        # Initialize Rotation
        if rotation is not None:
            self.rot_vec = torch.nn.Parameter(torch.as_tensor(rotation), requires_grad=rot_grad)
        else:
            self.rot_vec = torch.nn.Parameter(torch.zeros(3), requires_grad=rot_grad)

        # --- 4. Rotation Hook ---
        if rot_grad and (rot_mask is not None):
            self.register_buffer('rot_mask', torch.as_tensor(rot_mask, dtype=dtype))
            self.rot_vec.register_hook(lambda grad: grad * self.rot_mask)

    def _compute_matrix(self):
        x, y, z = self.rot_vec
        K = torch.zeros((3, 3), device=self.rot_vec.device, dtype=self.rot_vec.dtype)
        K[0, 1], K[0, 2] = -z, y
        K[1, 0], K[1, 2] = z, -x
        K[2, 0], K[2, 1] = -y, x
        return torch.linalg.matrix_exp(K)

    @property
    def rot(self):
        """
        Accessing .rot will automatically run _compute_matrix().
        It acts like a variable, not a function.
        """
        return self._compute_matrix()

    def transform(self, rays):

        return self.transform_(rays.pos, rays.dir)

    def transform_(self, _pos, _dir):
        """
        Applies the transformation (Local -> Global).
        New Pos = (Pos @ R.T) + T
        New Dir = (Dir @ R.T)

        Returns a NEW Rays object (non-destructive).
        """
        # Create a copy of the rays to avoid modifying the input in-place
        # We assume Rays class has a mechanism to copy or we create new one
        # Ideally, we create new tensors and instantiate a new Rays object.

        # Apply Rotation (P @ R.T is standard for row-vector multiplication)

        # Inverse Translation
        shifted_pos = _pos - self.trans[None, :]

        local_pos = shifted_pos @ self.rot
        local_dir = _dir @ self.rot

        return local_pos, local_dir

    def invTransform(self, rays):

        return self.invTransform_(rays.pos, rays.dir)

    def invTransform_(self, _pos, _dir):
        """
        Applies the INVERSE transformation (Global -> Local).
        New Pos = (Pos - T) @ R
        New Dir = Dir @ R

        Returns a NEW Rays object.
        """

        # Inverse Rotation (Multiply by R on right is equivalent to multiplying by R.T on left)
        # Since R is orthogonal, R.inv = R.T.
        # Logic: (P_global - T) * R_global_to_local
        # Here self.rot is Local_to_Global. So we need to multiply by R_inverse.
        # @ R.T rotates forward. v @ R rotates backward.

        global_pos = (_pos @ self.rot.T) + self.trans[None, :]
        global_dir = (_dir @ self.rot.T)

        return global_pos, global_dir

    def paraxial(self):

        affine_vector = -torch.stack([self.trans[0], self.rot_vec[0], self.trans[1], self.rot_vec[1], -torch.tensor(1.0)]).unsqueeze(1)

        zeros = torch.eye(5 , device=self.trans.device, dtype=self.trans.dtype)[:, :4]

        Mat = torch.cat([zeros, affine_vector], dim=1)

        return Mat

    def paraxial_inv(self):

        affine_vector = torch.stack([self.trans[0], self.rot_vec[0], self.trans[1], self.rot_vec[1], torch.tensor(1.0)]).unsqueeze(1)

        zeros = torch.eye(5 , device=self.trans.device, dtype=self.trans.dtype)[:, :4]

        Mat = torch.cat([zeros, affine_vector], dim=1)

        return Mat

class NoisyTransform(RayTransform):
    """
    Transform class that selectively adds random perturbations with a normal distribution to rotation and translation
    useful for optical design with realistic tolerances
    """
    def __init__(self, rotation: Optional[Vector3] = None, translation: Optional[Vector3] = None,
                 dtype: torch.dtype = torch.float32,
                 std_translation: Optional[Vector3] = (0, 0, 0), std_rotation: Optional[Vector3] = (0, 0, 0),
                 trans_grad: bool = False, trans_mask: Bool3 = None,
                 rot_grad: bool = False, rot_mask: Bool3 = None):
        """
        Args:
            rotation: [3, 3] Rotation matrix (Local -> Global).
            translation: [3] Translation vector (Object position).
        """
        super().__init__(rotation = rotation, translation = translation, dtype = dtype,
                         trans_grad = trans_grad, trans_mask = trans_mask,
                         rot_grad = rot_grad, rot_mask=rot_mask)

        self.cached_trans_noise = None
        self.cached_rot_noise = None

        self.zero = torch.nn.Parameter(torch.zeros(3, dtype=dtype), requires_grad = False)

        self.trans_scale = torch.nn.Parameter(torch.as_tensor(trans_grad, dtype=dtype), requires_grad=False)
        self.rot_scale = torch.nn.Parameter(torch.as_tensor(rot_grad, dtype=dtype), requires_grad=False)

        self.trans_dist = Normal(self.zero, self.trans_scale)
        self.rot_dist = Normal(self.zero, self.rot_scale)


    def _compute_matrix_batch(self, rot_vec_batch):

        N = rot_vec_batch.shape[0]

        x, y, z = self.rot_vec[:, 0], self.rot_vec[:, 1], self.rot_vec[:, 2]
        K = torch.zeros((N, 3, 3), device=self.rot_vec.device, dtype=self.rot_vec.dtype)
        K[:, 0, 1], K[:, 0, 2] = -z, y
        K[:, 1, 0], K[:, 1, 2] = z, -x
        K[:, 2, 0], K[:, 2, 1] = -y, x
        return torch.linalg.matrix_exp(K)

    def addNoise(self, N):

        trans_noise = self.trans[:, None] + self.trans_dist.sample((N, ))
        rot_noise = self.rot_dist[:, None] + self.rot_dist.sample((N, ))

        return trans_noise, rot_noise

    def transform_(self, _pos, _dir):

        trans_noise, rot_noise = self.addNoise(_pos.size(0))
        rot = self._compute_matrix_batch(rot_noise)

        shifted_pos = _pos - trans_noise

        local_pos = torch.bmm(shifted_pos, rot)
        local_dir = torch.bmm(_dir, rot)

        self.cached_trans_noise = torch.tensor([])
        self.cached_rot_noise = torch.tensor([])

        return local_pos, local_dir

    def invTransform_(self, _pos, _dir):

        batch_size = _pos.size(0)

        with torch.no_grad():
            if self.cached_trans_noise.size(0) == batch_size:
                trans_noise, rot_noise = self.cached_trans_noise, self.cached_rot_noise
            else:
                trans_noise, rot_noise = self.addNoise(batch_size)

        rot = self._compute_matrix_batch(rot_noise)

        unshifted_pos = torch.bmm(_pos, rot.transpose(1, 2))
        global_dir = torch.bmm(_dir, rot.transpose(1, 2))

        global_pos = unshifted_pos + trans_noise

        return global_pos, global_dir

