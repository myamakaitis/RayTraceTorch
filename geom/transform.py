import torch
import torch.nn as nn

class RayTransform(nn.Module):

    def __init__(self, rotation=None, translation=None, dtype=torch.float32,
                 trans_grad = False, trans_mask = None,
                 rot_grad = False, rot_mask = None):
        """
        Args:
            rotation: [3, 3] Rotation matrix (Local -> Global).
            translation: [3] Translation vector (Object position).
            device: 'cpu' or 'cuda'.
        """
        super().__init__()
        self._dtype = dtype

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
        shifted_pos = rays.pos - self.trans[None, :]

        local_pos = shifted_pos @ self.rot
        local_dir = rays.dir @ self.rot

        return local_pos, local_dir


    def invTransform(self, rays):
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

        global_pos = (rays.pos @ self.rot.T) + self.trans[None, :]
        global_dir = (rays.dir @ self.rot.T)

        return global_pos, global_dir

class NoisyTransform(RayTransform):
    """
    Transform class that selectively adds random perturbations with a normal distribution to rotation and translation
    useful for optical design with realistic tolerances
    """
    def __init__(self):
        raise NotImplementedError()