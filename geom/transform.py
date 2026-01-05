import torch

class RayTransform:

    def __init__(self, rotation=None, translation=None, device='cpu'):
        """
        Args:
            rotation: [3, 3] Rotation matrix (Local -> Global).
            translation: [3] Translation vector (Object position).
            device: 'cpu' or 'cuda'.
        """
        self.device = device

        # 1. Initialize Translation
        if translation is not None:
            self.trans = torch.as_tensor(translation, dtype=torch.float32, device=device)
        else:
            self.trans = torch.zeros(3, dtype=torch.float32, device=device)

        # 2. Initialize Rotation
        if rotation is not None:
            self.rot = torch.as_tensor(rotation, dtype=torch.float32, device=device)
            self._validate_rotation()
        else:
            self.rot = torch.eye(3, dtype=torch.float32, device=device)

    def _validate_rotation(self):
        """Checks if R.T @ R == Identity."""
        # Matrix multiplication: R.T @ R
        check = self.rot.T @ self.rot
        identity = torch.eye(3, device=self.device)

        if not torch.allclose(check, identity, atol=1e-5):
            raise ValueError("Provided rotation matrix is not orthogonal (R.T @ R != I).")

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
        local_pos = (rays.pos @ self.rot.T) + self.trans
        local_dir = (rays.dir @ self.rot.T)

        return local_pos, local_dir

    def invTransform(self, rays):
        """
        Applies the INVERSE transformation (Global -> Local).
        New Pos = (Pos - T) @ R
        New Dir = Dir @ R

        Returns a NEW Rays object.
        """
        # Inverse Translation
        shifted_pos = rays.pos - self.trans

        # Inverse Rotation (Multiply by R on right is equivalent to multiplying by R.T on left)
        # Since R is orthogonal, R.inv = R.T.
        # Logic: (P_global - T) * R_global_to_local
        # Here self.rot is Local_to_Global. So we need to multiply by R_inverse.
        # @ R.T rotates forward. v @ R rotates backward.

        global_pos = shifted_pos @ self.rot
        global_dir = rays.dir @ self.rot

        return global_pos, global_dir

    def to(self, device):
        self.device = device
        self.rot = self.rot.to(device)
        self.trans = self.trans.to(device)
        return self