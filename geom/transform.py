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

        # Inverse Translation
        shifted_pos = rays.pos - self.trans

        global_pos = shifted_pos @ self.rot
        global_dir = rays.dir @ self.rot

        return global_pos, global_dir


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

        local_pos = (rays.pos @ self.rot.T) + self.trans
        local_dir = (rays.dir @ self.rot.T)

        return local_pos, local_dir

    def to(self, device):
        self.device = device
        self.rot = self.rot.to(device)
        self.trans = self.trans.to(device)
        return self

def eulerToRotMat(euler_angles):
    """
    Converts a batch of Euler angles (roll, pitch, yaw) to rotation matrices
    using the ZYX convention (extrinsic rotation: X, then Y, then Z).

    Args:
        euler_angles (torch.Tensor): Shape (..., 3) containing (roll, pitch, yaw)
                                     in radians. Supports batch dimensions.

    Returns:
        torch.Tensor: Shape (..., 3, 3) rotation matrices.
    """
    # Extract the angles (supports arbitrary batch dimensions)
    # theta_x = roll, theta_y = pitch, theta_z = yaw
    theta_x = euler_angles[..., 0]
    theta_y = euler_angles[..., 1]
    theta_z = euler_angles[..., 2]

    # Precompute sines and cosines
    c_x = torch.cos(theta_x)
    s_x = torch.sin(theta_x)
    c_y = torch.cos(theta_y)
    s_y = torch.sin(theta_y)
    c_z = torch.cos(theta_z)
    s_z = torch.sin(theta_z)

    # Create a tensor of zeros/ones with the same shape/device as the angles
    # This ensures we handle different devices (CPU/GPU) automatically
    zeros = torch.zeros_like(theta_x)
    ones = torch.ones_like(theta_x)

    # Construct the Rotation Matrices

    # Rotation Matrix for X-axis (Roll)
    # [ 1   0    0  ]
    # [ 0  c_x -s_x ]
    # [ 0  s_x  c_x ]
    R_x = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, c_x, -s_x], dim=-1),
        torch.stack([zeros, s_x, c_x], dim=-1)
    ], dim=-2)

    # Rotation Matrix for Y-axis (Pitch)
    # [ c_y  0  s_y ]
    # [  0   1   0  ]
    # [-s_y  0  c_y ]
    R_y = torch.stack([
        torch.stack([c_y, zeros, s_y], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-s_y, zeros, c_y], dim=-1)
    ], dim=-2)

    # Rotation Matrix for Z-axis (Yaw)
    # [ c_z -s_z  0 ]
    # [ s_z  c_z  0 ]
    # [  0    0   1 ]
    R_z = torch.stack([
        torch.stack([c_z, -s_z, zeros], dim=-1),
        torch.stack([s_z, c_z, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)

    # Combine rotations: R = R_z @ R_y @ R_x
    # This corresponds to rotating around X, then Y, then Z (in fixed frame)
    # or Z, then Y, then X (in moving frame)
    R = torch.matmul(R_z, torch.matmul(R_y, R_x))

    return R