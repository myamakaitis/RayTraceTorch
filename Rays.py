class Ray:
    def __init__(self, position, direction, wavelength, intensity=None, color=None):
        """
        Ray class with history tracking.

        Parameters:
        - position: torch.Tensor, shape [N, 3], batched 3D positions of the rays.
        - direction: torch.Tensor, shape [N, 3], batched 3D direction vectors of the rays.
        - wavelength: torch.Tensor, shape [N], batched wavelengths of the rays.
        - intensity: torch.Tensor, shape [N], batched intensities of the rays (optional).
        - color: torch.Tensor, shape [N, 3], batched RGB colors of the rays (optional).
        """
        self.position = position
        self.direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # Normalize directions
        self.wavelength = wavelength
        self.intensity = intensity if intensity is not None else torch.ones_like(wavelength)
        self.color = color if color is not None else torch.ones((position.shape[0], 3))

        # Initialize history
        self.history = []  # List of dictionaries to store historical states

    def record_state(self, surface_name=None):
        """
        Record the ray's current state in its history.

        Parameters:
        - surface_name: str or None, name of the surface the ray interacted with (if any).
        """
        self.history.append({
            "position": self.position.clone(),
            "direction": self.direction.clone(),
            "wavelength": self.wavelength.clone(),
            "intensity": self.intensity.clone() if self.intensity is not None else None,
            "color": self.color.clone() if self.color is not None else None,
            "surface_name": surface_name
        })

    def get_history(self):
        """
        Get the full history of the ray's states.

        Returns:
        - history: list of dictionaries, each containing the ray's state and surface interaction.
        """
        return self.history