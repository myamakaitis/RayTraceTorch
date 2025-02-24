import torch

def Reflect(ray_direction, normal, *_):
    """
    Compute the reflected ray direction.

    Parameters:
    - ray_direction: torch.Tensor, direction of the incoming ray (3D vector).
    - normal: torch.Tensor, surface normal at the intersection point (3D vector).
    - *args: Additional arguments (unused in this function).

    Returns:
    - reflected_direction: torch.Tensor, direction of the reflected ray (3D vector).
    """
    return ray_direction - 2 * torch.dot(ray_direction, normal) * normal

def Refract(ray_direction, normal, n_front, n_back, *_):
    """
    Compute the refracted ray direction using Snell's law.

    Parameters:
    - ray_direction: torch.Tensor, direction of the incoming ray (3D vector).
    - normal: torch.Tensor, surface normal at the intersection point (3D vector).
    - n1: float or torch.Tensor, refractive index of the medium the ray is coming from.
    - n2: float or torch.Tensor, refractive index of the medium the ray is entering.
    - *args: Additional arguments (unused in this function).

    Returns:
    - refracted_direction: torch.Tensor, direction of the refracted ray (3D vector).
    """


    # Refractive indices


    cos_theta1 = -torch.dot(ray_direction, normal)

    entering = cos_theta1 > 0

    # n1 = torch.where(entering, self.refractive_index_front, self.refractive_index_back)
    # n2 = torch.where(entering, self.refractive_index_back, self.refractive_index_front)

    sin_theta1_sq = 1.0 - cos_theta1**2
    sin_theta2_sq = (n1 / n2)**2 * sin_theta1_sq

    if sin_theta2_sq > 1.0:
        # Total internal reflection
        return Reflect(ray_direction, normal)

    cos_theta2 = torch.sqrt(1.0 - sin_theta2_sq)
    return (n1 / n2) * ray_direction + (n1 / n2 * cos_theta1 - cos_theta2) * normal

class Surface:
    def __init__(self, name, SurfaceFunc, position, rotation_matrix, refractive_index_front=1.0, refractive_index_back=1.0):
        """
        Base class for optical surfaces.

        Parameters:
        - name: str, user-defined name for the surface.
        - SurfaceFunc: function, defines how the surface interacts with rays (e.g., Reflect, Refract).
        - position: torch.Tensor, 3D position of the surface in the scene (global coordinates).
        - rotation_matrix: torch.Tensor, 3x3 rotation matrix defining the surface's orientation.
        - refractive_index_front: float or torch.Tensor, refractive index in front of the surface.
        - refractive_index_back: float or torch.Tensor, refractive index behind the surface.
        """
        self.name = name
        self.SurfaceFunc = SurfaceFunc
        self.position = position
        self.rotation_matrix = rotation_matrix
        self.refractive_index_front = refractive_index_front
        self.refractive_index_back = refractive_index_back

    def toLocalXYZ(self, ray):
        """
        Transform a ray from the scene's global coordinate system to the surface's local coordinate system.
        Temporarily overwrites the ray's position and direction.

        Parameters:
        - ray: Ray object, the ray to transform.
        """
        # Translate the ray's position to local coordinates
        ray.position = ray.position - self.position

        # Rotate the ray's position and direction using the rotation matrix
        ray.position = torch.matmul(self.rotation_matrix.T, ray.position)
        ray.direction = torch.matmul(self.rotation_matrix.T, ray.direction)

    def toSceneXYZ(self, ray):
        """
        Transform a ray from the surface's local coordinate system back to the scene's global coordinate system.
        Temporarily overwrites the ray's position and direction.

        Parameters:
        - ray: Ray object, the ray to transform.
        """
        # Rotate the ray's position and direction back to global coordinates
        ray.position = torch.matmul(self.rotation_matrix, ray.position)
        ray.direction = torch.matmul(self.rotation_matrix, ray.direction)

        # Translate the ray's position back to global coordinates
        ray.position = ray.position + self.position

    def intersectScene(self, ray):

        self.toLocalXYZ(ray)
        self.intersect(ray)
        self.toSceneXYZ(ray)

    def intersect(self, ray):
        """
        Compute the intersection point of a ray with the surface in the surface's local coordinate system.

        Parameters:
        - ray: Ray object, the ray to intersect with the surface.

        Returns:
        - intersection_point: torch.Tensor, point where the ray intersects the surface (in local coordinates).
        - t: torch.Tensor, ray parameter at the intersection point.
        """
        # Transform rays to local coordinates

        raise NotImplementedError("Intersection method must be implemented by subclass.")

    def surfaceNormal(self, point):
        """
        Compute the surface normal at a given point on the surface (in local coordinates).

        Parameters:
        - point: torch.Tensor, point on the surface (in local coordinates).

        Returns:
        - normal: torch.Tensor, surface normal at the point (in local coordinates).
        """
        raise NotImplementedError("Surface normal method must be implemented by subclass.")

    def surfaceInteraction(self, rays):
        """
        Propagate rays through this surface. Assumes all rays intersect the surface.

        Parameters:
        - rays: Ray object, the rays to propagate.

        Returns:
        - rays: Ray object, the propagated rays.
        """

        self.toLocalXYZ(rays)

        # Find intersection points and surface normals
        intersection_point, normal, t, valid = self.intersect(rays)

        # Compute new directions using SurfaceFunc
        new_direction = self.SurfaceFunc(rays.direction, normal, n1, n2)

        # Update ray positions and directions
        rays.position = intersection_point
        rays.direction = new_direction

        # Transform rays back to global coordinates
        self.toSceneXYZ(rays)

        return rays

class InfinitePlane(Surface):
    def __init__(self, name, SurfaceFunc, position, normal, width, height, refractive_index_front=1.0,
                 refractive_index_back=1.0):
        """
        Finite rectangular plane surface.

        Parameters:
        - name: str, user-defined name for the surface.
        - SurfaceFunc: function, defines how the surface interacts with rays.
        - position: torch.Tensor, shape [3], the center of the rectangle.
        - normal: torch.Tensor, shape [3], normal vector of the plane.
        - width: float, width of the rectangle.
        - height: float, height of the rectangle.
        - refractive_index_front: float or torch.Tensor, refractive index in front of the surface.
        - refractive_index_back: float or torch.Tensor, refractive index behind the surface.
        """
        super().__init__(name, SurfaceFunc, position, torch.eye(3), refractive_index_front, refractive_index_back)
        self.normal = normal / torch.norm(normal)  # Ensure the normal is a unit vector
        self.width = width
        self.height = height

        # Define local coordinate system for the plane
        self.tangent1 = torch.tensor([1.0, 0.0, 0.0], device=normal.device)  # Arbitrary tangent vector
        self.tangent2 = torch.cross(self.normal, self.tangent1)  # Second tangent vector
        self.tangent1 = torch.cross(self.tangent2, self.normal)  # Recompute to ensure orthogonality

