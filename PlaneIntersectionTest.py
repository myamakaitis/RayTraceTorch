import torch

def intersect_ray_plane(ray_origin, ray_dir, plane_normal, plane_point):
    # Compute intersection t value
    denom = torch.dot(ray_dir, plane_normal)
    if torch.abs(denom) < 1e-6:
        return None  # No intersection or parallel ray

    t = torch.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None  # Intersection is behind the ray

    intersection = ray_origin + t * ray_dir
    return intersection

def snells_law(ray_dir, normal, n1, n2):
    # Compute refraction using Snell's Law
    cos_theta_i = -torch.dot(normal, ray_dir)
    sin_theta_i2 = 1 - cos_theta_i**2
    n_ratio = n1 / n2
    sin_theta_t2 = n_ratio**2 * sin_theta_i2

    if sin_theta_t2 > 1:
        return None  # Total internal reflection

    cos_theta_t = torch.sqrt(1 - sin_theta_t2)
    refracted_dir = n_ratio * ray_dir + (n_ratio * cos_theta_i - cos_theta_t) * normal
    return refracted_dir / torch.norm(refracted_dir)

# Define ray and plane
ray_origin = torch.tensor([-10.0, 0.5, -0.5], requires_grad=True)
ray_dir = torch.tensor([0.9886, 0.15, 0.0], requires_grad=True)  # Moving towards the plane
plane_normal = torch.tensor([-1.0, 0.0, 0.0])
plane_point = torch.tensor([0.0, 0.0, 0.0])  # Plane passes through the origin

# Intersection
intersection = intersect_ray_plane(ray_origin, ray_dir, plane_normal, plane_point)
if intersection is not None:
    n1, n2 = 1.0, 1.5
    new_direction = snells_law(ray_dir, plane_normal, n1, n2)

    print("Intersection:", intersection)
    print("New Direction:", new_direction)

    # Example: Compute gradient w.r.t. ray origin
    intersection.sum().backward()
    print("Gradient of intersection w.r.t. ray origin:", ray_origin.grad)

    new_direction.sum().backward()
    print("Gradient of intersection w.r.t. ray origin:", ray_origin.grad)