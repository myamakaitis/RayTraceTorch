import torch


def intersect_ray_sphere(ray_origin, ray_dir, sphere_center, sphere_radius):
    # Compute quadratic equation coefficients
    oc = ray_origin - sphere_center
    a = torch.dot(ray_dir, ray_dir)
    b = 2.0 * torch.dot(oc, ray_dir)
    c = torch.dot(oc, oc) - sphere_radius ** 2

    # Compute discriminant
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None  # No intersection

    # Compute nearest intersection
    t = (-b - torch.sqrt(discriminant)) / (2.0 * a)
    if t < 0:
        return None  # Intersection is behind the ray

    intersection = ray_origin + t * ray_dir
    return intersection


def snells_law(ray_dir, normal, n1, n2):
    # Compute refraction using Snell's Law
    cos_theta_i = -torch.dot(normal, ray_dir)
    sin_theta_i2 = 1 - cos_theta_i ** 2
    n_ratio = n1 / n2
    sin_theta_t2 = n_ratio ** 2 * sin_theta_i2

    if sin_theta_t2 > 1:
        return None  # Total internal reflection

    cos_theta_t = torch.sqrt(1 - sin_theta_t2)
    refracted_dir = n_ratio * ray_dir + (n_ratio * cos_theta_i - cos_theta_t) * normal
    return refracted_dir / torch.norm(refracted_dir)


# Define ray and sphere
ray_origin = torch.tensor([0.0, 0.0, -5.0], requires_grad=True)
ray_dir = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)  # Moving towards the sphere
sphere_center = torch.tensor([0.0, 0.0, 0.0])
sphere_radius = torch.tensor(2.0, requires_grad=True)

# Intersection
intersection = intersect_ray_sphere(ray_origin, ray_dir, sphere_center, sphere_radius)
if intersection is not None:
    normal = (intersection - sphere_center) / torch.norm(intersection - sphere_center)
    n1, n2 = 1.0, torch.tensor(1.5, requires_grad=True)
    new_direction = snells_law(ray_dir, normal, n1, n2)

    print("Intersection:", intersection)
    print("New Direction:", new_direction)

    # Compute gradients w.r.t. sphere radius and index of refraction
    intersection.sum().backward(retain_graph=True)
    print("Gradient of intersection w.r.t. sphere radius:", sphere_radius.grad)

    new_direction.sum().backward()
    print("Gradient of refraction w.r.t. index of refraction:", n2.grad)