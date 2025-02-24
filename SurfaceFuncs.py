import torch as tr

def Reflect(ray_dir, surf_norm, *_):

    s = tr.dot(ray_dir, surf_norm)

    ray_transmit = ray_dir - 2*s*surf_norm

    return ray_transmit

def Refract(ray_dir, surf_norm, n2, n1, *_):

    mu = n1/n2

    c = -tr.dot(ray_dir, surf_norm)

    if c < 0:
        mu **= -1
        surf_norm *= -1
        c *= -1

    discriminant = 1 - mu**2 * (1 - c**2)

    if discriminant < 0:
        return Reflect(ray_dir, surf_norm)

    ray_transmit = mu*ray_dir + (mu*c - tr.sqrt(discriminant))*surf_norm

    return ray_transmit

def Stop(ray_dir, *_):

    return 0*ray_dir
