from Rays import Ray
import torch

class Scene:
    def __init__(self, max_bounces=10):
        """
        Scene class to manage rays and surfaces, with history tracking.

        Parameters:
        - max_bounces: int, maximum number of bounces for each ray.
        """
        self.surfaces = []  # List of surfaces in the scene
        self.rays = None  # Batched Ray object for all rays in the scene
        self.max_bounces = max_bounces

    def add_surface(self, surface):
        """
        Add a surface to the scene.

        Parameters:
        - surface: Surface object, the surface to add.
        """
        self.surfaces.append(surface)

    def add_rays(self, rays):
        """
        Add rays to the scene. Concatenates new rays with existing rays.

        Parameters:
        - rays: Ray object or list of Ray objects, the rays to add.
        """
        if isinstance(rays, Ray):
            rays = [rays]

        # Combine new rays into a single batched Ray object
        new_rays = Ray(
            position=torch.cat([ray.position for ray in rays]),
            direction=torch.cat([ray.direction for ray in rays]),
            wavelength=torch.cat([ray.wavelength for ray in rays]),
            intensity=torch.cat([ray.intensity for ray in rays]) if rays[0].intensity is not None else None,
            color=torch.cat([ray.color for ray in rays]) if rays[0].color is not None else None
        )

        # Concatenate with existing rays (if any)
        if self.rays is None:
            self.rays = new_rays
        else:
            self.rays.position = torch.cat([self.rays.position, new_rays.position])
            self.rays.direction = torch.cat([self.rays.direction, new_rays.direction])
            self.rays.wavelength = torch.cat([self.rays.wavelength, new_rays.wavelength])
            if self.rays.intensity is not None and new_rays.intensity is not None:
                self.rays.intensity = torch.cat([self.rays.intensity, new_rays.intensity])
            if self.rays.color is not None and new_rays.color is not None:
                self.rays.color = torch.cat([self.rays.color, new_rays.color])

    def mask_rays(self, mask):
        """
        Create a masked subset of rays based on the given mask.

        Parameters:
        - mask: torch.Tensor, boolean mask indicating which rays to include.

        Returns:
        - masked_rays: Ray object, the masked subset of rays.
        """
        return Ray(
            position=self.rays.position[mask],
            direction=self.rays.direction[mask],
            wavelength=self.rays.wavelength[mask],
            intensity=self.rays.intensity[mask] if self.rays.intensity is not None else None,
            color=self.rays.color[mask] if self.rays.color is not None else None
        )

    def propagate_rays(self):
        """
        Propagate all rays through the scene and record their history.
        """
        if self.rays is None:
            raise ValueError("No rays in the scene to propagate.")

        self.rays.record_state(surface_name="Origin")

        for _ in range(self.max_bounces):
            # Initialize closest intersection data
            closest_t = torch.full((self.rays.position.shape[0],), float('inf'), device=self.rays.position.device)
            closest_surface_idx = torch.full((self.rays.position.shape[0],), -1, dtype=torch.long,
                                             device=self.rays.position.device)

            # Test intersections with all surfaces
            for i, surface in enumerate(self.surfaces):
                _, _, t, valid = surface.intersect(self.rays)
                update_mask = valid & (t < closest_t)
                if update_mask.any():
                    closest_t[update_mask] = t[update_mask]
                    closest_surface_idx[update_mask] = i

            # If no surface is intersected, stop propagation
            if (closest_surface_idx == -1).all():
                break

            # Propagate rays to their respective closest surfaces
            for i, surface in enumerate(self.surfaces):
                # Mask for rays that intersect this surface
                mask = closest_surface_idx == i
                if mask.any():
                    # Create a subset of rays that intersect this surface
                    masked_rays = self.mask_rays(mask)

                    # Record the ray's current state before propagation


                    # Propagate the masked rays through the surface
                    propagated_rays = surface.propagate_ray(masked_rays)

                    # Update the original ray object with the propagated rays
                    self.rays.position[mask] = propagated_rays.position
                    self.rays.direction[mask] = propagated_rays.direction

                    masked_rays.record_state(surface_name=surface.name)