import torch
import torch.nn as nn
from ..elements import Element
from ..rays import Bundle
from ..rays import Rays


class Scene(nn.Module):

    def __init__(self):
        super().__init__()

        self.elements = nn.ModuleList()
        self.bundles: nn.ModuleList = nn.ModuleList()
        self._bundle_N_rays: list[int] = []   # parallel to self.bundles — rays per bundle
        self.rays = None
        self.Nbounces = 100

        self._build_index_maps()

    # ------------------------------------------------------------------
    # Scene population
    # ------------------------------------------------------------------

    def add_element(self, element: Element):
        """Append an optical element to the scene."""
        self.elements.append(element)

    def add_bundle(self, bundle: Bundle, N_rays: int = 200):
        """Register a bundle and the number of rays to draw from it."""
        self.bundles.append(bundle)
        self._bundle_N_rays.append(N_rays)

    # ------------------------------------------------------------------
    # Clear helpers
    # ------------------------------------------------------------------

    def clear_elements(self):
        """Remove all elements and reset the surface-index buffers."""
        self.elements = nn.ModuleList()
        self._build_index_maps()

    def clear_bundles(self):
        """Remove all bundles and discard the current Rays object."""
        self.bundles = nn.ModuleList()
        self._bundle_N_rays = []
        self.rays = None

    def clear_rays(self):
        """Discard the merged Rays object without removing the bundles."""
        self.rays = None

    # ------------------------------------------------------------------
    # Ray construction
    # ------------------------------------------------------------------

    def _build_rays(self):
        """
        Sample every registered bundle and merge the results into a single
        Rays tensorclass stored at ``self.rays``.

        Each bundle is sampled independently so it preserves its own
        ``ray_id``, wavelength distribution, and spatial transform.  The
        resulting tensors are concatenated along the N (ray) dimension.
        """
        if len(self.bundles) == 0:
            self.rays = None
            return

        batches = [
            bundle.sample(N)
            for bundle, N in zip(self.bundles, self._bundle_N_rays)
        ]

        if len(batches) == 1:
            self.rays = batches[0]
            return

        # Concatenate all per-bundle fields along the ray dimension
        pos        = torch.cat([r.pos        for r in batches], dim=0)
        direction  = torch.cat([r.dir        for r in batches], dim=0)
        intensity  = torch.cat([r.intensity  for r in batches], dim=0)
        ray_id     = torch.cat([r.id         for r in batches], dim=0)
        wavelength = torch.cat([r.wavelength for r in batches], dim=0)

        self.rays = Rays(
            pos=pos, dir=direction, intensity=intensity,
            id=ray_id, wavelength=wavelength,
            batch_size=[pos.shape[0]],
        )

    # ------------------------------------------------------------------
    # Geometry bookkeeping
    # ------------------------------------------------------------------

    def _build_index_maps(self):
        """
        Flatten the Element → Surface hierarchy into contiguous index buffers
        for vectorised ray-casting lookups.
        """
        if hasattr(self, 'map_to_element') and isinstance(self.map_to_element, torch.Tensor):
            device = self.map_to_element.device
        else:
            device = torch.device('cpu')

        if len(self.elements) == 0:
            empty_long = torch.tensor([], dtype=torch.long)
            self.register_buffer('map_to_element', empty_long)
            self.register_buffer('map_to_surface', empty_long)
            self.total_surfaces = 0
            return

        elem_indices = []
        surf_indices = []

        for k, element in enumerate(self.elements):
            num_surfaces = len(element.shape)
            elem_indices.append(torch.full((num_surfaces,), k, dtype=torch.long, device=device))
            surf_indices.append(torch.arange(num_surfaces, dtype=torch.long, device=device))

        self.register_buffer('map_to_element', torch.cat(elem_indices))
        self.register_buffer('map_to_surface', torch.cat(surf_indices))
        self.total_surfaces = self.map_to_element.size(0)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self):
        """
        Propagate ``self.rays`` for up to ``Nbounces`` bounces.
        Calls ``_build_rays()`` automatically if rays have not been built yet.
        """
        if self.rays is None:
            self._build_rays()
        if self.rays is None:
            return
        self._build_index_maps()
        for _ in range(self.Nbounces):
            if not (self.rays.intensity > 0).any():
                break
            self.step()

    def ray_cast(self, rays):
        """
        Find the closest surface hit for every ray.

        Accepts either a plain ``Rays`` tensorclass or a ``Paths`` proxy —
        the proxy is unwrapped before being handed to element intersection
        code so that element physics never needs to be aware of ``Paths``.

        Returns
        -------
        ``(hit_mask, winner_element_ids, winner_surf_ids)`` or ``None`` if
        there are no hits.
        """
        if len(self.elements) == 0:
            return None

        # Unwrap Paths → Rays so elements only ever see the raw tensorclass
        raw = rays.unwrap() if hasattr(rays, 'unwrap') else rays

        with torch.no_grad():
            t_candidates = [element.intersectTest(raw) for element in self.elements]
            if not t_candidates:
                return None

            t_matrix = torch.cat(t_candidates, dim=1)   # [N_rays, Total_surfaces]
            min_t, global_hit_idx = torch.min(t_matrix, dim=1)
            hit_mask = min_t < float('inf')

            if not hit_mask.any():
                return None

            winner_element_ids = self.map_to_element[global_hit_idx]
            winner_surf_ids    = self.map_to_surface[global_hit_idx]

            return hit_mask, winner_element_ids, winner_surf_ids

    def step(self):
        """Perform one physics bounce for all active rays."""
        if self.rays is None:
            return

        result = self.ray_cast(self.rays)
        if result is None:
            return

        hit_mask, winner_element_ids, winner_surf_ids = result
        active_mask = hit_mask & (self.rays.intensity > 0)

        if not active_mask.any():
            return

        next_pos       = torch.zeros_like(self.rays.pos)
        next_dir       = torch.zeros_like(self.rays.dir)
        next_intensity = torch.zeros_like(self.rays.intensity)

        for k_tensor in torch.unique(winner_element_ids[active_mask]):
            k         = k_tensor.item()
            element   = self.elements[k]
            elem_mask = (winner_element_ids == k) & active_mask

            for j_tensor in torch.unique(winner_surf_ids[elem_mask]):
                j             = j_tensor.item()
                specific_mask = elem_mask & (winner_surf_ids == j)

                ray_subset = self.rays[specific_mask]
                out_pos, out_dir, out_intensity = element(ray_subset, j)

                next_pos       = next_pos.masked_scatter(specific_mask.unsqueeze(-1), out_pos)
                next_dir       = next_dir.masked_scatter(specific_mask.unsqueeze(-1), out_dir)
                next_intensity = next_intensity.masked_scatter(specific_mask, out_intensity)

        # scatter_update uses index_put with the boolean mask, so it expects
        # value tensors of shape [M, ...] where M = active_mask.sum(), not [N, ...].
        self.rays.scatter_update(
            active_mask,
            next_pos[active_mask],
            next_dir[active_mask],
            next_intensity[active_mask],
        )
