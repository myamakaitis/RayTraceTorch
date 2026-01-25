import torch
import torch.nn as nn
import numpy as np

class Scene(nn.Module):

    def __init__(self):
        super().__init__()

        self.elements = nn.ModuleList()
        self.rays = torch.tensor(())
        self.Nbounces = 100

        self._build_index_maps()

    def clear_elements(self):
        self.elements = nn.ModuleList()
        self._build_index_maps()

    def clear_rays(self):
        self.rays = nn.ModuleList()

    def _build_index_maps(self):
        """
        Flattens the hierarchy of Element -> Surfaces into global indices
        to allow vectorized lookups.
        """
        if hasattr(self, 'map_to_element') and isinstance(self.map_to_element, torch.Tensor):
            device = self.map_to_element.device
        else:
            device = torch.device('cpu')

        if len(self.elements) == 0:
            # Register empty buffers so downstream code (renderers) doesn't crash looking for attributes
            empty_long = torch.tensor([], dtype=torch.long)
            self.register_buffer('map_to_element', empty_long)
            self.register_buffer('map_to_surface', empty_long)
            self.total_surfaces = 0
            return

        elem_indices = []
        surf_indices = []

        for k, element in enumerate(self.elements):
            num_surfaces = len(element.shape)

            # 2. Create tensors directly on the correct device
            elem_indices.append(torch.full((num_surfaces,), k, dtype=torch.long, device=device))
            surf_indices.append(torch.arange(num_surfaces, dtype=torch.long, device=device))

        # Join them into 1D lookup tensors
        # register_buffer ensures they are moved to GPU with the model but not treated as params
        self.register_buffer('map_to_element', torch.cat(elem_indices))
        self.register_buffer('map_to_surface', torch.cat(surf_indices))
        self.total_surfaces = self.map_to_element.size(0)

    def simulate(self):
        """
        Main simulation loop. Propagates rays for Nbounces.
        """

        self._build_index_maps()
        for _ in range(self.Nbounces):
            # Check if any rays are still active before stepping
            if not (self.rays.intensity > 0).any():
                break
            self.step()

    def ray_cast(self, rays):
        """
        Centralized Intersection Logic.
        Calculates the closest intersection for a batch of rays against all elements.

        Returns:
            tuple: (hit_mask, winner_element_ids, winner_surf_ids) or None
        """
        # Handle empty scene
        if len(self.elements) == 0:
            return None

        # 1. Intersection Test (Detached / No Grad)
        with torch.no_grad():
            t_candidates = []
            for element in self.elements:
                # element.intersectTest returns [N_rays, N_surfaces_in_element]
                t_candidates.append(element.intersectTest(rays))

            if not t_candidates:
                return None

            # [N_rays, Total_Surfaces_In_Scene]
            t_matrix = torch.cat(t_candidates, dim=1)

            # Find closest hit
            min_t, global_hit_idx = torch.min(t_matrix, dim=1)

            # Geometric Hit Mask (t < inf)
            hit_mask = min_t < float('inf')

            if not hit_mask.any():
                return None

            # Retrieve ID mappings (assuming _build_index_maps was called)
            winner_element_ids = self.map_to_element[global_hit_idx]
            winner_surf_ids = self.map_to_surface[global_hit_idx]

            return hit_mask, winner_element_ids, winner_surf_ids

    def step(self):
        """
        Performs one physics bounce.
        """
        # 1. Reuse Centralized Ray Cast
        result = self.ray_cast(self.rays)
        if result is None: return

        hit_mask, winner_element_ids, winner_surf_ids = result

        # 2. Apply Physics Constraints
        # For simulation, we only care about rays that hit AND have intensity
        active_mask = hit_mask & (self.rays.intensity > 0)

        if not active_mask.any():
            return

        # 3. Physics Phase (Differentiable)
        next_pos = torch.zeros_like(self.rays.pos)
        next_dir = torch.zeros_like(self.rays.dir)
        next_intensity = torch.zeros_like(self.rays.intensity)

        unique_active_elements = torch.unique(winner_element_ids[active_mask])

        for k_tensor in unique_active_elements:
            k = k_tensor.item()
            element = self.elements[k]

            # Mask for active rays hitting Element K
            elem_mask = (winner_element_ids == k) & active_mask

            # Extract surfaces hit on this element
            surfs_hit_on_this_element = winner_surf_ids[elem_mask]
            unique_surfs = torch.unique(surfs_hit_on_this_element)

            for j_tensor in unique_surfs:
                j = j_tensor.item()
                specific_mask = elem_mask & (winner_surf_ids == j)

                # Compute Physics
                ray_subset = self.rays.subset(specific_mask)
                out_pos, out_dir, out_intensity = element(ray_subset, j)

                # Accumulate
                next_pos = next_pos.masked_scatter(specific_mask.unsqueeze(-1), out_pos)
                next_dir = next_dir.masked_scatter(specific_mask.unsqueeze(-1), out_dir)
                next_intensity = next_intensity.masked_scatter(specific_mask, out_intensity)

        self.rays.scatter_update(active_mask, next_pos, next_dir, next_intensity)
