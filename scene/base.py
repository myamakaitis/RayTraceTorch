import torch
import torch.nn as nn
import numpy as np

class Scene(nn.Module):

    def __init__(self):
        super().__init__()

        self.elements = nn.ModuleList()
        self.rays = None
        self.Nbounces = 100

        self._build_index_maps()

    def _build_index_maps(self):
        """
        Flattens the hierarchy of Element -> Surfaces into global indices
        to allow vectorized lookups.
        """
        elem_indices = []
        surf_indices = []

        for k, element in enumerate(self.elements):
            # We assume element.shape returns a Shape object which has a known number of surfaces
            # If explicit surface lists aren't exposed, we rely on the specific implementation
            # defined in the Architecture.md (e.g., Doublet has 3 surfaces)
            # Here we assume element.shape.surfaces is a list or similar iterable.
            num_surfaces = len(element.shape)

            elem_indices.append(torch.full((num_surfaces,), k, dtype=torch.long))
            surf_indices.append(torch.arange(num_surfaces, dtype=torch.long))

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

    def step(self):
        """
        Performs one bounce for all rays simultaneously.
        """
        # 1. INTERSECTION PHASE (Detached / No Grad)
        # We need to find WHICH element and WHICH surface every ray hit.
        with torch.no_grad():
            t_candidates = []

            # Collect intersection distances from all elements
            for element in self.elements:
                # element.intersectTest returns [N_rays, N_surfaces_in_element]
                t_candidates.append(element.intersectTest(self.rays))

            # Concatenate into one dense matrix: [N_rays, Total_Surfaces_In_Scene]
            t_matrix = torch.cat(t_candidates, dim=1)

            # Find the closest surface for each ray
            # min_t: distance to closest hit
            # global_hit_idx: index (0 to Total_Surfaces) of the hit
            min_t, global_hit_idx = torch.min(t_matrix, dim=1)

            # Identify rays that actually hit something (t < inf)
            hit_mask = (min_t < float('inf')) & (self.rays.intensity > 0)

            # If no rays hit anything, we are done
            if not hit_mask.any():
                return

            # Retrieve the specific Element ID and Internal Surface ID for the winners
            # We use the pre-computed lookups.
            winner_element_ids = self.map_to_element[global_hit_idx]
            winner_surf_ids = self.map_to_surface[global_hit_idx]

        # 2. PHYSICS PHASE (Differentiable)
        # We calculate the next state for the rays.
        # We initiate "accumulators" for the new ray states.
        # We use zeros_like to ensure device/dtype compatibility.

        next_pos = torch.zeros_like(self.rays.pos)
        next_dir = torch.zeros_like(self.rays.dir)
        next_intensity = torch.zeros_like(self.rays.intensity)

        unique_active_elements = torch.unique(winner_element_ids[hit_mask])

        for k_tensor in unique_active_elements:

            k = k_tensor.item()  # Element Index
            element = self.elements[k]

            # Mask for all rays hitting Element K
            # We intersect with hit_mask to ensure we don't include infinite misses
            elem_mask = (winner_element_ids == k) & hit_mask

            # OPTIMIZATION:
            # Instead of passing a tensor of surf_ids to the element (which breaks),
            # we find which specific surfaces of this element were hit.
            # E.g., for a lens, did we hit Surface 0 (Front) or Surface 2 (Edge)?

            # Extract the surface indices for hits on this element
            surfs_hit_on_this_element = winner_surf_ids[elem_mask]
            unique_surfs = torch.unique(surfs_hit_on_this_element)

            for j_tensor in unique_surfs:
                j = j_tensor.item()  # Surface Index (Scalar)

                # 2a. Precise Masking
                # Rays hitting Element K AND Surface J
                # This subset defines a group of rays undergoing the EXACT same physics/math.
                specific_mask = elem_mask & (winner_surf_ids == j)

                # 2b. Gather Data
                ray_subset = self.rays.subset(specific_mask)

                # 2c. Compute Physics (Standard non-vectorized-index call)
                # Now we pass 'j' as an int, which fits your Element.forward signature
                out_pos, out_dir, out_intensity = element(ray_subset, j)

                # 2d. Accumulate (Scatter)
                # Place results back into the global tensors
                next_pos = next_pos.masked_scatter(specific_mask.unsqueeze(-1), out_pos)
                next_dir = next_dir.masked_scatter(specific_mask.unsqueeze(-1), out_dir)
                next_intensity = next_intensity.masked_scatter(specific_mask, out_intensity)

            # --- 3. FINAL UPDATE ---
            # Update the main Rays object with the aggregated results
        self.rays.update(
            mask=hit_mask,
            new_pos=next_pos,
            new_dir=next_dir,
            new_intensity=next_intensity
        )