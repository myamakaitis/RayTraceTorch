import torch

from typing import Union

from ..geom import Surface, Shape
from .parent import Element
from ..phys import Transmit

class Sensor(Element):

    def __init__(self, shape: Union[Shape, Surface]):

        super().__init__()

        self.shape = shape
        self.surface_functions.extend([Transmit()]*len(shape))

        self.hitLocs = []
        self.hitIntensity = []
        self.hitID = []

    def forward(self, rays, surf_idx):
        # Geometric Intersection (Detailed & Differentiable)
        # shape.forward returns: t, global_hit_point, global_normal
        _, new_pos_global, normal_global, hit_local = self.shape(rays, surf_idx)

        # Surface Physics
        # Get the physics function assigned to this surface
        phys_func = self.surface_functions[surf_idx]

        # Apply physics
        # Returns: new_direction, intensity_modulation
        new_dir_global, intensity_mult = phys_func(hit_local, rays.dir, normal_global)

        self.hitLocs.append(hit_local)
        self.hitIntensity.append(rays.intensity)
        self.hitID.append(rays.id)

        return new_pos_global, new_dir_global, intensity_mult

    def getHitsTensors(self):

        return torch.cat(self.hitLocs, dim=0), torch.cat(self.hitIntensity, dim=0), torch.cat(self.hitID, dim=0),

    def getSpotSizeID_xy(self, ray_id, target_xy = None, norm_ord = 2):

        hit_xyz, hit_intensity, hit_id = self.getHitsTensors()

        id_mask = (hit_id == ray_id)

        hit_xyz_id, hit_intensity_id = hit_xyz[id_mask], hit_intensity[id_mask]

        intensity_sum = torch.sum(hit_intensity)

        # if target unspecified use centroid
        if target_xy is None:
            target_xy = torch.sum(hit_xyz_id[:, 0:2] * hit_intensity_id[:, None]) / intensity_sum

        diff_pos_id = hit_xyz_id[:, 0:2] - target_xy[None, :]

        spot_size_raised = torch.sum(hit_intensity * torch.abs(diff_pos_id)**norm_ord, dim = 0) / intensity_sum

        return spot_size_raised

    def getSpotSizeParallel_xy(self, query_ids, target_xy=None, norm_ord=2):
        """
        Vectorized calculation of spot sizes for a batch of ray IDs.

        Args:
            query_ids (Tensor): Shape (K,). The set of IDs to compute.
            target_xy (Tensor, optional): Shape (K, 2). Pre-specified targets per ID.
                                          If None, centroids are calculated.
            norm_ord (int): Norm order.

        Returns:
            spot_sizes (Tensor): Shape (K,). The result for each query_id.
        """
        # 1. Get all data
        hit_xyz, hit_intensity, hit_id = self.getHitsTensors()

        query_ids = torch.as_tensor(query_ids, device=hit_id.device, dtype=torch.int32)

        # 2. Filter: Keep only hits that belong to the requested query_ids
        # torch.isin requires PyTorch 1.13+. For older versions, use broadcast comparison.
        mask = torch.isin(hit_id, query_ids)

        # Apply mask to reduce data volume immediately
        h_xyz = hit_xyz[mask]  # (M, 3)
        h_int = hit_intensity[mask]  # (M,)
        h_id = hit_id[mask]  # (M,)

        # 3. Map: Find which index [0...K-1] in query_ids each hit corresponds to.
        # We use searchsorted, which requires the "haystack" (query_ids) to be sorted.
        sorted_q_ids, sort_idx = torch.sort(query_ids)

        # group_indices will be values in [0, K-1] pointing to the index in sorted_q_ids
        group_indices = torch.searchsorted(sorted_q_ids, h_id)

        # Initialize output tensors (K groups)
        num_groups = query_ids.shape[0]

        # 4. Reduce: Calculate Intensity Sum (Denominator)
        # shape: (K,)
        intensity_sum = torch.zeros(num_groups, device=hit_id.device, dtype=h_int.dtype)
        intensity_sum = intensity_sum.scatter_add(0, group_indices, h_int)

        # Handle division by zero (for IDs with no hits)
        safe_denom = intensity_sum.clone()
        safe_denom[safe_denom == 0] = 1.0

        # 5. Reduce: Calculate Centroids (if target_xy not provided)
        if target_xy is None:
            # Weighted sum of positions: sum(pos * intensity)
            # We weigh the first 2 columns (x, y) by intensity
            weighted_pos = h_xyz[:, 0:2] * h_int.unsqueeze(1)  # (M, 2)

            pos_sum = torch.zeros(num_groups, 2, device=hit_id.device, dtype=h_xyz.dtype)
            # scatter_add requires 'index' to match dimensions of 'src', so we expand index
            pos_sum.scatter_add_(0, group_indices.unsqueeze(1).expand(-1, 2), weighted_pos)

            # Calculate centroids
            centroids = pos_sum / safe_denom.unsqueeze(1)  # (K, 2)
        else:
            # If target_xy is provided, ensure it aligns with our sorted processing order
            # target_xy was likely provided in the order of original `query_ids`
            # We must reorder it to match `sorted_q_ids`
            centroids = target_xy[sort_idx]

        # 6. Compute Spot Size (Variance/Moment)
        # Gather the calculated centroids back to the individual hit level
        # expanded_centroids shape: (M, 2)
        expanded_centroids = centroids[group_indices]

        # Calculate difference per hit
        diff_pos = h_xyz[:, 0:2] - expanded_centroids

        # Weighted moment: sum(intensity * |diff|^norm)
        weighted_diff = h_int[:, None] * (torch.abs(diff_pos) ** norm_ord)

        # Reduce weighted diffs back to groups
        spot_size_sum = torch.zeros(num_groups, 2, device=hit_id.device, dtype=h_xyz.dtype)
        spot_size_sum = spot_size_sum.scatter_add(0, group_indices.unsqueeze(1).expand(-1, 2), weighted_diff)

        # Final division by total intensity (sum over the 2 dimensions if desired, or keep separate)
        # Your original code did dim=0 sum on the result, resulting in a scalar per ID.
        spot_size_raised = torch.sum(spot_size_sum, dim=1) / (2*safe_denom)

        # 7. Restore Order: The results correspond to sorted_q_ids.
        # We need to unsort them to match the input `query_ids` order.
        # We can use the inverse of the sort_idx.
        results_ordered = torch.empty_like(spot_size_raised)
        results_ordered[sort_idx] = spot_size_raised

        return results_ordered, intensity_sum


