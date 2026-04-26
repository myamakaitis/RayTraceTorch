import torch
import torch.nn as nn
from typing import List, Optional


# ---------------------------------------------------------------------------
# Log-barrier primitives
# ---------------------------------------------------------------------------

def _log_barrier_lb(x, lb: float):
    """One-sided log-barrier: penalises x → lb from above.  Requires x > lb."""
    return -torch.log(x - lb)


def _log_barrier_ub(x, ub: float):
    """One-sided log-barrier: penalises x → ub from below.  Requires x < ub."""
    return -torch.log(ub - x)


def _log_barrier(x, lb: float, ub: float):
    """Two-sided log-barrier for  lb < x < ub."""
    return -torch.log(x - lb) - torch.log(ub - x)


# ---------------------------------------------------------------------------
# Surface z-position helper
# ---------------------------------------------------------------------------

def _get_optical_z_list(elem):
    """
    Returns a list of global z-position tensors for each optical surface in elem.

    For multi-surface shapes (Singlet, Doublet, Triplet, CylSinglet) the global
    z of optical surface i is:
        elem.shape.transform.trans[2] + elem.shape.surfaces[i].transform.trans[2]

    For single-surface shapes (mirrors, apertures, ideal elements) the list
    contains only elem.shape.transform.trans[2].

    These tensors are differentiable nn.Parameters, enabling log-barrier
    gradients to flow back to thickness/spacing parameters.
    """
    shape  = elem.shape
    base_z = shape.transform.trans[2]

    if hasattr(shape, 'N_optical') and hasattr(shape, 'surfaces'):
        return [base_z + shape.surfaces[i].transform.trans[2]
                for i in range(shape.N_optical)]

    return [base_z]


# ---------------------------------------------------------------------------
# Constraint modules
# ---------------------------------------------------------------------------

class Constraint(nn.Module):
    """Base class for optimization constraints."""
    pass


class ThicknessConstraint(Constraint):
    """
    Log-barrier inequality constraint on the axial distance between consecutive
    optical surfaces within each element.

    For each pair of consecutive optical surfaces (i, i+1) inside an element:
        t_min  <  z_{i+1} - z_i  <  t_max   (if t_max is given)
        t_min  <  z_{i+1} - z_i             (if t_max is None)

    Single-surface elements (mirrors, apertures) have no intra-element pairs
    and contribute zero to the loss.

    Args:
        elements: List of Element objects to constrain.
        t_min:    Minimum allowed surface separation (must be > 0).
        t_max:    Maximum allowed surface separation.  None → only lower bound.
        weight:   Scalar multiplier on the total barrier loss.
    """

    def __init__(self, elements: List, t_min: float,
                 t_max: Optional[float] = None, weight: float = 1.0):
        super().__init__()
        self.elements = elements
        self.t_min    = t_min
        self.t_max    = t_max
        self.weight   = weight

    def forward(self):
        terms = []
        for elem in self.elements:
            z_list = _get_optical_z_list(elem)
            for i in range(len(z_list) - 1):
                thickness = z_list[i + 1] - z_list[i]
                if self.t_max is not None:
                    terms.append(_log_barrier(thickness, self.t_min, self.t_max))
                else:
                    terms.append(_log_barrier_lb(thickness, self.t_min))

        if not terms:
            # No constrained pairs (all single-surface elements).
            # Return a zero on the correct device/dtype.
            ref_z = _get_optical_z_list(self.elements[0])[0]
            return ref_z.new_zeros(())

        return self.weight * sum(terms)


class SpacingConstraint(Constraint):
    """
    Log-barrier inequality constraint on the axial gap between adjacent elements.

    For each consecutive pair (elem_i, elem_{i+1}) in the element list:
        z_first_optical(elem_{i+1}) - z_last_optical(elem_i)  >  d_min

    Args:
        elements: Ordered list of Element objects spanning the system.
        d_min:    Minimum required clear gap between adjacent elements.
        weight:   Scalar multiplier on the total barrier loss.
    """

    def __init__(self, elements: List, d_min: float, weight: float = 1.0):
        super().__init__()
        self.elements = elements
        self.d_min    = d_min
        self.weight   = weight

    def forward(self):
        terms = []
        for i in range(len(self.elements) - 1):
            z_exit  = _get_optical_z_list(self.elements[i])[-1]
            z_entry = _get_optical_z_list(self.elements[i + 1])[0]
            gap = z_entry - z_exit
            terms.append(_log_barrier_lb(gap, self.d_min))

        if not terms:
            ref_z = _get_optical_z_list(self.elements[0])[0]
            return ref_z.new_zeros(())

        return self.weight * sum(terms)


class SystemLengthConstraint(Constraint):
    """
    Log-barrier inequality constraint on total system length.

    System length is defined as:
        L = z_last_optical(elements[-1]) - z_first_optical(elements[0])

    Constraint:  L  <  L_max

    Args:
        elements: Ordered list of Element objects spanning the system.
        L_max:    Maximum allowed system length.
        weight:   Scalar multiplier on the barrier loss.
    """

    def __init__(self, elements: List, L_max: float, weight: float = 1.0):
        super().__init__()
        self.elements = elements
        self.L_max    = L_max
        self.weight   = weight

    def forward(self):
        z_first = _get_optical_z_list(self.elements[0])[0]
        z_last  = _get_optical_z_list(self.elements[-1])[-1]
        length  = z_last - z_first
        return self.weight * _log_barrier_ub(length, self.L_max)
