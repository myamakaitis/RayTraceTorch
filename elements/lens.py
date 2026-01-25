import torch
import torch.nn as nn

from geom import RayTransform
from .parent import Element
from ..phys import RefractSnell, RefractFresnel, Block
from ..geom import Singlet, CylSinglet, Doublet, Triplet
from .ideal import ParaxialRefractMat

class SingletLens(Element):

    def __init__(self, c1: float, c2: float,
                 d: float, t: float,
                 ior_glass: float, ior_media: float = 1.0,

                 c1_grad: bool = False, c2_grad: bool = False,
                 t_grad: bool = False, d_grad: bool = False,
                 ior_glass_grad: bool = False, ior_media_grad: bool = False,

                 fresnel: bool = False, inked: bool =False,
                 transform: RayTransform=None):

        super().__init__()

        self.ior_glass = nn.Parameter(torch.as_tensor(ior_glass), requires_grad=ior_glass_grad)
        self.ior_media = nn.Parameter(torch.as_tensor(ior_media), requires_grad=ior_media_grad)

        self.shape = Singlet(C1 = c1, C2 = c2, D=d, T=t,
                             C1_grad = c1_grad, C2_grad = c2_grad,
                             D_grad=d_grad, T_grad=t_grad,
                             transform=transform)

        if fresnel:
            refractFunc = RefractFresnel
        else:
            refractFunc = RefractSnell

        sf1 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf1.ior_in = self.ior_glass
        sf1.ior_out = self.ior_media
        self.surface_functions.append(sf1)

        sf2 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf2.ior_out = self.ior_glass
        sf2.ior_in = self.ior_media
        self.surface_functions.append(sf2)

        if inked:
            sf3 = Block()
        else:
            sf3 = refractFunc(ior_in=0.0, ior_out=0.0)
            sf3.ior_in = self.ior_glass
            sf3.ior_out = self.ior_media

        self.surface_functions.append(sf3)

    @property
    def Power(self):
        return self.power1 + self.power2 - self.power1*self.power2*(self.T/self.ior_glass)

    @property
    def power1(self):
        return self.shape.surfaces[0].c * (self.ior_glass - self.ior_media)

    @property
    def power2(self):
        return self.shape.surfaces[1].c * (self.ior_media - self.ior_glass)

    @property
    def f(self):
        """Effective focal length"""
        return 1 / self.Power

    @property
    def f_bfl(self):
        """Back focal length"""
        # BFL = f * (1 - T * phi1 / n)
        # phi1 = (ior_glass - ior_media) * c1
        n = self.ior_glass
        phi1 = (n - self.ior_media) * self.shape.surfaces[0].c

        return self.f * (1 - self.T * phi1 / n)

    @property
    def f_ffl(self):
        # FFL = -f * (1 - T * phi2 / n)
        # phi2 = (ior_glass - ior_media) * c
        n = self.ior_glass
        return -self.f * (1 - self.T * self.power2/ n)

    @property
    def R1(self):
        """Radius of front lens"""
        return 1 / self.shape.surfaces[0].c

    @property
    def R2(self):
        """Radius of rear lens"""
        return -1 / self.shape.surfaces[1].c

    @property
    def T(self):
        return self.shape.T

    @property
    def T_edge(self):
        return self.shape.T_edge

    @property
    def P1z(self):
        """element z-location of 1st principle plane"""
        # H1 is shifted from vertex of surface[0] by h1 = f * T * phi2 / n

        h1 = -self.f * (self.ior_glass - self.ior_media) * self.T * self.shape.surfaces[1].c / self.ior_glass

        return self.shape.surfaces[0].z + h1

    @property
    def P2z(self):
        """element z-location of 2nd principle plane"""
        # H2 is shifted from vertex of surface[1] by h2 = -f * T * phi1 / n
        h2 = -self.f * (self.ior_glass - self.ior_media) * self.T * self.shape.surfaces[0].c / self.ior_glass

        return self.shape.surfaces[1].z + h2

    def getParaxial(self):

        # Get the paraxial approximation of the lense, i.e. the two locations of the principle points
        # And the linear transforms at those points.
        # Distance matrix is filled in later

        Zs = [self.shape.z + self.shape.surfaces[0].z, self.shape.z + self.shape.surfaces[1].z]

        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        C1 = self.shape.surfaces[0].c
        C2 = self.shape.surfaces[1].c
        Mat1 = ParaxialRefractMat(C1, C1, self.ior_media, self.ior_glass)
        Mat2 = ParaxialRefractMat(C2, C2, self.ior_glass, self.ior_media)

        Mats = [T_inv @ Mat1 @ T, T_inv @ Mat2 @ T]

        return Zs, Mats


    def Bend(self, delta_c):
        """
        Adjust R1 and R2 without changing total Power (Focal Length).
        Adds 'delta_c' to c1 and solves for the new c2.

        Args:
            delta_c (float or Tensor): Change in curvature to apply to the front surface.
        """
        with torch.no_grad():
            current_power = self.Power
            n = self.ior_glass
            n0 = self.ior_media

            # Constants for the power equation: P = k * (C1 + C2 - D * C1 * C2)
            k = (n - n0)
            D = self.T * (n - n0) / n

            # The target sum S = C1 + C2(1 - D*C1) = P/k
            target_S = current_power / k

            # New C1
            c1_new = self.shape.C1 + delta_c

            # Solve for C2: C2 = (S - C1) / (1 - D*C1)
            denom = 1.0 - D * c1_new
            if torch.abs(denom) < 1e-6:
                raise ValueError("Bending results in singular configuration.")

            c2_new = (target_S - c1_new) / denom

            # Update the underlying shape parameters
            self.shape.C1.copy_(c1_new)
            self.shape.C2.copy_(c2_new)

    def __repr__(self):
        return f"{self.__name__} - f = {self.f}"


class CylSingletLens(SingletLens):

    def __init__(self, c1, c2, height, width, t,
                 ior_glass, ior_media = 1.0,
                 c1_grad = False, c2_grad = False,
                 t_grad = False, height_grad = False, width_grad = False,
                 ior_glass_grad = False, ior_media_grad = False,
                 fresnel = False, inked=False, transform: RayTransform=None):

        super().__init__(c1, c2, height, t,
                 ior_glass, ior_media = ior_media,
                 c1_grad = False, c2_grad = False,
                 t_grad = False, d_grad = False,
                 ior_glass_grad = ior_glass_grad, ior_media_grad = ior_media_grad,
                 fresnel = fresnel, inked=inked, transform=None)

        self.shape = CylSinglet(C1 = c1, C2 = c2, height=height, width=width, T=t,
                             C1_grad = c1_grad, C2_grad = c2_grad,
                             h_grad=height_grad, w_grad=width_grad, T_grad=t_grad,
                             transform=transform)

        for _ in range(3):
            self.surface_functions.append(self.surface_functions[-1])
        self.Nsurfaces = 6

    def getParaxial(self):

        # Get the paraxial approximation of the lense, i.e. the two locations of the principle points
        # And the linear transforms at those points.
        # Distance matrix is filled in later

        Zs = [self.shape.z + self.shape.surfaces[0].z, self.shape.z + self.shape.surfaces[1].z]

        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        C1 = self.shape.surfaces[0].c
        C2 = self.shape.surfaces[1].c
        Mat1 = ParaxialRefractMat(0.0, C1, self.ior_media, self.ior_glass)
        Mat2 = ParaxialRefractMat(0.0, C2, self.ior_glass, self.ior_media)

        Mats = [T_inv @ Mat1 @ T, T_inv @ Mat2 @ T]

        return Zs, Mats


class DoubletLens(Element):

    def __init__(self, c1, c2, c3, d, t1, t2,
                 ior_glass1, ior_glass2, ior_media=1.0,
                 c1_grad=False, c2_grad=False, c3_grad=False,
                 t1_grad=False, t2_grad=False, d_grad=False,
                 ior_glass1_grad=False, ior_glass2_grad=False, ior_media_grad=False,
                 fresnel=False, inked=True, transform: RayTransform=None):

        super().__init__()

        # Register Optical Parameters
        self.ior_glass1 = nn.Parameter(torch.as_tensor(ior_glass1), requires_grad=ior_glass1_grad)
        self.ior_glass2 = nn.Parameter(torch.as_tensor(ior_glass2), requires_grad=ior_glass2_grad)
        self.ior_media = nn.Parameter(torch.as_tensor(ior_media), requires_grad=ior_media_grad)

        # Initialize Geometry
        self.shape = Doublet(C1=c1, C2=c2, C3=c3,
                             D=d, T1=t1, T2=t2,
                             C1_grad=c1_grad, C2_grad=c2_grad, C3_grad=c3_grad,
                             D_grad=d_grad, T1_grad=t1_grad, T2_grad=t2_grad,
                             transform=transform)

        # Determine Refraction Model
        if fresnel:
            refractFunc = RefractFresnel
        else:
            refractFunc = RefractSnell

        # --- Surface 1: Media -> Glass 1 ---
        sf1 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf1.ior_in = self.ior_media
        sf1.ior_out = self.ior_glass1
        self.surface_functions.append(sf1)

        # --- Surface 2: Glass 1 -> Glass 2 (Cemented Interface) ---
        sf2 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf2.ior_in = self.ior_glass1
        sf2.ior_out = self.ior_glass2
        self.surface_functions.append(sf2)

        # --- Surface 3: Glass 2 -> Media ---
        sf3 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf3.ior_in = self.ior_glass2
        sf3.ior_out = self.ior_media
        self.surface_functions.append(sf3)

        # --- Surface 4: Edge (Mechanical) ---
        self.surface_functions.append(Block())
        self.surface_functions.append(Block())

    # --- Geometric Accessors ---
    @property
    def R1(self):
        """Radius of front surface"""
        return 1.0 / self.shape.surfaces[0].c
    @property
    def R2(self):
        """Radius of internal surface"""
        return 1.0 / self.shape.surfaces[1].c
    @property
    def R3(self):
        """Radius of surface 3"""
        return -1.0 / self.shape.surfaces[2].c

    @property
    def T1(self):
        return self.shape.T1
    @property
    def T2(self):
        return self.shape.T2

    def getParaxial(self):

        # Get the paraxial approximation of the lense, i.e. the two locations of the principle points
        # And the linear transforms at those points.
        # Distance matrix is filled in later
        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        Zs = [self.shape.z + self.shape.surfaces[i].z for i in range(3)]
        Cs = [self.shape.z + self.shape.surfaces[i].z for i in range(3)]

        iors = [self.ior_media, self.ior_glass1, self.ior_glass2, self.ior_media]

        Mats = []
        for i in range(3):

            M = ParaxialRefractMat(Cs[i], Cs[i], iors[i], iors[i+1])
            Mats.append(T_inv @ M @ T)

        return Zs, Mats


class TripletLens(Element):

    def __init__(self, c1, c2, c3, c4, d, t1, t2, t3,
                 ior_glass1, ior_glass2, ior_glass3, ior_media=1.0,
                 c1_grad=False, c2_grad=False, c3_grad=False, c4_grad=False,
                 t1_grad=False, t2_grad=False, t3_grad=False, d_grad=False,
                 ior_glass1_grad=False, ior_glass2_grad=False,
                 ior_glass3_grad=False, ior_media_grad=False,
                 fresnel=False, inked=True, transform: RayTransform=None):

        super().__init__()

        # Register Optical Parameters
        self.ior_glass1 = nn.Parameter(torch.as_tensor(ior_glass1), requires_grad=ior_glass1_grad)
        self.ior_glass2 = nn.Parameter(torch.as_tensor(ior_glass2), requires_grad=ior_glass2_grad)
        self.ior_glass3 = nn.Parameter(torch.as_tensor(ior_glass3), requires_grad=ior_glass3_grad)
        self.ior_media = nn.Parameter(torch.as_tensor(ior_media), requires_grad=ior_media_grad)

        # Initialize Geometry
        self.shape = Triplet(C1=c1, C2=c2, C3=c3, C4=c4,
                             D=d, T1=t1, T2=t2, T3=t3,
                             C1_grad=c1_grad, C2_grad=c2_grad,
                             C3_grad=c3_grad, C4_grad=c4_grad,
                             D_grad=d_grad,
                             T1_grad=t1_grad, T2_grad=t2_grad, T3_grad=t3_grad,
                             transform=transform)

        # Determine Refraction Model
        if fresnel:
            refractFunc = RefractFresnel
        else:
            refractFunc = RefractSnell

        # --- Surface 1: Media -> Glass 1 ---
        sf1 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf1.ior_in = self.ior_media
        sf1.ior_out = self.ior_glass1
        self.surface_functions.append(sf1)

        # --- Surface 2: Glass 1 -> Glass 2 ---
        sf2 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf2.ior_in = self.ior_glass1
        sf2.ior_out = self.ior_glass2
        self.surface_functions.append(sf2)

        # --- Surface 3: Glass 2 -> Glass 3 ---
        sf3 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf3.ior_in = self.ior_glass2
        sf3.ior_out = self.ior_glass3
        self.surface_functions.append(sf3)

        # --- Surface 4: Glass 3 -> Media ---
        sf4 = refractFunc(ior_in=0.0, ior_out=0.0)
        sf4.ior_in = self.ior_glass3
        sf4.ior_out = self.ior_media
        self.surface_functions.append(sf4)

        # --- Surface 5: Edge ---
        for _ in range(3):
            self.surface_functions.append(Block())

    # --- Geometric Accessors ---
    @property
    def R1(self):
        return 1.0 / self.shape.surfaces[0].c
    @property
    def R2(self):
        return 1.0 / self.shape.surfaces[1].c
    @property
    def R3(self):
        return -1.0 / self.shape.surfaces[2].c
    @property
    def R4(self):
        return -1.0 / self.shape.surfaces[3].c

    @property
    def T1(self):
        return self.shape.T1
    @property
    def T2(self):
        return self.shape.T2
    @property
    def T3(self):
        return self.shape.T3

    def getParaxial(self):

        # Get the paraxial approximation of the lense, i.e. the two locations of the principle points
        # And the linear transforms at those points.
        # Distance matrix is filled in later
        T = self.shape.transform.paraxial()
        T_inv = self.shape.transform.paraxial_inv()

        Zs = [self.shape.z + self.shape.surfaces[i].z for i in range(4)]
        Cs = [self.shape.z + self.shape.surfaces[i].z for i in range(4)]

        iors = [self.ior_media, self.ior_glass1, self.ior_glass2, self.ior_glass3, self.ior_media]

        Mats = []
        for i in range(4):

            M = ParaxialRefractMat(Cs[i], Cs[i], iors[i], iors[i+1])
            Mats.append(T_inv @ M @ T)

        return Zs, Mats