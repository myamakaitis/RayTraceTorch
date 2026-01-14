import torch
import torch.nn as nn

from .parent import Element
from ..phys import RefractSnell, RefractFresnel, Block
from ..geom import Singlet, CylSinglet

class SingletLens(Element):

    def __init__(self, c1, c2, d, t,
                 ior_glass, ior_media = 1.0,
                 c1_grad = False, c2_grad = False,
                 t_grad = False, d_grad = False,
                 ior_glass_grad = False, ior_media_grad = False,
                 fresnel = False, inked=False, transform=None):

        super().__init__()

        self.n_glass = nn.Parameter(torch.as_tensor(ior_glass), requires_grad=ior_glass_grad)
        self.n_media = nn.Parameter(torch.as_tensor(ior_media), requires_grad=ior_media_grad)

        self.shape = Singlet(C1 = c1, C2 = c2, D=d, T=t,
                             C1_grad = c1_grad, C2_grad = c2_grad,
                             D_grad=d_grad, T_grad=t_grad,
                             transform=transform)

        if fresnel:
            refractFunc = RefractFresnel
        else:
            refractFunc = RefractSnell

        sf1 = refractFunc(n_in=0.0, n_out=0.0)
        sf1.n_in = self.n_glass
        sf1.n_out = self.n_media
        self.surface_functions.append(sf1)

        sf2 = refractFunc(n_in=0.0, n_out=0.0)
        sf2.n_out = self.n_glass
        sf2.n_in = self.n_media
        self.surface_functions.append(sf2)

        if inked:
            sf3 = Block()
        else:
            sf3 = refractFunc(n_in=0.0, n_out=0.0)
            sf3.n_in = self.n_glass
            sf3.n_out = self.n_media

        self.surface_functions.append(sf3)


    def Power(self):
        return self.power1() + self.power2() - self.power1()*self.power2()*(self.T()/self.n_glass)


    def power1(self):
        return self.shape.surfaces[0].c * (self.n_glass - self.n_media)


    def power2(self):
        return self.shape.surfaces[1].c * (self.n_media - self.n_glass)


    def f(self):
        """Effective focal length"""
        return 1 / self.Power()


    def f_bfl(self):
        """Back focal length"""
        # BFL = f * (1 - T * phi1 / n)
        # phi1 = (n_glass - n_media) * c1
        n = self.n_glass
        phi1 = (n - self.n_media) * self.shape.surfaces[0].c

        return self.f() * (1 - self.T() * phi1 / n)


    def f_ffl(self):
        # FFL = -f * (1 - T * phi2 / n)
        # phi2 = (n_glass - n_media) * c
        n = self.n_glass
        return -self.f() * (1 - self.T() * self.power2()/ n)


    def R1(self):
        """Radius of front lens"""
        return 1 / self.shape.surfaces[0].c


    def R2(self):
        """Radius of rear lens"""
        return -1 / self.shape.surfaces[1].c


    def T(self):
        return self.shape.T


    def T_edge(self):
        return self.shape.T_edge


    def P1z(self):
        """element z-location of 1st principle plane"""
        # H1 is shifted from vertex of surface[0] by h1 = f * T * phi2 / n

        h1 = -self.f() * (self.n_glass - self.n_media) * self.T() * self.shape.surfaces[1].c / self.n_glass

        return self.shape.surfaces[0].transform.trans[2] + h1


    def P2z(self):
        """element z-location of 2nd principle plane"""
        # H2 is shifted from vertex of surface[1] by h2 = -f * T * phi1 / n
        h2 = -self.f() * (self.n_glass - self.n_media) * self.T() * self.shape.surfaces[0].c / self.n_glass

        return self.shape.surfaces[1].transform.trans[2] + h2

    def Bend(self):
        """
        Adjust R1 and R2 without changing total Power (Focal Length).
        Adds 'delta_c' to c1 and solves for the new c2.

        Args:
            delta_c (float or Tensor): Change in curvature to apply to the front surface.
        """
        with torch.no_grad():
            current_power = self.Power
            n = self.n_glass
            n0 = self.n_media

            # Constants for the power equation: P = k * (C1 + C2 - D * C1 * C2)
            k = (n - n0)
            D = self.T() * (n - n0) / n

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

    def status(self):
        print(f"---")
        print(f"R1: {self.R1()}")
        print(f"R2: {self.R2()}")
        print(f"n_medai: {self.n_media}")
        print(f"n_glass: {self.n_glass}")
        print(f"Center T: {self.T()}")
        print("---")
        print(f"EFL: {self.f()}")
        print(f"BLF: {self.f_bfl()}")
        print(f"FFL: {self.f_ffl()}")
        print(f"P1: {self.P1z()}")
        print(f"P2: {self.P2z()}")


class CylSingletLens(SingletLens):

    def __init__(self, c1, c2, height, width, t,
                 ior_glass, ior_media = 1.0,
                 c1_grad = False, c2_grad = False,
                 t_grad = False, h_grad = False, w_grad = False,
                 ior_glass_grad = False, ior_media_grad = False,
                 fresnel = False, inked=False, transform=None):

        super().__init__(c1, c2, height, t,
                 ior_glass, ior_media = ior_media,
                 c1_grad = False, c2_grad = False,
                 t_grad = False, d_grad = False,
                 ior_glass_grad = ior_glass_grad, ior_media_grad = ior_media_grad,
                 fresnel = fresnel, inked=inked, transform=None)

        self.shape = CylSinglet(C1 = c1, C2 = c2, height=height, width=width, T=t,
                             C1_grad = c1_grad, C2_grad = c2_grad,
                             h_grad=h_grad, w_grad=w_grad, T_grad=t_grad,
                             transform=transform)

        for _ in range(3):
            self.surface_functions.append(self.surface_functions[-1])
        self.Nsurfaces = 6


class DoubletLens(Element):

    def __init__(self, c1, c2, c3, d, t1, t2,
                 ior_glass1, ior_glass2, ior_media=1.0,
                 c1_grad=False, c2_grad=False, c3_grad=False,
                 t1_grad=False, t2_grad=False, d_grad=False,
                 ior_glass1_grad=False, ior_glass2_grad=False, ior_media_grad=False,
                 fresnel=False, inked=True, transform=None):

        super().__init__()

        # Register Optical Parameters
        self.n_glass1 = nn.Parameter(torch.as_tensor(ior_glass1), requires_grad=ior_glass1_grad)
        self.n_glass2 = nn.Parameter(torch.as_tensor(ior_glass2), requires_grad=ior_glass2_grad)
        self.n_media = nn.Parameter(torch.as_tensor(ior_media), requires_grad=ior_media_grad)

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
        sf1 = refractFunc(n_in=0.0, n_out=0.0)
        sf1.n_in = self.n_media
        sf1.n_out = self.n_glass1
        self.surface_functions.append(sf1)

        # --- Surface 2: Glass 1 -> Glass 2 (Cemented Interface) ---
        sf2 = refractFunc(n_in=0.0, n_out=0.0)
        sf2.n_in = self.n_glass1
        sf2.n_out = self.n_glass2
        self.surface_functions.append(sf2)

        # --- Surface 3: Glass 2 -> Media ---
        sf3 = refractFunc(n_in=0.0, n_out=0.0)
        sf3.n_in = self.n_glass2
        sf3.n_out = self.n_media
        self.surface_functions.append(sf3)

        # --- Surface 4: Edge (Mechanical) ---
        self.surface_functions.append(Block())
        self.surface_functions.append(Block())

    # --- Geometric Accessors ---

    def R1(self):
        """Radius of front surface"""
        return 1.0 / self.shape.surfaces[0].c

    def R2(self):
        """Radius of internal surface"""
        return 1.0 / self.shape.surfaces[1].c

    def R3(self):
        """Radius of rear surface"""
        return -1.0 / self.shape.surfaces[2].c

    def T1(self):
        return self.shape.T1

    def T2(self):
        return self.shape.T2


class TripletLens(Element):

    def __init__(self, c1, c2, c3, c4, d, t1, t2, t3,
                 ior_glass1, ior_glass2, ior_glass3, ior_media=1.0,
                 c1_grad=False, c2_grad=False, c3_grad=False, c4_grad=False,
                 t1_grad=False, t2_grad=False, t3_grad=False, d_grad=False,
                 ior_glass1_grad=False, ior_glass2_grad=False,
                 ior_glass3_grad=False, ior_media_grad=False,
                 fresnel=False, inked=True, transform=None):

        super().__init__()

        # Register Optical Parameters
        self.n_glass1 = nn.Parameter(torch.as_tensor(ior_glass1), requires_grad=ior_glass1_grad)
        self.n_glass2 = nn.Parameter(torch.as_tensor(ior_glass2), requires_grad=ior_glass2_grad)
        self.n_glass3 = nn.Parameter(torch.as_tensor(ior_glass3), requires_grad=ior_glass3_grad)
        self.n_media = nn.Parameter(torch.as_tensor(ior_media), requires_grad=ior_media_grad)

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
        sf1 = refractFunc(n_in=0.0, n_out=0.0)
        sf1.n_in = self.n_media
        sf1.n_out = self.n_glass1
        self.surface_functions.append(sf1)

        # --- Surface 2: Glass 1 -> Glass 2 ---
        sf2 = refractFunc(n_in=0.0, n_out=0.0)
        sf2.n_in = self.n_glass1
        sf2.n_out = self.n_glass2
        self.surface_functions.append(sf2)

        # --- Surface 3: Glass 2 -> Glass 3 ---
        sf3 = refractFunc(n_in=0.0, n_out=0.0)
        sf3.n_in = self.n_glass2
        sf3.n_out = self.n_glass3
        self.surface_functions.append(sf3)

        # --- Surface 4: Glass 3 -> Media ---
        sf4 = refractFunc(n_in=0.0, n_out=0.0)
        sf4.n_in = self.n_glass3
        sf4.n_out = self.n_media
        self.surface_functions.append(sf4)

        # --- Surface 5: Edge ---
        for _ in range(3):
            self.surface_functions.append(Block())

    # --- Geometric Accessors ---

    def R1(self):
        return 1.0 / self.shape.surfaces[0].c

    def R4(self):
        return -1.0 / self.shape.surfaces[3].c

    def T1(self):
        return self.shape.T1

    def T2(self):
        return self.shape.T2

    def T3(self):
        return self.shape.T3