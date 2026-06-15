import scipy.constants as spconsts
from torchgpe.bec2D.potentials import LinearPotential, Union, Callable, any_time_dependent_variable, time_dependent_variable
from torchgpe.utils.potentials import NonLinearPotential
from torchgpe.utils import ftn, iftn
from torch.nn.functional import pad
import numpy as np
import torch

class TwoComponentContact(NonLinearPotential):
    def __init__(self, a_s: float = 100):
        super().__init__()
        self.a_s = a_s

    def on_propagation_begin(self):
        aB = spconsts.codata.value("Bohr radius")
        self._a_s = self.a_s * aB

        ell = self.gas.adim_length

        self._g1 = 4 * np.pi * self.gas.gas1.N_particles * self._a_s / ell
        self._g2 = 4 * np.pi * self.gas.gas2.N_particles * self._a_s / ell

    def potential_function(self, X, Y, Z, psi, time=None):
        psi1 = psi[0]
        psi2 = psi[1]

        Vnl1 = self._g1 * torch.abs(psi1)**2
        Vnl2 = self._g2 * torch.abs(psi2)**2

        return torch.stack((Vnl1, Vnl2))
    
class TwoComponentVariableContact(NonLinearPotential):
    def __init__(self, a_s: float = 100):
        super().__init__()
        self.a_s = a_s

    def on_propagation_begin(self):
        aB = spconsts.codata.value("Bohr radius")
        self._a_s = self.a_s * aB

        ell = self.gas.adim_length

        self._g = 4 * np.pi * self.gas.N_total * self._a_s / ell

    def potential_function(self, X, Y, Z, psi, time=None):
        psi1 = psi[0]
        psi2 = psi[1]

        Vnl1 = self._g * torch.abs(psi1)**2
        Vnl2 = self._g * torch.abs(psi2)**2

        return torch.stack((Vnl1, Vnl2))

class Trap3D(LinearPotential):
    def __init__(
        self,
        omegax: Union[float, Callable],
        omegay: Union[float, Callable],
        omegaz: Union[float, Callable],
    ):
        super().__init__()
        self.omegax = omegax
        self.omegay = omegay
        self.omegaz = omegaz

    def on_propagation_begin(self):
        self.is_time_dependent = any_time_dependent_variable(
            self.omegax,
            self.omegay,
            self.omegaz,
        )
        self._omegax = time_dependent_variable(self.omegax)
        self._omegay = time_dependent_variable(self.omegay)
        self._omegaz = time_dependent_variable(self.omegaz)

    def get_potential(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, time=None):
        return 2 * (np.pi / self.gas.adim_pulse) ** 2 * (
            (self._omegax(time) * X) ** 2
            + (self._omegay(time) * Y) ** 2
            + (self._omegaz(time) * Z) ** 2
        )

class TwoComponentDipoleDipole3D(NonLinearPotential):
    """
    3D dipole-dipole interaction for your TwoComponentGas implementation.

    Assumes:
        gas.psi[0] = psi1
        gas.psi[1] = psi2

    Returns:
        torch.stack((Vdd1, Vdd2))

    where:
        Vdd1 = Phi_11[n1] + Phi_12[n2]
        Vdd2 = Phi_22[n2] + Phi_21[n1]

    Fourier-space kernel:
        K_dd(k) = 3 (e . k_hat)^2 - 1

    Dimensionless coupling:
        g_dd,ij = 4*pi*N_j*a_dd,ij / ell

    where:
        N_j      = atom number of the source component
        a_dd,ij = dipolar length in meters
        ell      = gas.adim_length

    This matches the same unit-normalized convention as:
        g_contact = 4*pi*N*a_s/ell
    """

    def __init__(
        self,
        a_dd11_bohr: float = 0.0,
        a_dd22_bohr: float = 0.0,
        a_dd12_bohr: float = 0.0,
        polarization=(0.0, 0.0, 1.0),
    ):
        super().__init__()

        self.a_dd11_bohr = a_dd11_bohr
        self.a_dd22_bohr = a_dd22_bohr
        self.a_dd12_bohr = a_dd12_bohr

        self.polarization = polarization

    def on_propagation_begin(self):
        """
        Called automatically by gas.ground_state(...) before propagation.
        """
        if not hasattr(self.gas, "gas1") or not hasattr(self.gas, "gas2"):
            raise TypeError(
                "TwoComponentDipoleDipole3D expects a TwoComponentGas object "
                "with gas.gas1 and gas.gas2."
            )

        self.base = self.gas.gas1

        self.N = self.base.N_grid

        # Your Gas class pads real-space arrays from N to:
        # N + 2*(N//2), which is 2N for even N.
        self.M = self.base.Kx.shape[0]
        self.h = (self.M - self.N) // 2

        self.device = self.base.device
        self.float_dtype = self.base.float_dtype
        self.complex_dtype = self.base.complex_dtype

        a0 = spconsts.codata.value("Bohr radius")
        ell = self.base.adim_length

        N1 = self.gas.gas1.N_particles
        N2 = self.gas.gas2.N_particles

        a_dd11 = self.a_dd11_bohr * a0
        a_dd22 = self.a_dd22_bohr * a0
        a_dd12 = self.a_dd12_bohr * a0

        # Self-dipolar terms
        self._gdd11 = 4.0 * np.pi * N1 * a_dd11 / ell
        self._gdd22 = 4.0 * np.pi * N2 * a_dd22 / ell

        # Cross-dipolar terms.
        # Component 1 feels density of component 2, so source atom number is N2.
        # Component 2 feels density of component 1, so source atom number is N1.
        self._gdd12_on_1 = 4.0 * np.pi * N2 * a_dd12 / ell
        self._gdd12_on_2 = 4.0 * np.pi * N1 * a_dd12 / ell

        self._build_kspace_kernel()

    def _build_kspace_kernel(self):
        """
        Build the shifted Fourier-space dipolar kernel.

        Since we use torchgpe.utils.ftn / iftn, we use the already-shifted
        Kx, Ky, Kz stored in the gas object directly.
        """
        Kx = self.base.Kx
        Ky = self.base.Ky
        Kz = self.base.Kz

        e = torch.as_tensor(
            self.polarization,
            dtype=self.float_dtype,
            device=self.device,
        )

        e_norm = torch.linalg.norm(e)
        if e_norm == 0:
            raise ValueError("polarization vector cannot be zero")

        e = e / e_norm

        k2 = Kx**2 + Ky**2 + Kz**2
        k_dot_e = e[0] * Kx + e[1] * Ky + e[2] * Kz

        kernel = torch.zeros_like(k2, dtype=self.float_dtype, device=self.device)

        mask = k2 > 0
        kernel[mask] = 3.0 * k_dot_e[mask] ** 2 / k2[mask] - 1.0

        # Free-space convention: the angular average of the DDI is zero,
        # so set the k=0 value to zero.
        kernel[~mask] = 0.0

        self._kernel = kernel

    def _pad_density(self, density):
        """
        Pad density from shape (N, N, N) to the same shape as gas.Kx.

        For a 3D tensor, torch.nn.functional.pad uses:
            (z_left, z_right, y_left, y_right, x_left, x_right)

        Since your grid has the same N_grid along each direction, the same
        h works for all axes.
        """
        h = self.h

        density_padded = pad(
            density,
            (h, h, h, h, h, h),
            mode="constant",
            value=0.0,
        )

        if density_padded.shape != self._kernel.shape:
            raise RuntimeError(
                f"Padded density shape {density_padded.shape} does not match "
                f"kernel shape {self._kernel.shape}."
            )

        return density_padded

    def _crop_potential(self, potential_padded):
        """
        Crop padded potential back to the physical real-space grid.
        """
        h = self.h
        n = self.N

        return potential_padded[h:h+n, h:h+n, h:h+n]

    def _dipolar_convolution(self, density, gdd):
        """
        Compute:

            Phi_dd = gdd * iftn[ K_dd(k) * ftn[n(r)] ]

        using the same ftn/iftn convention as your Gas class.
        """
        if abs(gdd) == 0.0:
            return torch.zeros_like(density)

        density_padded = self._pad_density(density)

        # ftn/iftn should use the same shifted convention as gas.psik.
        density_k = ftn(density_padded.to(self.complex_dtype))

        phi_padded = iftn(self._kernel * density_k).real

        phi = self._crop_potential(phi_padded)

        return gdd * phi

    def potential_function(self, X, Y, Z, psi, time=None):
        """
        Called by your propagation code as:

            potential.potential_function(*gas.coordinates, gas.psi)

        It must return shape:
            (2, N_grid, N_grid, N_grid)
        """
        psi1 = psi[0]
        psi2 = psi[1]

        n1 = torch.abs(psi1) ** 2
        n2 = torch.abs(psi2) ** 2

        Vdd1 = (
            self._dipolar_convolution(n1, self._gdd11)
            + self._dipolar_convolution(n2, self._gdd12_on_1)
        )

        Vdd2 = (
            self._dipolar_convolution(n2, self._gdd22)
            + self._dipolar_convolution(n1, self._gdd12_on_2)
        )

        return torch.stack((Vdd1, Vdd2))