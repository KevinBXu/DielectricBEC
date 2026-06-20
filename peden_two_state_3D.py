import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.constants as spconsts

from torch.nn.functional import pad

from torchgpe.utils import ftn, iftn
from torchgpe.utils.callbacks import Callback
from torchgpe.utils.potentials import LinearPotential, NonLinearPotential

from two_component_variable_gas import TwoComponentGas

from two_state_propagation import as_two_component_potential


DEBYE_TO_CM = 3.33564e-30  # Coulomb meter


def set_two_component_molecular_mass(gas, mass_amu):
    """
    Your Gas class gets its mass from torchgpe's atomic element table.
    For polar molecules such as RbCs, ThO, or SrF, patch the molecular mass
    after constructing TwoComponentGas.

    Parameters
    ----------
    gas:
        Your TwoComponentGas object.

    mass_amu:
        Molecular mass in atomic mass units.
        Example:
            RbCs ~ 220 amu
            SrF  ~ 107 amu
            ThO  ~ 248 amu
    """
    mass_kg = mass_amu * spconsts.atomic_mass

    for g in (gas.gas1, gas.gas2):
        g.mass = mass_kg
        g.adim_pulse = spconsts.hbar / (mass_kg * g.adim_length**2)

    gas.adim_pulse = gas.gas1.adim_pulse

    return mass_kg


class Trap3D(LinearPotential):
    """
    Dimensionless 3D harmonic trap.

    Physical trap:
        V_phys = 1/2 m (omega_x^2 x^2 + omega_y^2 y^2 + omega_z^2 z^2)

    Dimensionless coordinates:
        X = x / ell

    Dimensionless potential:
        V_dim = V_phys / (hbar * omega_ell)
              = 1/2 [ (omega_x/omega_ell)^2 X^2
                    + (omega_y/omega_ell)^2 Y^2
                    + (omega_z/omega_ell)^2 Z^2 ]

    where:
        omega_ell = hbar / (m ell^2)
    """

    def __init__(self, omegax, omegay, omegaz):
        super().__init__()
        self.omegax = float(omegax)
        self.omegay = float(omegay)
        self.omegaz = float(omegaz)

    def on_propagation_begin(self):
        base = self.gas.gas1 if hasattr(self.gas, "gas1") else self.gas

        self.omega_ell = base.adim_pulse

        self.lam_x = self.omegax / self.omega_ell
        self.lam_y = self.omegay / self.omega_ell
        self.lam_z = self.omegaz / self.omega_ell

    def get_potential(self, X, Y, Z, time=None):
        return 0.5 * (
            self.lam_x**2 * X**2
            + self.lam_y**2 * Y**2
            + self.lam_z**2 * Z**2
        )


class PedenDirectDDI3D(NonLinearPotential):
    """
    Direct state-dependent electric dipole-dipole interaction.

    This implements the no-exchange, no-mixing version of the two-state
    molecular model.

    Components:
        psi[0] = |up> component
        psi[1] = |down> component

    Dipole moments:
        d_up_debye
        d_down_debye

    They may have opposite signs. For a Lambda-doublet-like two-state model,
    a common simplified choice is:
        d_up_debye = +d0
        d_down_debye = -d0

    Mean-field potentials:
        V_up   = Phi_up,up[n_up]     + Phi_up,down[n_down]
        V_down = Phi_down,down[n_down] + Phi_down,up[n_up]

    Fourier-space kernel:
        K_dd(k) = 3 (e . k_hat)^2 - 1

    Dimensionless coupling used with that kernel:
        g_ij = N_j * m * d_i * d_j / (3 eps0 hbar^2 ell)

    where:
        i = component being acted on
        j = source-density component
        N_j = molecule number in source component

    This convention matches your old code, where each component wavefunction
    is normalized separately to 1.
    """

    def __init__(
        self,
        d_up_debye,
        d_down_debye,
        polarization=(0.0, 0.0, 1.0),
    ):
        super().__init__()

        self.d_up_debye = float(d_up_debye)
        self.d_down_debye = float(d_down_debye)
        self.polarization = polarization

    def on_propagation_begin(self):
        if not hasattr(self.gas, "gas1") or not hasattr(self.gas, "gas2"):
            raise TypeError("PedenDirectDDI3D expects your TwoComponentGas object.")

        self.base = self.gas.gas1

        self.N = self.base.N_grid
        self.M = self.base.Kx.shape[0]
        self.h = (self.M - self.N) // 2

        self.device = self.base.device
        self.float_dtype = self.base.float_dtype
        self.complex_dtype = self.base.complex_dtype

        self.mass = self.base.mass
        self.ell = self.base.adim_length

        N = self.gas.N_total

        self.d_up = self.d_up_debye * DEBYE_TO_CM
        self.d_down = self.d_down_debye * DEBYE_TO_CM

        self._g_up_up = self._dimensionless_g(
            source_N=self.gas.N_total,
            target_d=self.d_up,
            source_d=self.d_up,
        )

        self._g_down_down = self._dimensionless_g(
            source_N=self.gas.N_total,
            target_d=self.d_down,
            source_d=self.d_down,
        )

        self._g_up_from_down = self._dimensionless_g(
            source_N=self.gas.N_total,
            target_d=self.d_up,
            source_d=self.d_down,
        )

        self._g_down_from_up = self._dimensionless_g(
            source_N=self.gas.N_total,
            target_d=self.d_down,
            source_d=self.d_up,
        )

        self._build_kernel()

    def _dimensionless_g(self, source_N, target_d, source_d):
        return (
            source_N
            * self.mass
            * target_d
            * source_d
            / (3.0 * spconsts.epsilon_0 * spconsts.hbar**2 * self.ell)
        )

    def _build_kernel(self):
        """
        Uses your gas.Kx, gas.Ky, gas.Kz directly because your Gas class stores
        shifted padded momentum grids and your code uses ftn/iftn.
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

        # Free-space convention: zero angular average at k = 0.
        kernel[~mask] = 0.0

        self._kernel = kernel

    def _pad_density(self, density):
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
                f"DDI kernel shape {self._kernel.shape}."
            )

        return density_padded

    def _crop(self, field_padded):
        h = self.h
        n = self.N
        return field_padded[h:h+n, h:h+n, h:h+n]

    def _convolve_ddi(self, density, g):
        if abs(g) == 0.0:
            return torch.zeros_like(density)

        density_padded = self._pad_density(density)
        density_k = ftn(density_padded.to(self.complex_dtype))

        phi_padded = iftn(self._kernel * density_k).real
        phi = self._crop(phi_padded)

        return g * phi

    def potential_function(self, X, Y, Z, psi, time=None):
        """
        Return shape:
            (2, N_grid, N_grid, N_grid)

        This is exactly what your old two-component propagation expects for a
        diagonal nonlinear potential.
        """
        psi_up = psi[0]
        psi_down = psi[1]

        n_up = torch.abs(psi_up) ** 2
        n_down = torch.abs(psi_down) ** 2

        V_up = (
            self._convolve_ddi(n_up, self._g_up_up)
            + self._convolve_ddi(n_down, self._g_up_from_down)
        )

        V_down = (
            self._convolve_ddi(n_down, self._g_down_down)
            + self._convolve_ddi(n_up, self._g_down_from_up)
        )

        return torch.stack((V_up, V_down))

    def print_couplings(self):
        print("Dimensionless DDI couplings")
        print("---------------------------")
        print(f"g_up_up        = {self._g_up_up:.6e}")
        print(f"g_down_down    = {self._g_down_down:.6e}")
        print(f"g_up_from_down = {self._g_up_from_down:.6e}")
        print(f"g_down_from_up = {self._g_down_from_up:.6e}")


class ComponentDiagnostics(Callback):
    """
    Lightweight monitor for the old fixed-population code.

    Since the old code normalizes each component separately, the component
    norms should remain close to 1.
    """

    def __init__(self, compute_every=100, print_every=1000):
        super().__init__()
        self.compute_every = compute_every
        self.print_every = print_every

        self.epochs = []
        self.norm_up = []
        self.norm_down = []
        self.rms_x_up_um = []
        self.rms_y_up_um = []
        self.rms_z_up_um = []
        self.rms_x_down_um = []
        self.rms_y_down_um = []
        self.rms_z_down_um = []

    def on_propagation_begin(self):
        self.base = self.gas.gas1 if hasattr(self.gas, "gas1") else self.gas
        self.dV = self.base.dx * self.base.dy * self.base.dz
        self.ell_um = self.base.adim_length * 1e6

    def _norm_and_rms(self, psi):
        n = torch.abs(psi) ** 2
        norm = torch.sum(n) * self.dV

        x2 = torch.sum(n * self.base.X**2) * self.dV / norm
        y2 = torch.sum(n * self.base.Y**2) * self.dV / norm
        z2 = torch.sum(n * self.base.Z**2) * self.dV / norm

        return (
            norm.real,
            torch.sqrt(x2.real) * self.ell_um,
            torch.sqrt(y2.real) * self.ell_um,
            torch.sqrt(z2.real) * self.ell_um,
        )

    def on_epoch_end(self, epoch):
        if epoch % self.compute_every != 0:
            return

        psi = self.gas.psi
        norm_u, xu, yu, zu = self._norm_and_rms(psi[0])
        norm_d, xd, yd, zd = self._norm_and_rms(psi[1])

        self.epochs.append(epoch)
        self.norm_up.append(float(norm_u.detach().cpu()))
        self.norm_down.append(float(norm_d.detach().cpu()))
        self.rms_x_up_um.append(float(xu.detach().cpu()))
        self.rms_y_up_um.append(float(yu.detach().cpu()))
        self.rms_z_up_um.append(float(zu.detach().cpu()))
        self.rms_x_down_um.append(float(xd.detach().cpu()))
        self.rms_y_down_um.append(float(yd.detach().cpu()))
        self.rms_z_down_um.append(float(zd.detach().cpu()))

        if epoch % self.print_every == 0:
            print(
                f"epoch {epoch:8d} | "
                f"norm_up={self.norm_up[-1]:.6f}, "
                f"norm_down={self.norm_down[-1]:.6f} | "
                f"rms_up=({self.rms_x_up_um[-1]:.3f}, "
                f"{self.rms_y_up_um[-1]:.3f}, "
                f"{self.rms_z_up_um[-1]:.3f}) um | "
                f"rms_down=({self.rms_x_down_um[-1]:.3f}, "
                f"{self.rms_y_down_um[-1]:.3f}, "
                f"{self.rms_z_down_um[-1]:.3f}) um"
            )


def initialize_two_component_gaussian(gas, omega_x, omega_y, omega_z):
    """
    Initial wavefunction using the noninteracting harmonic oscillator widths.

    This is just a smooth starting guess; imaginary time propagation will reshape
    it under the dipolar interaction.
    """
    base = gas.gas1

    sigma_x = np.sqrt(base.adim_pulse / omega_x)
    sigma_y = np.sqrt(base.adim_pulse / omega_y)
    sigma_z = np.sqrt(base.adim_pulse / omega_z)

    psi0 = torch.exp(
        -0.5
        * (
            (base.X / sigma_x) ** 2
            + (base.Y / sigma_y) ** 2
            + (base.Z / sigma_z) ** 2
        )
    )

    gas.psi = torch.stack((psi0, psi0))

    print("Initial Gaussian widths")
    print("-----------------------")
    print(f"sigma_x = {sigma_x * base.adim_length * 1e6:.4f} um")
    print(f"sigma_y = {sigma_y * base.adim_length * 1e6:.4f} um")
    print(f"sigma_z = {sigma_z * base.adim_length * 1e6:.4f} um")


def plot_central_x_slices(gas, normalize=True):
    """
    Plot |psi_up(x,0,0)|^2 and |psi_down(x,0,0)|^2.

    Because your meshgrid uses indexing='xy', the x line is:
        n[iy0, :, iz0]
    """
    base = gas.gas1
    psi = gas.psi

    n_up = torch.abs(psi[0]) ** 2
    n_down = torch.abs(psi[1]) ** 2

    iy0 = torch.argmin(torch.abs(base.y)).item()
    iz0 = torch.argmin(torch.abs(base.z)).item()

    x_um = base.x.detach().cpu().numpy() * base.adim_length * 1e6

    line_up = n_up[iy0, :, iz0].detach().cpu().numpy()
    line_down = n_down[iy0, :, iz0].detach().cpu().numpy()

    if normalize:
        line_up = line_up / np.max(line_up)
        line_down = line_down / np.max(line_down)
        ylabel = "central density / peak"
    else:
        ylabel = r"$|\psi(x,0,0)|^2$"

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_um, line_up, label=r"$|\uparrow\rangle$")
    plt.plot(x_um, line_down, label=r"$|\downarrow\rangle$")
    plt.xlabel(r"$x$ [$\mu$m]")
    plt.ylabel(ylabel)
    plt.title("Central density slices")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_xy_density(gas, component=0, normalize=True):
    """
    Plot central z-slice density in the xy plane.
    """
    base = gas.gas1

    psi = gas.psi[component]
    n = torch.abs(psi) ** 2

    iz0 = torch.argmin(torch.abs(base.z)).item()

    img = n[:, :, iz0].detach().cpu().numpy()

    if normalize:
        img = img / np.max(img)

    extent = [
        base.x[0].item() * base.adim_length * 1e6,
        base.x[-1].item() * base.adim_length * 1e6,
        base.y[0].item() * base.adim_length * 1e6,
        base.y[-1].item() * base.adim_length * 1e6,
    ]

    plt.figure(figsize=(5.5, 5))
    plt.imshow(img, origin="lower", extent=extent, aspect="equal")
    plt.xlabel(r"$x$ [$\mu$m]")
    plt.ylabel(r"$y$ [$\mu$m]")
    plt.title(f"Component {component + 1}: central xy density")
    plt.colorbar(label="normalized density" if normalize else r"$|\psi|^2$")
    plt.tight_layout()
    plt.show()

def plot_xz_density(gas, component=0, normalize=True):
    """
    Plot central z-slice density in the xy plane.
    """
    base = gas.gas1

    psi = gas.psi[component]
    n = torch.abs(psi) ** 2

    iy0 = torch.argmin(torch.abs(base.y)).item()

    img = n[iy0, :, :].detach().cpu().numpy()

    if normalize:
        img = img / np.max(img)

    extent = [
        base.x[0].item() * base.adim_length * 1e6,
        base.x[-1].item() * base.adim_length * 1e6,
        base.z[0].item() * base.adim_length * 1e6,
        base.z[-1].item() * base.adim_length * 1e6,
    ]

    plt.figure(figsize=(5.5, 5))
    plt.imshow(img, origin="lower", extent=extent, aspect="equal")
    plt.xlabel(r"$x$ [$\mu$m]")
    plt.ylabel(r"$z$ [$\mu$m]")
    plt.title(f"Component {component + 1}: central xy density")
    plt.colorbar(label="normalized density" if normalize else r"$|\psi|^2$")
    plt.tight_layout()
    plt.show()


from torchgpe.utils.callbacks import Callback


class PopulationMonitor(Callback):
    def __init__(self, compute_every=10, print_every=100):
        super().__init__()

        self.compute_every = compute_every
        self.print_every = print_every

        self.epochs = []
        self.f1 = []
        self.f2 = []
        self.N1 = []
        self.N2 = []

    def on_epoch_end(self, epoch):
        if epoch % self.compute_every != 0:
            return

        fractions = self.gas.component_fractions.detach().cpu()
        numbers = self.gas.component_numbers.detach().cpu()

        self.epochs.append(epoch)

        self.f1.append(float(fractions[0]))
        self.f2.append(float(fractions[1]))

        self.N1.append(float(numbers[0]))
        self.N2.append(float(numbers[1]))

        if epoch % self.print_every == 0:
            print(
                f"epoch {epoch:8d} | "
                f"f1 = {self.f1[-1]:.6f} | "
                f"f2 = {self.f2[-1]:.6f} | "
                f"N1 = {self.N1[-1]:.3f} | "
                f"N2 = {self.N2[-1]:.3f}"
            )
    
import numpy as np
import scipy.constants as const


DEBYE_TO_CM = 3.33564e-30  # C m


def print_beta_and_D(
    N_total,
    d_debye,
    E_field_V_per_m,
    Delta_Hz,
    omega_z_rad_s,
    mass_kg,
    area_m2=None,
    n2D_m2=None,
):
    """
    Print dimensionless beta and D for the two-state molecular BEC model.

    Parameters
    ----------
    N_total:
        Total number of molecules.

    d_debye:
        Effective dipole moment d in Debye.

    E_field_V_per_m:
        Applied electric field in V/m.

    Delta_Hz:
        Zero-field splitting in Hz, not angular frequency.
        The code converts this to energy using h * Delta_Hz.

    omega_z_rad_s:
        Axial trap angular frequency in rad/s.
        Example: omega_z_rad_s = 2*np.pi*4000

    mass_kg:
        Molecular mass in kg.

    area_m2:
        In-plane condensate area in m^2. Used to compute n2D = N / area.

    n2D_m2:
        Optional direct 2D areal density in m^-2.
        If provided, this overrides area_m2.

    Notes
    -----
    beta = d E / Delta

    D = n2D d^2 / (3 eps0 Delta l_z)

    where:
        Delta = h * Delta_Hz
        l_z = sqrt(hbar / (m omega_z))
    """

    d = d_debye * DEBYE_TO_CM
    Delta_energy = const.h * Delta_Hz
    l_z = np.sqrt(const.hbar / (mass_kg * omega_z_rad_s))

    if n2D_m2 is None:
        if area_m2 is None:
            raise ValueError("Provide either area_m2 or n2D_m2.")
        n2D_m2 = N_total / area_m2

    beta = d * E_field_V_per_m / Delta_energy

    D = n2D_m2 * d**2 / (3 * const.epsilon_0 * Delta_energy * l_z)

    print("Dimensionless molecular-BEC parameters")
    print("--------------------------------------")
    print(f"N_total          = {N_total:.6e}")
    print(f"d                = {d_debye:.6g} Debye")
    print(f"E field          = {E_field_V_per_m:.6e} V/m")
    print(f"Delta            = {Delta_Hz:.6e} Hz")
    print(f"omega_z          = {omega_z_rad_s:.6e} rad/s")
    print(f"mass             = {mass_kg:.6e} kg")
    print(f"l_z              = {l_z:.6e} m")
    print(f"n2D              = {n2D_m2:.6e} m^-2")
    print()
    print(f"beta = d E / Delta              = {beta:.6e}")
    print(f"D    = n2D d^2/(3 eps0 Delta lz) = {D:.6e}")

    return {
        "beta": beta,
        "D": D,
        "l_z_m": l_z,
        "n2D_m2": n2D_m2,
        "Delta_energy_J": Delta_energy,
        "d_Cm": d,
    }

def print_beta_and_D_from_gas(
    gas,
    d_debye,
    E_field_V_per_m,
    Delta_Hz,
    omega_z_rad_s,
):
    """
    Convenience wrapper for your TwoComponentGas object.

    Uses:
        N_total = gas.N_total if available,
                  otherwise gas.N1 + gas.N2
        mass    = gas.gas1.mass
        area    = physical Lx * Ly from the dimensionless grid
    """

    N_total = getattr(gas, "N_total", gas.N1 + gas.N2)

    mass_kg = gas.gas1.mass

    # gas.x and gas.y are dimensionless; multiply by adim_length to get meters.
    Lx_m = (gas.X[1,-1,1] - gas.X[1,0,1]).detach().cpu().item() * gas.adim_length
    Ly_m = (gas.Y[-1,1,1] - gas.Y[0,1,1]).detach().cpu().item() * gas.adim_length

    area_m2 = Lx_m * Ly_m

    return print_beta_and_D(
        N_total=N_total,
        d_debye=d_debye,
        E_field_V_per_m=E_field_V_per_m,
        Delta_Hz=Delta_Hz,
        omega_z_rad_s=omega_z_rad_s,
        mass_kg=mass_kg,
        area_m2=area_m2,
    )

import numpy as np
import scipy.constants as const


DEBYE_TO_C_M = 3.33564e-30  # C m


def print_beta_and_D_from_stark_shift(
    N_total,
    d_debye,
    stark_shift_Hz,
    Delta_Hz,
    omega_z_rad_s,
    mass_kg,
    area_m2=None,
    n2D_m2=None,
):
    """
    Print beta and D using the Stark shift dE directly.

    Parameters
    ----------
    N_total:
        Total number of molecules.

    d_debye:
        Dipole moment in Debye. Used only for D.

    stark_shift_Hz:
        Stark shift dE/h in Hz.
        This is the energy shift divided by Planck's constant.

    Delta_Hz:
        Zero-field splitting Delta/h in Hz.

    omega_z_rad_s:
        Axial trap angular frequency in rad/s.

    mass_kg:
        Molecular mass in kg.

    area_m2:
        In-plane condensate area in m^2.
        Used to compute n2D = N_total / area_m2.

    n2D_m2:
        Optional areal density in m^-2.
        If provided, overrides area_m2.
    """

    d_Cm = d_debye * DEBYE_TO_C_M

    Delta_energy_J = const.h * Delta_Hz
    stark_energy_J = const.h * stark_shift_Hz

    l_z = np.sqrt(const.hbar / (mass_kg * omega_z_rad_s))

    if n2D_m2 is None:
        if area_m2 is None:
            raise ValueError("Provide either area_m2 or n2D_m2.")
        n2D_m2 = N_total / area_m2

    beta = stark_energy_J / Delta_energy_J

    D = N_total * d_Cm**2 / (
        (const.hbar * omega_z_rad_s) * l_z ** 3
    )

    print("Dimensionless parameters")
    print("------------------------")
    print(f"N_total          = {N_total:.6e}")
    print(f"d                = {d_debye:.6g} Debye")
    print(f"stark shift dE/h = {stark_shift_Hz:.6e} Hz")
    print(f"Delta/h          = {Delta_Hz:.6e} Hz")
    print(f"omega_z          = {omega_z_rad_s:.6e} rad/s")
    print(f"mass             = {mass_kg:.6e} kg")
    print(f"l_z              = {l_z:.6e} m")
    print(f"n2D              = {n2D_m2:.6e} m^-2")
    print()
    print(f"beta = dE / Delta                 = {beta:.6e}")
    print(f"D    = n2D d^2/(3 eps0 Delta lz)  = {D:.6e}")

    return {
        "beta": beta,
        "D": D,
        "l_z_m": l_z,
        "n2D_m2": n2D_m2,
        "Delta_energy_J": Delta_energy_J,
        "stark_energy_J": stark_energy_J,
        "d_Cm": d_Cm,
    }

def print_beta_and_D_from_gas_stark_shift(
    gas,
    d_debye,
    stark_shift_Hz,
    Delta_Hz,
    omega_z_rad_s,
):
    """
    Read N_total, mass, and in-plane area from your gas object.
    """

    N_total = getattr(gas, "N_total", gas.N1 + gas.N2)

    mass_kg = gas.gas1.mass

    # Your grid coordinates are dimensionless.
    # Multiply by adim_length to recover physical meters.
    Lx_m = (gas.gas1.x[-1] - gas.gas1.x[0]).detach().cpu().item() * gas.adim_length
    Ly_m = (gas.gas1.y[-1] - gas.gas1.y[0]).detach().cpu().item() * gas.adim_length

    area_m2 = Lx_m * Ly_m

    return print_beta_and_D_from_stark_shift(
        N_total=N_total,
        d_debye=d_debye,
        stark_shift_Hz=stark_shift_Hz,
        Delta_Hz=Delta_Hz,
        omega_z_rad_s=omega_z_rad_s,
        mass_kg=mass_kg,
        area_m2=area_m2,
    )

def print_imaginary_time_stiffness(gas, trap, ddi, single_molecule, time_step):
    """
    time_step should be the physical time_step you pass to ground_state,
    e.g. -1e-9j.
    """
    dt_dim = time_step * gas.adim_pulse
    tau = -dt_dim.imag

    trap.set_gas(gas)
    trap.on_propagation_begin()

    ddi.set_gas(gas)
    ddi.on_propagation_begin()

    single_molecule.set_gas(gas)
    single_molecule.on_propagation_begin()

    V_trap = as_two_component_potential(
        trap.get_potential(*gas.coordinates),
        gas,
    )

    V_ddi = as_two_component_potential(
        ddi.potential_function(*gas.coordinates, gas.psi),
        gas,
    )

    V_total = V_trap + V_ddi

    V_min = torch.amin(V_total.real).detach().cpu().item()
    V_max = torch.amax(V_total.real).detach().cpu().item()
    V_span = V_max - V_min

    H_span = (
        single_molecule.evals[-1] - single_molecule.evals[0]
    ).detach().cpu().item()

    print("Imaginary-time stiffness diagnostics")
    print("-----------------------------------")
    print(f"physical time_step        = {time_step}")
    print(f"dimensionless tau         = {tau:.6e}")
    print(f"V_min                     = {V_min:.6e}")
    print(f"V_max                     = {V_max:.6e}")
    print(f"V_span                    = {V_span:.6e}")
    print(f"tau * V_span              = {tau * V_span:.6e}")
    print(f"H eigenvalue spread       = {H_span:.6e}")
    print(f"tau * H eigenvalue spread = {tau * H_span:.6e}")

def rms_rho_over_ellz(gas, ellz=None, component=None):
    """
    Returns sqrt(<rho^2>) / ellz.

    component=None: total two-component density
    component=0 or 1: width of one component only
    """
    psi = gas.psi
    dV = gas.dx * gas.dy * gas.dz

    if component is None:
        density = torch.abs(psi[0])**2 + torch.abs(psi[1])**2
    else:
        density = torch.abs(psi[component])**2

    norm = torch.sum(density) * dV

    rho2 = gas.X**2 + gas.Y**2
    mean_rho2 = torch.sum(density * rho2) * dV / norm

    rms_rho_dimensionless = torch.sqrt(mean_rho2.real)

    # gas.X and gas.Y are in units of gas.adim_length.
    # If adim_length = ellz, this conversion factor is 1.
    if ellz is None:
        return rms_rho_dimensionless.detach().cpu().item()
    else:
        return (
            rms_rho_dimensionless
            * gas.adim_length
            / ellz
        ).detach().cpu().item()
    
if __name__ == "__main__":
    torch.set_default_dtype(torch.double)

    # -------------------------
    # Molecule and simulation
    # -------------------------
    # Example: 87Rb133Cs mass ~ 220 amu.
    # The element="87Rb" is only used to construct your existing Gas object.
    # We patch the mass immediately after construction.
    mass_amu = 220.0

    N_up = int(2.5e4)
    N_down = int(2.5e4)

    gas = TwoComponentGas(
        element="87Rb",
        N_particles=(N_up, N_down),
        N_grid=128,
        grid_size=(16e-6, 16e-6, 3e-6),
        adimensionalization_length=1e-6,
    )

    set_two_component_molecular_mass(gas, mass_amu=mass_amu)

    # Quasi-2D / pancake trap.
    # Frequencies must be angular frequencies, rad/s.
    omega_x = 2 * np.pi * 400
    omega_y = 2 * np.pi * 400
    omega_z = 2 * np.pi * 4000

    trap = Trap3D(
        omegax=omega_x,
        omegay=omega_y,
        omegaz=omega_z,
    )

    # -------------------------
    # Two dipole states
    # -------------------------
    # Simplified dipole-basis model:
    #   |up>   has d = +d0
    #   |down> has d = -d0
    #
    # Replace d0_debye with the dipole moment appropriate for your molecule/state.
    # If the run becomes unstable, reduce d0_debye, N, or the time step.
    d0_debye = 1.0

    ddi = PedenDirectDDI3D(
        d_up_debye=+d0_debye,
        d_down_debye=-d0_debye,
        polarization=(0.0, 0.0, 1.0),
    )

    # Initialize.
    initialize_two_component_gaussian(
        gas,
        omega_x=omega_x,
        omega_y=omega_y,
        omega_z=omega_z,
    )

    # Initialize once to print couplings before the run.
    ddi.set_gas(gas)
    ddi.on_propagation_begin()
    ddi.print_couplings()

    diagnostics = ComponentDiagnostics(
        compute_every=100,
        print_every=1000,
    )

    # -------------------------
    # Ground state
    # -------------------------
    # With strong electric dipoles, start cautiously.
    # If stable, you can increase |time_step| gradually.
    gas.ground_state(
        potentials=[trap, ddi],
        time_step=-1e-8j,
        N_iterations=50_000,
        callbacks=[diagnostics],
        leave_progress_bar=True,
    )

    plot_central_x_slices(gas, normalize=True)
    plot_xy_density(gas, component=0, normalize=True)
    plot_xy_density(gas, component=1, normalize=True)