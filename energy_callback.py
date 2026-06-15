import math
import torch
from torchgpe.utils.callbacks import Callback


class TwoComponentEnergyMonitor3D(Callback):
    """
    Energy monitor for your TwoComponentGas implementation.

    Assumes:
        psi[0] = component 1
        psi[1] = component 2

    Energy:
        E = int [
              1/2 |grad psi1|^2
            + 1/2 |grad psi2|^2
            + V1 |psi1|^2
            + V2 |psi2|^2
            + g11/2 |psi1|^4
            + g22/2 |psi2|^4
            + g12 |psi1|^2 |psi2|^2
        ] dV
    """

    def __init__(
        self,
        g11,
        g22,
        g12,
        V1=None,
        V2=None,
        compute_every=100,
        print_every=1000,
        track_wavefunction_change=True,
    ):
        super().__init__()

        self.g11 = g11
        self.g22 = g22
        self.g12 = g12

        # V1 and V2 can be tensors, callables, or None.
        # If None, they are treated as zero.
        self.V1 = V1
        self.V2 = V2

        self.compute_every = compute_every
        self.print_every = print_every
        self.track_wavefunction_change = track_wavefunction_change

        self.epochs = []
        self.energy_total = []
        self.energy_kinetic = []
        self.energy_potential = []
        self.energy_interaction = []
        self.norm1 = []
        self.norm2 = []
        self.relative_change = []

        self._previous_psi = None

    def _scalar_float(self, x):
        if torch.is_tensor(x):
            return float(x.detach().cpu())
        return float(x)

    def _make_potential_tensor(self, V, reference):
        """
        Convert V into a tensor matching the grid/device/dtype of reference.
        V may be:
            None
            a torch.Tensor
            a callable V(X, Y, Z)
            a scalar
        """
        gas = self.gas

        if V is None:
            return torch.zeros_like(reference.real)

        if callable(V):
            V = V(gas.X, gas.Y, gas.Z)

        if hasattr(V, "get_potential"):
            V = V.get_potential(gas.X, gas.Y, gas.Z)

        if not torch.is_tensor(V):
            V = torch.as_tensor(V, device=reference.device, dtype=reference.real.dtype)

        return V.to(device=reference.device, dtype=reference.real.dtype)

    def _grad_abs_sq(self, psi):
        """
        Computes |grad psi|^2 using finite differences.

        Important: your gas3D.py uses torch.meshgrid(..., indexing="xy"),
        so the tensor axis order is effectively [y, x, z], not [x, y, z].
        Therefore spacing should be (dy, dx, dz).
        """
        gas = self.gas

        dx = self._scalar_float(gas.dx)
        dy = self._scalar_float(gas.dy)
        dz = self._scalar_float(gas.dz)

        grad_re = torch.gradient(
            psi.real,
            spacing=(dy, dx, dz),
            dim=(0, 1, 2),
        )

        grad_im = torch.gradient(
            psi.imag,
            spacing=(dy, dx, dz),
            dim=(0, 1, 2),
        )

        grad_sq = 0.0
        for gre, gim in zip(grad_re, grad_im):
            grad_sq = grad_sq + gre**2 + gim**2

        return grad_sq

    def on_propagation_begin(self):
        self._previous_psi = None

    def on_epoch_end(self, epoch):
        if epoch % self.compute_every != 0:
            return

        gas = self.gas

        psi = gas.psi
        psi1 = psi[0]
        psi2 = psi[1]

        n1 = torch.abs(psi1) ** 2
        n2 = torch.abs(psi2) ** 2

        dV = gas.dx * gas.dy * gas.dz

        V1 = self._make_potential_tensor(self.V1, psi1)
        V2 = self._make_potential_tensor(self.V2, psi2)

        # Kinetic energy
        Ekin1 = 0.5 * torch.sum(self._grad_abs_sq(psi1)) * dV
        Ekin2 = 0.5 * torch.sum(self._grad_abs_sq(psi2)) * dV
        Ekin = Ekin1 + Ekin2

        # Linear potential energy
        Epot = (
            torch.sum(V1 * n1)
            + torch.sum(V2 * n2)
        ) * dV

        # Contact interaction energy
        Eint = (
            0.5 * self.g11 * torch.sum(n1**2)
            + 0.5 * self.g22 * torch.sum(n2**2)
            + self.g12 * torch.sum(n1 * n2)
        ) * dV

        Etot = Ekin + Epot + Eint

        # Component norms. In your implementation these should stay close to 1,
        # because each underlying Gas.psi setter normalizes the component.
        N1 = torch.sum(n1) * dV
        N2 = torch.sum(n2) * dV

        if self.track_wavefunction_change and self._previous_psi is not None:
            diff = psi - self._previous_psi
            rel_change = (
                torch.linalg.norm(diff.reshape(-1))
                / torch.linalg.norm(psi.reshape(-1))
            )
            rel_change_value = rel_change.detach().cpu().item()
        else:
            rel_change_value = math.nan

        self._previous_psi = psi.detach().clone()

        Etot_value = Etot.real.detach().cpu().item()
        Ekin_value = Ekin.real.detach().cpu().item()
        Epot_value = Epot.real.detach().cpu().item()
        Eint_value = Eint.real.detach().cpu().item()
        N1_value = N1.real.detach().cpu().item()
        N2_value = N2.real.detach().cpu().item()

        self.epochs.append(epoch)
        self.energy_total.append(Etot_value)
        self.energy_kinetic.append(Ekin_value)
        self.energy_potential.append(Epot_value)
        self.energy_interaction.append(Eint_value)
        self.norm1.append(N1_value)
        self.norm2.append(N2_value)
        self.relative_change.append(rel_change_value)

        if epoch % self.print_every == 0:
            print(
                f"epoch {epoch:8d} | "
                f"E = {Etot_value:.12e} | "
                f"Ekin = {Ekin_value:.6e} | "
                f"Epot = {Epot_value:.6e} | "
                f"Eint = {Eint_value:.6e} | "
                f"N1 = {N1_value:.6e} | "
                f"N2 = {N2_value:.6e} | "
                f"dpsi = {rel_change_value:.3e}"
            )

import math
import torch
import numpy as np
from torch.nn.functional import pad

from torchgpe.utils import ftn, iftn
from torchgpe.utils.callbacks import Callback


class TwoComponentDipolarEnergyMonitor3D(Callback):
    """
    Energy monitor for a two-component 3D gas with dipole-dipole interactions
    and no contact interaction.

    Energy terms:
        E_kin = sum_j N_j int 1/2 |grad psi_j|^2 dV
        E_pot = sum_j N_j int V_j |psi_j|^2 dV
        E_dd  = 1/2 sum_j N_j int n_j Phi_dd,j dV

    where:
        n_j = |psi_j|^2

    This assumes your wavefunctions are normalized component-wise:
        int |psi_1|^2 dV = 1
        int |psi_2|^2 dV = 1

    Parameters
    ----------
    dipole:
        The same dipole-dipole NonLinearPotential object passed to
        gas.ground_state(...).

    linear_potentials:
        List of linear potentials, e.g. [trap].

    physical_energy:
        If True, weight component energies by N1 and N2.
        This is the physical total energy convention.

    per_particle:
        If True and physical_energy=True, report E / (N1 + N2).
        This is often easier to read during convergence monitoring.

    compute_every:
        Compute energy every this many iterations.

    print_every:
        Print energy every this many iterations.

    use_spectral_kinetic:
        If True, compute kinetic energy using spectral derivatives with
        the same ftn/iftn convention as your Gas class.
    """

    def __init__(
        self,
        dipole,
        linear_potentials=None,
        physical_energy=True,
        per_particle=True,
        compute_every=100,
        print_every=1000,
        use_spectral_kinetic=True,
        track_wavefunction_change=True,
    ):
        super().__init__()

        self.dipole = dipole
        self.linear_potentials = [] if linear_potentials is None else linear_potentials

        self.physical_energy = physical_energy
        self.per_particle = per_particle
        self.compute_every = compute_every
        self.print_every = print_every
        self.use_spectral_kinetic = use_spectral_kinetic
        self.track_wavefunction_change = track_wavefunction_change

        self.epochs = []
        self.energy_total = []
        self.energy_kinetic = []
        self.energy_potential = []
        self.energy_dipole = []
        self.norm1 = []
        self.norm2 = []
        self.relative_change = []

        self._previous_psi = None

    def on_propagation_begin(self):
        if not hasattr(self.gas, "gas1") or not hasattr(self.gas, "gas2"):
            raise TypeError(
                "TwoComponentDipolarEnergyMonitor3D expects your TwoComponentGas object."
            )

        self.base = self.gas.gas1

        self.N_grid = self.base.N_grid
        self.M_grid = self.base.Kx.shape[0]
        self.h = (self.M_grid - self.N_grid) // 2

        self.device = self.base.device
        self.float_dtype = self.base.float_dtype
        self.complex_dtype = self.base.complex_dtype

        self.dV = self.base.dx * self.base.dy * self.base.dz

        self.N1 = float(self.gas.gas1.N_particles)
        self.N2 = float(self.gas.gas2.N_particles)
        self.Ntot = self.N1 + self.N2

        # Initialize potentials if needed. If these same objects were passed
        # to ground_state(...), this is usually redundant but harmless.
        for pot in self.linear_potentials:
            pot.set_gas(self.gas)
            pot.on_propagation_begin()

        self.dipole.set_gas(self.gas)
        self.dipole.on_propagation_begin()

        self._V_linear = self._build_linear_potential_tensor()

        self._previous_psi = None

    def _build_linear_potential_tensor(self):
        """
        Returns a tensor of shape:
            (2, N_grid, N_grid, N_grid)

        If a linear potential returns shape (N, N, N), it is applied to
        both components.
        """
        shape = (2,) + tuple(self.base.X.shape)

        V_total = torch.zeros(
            shape,
            dtype=self.float_dtype,
            device=self.device,
        )

        for pot in self.linear_potentials:
            V = pot.get_potential(*self.gas.coordinates)

            if not torch.is_tensor(V):
                V = torch.as_tensor(V, dtype=self.float_dtype, device=self.device)

            V = V.to(dtype=self.float_dtype, device=self.device)

            if V.ndim == 3:
                V_total = V_total + torch.stack((V, V))
            elif V.ndim == 4 and V.shape[0] == 2:
                V_total = V_total + V
            else:
                raise ValueError(
                    "Linear potential must return shape (N,N,N) or (2,N,N,N). "
                    f"Got shape {tuple(V.shape)}."
                )

        return V_total

    def _pad_field(self, field):
        """
        Pad from shape (N,N,N) to the same shape as gas.Kx, gas.Ky, gas.Kz.

        torch.nn.functional.pad uses:
            (z_left, z_right, y_left, y_right, x_left, x_right)
        """
        h = self.h

        return pad(
            field,
            (h, h, h, h, h, h),
            mode="constant",
            value=0.0,
        )

    def _crop_field(self, field_padded):
        h = self.h
        n = self.N_grid

        return field_padded[h:h+n, h:h+n, h:h+n]

    def _spectral_grad_abs_sq(self, psi):
        """
        Compute |grad psi|^2 using spectral derivatives.

        This uses the same shifted Kx, Ky, Kz and ftn/iftn convention as
        your Gas.psik implementation.
        """
        psi_padded = self._pad_field(psi)
        psi_k = ftn(psi_padded)

        dpsi_dx = self._crop_field(iftn(1j * self.base.Kx * psi_k))
        dpsi_dy = self._crop_field(iftn(1j * self.base.Ky * psi_k))
        dpsi_dz = self._crop_field(iftn(1j * self.base.Kz * psi_k))

        return (
            torch.abs(dpsi_dx) ** 2
            + torch.abs(dpsi_dy) ** 2
            + torch.abs(dpsi_dz) ** 2
        )

    def _finite_difference_grad_abs_sq(self, psi):
        """
        Backup finite-difference kinetic-energy estimate.

        Because your meshgrid uses indexing='xy', the tensor axis order is:
            axis 0 -> y
            axis 1 -> x
            axis 2 -> z

        so spacing is:
            (dy, dx, dz)
        """
        dx = float(self.base.dx.detach().cpu())
        dy = float(self.base.dy.detach().cpu())
        dz = float(self.base.dz.detach().cpu())

        grad_re = torch.gradient(
            psi.real,
            spacing=(dy, dx, dz),
            dim=(0, 1, 2),
        )

        grad_im = torch.gradient(
            psi.imag,
            spacing=(dy, dx, dz),
            dim=(0, 1, 2),
        )

        grad_sq = 0.0

        for gre, gim in zip(grad_re, grad_im):
            grad_sq = grad_sq + gre**2 + gim**2

        return grad_sq

    def _grad_abs_sq(self, psi):
        if self.use_spectral_kinetic:
            return self._spectral_grad_abs_sq(psi)

        return self._finite_difference_grad_abs_sq(psi)

    def _energy_weights(self):
        """
        Returns component weights and final scale factor.

        physical_energy=True:
            E = N1 E1 + N2 E2 + interactions

        per_particle=True:
            report E / (N1 + N2)

        physical_energy=False:
            report unweighted numerical energy-like quantity.
        """
        if self.physical_energy:
            w1 = self.N1
            w2 = self.N2
            scale = self.Ntot if self.per_particle else 1.0
        else:
            w1 = 1.0
            w2 = 1.0
            scale = 1.0

        return w1, w2, scale

    def on_epoch_end(self, epoch):
        if epoch % self.compute_every != 0:
            return

        with torch.no_grad():
            psi = self.gas.psi

            psi1 = psi[0]
            psi2 = psi[1]

            n1 = torch.abs(psi1) ** 2
            n2 = torch.abs(psi2) ** 2

            w1, w2, scale = self._energy_weights()

            # Norms should remain close to 1 for each component.
            N1_norm = torch.sum(n1) * self.dV
            N2_norm = torch.sum(n2) * self.dV

            # Kinetic energy
            Ekin1 = 0.5 * torch.sum(self._grad_abs_sq(psi1)) * self.dV
            Ekin2 = 0.5 * torch.sum(self._grad_abs_sq(psi2)) * self.dV

            Ekin = (w1 * Ekin1 + w2 * Ekin2) / scale

            # Linear trap energy
            V1 = self._V_linear[0]
            V2 = self._V_linear[1]

            Epot1 = torch.sum(V1 * n1) * self.dV
            Epot2 = torch.sum(V2 * n2) * self.dV

            Epot = (w1 * Epot1 + w2 * Epot2) / scale

            # Dipolar mean-field potential.
            # This should return shape (2, N, N, N).
            Vdd = self.dipole.potential_function(
                *self.gas.coordinates,
                psi,
            )

            if Vdd.shape[0] != 2:
                raise ValueError(
                    "dipole.potential_function(...) must return shape "
                    "(2, N_grid, N_grid, N_grid)."
                )

            Vdd1 = Vdd[0]
            Vdd2 = Vdd[1]

            # Dipolar interaction energy.
            #
            # The factor 1/2 avoids double-counting pair interactions.
            Edd1 = torch.sum(n1 * Vdd1) * self.dV
            Edd2 = torch.sum(n2 * Vdd2) * self.dV

            Edd = 0.5 * (w1 * Edd1 + w2 * Edd2) / scale

            Etot = Ekin + Epot + Edd

            if self.track_wavefunction_change and self._previous_psi is not None:
                diff = psi - self._previous_psi
                rel_change = (
                    torch.linalg.norm(diff.reshape(-1))
                    / torch.linalg.norm(psi.reshape(-1))
                )
                rel_change_value = rel_change.detach().cpu().item()
            else:
                rel_change_value = math.nan

            self._previous_psi = psi.detach().clone()

            Etot_value = Etot.real.detach().cpu().item()
            Ekin_value = Ekin.real.detach().cpu().item()
            Epot_value = Epot.real.detach().cpu().item()
            Edd_value = Edd.real.detach().cpu().item()

            N1_value = N1_norm.real.detach().cpu().item()
            N2_value = N2_norm.real.detach().cpu().item()

            self.epochs.append(epoch)
            self.energy_total.append(Etot_value)
            self.energy_kinetic.append(Ekin_value)
            self.energy_potential.append(Epot_value)
            self.energy_dipole.append(Edd_value)
            self.norm1.append(N1_value)
            self.norm2.append(N2_value)
            self.relative_change.append(rel_change_value)

            if epoch % self.print_every == 0:
                label = "E/N" if self.physical_energy and self.per_particle else "E"

                print(
                    f"epoch {epoch:8d} | "
                    f"{label} = {Etot_value:.12e} | "
                    f"Ekin = {Ekin_value:.6e} | "
                    f"Epot = {Epot_value:.6e} | "
                    f"Edd = {Edd_value:.6e} | "
                    f"N1 = {N1_value:.6e} | "
                    f"N2 = {N2_value:.6e} | "
                    f"dpsi = {rel_change_value:.3e}"
                )


import torch
from torchgpe.utils.callbacks import Callback


def as_two_component_potential(V, gas):
    if not torch.is_tensor(V):
        V = torch.as_tensor(
            V,
            dtype=gas.float_dtype,
            device=gas.device,
        )

    V = V.to(device=gas.device, dtype=gas.float_dtype)

    if V.ndim == 0:
        V0 = torch.zeros_like(gas.X) + V
        return torch.stack((V0, V0))

    if V.ndim == 3:
        return torch.stack((V, V))

    if V.ndim == 4 and V.shape[0] == 2:
        return V

    raise ValueError(
        "Potential must have shape (), (N,N,N), or (2,N,N,N). "
        f"Got shape {tuple(V.shape)}."
    )


class EffectiveDetuningMonitor(Callback):
    def __init__(
        self,
        linear_potentials,
        nonlinear_potentials,
        single_molecule,
        compute_every=10,
        print_every=10,
    ):
        super().__init__()

        self.linear_potentials = linear_potentials
        self.nonlinear_potentials = nonlinear_potentials
        self.single_molecule = single_molecule

        self.compute_every = compute_every
        self.print_every = print_every

        self.epochs = []
        self.f1 = []
        self.f2 = []
        self.delta_center = []
        self.delta_weighted = []
        self.delta_max_abs = []
        self.omega_coupling = []
        self.max_density = []

    def _build_diagonal_potential(self):
        V = torch.zeros(
            (2,) + tuple(self.gas.X.shape),
            dtype=self.gas.float_dtype,
            device=self.gas.device,
        )

        for pot in self.linear_potentials:
            V = V + as_two_component_potential(
                pot.get_potential(*self.gas.coordinates),
                self.gas,
            )

        for pot in self.nonlinear_potentials:
            V = V + as_two_component_potential(
                pot.potential_function(*self.gas.coordinates, self.gas.psi),
                self.gas,
            )

        return V

    def on_epoch_end(self, epoch):
        if epoch % self.compute_every != 0:
            return

        psi = self.gas.psi
        dV = self.gas.dx * self.gas.dy * self.gas.dz

        n1 = torch.abs(psi[0])**2
        n2 = torch.abs(psi[1])**2
        ntot = n1 + n2

        f1 = torch.sum(n1) * dV
        f2 = torch.sum(n2) * dV

        V = self._build_diagonal_potential()

        H = self.single_molecule.H_dim

        # Effective diagonal difference:
        # component 1 diagonal minus component 2 diagonal.
        delta_eff = (
            V[0]
            - V[1]
            + (H[0, 0] - H[1, 1]).real
        )

        # Coupling scale. For H12 = -Omega/2, this equals Omega.
        omega_eff = 2.0 * torch.abs(H[0, 1])

        ix0 = torch.argmin(torch.abs(self.gas.X[1,:,1])).item()
        iy0 = torch.argmin(torch.abs(self.gas.Y[:,1,1])).item()
        iz0 = torch.argmin(torch.abs(self.gas.Z[1,1,:])).item()

        # meshgrid(indexing="xy") means tensor order is [y, x, z].
        delta_center = delta_eff[iy0, ix0, iz0]

        weighted_delta = torch.sum(delta_eff * ntot) * dV / torch.sum(ntot * dV)

        max_density = torch.max(ntot)
        max_abs_delta = torch.max(torch.abs(delta_eff))

        self.epochs.append(epoch)
        self.f1.append(f1.detach().cpu().item())
        self.f2.append(f2.detach().cpu().item())
        self.delta_center.append(delta_center.detach().cpu().item())
        self.delta_weighted.append(weighted_delta.detach().cpu().item())
        self.delta_max_abs.append(max_abs_delta.detach().cpu().item())
        self.omega_coupling.append(omega_eff.detach().cpu().item())
        self.max_density.append(max_density.detach().cpu().item())

        if epoch % self.print_every == 0:
            ratio = max_abs_delta / (omega_eff + 1e-30)

            print(
                f"epoch {epoch:8d} | "
                f"f1={self.f1[-1]:.6f}, "
                f"f2={self.f2[-1]:.6f}, "
                f"f_sum={self.f1[-1] + self.f2[-1]:.6f} | "
                f"Delta_center={self.delta_center[-1]:.3e} | "
                f"Delta_weighted={self.delta_weighted[-1]:.3e} | "
                f"max|Delta|/Omega={ratio.detach().cpu().item():.3e} | "
                f"max density={self.max_density[-1]:.3e}"
            )