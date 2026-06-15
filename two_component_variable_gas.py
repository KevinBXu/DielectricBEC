import torch

from typing import List, Union
import torch
import numpy as np
from torch.nn.functional import pad
import scipy.constants as spconsts

from torchgpe.utils import normalize_wavefunction, ftn, iftn
from torchgpe.utils.elements import elements_dict
from torchgpe.utils.potentials import Potential, LinearPotential
from torchgpe.utils.propagation import imaginary_time_propagation, real_time_propagation
from torchgpe.utils.callbacks import Callback

from two_state_propagation import two_state_imaginary_time_propagation, two_state_real_time_propagation
from gas3D import Gas, Zero, UPDATED_PSI, UPDATED_PSIK, UPDATED_BOTH

class TwoComponentGas:
    def __init__(
        self,
        element="87Rb",
        N_particles=(5e4, 5e4),
        N_grid=128,
        grid_size=20e-6,
        device=None,
        float_dtype=torch.double,
        complex_dtype=torch.complex128,
        adimensionalization_length=1e-6,
    ):
        self.N1 = N_particles[0]
        self.N2 = N_particles[1]

        self.N_total = int(self.N1 + self.N2)

        self.gas1 = Gas(
            element=element,
            N_particles=self.N1,
            N_grid=N_grid,
            grid_size=grid_size,
            device=device,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            adimensionalization_length=adimensionalization_length,
        )

        self.gas2 = Gas(
            element=element,
            N_particles=self.N2,
            N_grid=N_grid,
            grid_size=grid_size,
            device=device,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            adimensionalization_length=adimensionalization_length,
        )

        self.element = element
        self.N_grid = N_grid
        self.grid_size = grid_size
        self.device = self.gas1.device
        self.float_dtype = self.gas1.float_dtype
        self.complex_dtype = self.gas1.complex_dtype

        # Share useful grids from gas1
        self.X = self.gas1.X
        self.Y = self.gas1.Y
        self.Z = self.gas1.Z
        self.Kx = self.gas1.Kx
        self.Ky = self.gas1.Ky
        self.Kz = self.gas1.Kz
        self.dx = self.gas1.dx
        self.dy = self.gas1.dy
        self.dz = self.gas1.dz
        self.adim_length = self.gas1.adim_length
        self.adim_pulse = self.gas1.adim_pulse

        self._updated_wavefunction = self.gas1._updated_wavefunction

    def ground_state(self, potentials: List[Potential] = [], time_step: complex = -1e-6j, N_iterations: int = int(1e3), callbacks: List[Callback] = [], leave_progress_bar=True):
        """Compute the ground state's wave function.

        Use the split-step Fourier method with imaginary time propagation (ITP) to compute the ground state's wave function of the gas. 
        The potentials acting on the system are specified via the :py:attr:`potentials` parameter. 

        Args:
            potentials (List[:class:`~gpe.utils.potentials.Potential`]): Optional. The list of potentials acting on the system. Defaults to [].
            time_step (complex): Optional. The time step to be used in the ITP. 
            N_iterations (int): Optional. The number of steps of ITP to perform. 
            callbacks (List[:class:`~gpe.utils.callbacks.Callback`]): Optional. List of callbacks to be evaluated during the evolution. Defaults to [].
            leave_progress_bar (bool): Optional. Whether to leave the progress bar on screen after the propagation ends. Defaults to True.

        Raises:
            Exception: If time dependent potentials are specified 
            Exception: If the time step is not a purely imaginary number
            Exception: If the imaginary part of the time step is not positive
            Exception: If neither the wave function in real space nor in the one in momentum space have been initialized
        """

        # for i in range(1, 1000):
        #     print("calling")
        
        # Initial setup of the potentials
        for potential in potentials:
            potential.set_gas(self)
            potential.on_propagation_begin()

        # --- Process parameters ---

        if any(potential.is_time_dependent for potential in potentials):
            raise Exception(
                "Time dependent potentials can't be used in imaginary time propagation")

        if time_step.real != 0:
            raise Exception(
                "Imaginary time propagation requires a purely imaginary time step")
        if np.imag(time_step) >= 0:
            raise Exception(
                "The imaginary part of the time step must be negative")

        if self._updated_wavefunction is None:
            raise Exception(
                "The initial wave function must be initialized by either setting the psi or psik attributes")

        N_iterations = int(N_iterations)

        # Adimensionalize the time_step
        adim_time_step = time_step * self.adim_pulse

        # If no potential has been specified, use an identically zero one
        if len(potentials) == 0:
            potentials = [Zero(None)]

        # Generate a dictionary of runtime settings for the simulations to be given
        # to the callbacks. This list is not complete at the moment
        propagation_parameters = {
            "potentials": potentials,
            "time_step": time_step,
            "N_iterations": N_iterations,
        }

        # Initial setup of the callbacks
        for callback in callbacks:
            callback.set_gas(self)
            callback.set_propagation_params(propagation_parameters)

        two_state_imaginary_time_propagation(
            self, potentials, adim_time_step, N_iterations, callbacks, leave_progress_bar)
            
    def propagate(self, final_time: float, time_step: float = 1e-6, potentials: List[Potential] = [], callbacks: List[Callback] = [], leave_progress_bar=True):
        """Propagate the wave function in real time.

        Use the split-step Fourier method with real time propagation (RTP) to propagate the gas wave function to :py:attr:`final_time`. 
        The potentials acting on the system are specified via the :py:attr:`potentials` parameter. 

        Note:
            The time step is adjusted such that :py:attr:`final_time` is always reached.

        Args:
            final_time (float): The final time up to which the wave function whould be propagated.
            time_step (float): Optional. The time step to be used in the RTP. Defaults to :math:`10^{-6}`.
            potentials (List[:class:`~gpe.utils.potentials.Potential`]): Optional. The list of potentials acting on the system. Defaults to [].
            callbacks (List[:class:`~gpe.utils.callbacks.Callback`]): Optional. List of callbacks to be evaluated during the evolution. Defaults to [].
            leave_progress_bar (bool): Optional. Whether to leave the progress bar on screen after the propagation ends. Defaults to True.

        Raises:
            Exception: If the time step is not a floating point number
            Exception: If the time step is not positive
            Exception: If neither the wave function in real space nor in the one in momentum space have been initialized
        """

        # Initial setup of the potentials
        for potential in potentials:
            potential.set_gas(self)
            potential.on_propagation_begin()

        # --- Process parameters ---

        if not issubclass(type(time_step), (float, )):
            raise Exception(
                "The provided time step is not a floating point number.")
        if time_step <= 0:
            raise Exception("Propagation requires a positive time step")

        # Adjust the time step such that the final time is always reached
        N_iterations = round(final_time/time_step)
        time_step = final_time/N_iterations

        # Array of times to be passed to the time dependent potentials
        times = torch.linspace(0, final_time, N_iterations)

        # Adimensionalize the time_step
        adim_time_step = time_step * self.adim_pulse

        if self._updated_wavefunction is None:
            raise Exception(
                "The initial wave function must be initialized by setting either the psi or psik attributes")

        # If no potential has been specified, use an identically zero one
        if len(potentials) == 0:
            potentials = [Zero(None)]

        # Generate a dictionary of runtime settings for the simulations to be given
        # to the callbacks. This list is not complete at the moment
        propagation_parameters = {
            "potentials": potentials,
            "time_step": time_step,
            "N_iterations": N_iterations,
            "final_time": final_time
        }

        # Initial setup of the callbacks
        for callback in callbacks:
            callback.set_gas(self)
            callback.set_propagation_params(propagation_parameters)

        two_state_real_time_propagation(
            self, potentials, adim_time_step, times, callbacks, leave_progress_bar)

    @property
    def density(self):
        """The density of the gas in real space
        """
        return torch.abs(self.psi)**2

    @property
    def densityk(self):
        """The density of the gas in momentum space
        """
        return torch.abs(self.psik)**2

    @property
    def phase(self):
        """The phase (in radians) of the real space wave function
        """
        return torch.angle(self.psi)

    def _spinor_norm_real_space(self, psi):
        """
        psi has shape:
            (2, N_grid, N_grid, N_grid)

        Returns:
            sqrt(int (|psi1|^2 + |psi2|^2) dV)
        """
        dV = self.dx * self.dy * self.dz
        density_total = torch.abs(psi[0])**2 + torch.abs(psi[1])**2
        return torch.sqrt(torch.sum(density_total) * dV)

    def _spinor_norm_real_space(self, psi):
        """
        Total spinor norm:
            int (|psi1|^2 + |psi2|^2) dV
        """
        dV = self.dx * self.dy * self.dz
        density = torch.abs(psi[0])**2 + torch.abs(psi[1])**2
        return torch.sqrt(torch.sum(density) * dV)

    def _spinor_norm_momentum_space(self, psik):
        """
        Momentum-space total spinor norm.
        This mirrors the normalization style used by your original Gas.psik setter.
        """
        dK = self.gas1.dkx * self.gas1.dky * self.gas1.dkz
        density_k = torch.abs(psik[0])**2 + torch.abs(psik[1])**2
        return torch.sqrt(torch.sum(density_k) * dK)

    def _normalize_spinor_real_space(self, psi):
        if psi.dtype != self.complex_dtype:
            psi = psi.type(self.complex_dtype)

        psi = psi.to(device=self.device)

        norm = self._spinor_norm_real_space(psi)

        if torch.abs(norm).item() == 0.0:
            raise ValueError("Cannot normalize a zero two-component spinor.")

        return psi / norm

    def _normalize_spinor_momentum_space(self, psik):
        if psik.dtype != self.complex_dtype:
            psik = psik.type(self.complex_dtype)

        psik = psik.to(device=self.device)

        norm = self._spinor_norm_momentum_space(psik)

        if torch.abs(norm).item() == 0.0:
            raise ValueError("Cannot normalize a zero two-component spinor in momentum space.")

        return psik / norm

    @property
    def psi(self):
        """
        Return the two-component real-space spinor.

        Important:
        This avoids calling gas1.psi and gas2.psi because those normalize
        the two components separately.
        """
        if self._updated_wavefunction == UPDATED_PSIK:
            n = self.N_grid
            h = n // 2

            psi1_full = iftn(self.gas1._psik)
            psi2_full = iftn(self.gas2._psik)

            psi1 = psi1_full[h:h+n, h:h+n, h:h+n]
            psi2 = psi2_full[h:h+n, h:h+n, h:h+n]

            psi = torch.stack((psi1, psi2))
            psi = self._normalize_spinor_real_space(psi)

            self.gas1._psi = psi[0]
            self.gas2._psi = psi[1]

            self.gas1._updated_wavefunction = UPDATED_BOTH
            self.gas2._updated_wavefunction = UPDATED_BOTH
            self._updated_wavefunction = UPDATED_BOTH

        return torch.stack((self.gas1._psi, self.gas2._psi))

    @psi.setter
    def psi(self, value):
        """
        Set the spinor and normalize the total two-component wavefunction,
        not each component separately.
        """
        psi = self._normalize_spinor_real_space(value)

        self.gas1._psi = psi[0]
        self.gas2._psi = psi[1]

        self.gas1._updated_wavefunction = UPDATED_PSI
        self.gas2._updated_wavefunction = UPDATED_PSI
        self._updated_wavefunction = UPDATED_PSI

    @property
    def psik(self):
        """
        Return the two-component momentum-space spinor.

        Important:
        This avoids calling gas1.psik and gas2.psik because those normalize
        the two components separately.
        """
        if self._updated_wavefunction == UPDATED_PSI:
            n = self.N_grid
            h = n // 2

            padded_psi1 = pad(
                self.gas1._psi,
                (h, h, h, h, h, h),
                mode="constant",
                value=0,
            )

            padded_psi2 = pad(
                self.gas2._psi,
                (h, h, h, h, h, h),
                mode="constant",
                value=0,
            )

            psik1 = ftn(padded_psi1)
            psik2 = ftn(padded_psi2)

            psik = torch.stack((psik1, psik2))
            psik = self._normalize_spinor_momentum_space(psik)

            self.gas1._psik = psik[0]
            self.gas2._psik = psik[1]

            self.gas1._updated_wavefunction = UPDATED_BOTH
            self.gas2._updated_wavefunction = UPDATED_BOTH
            self._updated_wavefunction = UPDATED_BOTH

        return torch.stack((self.gas1._psik, self.gas2._psik))

    @psik.setter
    def psik(self, value):
        """
        Set the spinor in momentum space and normalize the total spinor norm.
        """
        psik = self._normalize_spinor_momentum_space(value)

        self.gas1._psik = psik[0]
        self.gas2._psik = psik[1]

        self.gas1._updated_wavefunction = UPDATED_PSIK
        self.gas2._updated_wavefunction = UPDATED_PSIK
        self._updated_wavefunction = UPDATED_PSIK

    @property
    def component_fractions(self):
        """
        Current component fractions:
            f1 = int |psi1|^2 dV
            f2 = int |psi2|^2 dV

        With total spinor normalization, f1 + f2 should be close to 1.
        """
        psi = self.psi
        dV = self.dx * self.dy * self.dz

        f1 = torch.sum(torch.abs(psi[0])**2) * dV
        f2 = torch.sum(torch.abs(psi[1])**2) * dV

        return torch.stack((f1.real, f2.real))

    @property
    def component_numbers(self):
        """
        Current molecule numbers in each component.
        """
        return self.N_total * self.component_fractions

    @property
    def coordinates(self):
        """The coordinates of the gas

        Returns a tuple containing the coordinates of the gas in real space.
        """
        return (self.X, self.Y, self.Z)
    
    @property
    def momenta(self):
        """The momenta of the gas

        Returns a tuple containing the momenta of the gas in momentum space.
        """
        return (self.Kx, self.Ky, self.Kz)