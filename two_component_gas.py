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
from gas3D import Gas, Zero

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

    @property
    def psi(self):
        psi_both = torch.stack((self.gas1.psi, self.gas2.psi))
        self._updated_wavefunction = self.gas1._updated_wavefunction
        return psi_both

    @psi.setter
    def psi(self, value):
        self.gas1.psi = value[0]
        self.gas2.psi = value[1]
        self._updated_wavefunction = self.gas1._updated_wavefunction

    @property
    def psik(self):
        psik_both = torch.stack((self.gas1.psik, self.gas2.psik))
        self._updated_wavefunction = self.gas1._updated_wavefunction
        return psik_both

    @psik.setter
    def psik(self, value):
        self.gas1.psik = value[0]
        self.gas2.psik = value[1]
        self._updated_wavefunction = self.gas1._updated_wavefunction

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