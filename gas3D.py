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

UPDATED_PSI = 0
UPDATED_PSIK = 1
UPDATED_BOTH = 2

class Zero(LinearPotential):
    """Zero potential. It is equivalent to not applying any potential at all.
    """

    def __init__(self):
        super().__init__()

    def get_potential(self, X: torch.tensor, Y: torch.tensor, Z: torch.tensor, time: float = None):
        return torch.zeros_like(X)

class Gas():
    """Quantum gas.

    The parameters :py:attr:`N_grid` and :py:attr:`grid_size` specify a computational grid on which the wavefunction
    is defined and evolved. :class:`Gas` exposes methods to perform real time propagation and to compute the ground 
    state's wave function via imaginary time propagation.

    Args:
        element (str): Optional. The element the gas is made of. Defaults to "87Rb".
        N_particles (int): Optional. The number of particles in the gas. Defaults to :math:`10^6`.
        N_grid (int): Optional. The number of points on each side of the computational grid. Defaults to :math:`2^8`.
        grid_size (float): Optional. The side of the computational grid. Defaults to :math:`10^{-6}`.
        device (torch.device or None): Optional. The device where to store tensors. Defaults to None, meaning that GPU will be used if available.
        float_dtype (:py:attr:`torch.dtype`): Optional. The dtype used to represent floating point numbers. Defaults to :py:attr:`torch.double`.
        complex_dtype (:py:attr:`torch.dtype`): Optional. The dtype used to represent complex numbers. Defaults to :py:attr:`torch.complex128`.
        adimensionalization_length (float): Optional. The unit of length to be used during the simulations. Defaults to :math:`10^{-6}`.
    """

    def __init__(
        self,
        element: str = "87Rb",
        N_particles: int = int(1e6),
        N_grid: int = 2**7,
        grid_size: float | tuple[float, float, float] = 1e-6,
        device: Union[torch.device, None] = None,
        float_dtype: torch.dtype = torch.double,
        complex_dtype: torch.dtype = torch.complex128,
        adimensionalization_length: float = 1e-6,
    ) -> None:

        self.element = element
        self.mass = elements_dict[self.element]["m"]
        self.d2_pulse = elements_dict[self.element]["omega d2"]

        if N_particles != int(N_particles):
            raise TypeError("The number of particles must be an integer")
        self.N_particles = int(N_particles)

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype

        self.adim_length = adimensionalization_length
        self.adim_pulse = spconsts.hbar / (self.mass * self.adim_length**2)

        if isinstance(grid_size, tuple):
            grid_size_x, grid_size_y, grid_size_z = grid_size
        else:
            grid_size_x = grid_size_y = grid_size_z = grid_size

        self.grid_size_x = grid_size_x / self.adim_length
        self.grid_size_y = grid_size_y / self.adim_length
        self.grid_size_z = grid_size_z / self.adim_length

        self.N_grid = int(N_grid)

        # Real-space grid
        self.x = torch.linspace(
            -self.grid_size_x / 2,
            self.grid_size_x / 2,
            self.N_grid,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.y = torch.linspace(
            -self.grid_size_y / 2,
            self.grid_size_y / 2,
            self.N_grid,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.z = torch.linspace(
            -self.grid_size_z / 2,
            self.grid_size_z / 2,
            self.N_grid,
            dtype=self.float_dtype,
            device=self.device,
        )

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

        coordinates = torch.meshgrid(self.x, self.y, self.z, indexing="xy")
        self.X = coordinates[0]
        self.Y = coordinates[1]
        self.Z = coordinates[2]
        del coordinates

        # Momentum-space grid.
        # This mirrors the 2D code: real-space psi is padded before FFT,
        # so the momentum grid has length 2 * N_grid.
        self.kx = 2 * np.pi * torch.fft.fftshift(
            torch.fft.fftfreq(
                self.N_grid + 2 * (self.N_grid // 2),
                self.dx,
                dtype=self.float_dtype,
                device=self.device,
            )
        )
        self.ky = 2 * np.pi * torch.fft.fftshift(
            torch.fft.fftfreq(
                self.N_grid + 2 * (self.N_grid // 2),
                self.dy,
                dtype=self.float_dtype,
                device=self.device,
            )
        )
        self.kz = 2 * np.pi * torch.fft.fftshift(
            torch.fft.fftfreq(
                self.N_grid + 2 * (self.N_grid // 2),
                self.dz,
                dtype=self.float_dtype,
                device=self.device,
            )
        )

        self.dkx = self.kx[1] - self.kx[0]
        self.dky = self.ky[1] - self.ky[0]
        self.dkz = self.kz[1] - self.kz[0]

        momenta = torch.meshgrid(self.kx, self.ky, self.kz, indexing="xy")
        self.Kx = momenta[0]
        self.Ky = momenta[1]
        self.Kz = momenta[2]
        del momenta

        self._psi = torch.zeros_like(self.X, dtype=self.complex_dtype)
        self._psik = torch.zeros_like(self.Kx, dtype=self.complex_dtype)
        self._updated_wavefunction = None

    def ground_state(self, potentials: List[Potential] = [], time_step: complex = -1e-6j, N_iterations: int = int(1e3), callbacks: List[Callback] = [], leave_progress_bar=True):
        """Compute the ground state's wave function.

        Use the split-step Fourier method with imaginary time propagation (ITP) to compute the ground state's wave function of the gas. 
        The potentials acting on the system are specified via the :py:attr:`potentials` parameter. 

        Args:
            potentials (List[:class:`~gpe.utils.potentials.Potential`]): Optional. The list of potentials acting on the system. Defaults to [].
            time_step (complex): Optional. The time step to be used in the ITP. Defaults to 
            N_iterations (int): Optional. The number of steps of ITP to perform. Defaults to 
            callbacks (List[:class:`~gpe.utils.callbacks.Callback`]): Optional. List of callbacks to be evaluated during the evolution. Defaults to [].
            leave_progress_bar (bool): Optional. Whether to leave the progress bar on screen after the propagation ends. Defaults to True.

        Raises:
            Exception: If time dependent potentials are specified 
            Exception: If the time step is not a purely imaginary number
            Exception: If the imaginary part of the time step is not positive
            Exception: If neither the wave function in real space nor in the one in momentum space have been initialized
        """
        
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

        imaginary_time_propagation(
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

        real_time_propagation(
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

    # --- Manage the update of psi and psik ---

    @property
    def psi(self):
        """The real space wave function of the gas.

        Returns the most updated real space wave function of the gas. If the last updated wave function is the one in momentum space, 
        computes and stores the real space wave function as its iFFT before returning it. 
        When a value is assigned to psi, takes care of the normalization before storing it.
        """

        # If the last updated wave function is psik, compute psi
        if self._updated_wavefunction == UPDATED_PSIK:
            n = self.N_grid
            h = n // 2
            psi_full = iftn(self._psik)
            self.psi = psi_full[h:h+n, h:h+n, h:h+n]
            self._updated_wavefunction = UPDATED_BOTH
        return self._psi

    @psi.setter
    def psi(self, value):
        if value.dtype != self.complex_dtype:
            value = value.type(self.complex_dtype)
        self._psi = normalize_wavefunction(value, self.dx, self.dy, self.dz)
        self._updated_wavefunction = UPDATED_PSI

    @property
    def psik(self):
        """The momentum space wave function of the gas.

        Returns the most updated momentum space wave function of the gas. If the last updated wave function is the one in real space, 
        computes and stores the momentum space wave function as its iFFT before returning it. 
        When a value is assigned to psik, takes care of the normalization before storing it.
        """

        # If the last updated wave function is psi, compute psik
        if self._updated_wavefunction == UPDATED_PSI:
            n = self.N_grid
            h = n // 2
            # For a 3D tensor, torch.nn.functional.pad uses:
            # (z_left, z_right, y_left, y_right, x_left, x_right)
            padded_psi = pad(
                self._psi,
                (h, h, h, h, h, h),
                mode="constant",
                value=0,
            )
            self.psik = ftn(padded_psi)
            self._updated_wavefunction = UPDATED_BOTH
        return self._psik

    @psik.setter
    def psik(self, value):
        if value.dtype != self.complex_dtype:
            value = value.type(self.complex_dtype)
        self._psik = normalize_wavefunction(value, self.dkx, self.dky, self.dkz)
        self._updated_wavefunction = UPDATED_PSIK


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
