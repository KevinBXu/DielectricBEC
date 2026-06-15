from torchgpe.utils.potentials import LinearPotential, NonLinearPotential
import torch
from tqdm.auto import tqdm, trange

def is_spinor_coupling(potential):
    return hasattr(potential, "apply_spinor")

def as_two_component_potential(V, gas):
    """
    Convert potential to shape:
        (2, N, N, N)
    """
    if not torch.is_tensor(V):
        V = torch.zeros_like(gas.X)

    V = V.to(device=gas.device)

    if V.ndim == 3:
        return torch.stack((V, V))

    if V.ndim == 4 and V.shape[0] == 2:
        return V

    raise ValueError(
        "Potential must have shape (N,N,N) or (2,N,N,N). "
        f"Got shape {tuple(V.shape)}."
    )

def zero_two_component_potential(gas):
    return torch.zeros(
        (2,) + tuple(gas.X.shape),
        dtype=gas.float_dtype,
        device=gas.device,
    )

def potential_propagator(
    gas,
    time_step,
    total_static_linear_potential,
    dynamic_linear_potentials,
    static_nonlinear_potentials,
    dynamic_nonlinear_potentials,
    time,
):
    """
    Diagonal potential propagator.

    For imaginary time, use a global scalar shift:
        V -> V - min(V)

    This prevents exp(+large) overflow from negative potentials.
    """

    V_total = as_two_component_potential(total_static_linear_potential, gas)

    for potential in static_nonlinear_potentials:
        Vnl = potential.potential_function(*gas.coordinates, gas.psi)
        V_total = V_total + as_two_component_potential(Vnl, gas)

    for potential in dynamic_linear_potentials:
        Vdyn = potential.get_potential(*gas.coordinates, time)
        V_total = V_total + as_two_component_potential(Vdyn, gas)

    for potential in dynamic_nonlinear_potentials:
        Vdyn_nl = potential.potential_function(*gas.coordinates, gas.psi, time)
        V_total = V_total + as_two_component_potential(Vdyn_nl, gas)

    if not torch.isfinite(V_total).all():
        raise FloatingPointError("NaN/Inf detected in total diagonal potential.")

    dt = torch.as_tensor(
        time_step,
        dtype=gas.complex_dtype,
        device=gas.device,
    )

    # Imaginary-time case: time_step = -i tau.
    if abs(dt.real.detach().cpu().item()) < 1e-15 and dt.imag.detach().cpu().item() < 0:
        tau = -dt.imag.real

        V_real = V_total.real

        # One global scalar shift, not component-wise.
        V_min = torch.amin(V_real)
        V_shifted = V_real - V_min

        exponent_span = (tau * torch.amax(V_shifted)).detach().cpu().item()

        if exponent_span > 80:
            print(
                "Warning: diagonal imaginary-time potential step is stiff. "
                f"tau * potential span = {exponent_span:.3e}. "
                "Reduce time_step or ramp the interactions."
            )

        return torch.exp(-tau * V_shifted)

    # Real-time case.
    return torch.exp(-1j * V_total * dt)

def two_state_imaginary_time_propagation(gas, potentials, time_step, N_iterations, callbacks, leave_progress_bar=True):
    """Performs imaginary time propagation of a wave function.

    Args:
        gas (Gas): The gas whose wave function has to be propagated.
        potentials (list): The list of potentials to apply.
        time_step (float): The time step to use.
        N_iterations (int): The number of iterations to perform.
        callbacks (list): The list of callbacks to call at the end of each iteration.
        leave_progress_bar (bool, optional): Whether to leave the progress bar after the propagation is complete. Defaults to True.
    """
    # Divide the potentials in linear and nonlinear to precompute the linear ones
    spinor_couplings = [potential for potential in potentials
        if is_spinor_coupling(potential)]
    linear_potentials = [potential for potential in potentials if issubclass(
        type(potential), LinearPotential)]
    nonlinear_potentials = [potential for potential in potentials if issubclass(
        type(potential), NonLinearPotential)]

    for callback in callbacks:
        callback.on_propagation_begin()

    # Precompute kinetic propagator and the total linear potential
    kinetic = 0.5 * sum(momentum**2 for momentum in gas.momenta)
    kinetic_propagator = torch.exp(-0.5j * kinetic * time_step)
    kinetic_propagator = torch.stack((kinetic_propagator, kinetic_propagator))
    
    if len(linear_potentials) == 0:
        total_linear_potential = torch.zeros_like(gas.X)
    else:
        total_linear_potential = sum(
            potential.get_potential(*gas.coordinates)
            for potential in linear_potentials
        )
    total_linear_potential = as_two_component_potential(total_linear_potential, gas)

    # Create a progress bar to monitor the evolution
    pbar = trange(N_iterations, smoothing=0, desc="Ground state",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=leave_progress_bar)

    for epoch in pbar:
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # One step of the split-step Fourier method
        two_state_propagation_step(
            gas,
            total_linear_potential,
            [],
            nonlinear_potentials,
            [],
            spinor_couplings,
            kinetic_propagator,
            time_step,
        )

        for callback in callbacks:
            callback.on_epoch_end(epoch)

    for callback in callbacks:
        callback.on_propagation_end()


def two_state_real_time_propagation(gas, potentials, time_step, times, callbacks, leave_progress_bar=True):
    """Performs real time propagation of a wave function.

    Args:
        gas (Gas): The gas whose wave function has to be propagated.
        potentials (list): The list of potentials to apply.
        time_step (float): The time step to use.
        times (list): The list of times to propagate to.
        callbacks (list): The list of callbacks to call at the end of each iteration.
        leave_progress_bar (bool, optional): Whether to leave the progress bar after the propagation is complete. Defaults to True.
    """
    # Divide the potentials in linear and nonlinear, time dependent and time independent to precompute the static linear ones
    static_linear_potentials = [potential for potential in potentials if issubclass(
        type(potential), LinearPotential) and not potential.is_time_dependent]
    dynamic_linear_potentials = [potential for potential in potentials if issubclass(
        type(potential), LinearPotential) and potential.is_time_dependent]
    static_nonlinear_potentials = [potential for potential in potentials if issubclass(
        type(potential), NonLinearPotential) and not potential.is_time_dependent]
    dynamic_nonlinear_potentials = [potential for potential in potentials if issubclass(
        type(potential), NonLinearPotential) and potential.is_time_dependent]
    spinor_couplings = [potential for potential in potentials
        if is_spinor_coupling(potential)]

    for callback in callbacks:
        callback.on_propagation_begin()

    # Precompute kinetic propagator and the total static linear potential
    kinetic = 0.5 * sum(momentum**2 for momentum in gas.momenta)
    kinetic_propagator = torch.exp(-0.5j * kinetic * time_step)
    kinetic_propagator = torch.stack((kinetic_propagator, kinetic_propagator))
    
    if len(static_linear_potentials) == 0:
        total_static_linear_potential = torch.zeros_like(gas.X)
    else:
        total_static_linear_potential = sum(
            potential.get_potential(*gas.coordinates)
            for potential in static_linear_potentials
        )

    total_static_linear_potential = as_two_component_potential(
        total_static_linear_potential,
        gas,
    )

    # Create a progress bar to monitor the evolution
    pbar = tqdm(times, smoothing=0, desc="Propagation",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', leave=leave_progress_bar)

    for epoch, t in enumerate(pbar):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # One step of the split-step Fourier method
        two_state_propagation_step(
            gas,
            total_static_linear_potential,
            dynamic_linear_potentials,
            static_nonlinear_potentials,
            dynamic_nonlinear_potentials,
            spinor_couplings,
            kinetic_propagator,
            time_step,
            t,
        )

        for callback in callbacks:
            callback.on_epoch_end(epoch)

    for callback in callbacks:
        callback.on_propagation_end()

def two_state_propagation_step(
    gas,
    total_static_linear_potential,
    dynamic_linear_potentials,
    static_nonlinear_potentials,
    dynamic_nonlinear_potentials,
    spinor_couplings,
    kinetic_propagator,
    time_step,
    time=None,
):
    if spinor_couplings is None:
        spinor_couplings = []

    gas.psik = gas.psik * kinetic_propagator

    if len(spinor_couplings) == 0:
        gas.psi = gas.psi * potential_propagator(
            gas,
            time_step,
            total_static_linear_potential,
            dynamic_linear_potentials,
            static_nonlinear_potentials,
            dynamic_nonlinear_potentials,
            time,
        )

    else:
        gas.psi = gas.psi * potential_propagator(
            gas,
            0.5 * time_step,
            total_static_linear_potential,
            dynamic_linear_potentials,
            static_nonlinear_potentials,
            dynamic_nonlinear_potentials,
            time,
        )

        for coupling in spinor_couplings:
            coupling.apply_spinor(gas, time_step, time)

        gas.psi = gas.psi * potential_propagator(
            gas,
            0.5 * time_step,
            total_static_linear_potential,
            dynamic_linear_potentials,
            static_nonlinear_potentials,
            dynamic_nonlinear_potentials,
            time,
        )

    gas.psik = gas.psik * kinetic_propagator

    if not torch.isfinite(gas.psi).all():
        raise FloatingPointError("NaN/Inf detected in gas.psi after propagation step.")