import torch


class SingleMoleculeHamiltonian:
    """
    Two-state single-molecule Hamiltonian with stable imaginary-time propagation.

    H_rad_s should be in angular-frequency units, rad/s.

    The propagation code passes an already dimensionless time_step, because
    TwoComponentGas.ground_state multiplies the physical time_step by
    gas.adim_pulse before calling the propagation routine.

    This class converts:
        H_dim = H_rad_s / gas.adim_pulse
    """

    is_time_dependent = False

    def __init__(
        self,
        delta=0.0,
        Omega=0.0,
        phase=0.0,
        H_rad_s=None,
    ):
        self.delta = delta
        self.Omega = Omega
        self.phase = phase
        self.H_rad_s = H_rad_s

    def set_gas(self, gas):
        self.gas = gas

    def on_propagation_begin(self):
        device = self.gas.device
        cdtype = self.gas.complex_dtype
        rdtype = self.gas.float_dtype

        if self.H_rad_s is not None:
            H = torch.as_tensor(
                self.H_rad_s,
                dtype=cdtype,
                device=device,
            )
        else:
            delta = torch.as_tensor(self.delta, dtype=rdtype, device=device)
            Omega = torch.as_tensor(self.Omega, dtype=rdtype, device=device)
            phase = torch.as_tensor(self.phase, dtype=rdtype, device=device)

            H = torch.zeros((2, 2), dtype=cdtype, device=device)

            H[0, 0] = 0.5 * delta
            H[1, 1] = -0.5 * delta

            H[0, 1] = -0.5 * Omega * torch.exp(-1j * phase)
            H[1, 0] = -0.5 * Omega * torch.exp(1j * phase)

        if not torch.allclose(H, H.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("Single-molecule Hamiltonian must be Hermitian.")

        # Convert rad/s to dimensionless GP energy units.
        self.H_dim = H / self.gas.adim_pulse

        # Precompute eigenbasis.
        self.evals, self.evecs = torch.linalg.eigh(self.H_dim)

        print("Single-molecule Hamiltonian diagnostics")
        print("--------------------------------------")
        print("H_dim eigenvalues:", self.evals.detach().cpu().numpy())
        print(
            "H_dim eigenvalue spread:",
            (self.evals[-1] - self.evals[0]).detach().cpu().item(),
        )

    def apply_spinor(self, gas, time_step, time=None):
        """
        Apply exp(-i H dt) to the two-component spinor.

        For imaginary time, time_step = -i d_tau, so:
            exp(-i H time_step) = exp(-H d_tau)

        We shift H by its lowest eigenvalue. This multiplies the spinor by
        a global scalar, which is removed by total normalization, but it
        prevents overflow.
        """
        dt = torch.as_tensor(
            time_step,
            dtype=gas.complex_dtype,
            device=gas.device,
        )

        psi = gas.psi
        old_shape = psi.shape
        psi_flat = psi.reshape(2, -1)

        # Imaginary time: dt is purely imaginary and negative.
        if abs(dt.real.detach().cpu().item()) < 1e-15 and dt.imag.detach().cpu().item() < 0:
            tau = -dt.imag.real

            # Shift lowest eigenvalue to zero.
            # Then all exp factors are <= 1.
            evals_shifted = self.evals - self.evals[0]

            max_exponent = (tau * evals_shifted[-1]).detach().cpu().item()
            if max_exponent > 80:
                print(
                    "Warning: single-molecule imaginary-time step is stiff. "
                    f"tau * eigenvalue spread = {max_exponent:.3e}. "
                    "Reduce time_step if convergence is poor."
                )

            factors = torch.exp(-tau * evals_shifted)

        else:
            # Real-time unitary evolution.
            factors = torch.exp(-1j * self.evals * dt)

        U = self.evecs @ torch.diag(factors.to(gas.complex_dtype)) @ self.evecs.conj().T

        new_psi = (U @ psi_flat).reshape(old_shape)

        if not torch.isfinite(new_psi).all():
            raise FloatingPointError("NaN/Inf created by SingleMoleculeHamiltonian.")

        gas.psi = new_psi