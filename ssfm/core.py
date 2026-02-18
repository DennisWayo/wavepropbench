import numpy as np

def ssfm_propagate(A0, beta2, gamma, dz, n_steps, dt):
    """
    Split-Step Fourier Method propagation.
    Preserves original algorithm ordering.
    """
    N = len(A0)
    w = np.fft.fftfreq(N, dt) * 2 * np.pi

    A = A0.astype(np.complex128)

    if beta2 != 0:
        H = np.exp(-1j * 0.5 * beta2 * w**2 * dz)
    else:
        H = None

    for _ in range(n_steps):

        # Nonlinear half-step
        if gamma != 0:
            A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)

        # Dispersion full-step
        if H is not None:
            A_freq = np.fft.fft(A)
            A_freq *= H
            A = np.fft.ifft(A_freq)

        # Nonlinear half-step
        if gamma != 0:
            A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)

    return A