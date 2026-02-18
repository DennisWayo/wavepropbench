import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Parameters
N = 2**12              # Number of sample points
T = 100.0              # Time window (ps)
t = np.linspace(-T/2, T/2, N)
dt = t[1] - t[0]
f = np.fft.fftfreq(N, d=dt)
w = 2 * np.pi * f      # Angular frequency

# SSFM Physical constants
beta2 = -1.0           # GVD parameter (ps^2/km)
gamma = 1.0            # Nonlinearity (1/W/km)
z = 1.0                # Propagation length (km)
n_steps = 100
dz = z / n_steps

# Raman response function
tau1, tau2 = 12.2, 32.0
h_R_t = ((tau1**2 + tau2**2) / (tau1 * tau2**2)) * np.exp(-t / tau2) * np.sin(t / tau1)
h_R_t /= trapezoid(h_R_t, t)
H_R_w = np.fft.fft(h_R_t)

# Input pulse (higher-order soliton)
P0 = 1.0
T0 = 1.0
A0 = np.sqrt(P0) / np.cosh(t / T0)

# Initialize fields
A = A0.astype(np.complex128)
dispersion_op = np.exp(-0.5j * beta2 * (w**2) * dz)

# SSFM loop with Raman and Kerr effects
for _ in range(n_steps):
    conv = np.fft.ifft(H_R_w * np.fft.fft(np.abs(A)**2)) * A
    NL = gamma * (1 - 0.18) * np.abs(A)**2 * A + gamma * 0.18 * conv
    A *= np.exp(1j * NL * dz / 2)

    A_freq = np.fft.fft(A)
    A_freq *= dispersion_op
    A = np.fft.ifft(A_freq)

    conv = np.fft.ifft(H_R_w * np.fft.fft(np.abs(A)**2)) * A
    NL = gamma * (1 - 0.18) * np.abs(A)**2 * A + gamma * 0.18 * conv
    A *= np.exp(1j * NL * dz / 2)

# Plotting
plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.plot(t, np.abs(A0)**2, 'b--', label='Input $|A_0|^2$')
plt.plot(t, np.abs(A)**2, 'orange', label='Output $|A(z)|^2$')
plt.title('Hybrid Soliton Propagation via SSFM\n(DMS + Raman + Higher-Order Soliton)')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(np.fft.fftshift(f),
         np.fft.fftshift(np.abs(np.fft.fft(A0))**2) / np.max(np.abs(np.fft.fft(A0))**2),
         'b--', label='Input Spectrum')
plt.plot(np.fft.fftshift(f),
         np.fft.fftshift(np.abs(np.fft.fft(A))**2) / np.max(np.abs(np.fft.fft(A))**2),
         'orange', label='Output Spectrum')
plt.title('Spectral Broadening & Shifting from Hybrid Effects')
plt.xlabel('Frequency Offset Δf (THz)')
plt.ylabel('Normalized Spectral Power')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("../figures/hybrid_soliton_spect_broad.png", dpi=300)
plt.show()