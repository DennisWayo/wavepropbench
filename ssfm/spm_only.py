import numpy as np
import matplotlib.pyplot as plt
from ssfm.core import ssfm_propagate

# Grid
N = 2**14
T_max = 1e-11
dt = 2 * T_max / N
t = np.linspace(-T_max, T_max, N)

# Pulse
tau = 150e-15 / (2 * np.sqrt(2 * np.log(2)))
A0 = np.exp(-t**2 / (2 * tau**2))
A0 *= 2.0

# Parameters
beta2 = 0
gamma = 5.0
z_max = 0.01
n_steps = 1000
dz = z_max / n_steps

# Propagate
A_output = ssfm_propagate(A0, beta2, gamma, dz, n_steps, dt)

# Spectra
S_in = np.fft.fftshift(np.fft.fft(A0))
S_out = np.fft.fftshift(np.fft.fft(A_output))
f = np.fft.fftshift(np.fft.fftfreq(N, dt)) / 1e12

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 7))

axs[0].plot(t * 1e12, np.abs(A0)**2, '--', label="Input |A₀|²")
axs[0].plot(t * 1e12, np.abs(A_output)**2, label="Output |A(z)|²")
axs[0].set_xlabel("Time (ps)")
axs[0].set_ylabel("Normalized Intensity")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(f, np.abs(S_in)**2 / np.max(np.abs(S_in)**2), '--', label="Input Spectrum")
axs[1].plot(f, np.abs(S_out)**2 / np.max(np.abs(S_out)**2), label="Output Spectrum")
axs[1].set_xlabel("Frequency Offset Δf (THz)")
axs[1].set_ylabel("Normalized Spectral Power")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("../figures/SSFM_SPM_Only.png", dpi=300)
plt.show()

print("SPM-only case completed.")