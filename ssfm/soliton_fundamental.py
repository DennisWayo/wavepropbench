import numpy as np
import matplotlib.pyplot as plt

# Parameters
T0 = 0.1e-12              # Initial pulse width (100 fs)
P0 = 1.0                  # Peak power (normalized)
beta2 = -1e-26            # GVD (s^2/m)
gamma = 1.0               # Nonlinear coefficient (1/W·m)
L = 0.01                  # Propagation distance (10 mm)
nt = 2048                 # Number of time samples
time_window = 2e-11       # Total time window (20 ps)

# Time and frequency axes
t = np.linspace(-time_window/2, time_window/2, nt)
dt = t[1] - t[0]
w = 2 * np.pi * np.fft.fftfreq(nt, d=dt)

# Initial pulse (sech for soliton)
A0 = np.sqrt(P0) * (1 / np.cosh(t / T0))

# Propagation setup
n_steps = 200
dz = L / n_steps
A = A0.astype(np.complex128)
H = np.exp(-0.5j * beta2 * w**2 * dz)

# SSFM propagation
for _ in range(n_steps):
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)
    A_freq = np.fft.fft(A)
    A_freq *= H
    A = np.fft.ifft(A_freq)
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)

A_output = A

# Spectrum
freq = np.fft.fftshift(w) / (2 * np.pi * 1e12)  # THz
input_spec = np.fft.fftshift(np.abs(np.fft.fft(A0))**2)
output_spec = np.fft.fftshift(np.abs(np.fft.fft(A_output))**2)
input_spec /= np.max(input_spec)
output_spec /= np.max(output_spec)

# Visualization
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(t * 1e12, np.abs(A0)**2, 'b--', label='Input |A₀|²')
axs[0].plot(t * 1e12, np.abs(A_output)**2, 'orange', label='Output |A(z)|²')
axs[0].set_title("Fundamental Soliton Propagation via SSFM")
axs[0].set_xlabel("Time (ps)")
axs[0].set_ylabel("Normalized Intensity")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(freq, input_spec, 'b--', label='Input Spectrum')
axs[1].plot(freq, output_spec, 'orange', label='Output Spectrum')
axs[1].set_title("Spectrum Preservation for Soliton")
axs[1].set_xlabel("Frequency Offset Δf (THz)")
axs[1].set_ylabel("Normalized Spectral Power")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("../figures/soliton_prop_spectrum_soliton.png", dpi=300)
plt.show()