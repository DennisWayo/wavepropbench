import numpy as np
import matplotlib.pyplot as plt

# Time grid
T = 10
nt = 2**12
t = np.linspace(-T, T, nt)
dt = t[1] - t[0]

# Frequency grid
w = 2 * np.pi * np.fft.fftfreq(nt, d=dt)

# Physical constants for SSFM
beta2 = -1.0      # (ps^2/km)
gamma = 0.0       # (1/W/km)
z_max = 1.0       # (km)
n_steps = 100
dz = z_max / n_steps

# Input pulse (N=3)
N_order = 3
A0 = N_order / np.cosh(t)
A = A0.copy().astype(np.complex128)

# Linear operator
dispersion = np.exp(-0.5j * beta2 * w**2 * dz)

# SSFM loop
for _ in range(n_steps):
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)
    A_freq = np.fft.fft(A)
    A_freq *= dispersion
    A = np.fft.ifft(A_freq)
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)

A_out = A
freq = np.fft.fftshift(np.fft.fftfreq(nt, d=dt)) * 1e-3  # keep your original scaling

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, np.abs(A0)**2, 'b--', label='Input |A₀|²')
plt.plot(t, np.abs(A_out)**2, 'orange', label='Output |A(z)|²')
plt.title('Higher-Order Soliton Fission (N=3) via SSFM')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(freq,
         np.fft.fftshift(np.abs(np.fft.fft(A0))**2 / np.max(np.abs(np.fft.fft(A0))**2)),
         'b--', label='Input Spectrum')
plt.plot(freq,
         np.fft.fftshift(np.abs(np.fft.fft(A_out))**2 / np.max(np.abs(np.fft.fft(A_out))**2)),
         'orange', label='Output Spectrum')
plt.xlabel('Frequency (THz)')
plt.ylabel('Normalized Power')
plt.title('Spectral Broadening from Soliton Fission')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("../figures/fission_spectral.png", dpi=300)
plt.show()