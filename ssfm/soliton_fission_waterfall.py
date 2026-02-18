import numpy as np
import matplotlib.pyplot as plt

# Parameters
T_max = 10
N_t = 2048
t = np.linspace(-T_max, T_max, N_t)
dt = t[1] - t[0]

beta2 = -1
gamma = 1
N_order = 3

# Spatial domain
L = np.pi / 2
N_z = 100
dz = L / N_z
z_array = np.linspace(0, L, N_z)

# Frequency grid
w = 2 * np.pi * np.fft.fftfreq(N_t, d=dt)
H = np.exp(-0.5j * beta2 * w**2 * dz)

# Initial condition
A0 = N_order * np.cosh(t)**-1
A = A0.copy().astype(np.complex128)

# Storage for z-evolution
A_z = np.zeros((N_z, N_t), dtype=np.complex128)
A_z[0, :] = A

# SSFM loop
for i in range(1, N_z):
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)
    A_freq = np.fft.fft(A)
    A_freq *= H
    A = np.fft.ifft(A_freq)
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz / 2)
    A_z[i, :] = A

# Plotting
plt.figure(figsize=(10, 6))
extent = [t[0], t[-1], z_array[-1], z_array[0]]
plt.imshow(np.abs(A_z)**2, extent=extent, aspect='auto', cmap='inferno')
plt.colorbar(label='|A(t,z)|²')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation Distance z (km)')
plt.title('Z-Evolution Waterfall of Higher-Order Soliton Fission (N=3)')
plt.tight_layout()
plt.savefig("../figures/z_evol_waterfall_fission.png", dpi=500)
plt.show()