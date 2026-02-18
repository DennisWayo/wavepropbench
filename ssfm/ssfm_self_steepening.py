import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq

# Simulation Parameters
T0 = 1.0
N = 5
beta2 = -1.0
gamma = 1.0
tau_shock = 0.02

z_max = 2.0
nz = 1000
nt = 2048
t_max = 8

dz = z_max / nz
t = np.linspace(-t_max, t_max, nt)
dt = t[1] - t[0]
w = 2 * np.pi * fftshift(fftfreq(nt, dt))

# Initial Pulse
A0 = N / np.cosh(t / T0)
A = A0.copy().astype(np.complex128)
A_z = np.zeros((nz, nt), dtype=np.complex128)
A_z[0] = A

linear_half_step = np.exp(0.5j * beta2 * w**2 * dz)

for i in range(1, nz):
    A_freq = fft(A)
    A = ifft(A_freq * linear_half_step)

    I = np.abs(A)**2
    dI_dt = np.gradient(I, dt)
    A *= np.exp(1j * gamma * I * dz) * (1 + 1j * gamma * tau_shock * dI_dt * dz)

    A_freq = fft(A)
    A = ifft(A_freq * linear_half_step)

    A_z[i] = A

plt.figure(figsize=(10, 6))
plt.imshow(np.abs(A_z)**2, extent=[t[0], t[-1], z_max, 0],
           aspect='auto', cmap='inferno')
plt.colorbar(label=r'$|A(t,z)|^2$')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation Distance z')
plt.title(f'Z-Evolution: Soliton with Self-Steepening (N={N})')
plt.tight_layout()
plt.savefig("../figures/z_evol_self_steep.png", dpi=300)
plt.show()