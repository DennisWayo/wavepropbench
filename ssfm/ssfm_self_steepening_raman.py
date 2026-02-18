import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import fftconvolve

t_max = 20
nt = 2048
dt = t_max / nt
t = np.linspace(-t_max/2, t_max/2, nt)
w = fftshift(fftfreq(nt, dt)) * 2 * np.pi

dz = 0.0005
nz = 8000
z = np.linspace(0, dz*nz, nz)

beta2 = -1
gamma = 1
tau_shock = 0.05
fR = 0.3

def raman_response(t):
    tau1 = 0.0122
    tau2 = 0.032
    h = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-t / tau2) * np.sin(t / tau1)
    return np.where(t > 0, h, 0.0)

h_R = raman_response(t)
h_R /= np.trapz(h_R, t)

N_order = 5
A0 = N_order / np.cosh(t)

A_z = np.zeros((nz, nt), dtype=np.complex128)
A = A0.copy().astype(np.complex128)
A_z[0, :] = A

dispersion_operator = np.exp(0.5j * beta2 * w**2 * dz)

for i in range(1, nz):
    A_freq = fft(A)
    A_freq *= dispersion_operator
    A = ifft(A_freq)

    I = np.abs(A)**2
    dI_dt = np.gradient(I, dt)
    conv = fftconvolve(I, h_R, mode='same') * dt

    Raman_term = gamma * fR * conv * A
    Kerr_term = gamma * (1 - fR) * I * A
    shock_term = 1j * gamma * tau_shock * dI_dt * A

    A += dz * 1j * (Kerr_term + Raman_term + shock_term)

    A_freq = fft(A)
    A_freq *= dispersion_operator
    A = ifft(A_freq)

    A_z[i, :] = A

intensity = np.abs(A_z)**2
intensity_clean = np.nan_to_num(intensity)

plt.figure(figsize=(10, 6))
plt.imshow(intensity_clean, extent=[t.min(), t.max(), z[-1], z[0]],
           aspect='auto', cmap='inferno')
plt.colorbar(label=r'$|A(t, z)|^2$')
plt.xlabel('Time (ps)')
plt.ylabel('Propagation Distance z')
plt.title(f'Z-Evolution: Soliton Fission with Self-Steepening and Raman (N={N_order})')
plt.tight_layout()
plt.savefig("../figures/z_evol_self_steep_raman.png", dpi=300)
plt.show()