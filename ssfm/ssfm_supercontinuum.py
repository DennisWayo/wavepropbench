import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft
from scipy.signal import fftconvolve

gamma = 1.0
fR = 0.18
dz = 0.05
z_max = 5.0
n_steps = int(z_max / dz)

t = np.linspace(-10, 10, 2048)
dt = t[1] - t[0]
A = np.exp(-t**2).astype(np.complex128)

h_R = (1 - 2 * t**2) * np.exp(-t**2)
h_R /= np.trapz(h_R, t)

evolution = [A.copy()]
max_phase = np.pi

for step in range(n_steps):
    I = np.abs(A)**2
    Raman = gamma * fR * fftconvolve(I, h_R, mode='same') * A
    NL = gamma * (1 - fR) * I * A + Raman

    NL_mag = np.abs(NL * dz)
    NL[NL_mag > max_phase] *= max_phase / NL_mag[NL_mag > max_phase]

    A *= np.exp(1j * NL * dz)
    evolution.append(A.copy())

evolution = np.array(evolution)

plt.figure(figsize=(10, 5))
plt.imshow(np.abs(evolution)**2, extent=[t[0], t[-1], 0, z_max],
           aspect='auto', cmap='inferno')
plt.xlabel("Time (ps)")
plt.ylabel("Propagation Distance z")
plt.title("Supercontinuum Evolution in Time Domain")
plt.colorbar(label=r"$|A(t, z)|^2$")
plt.tight_layout()
plt.savefig("../figures/supercontinuum_evolution.png", dpi=300)
plt.show()

spectrum = np.abs(fftshift(fft(evolution[-1])))**2
spectrum /= np.max(spectrum)
freq = fftshift(np.fft.fftfreq(len(t), d=dt))

plt.figure(figsize=(8, 4))
plt.plot(freq, spectrum)
plt.xlabel("Frequency Offset Δf (THz)")
plt.ylabel("Normalized Spectral Intensity")
plt.title("Supercontinuum Output Spectrum")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/supercontinuum_output.png", dpi=300)
plt.show()