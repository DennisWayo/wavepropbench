import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

ez_vac = np.load("../data/ez_vac.npy")
ez_diel = np.load("../data/ez_diel.npy")

min_len = min(len(ez_vac), len(ez_diel))
ez_vac = ez_vac[:min_len]
ez_diel = ez_diel[:min_len]

dt = 1.0

Ez_vac_fft = fftshift(fft(ez_vac))
Ez_diel_fft = fftshift(fft(ez_diel))
freq = fftshift(fftfreq(min_len, d=dt))

Ez_vac_fft_norm = np.abs(Ez_vac_fft) / np.max(np.abs(Ez_vac_fft))
Ez_diel_fft_norm = np.abs(Ez_diel_fft) / np.max(np.abs(Ez_diel_fft))

plt.figure(figsize=(10, 4))
plt.plot(freq, Ez_vac_fft_norm, '--', label='Vacuum (n=1.0)')
plt.plot(freq, Ez_diel_fft_norm, alpha=0.8, label='Dielectric (n=1.5)')
plt.title('FFT Spectrum Comparison at Probe')
plt.xlabel('Frequency Offset Δf (THz)')
plt.ylabel('Normalized |FFT(Ez)|')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../figures/fft_spectrum_comparison.png", dpi=300)
plt.show()