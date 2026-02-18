import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

meep_env = np.load("../data/envelope_xt.npy")
ssfm_out = np.load("../data/A_output.npy")
x_meep = np.load("../data/x_coords.npy")
t_meep = np.load("../data/t_xt.npy")
t_ssfm = np.load("../data/t_ssfm.npy")

idx_probe = len(x_meep) // 2
meep_line = meep_env[:, idx_probe]

min_len = min(len(meep_line), len(ssfm_out))
meep_trimmed = meep_line[:min_len]
ssfm_trimmed = np.abs(ssfm_out[:min_len])

meep_norm = meep_trimmed / np.max(np.abs(meep_trimmed))
ssfm_norm = ssfm_trimmed / np.max(np.abs(ssfm_trimmed))

corr = correlate(meep_norm, ssfm_norm, mode='full')
shift_idx = np.argmax(corr) - len(ssfm_norm) + 1
ssfm_aligned = np.roll(ssfm_norm, shift_idx)

rel_L2_shifted = np.linalg.norm(meep_norm - ssfm_aligned) / np.linalg.norm(ssfm_aligned)
print(f"Relative L² error after alignment: {rel_L2_shifted:.4f}")

plt.figure(figsize=(8, 4))
plt.semilogy(meep_norm, label='MEEP Envelope')
plt.semilogy(ssfm_aligned, label='SSFM Output (Aligned)')
plt.title("Log-Scale Amplitude Comparison (Aligned)")
plt.xlabel("Time Index")
plt.ylabel("Normalized Amplitude (log scale)")
plt.legend()
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig("../figures/log_scale_amplitude_comparison.png", dpi=500)
plt.show()