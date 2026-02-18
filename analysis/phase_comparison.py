import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, detrend
from scipy.interpolate import interp1d

# === Load SSFM and MEEP data ===
A_ssfm = np.load("../data/A_output.npy")       # Complex SSFM output
t_ssfm = np.load("../data/t_ssfm.npy")         # Time axis for SSFM
ez_xt = np.load("../data/ez_xt_meep.npy")      # MEEP Ez(x,t) field
t_meep = np.load("../data/t_xt.npy")           # Time axis from MEEP

# === Extract Ez(t) at center of MEEP domain (x = center) ===
mid_index = ez_xt.shape[1] // 2
ez_t = ez_xt[:, mid_index]
A_meep = ez_t

# === Compute analytic signals ===
analytic_ssfm = hilbert(np.real(A_ssfm))
analytic_meep = hilbert(ez_t - np.mean(ez_t))

# === Extract phase and detrend ===
phase_ssfm = np.unwrap(np.angle(analytic_ssfm))
phase_ssfm -= np.mean(phase_ssfm)

phase_meep = np.unwrap(np.angle(analytic_meep))
phase_meep_detrended = detrend(phase_meep)

# === Interpolate MEEP phase to SSFM time grid ===
interp_phase_meep = interp1d(
    t_meep,
    phase_meep_detrended,
    kind='linear',
    bounds_error=False,
    fill_value='extrapolate'
)

phase_meep_interp = interp_phase_meep(t_ssfm)
phase_meep_interp -= np.mean(phase_meep_interp)

# === Relative Phase Error ===
rel_phase_error = np.linalg.norm(
    phase_ssfm - phase_meep_interp
) / np.linalg.norm(phase_meep_interp)

print(f"Relative Phase L² Error: {rel_phase_error:.4f}")

# === Visualization ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("SSFM vs MEEP Phase Comparison", fontsize=14)

axs[0, 0].plot(t_ssfm, np.real(A_ssfm))
axs[0, 0].set_title("SSFM: Real Part")

axs[0, 1].plot(t_meep, np.real(A_meep))
axs[0, 1].set_title("MEEP: Real Part")

axs[1, 0].plot(t_ssfm, np.abs(analytic_ssfm))
axs[1, 0].set_title("SSFM: Envelope")

axs[1, 1].plot(t_meep, np.abs(analytic_meep))
axs[1, 1].set_title("MEEP: Envelope")

for ax in axs.flat:
    ax.grid(True)

plt.tight_layout()
plt.savefig("../figures/phase_comparison.png", dpi=300)
plt.show()