import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

# === Load Data ===
t_ssfm = np.load("t_ssfm.npy")
A_output = np.load("A_output.npy")
envelope_xt = np.load("envelope_xt.npy")

# === Extract center slice ===
meep_slice = envelope_xt[envelope_xt.shape[0] // 2]
meep_slice_smooth = gaussian_filter1d(meep_slice, sigma=10)

t_meep = t_ssfm[:meep_slice.shape[0]]

analytic_ssfm = A_output[:meep_slice.shape[0]]
analytic_meep = meep_slice_smooth + 0j

# === Metric Functions ===
def compute_fwhm(t, envelope):
    env = np.abs(envelope)
    half_max = np.max(env) / 2
    indices = np.where(env >= half_max)[0]
    return t[indices[-1]] - t[indices[0]]

def compute_rms_width(t, envelope):
    envelope = np.abs(envelope)
    envelope /= trapezoid(envelope, t)
    t_mean = trapezoid(t * envelope, t)
    t2_mean = trapezoid((t**2) * envelope, t)
    return np.sqrt(t2_mean - t_mean**2)

def plot_with_gaussian(t, envelope, label, color):
    envelope = np.abs(envelope)
    envelope /= np.max(envelope)
    rms = compute_rms_width(t, envelope)
    t_mean = trapezoid(t * envelope, t)
    gaussian = norm.pdf(t, loc=t_mean, scale=rms)
    gaussian /= np.max(gaussian)
    plt.plot(t, envelope, label=f"{label} Envelope", color=color)
    plt.plot(t, gaussian, '--', color=color, alpha=0.6)

# === Compute Metrics ===
fwhm_ssfm = compute_fwhm(t_ssfm[:2000], analytic_ssfm)
rms_ssfm = compute_rms_width(t_ssfm[:2000], analytic_ssfm)
fwhm_meep = compute_fwhm(t_meep, analytic_meep)
rms_meep = compute_rms_width(t_meep, analytic_meep)
rms_error = abs(rms_ssfm - rms_meep) / rms_meep

print("Temporal Broadening Metrics:")
print(f"SSFM  - FWHM: {fwhm_ssfm:.2f} fs,  RMS: {rms_ssfm:.2f} fs")
print(f"MEEP  - FWHM: {fwhm_meep:.2f} fs,  RMS: {rms_meep:.2f} fs")
print(f"Relative RMS Width Error: {rms_error:.4f}")

# === Plot ===
plt.figure(figsize=(10, 5))
plot_with_gaussian(t_ssfm[:2000], analytic_ssfm, "SSFM", 'tab:blue')
plot_with_gaussian(t_meep, analytic_meep, "MEEP", 'tab:orange')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/temporal_metrics.png", dpi=300)
plt.show()