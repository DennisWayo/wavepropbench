import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# === Load Data ===
envelope_xt = np.load("../data/envelope_xt.npy")
t_ssfm = np.load("../data/t_ssfm.npy")

meep_slice = envelope_xt[envelope_xt.shape[0] // 2]
t_meep = t_ssfm[:meep_slice.shape[0]]

# --- Peak Detection ---
peaks, props = find_peaks(
    meep_slice,
    height=0.3 * np.max(meep_slice),
    distance=20
)

peak_times = t_meep[peaks]
peak_amps = meep_slice[peaks]

plt.figure(figsize=(8, 4))
plt.plot(t_meep, meep_slice, label="MEEP Envelope")
plt.plot(peak_times, peak_amps, "rx", label=f"Peaks: {len(peaks)}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/peak_detection.png", dpi=300)
plt.show()

# --- Interval Statistics ---
if len(peak_times) >= 2:
    intervals = np.diff(peak_times)
    print(f"Average Peak-to-Peak Interval: {np.mean(intervals):.2f} fs")

# --- Spectral Centroid ---
dt = t_meep[1] - t_meep[0]
N = len(meep_slice)

frequencies = fftfreq(N, d=dt)
spectrum = np.abs(fft(meep_slice))**2
spectrum /= np.sum(spectrum)

mask = frequencies > 0
frequencies = frequencies[mask]
spectrum = spectrum[mask]

centroid = np.sum(frequencies * spectrum)
print(f"Spectral Centroid (Positive Only): {centroid:.4f}")