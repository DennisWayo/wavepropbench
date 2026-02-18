import numpy as np
import matplotlib.pyplot as plt
from ssfm.core import ssfm_propagate

# -------------------------------
# Grid
# -------------------------------
nt = 2048
t_max = 100
dt = 2 * t_max / nt
t = np.linspace(-t_max, t_max, nt)

# -------------------------------
# Input Pulse
# -------------------------------
t0 = 10
A0 = np.exp(-t**2 / (2 * t0**2))

# -------------------------------
# NLSE Parameters
# -------------------------------
beta2 = -1.5
gamma = 0.5
L = 20
nz = 100
dz = L / nz

# -------------------------------
# Propagation
# -------------------------------
A = ssfm_propagate(A0, beta2, gamma, dz, nz, dt)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8, 4))
plt.plot(t * 1e12, np.abs(A0)**2, '--', label='Input |A₀|²')
plt.plot(t * 1e12, np.abs(A)**2, label='Output |A|²')
plt.title("SSFM Pulse Propagation")
plt.xlabel("Time (ps)")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/SSFM_Pulse_.png", dpi=300)
plt.show()

# -------------------------------
# Save
# -------------------------------
np.save("../data/A_input.npy", A0)
np.save("../data/A_output.npy", A)
np.save("../data/t_ssfm.npy", t)

print("Linear SSFM case completed.")