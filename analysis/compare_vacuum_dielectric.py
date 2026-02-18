import numpy as np
import matplotlib.pyplot as plt

ez_vac = np.load("../data/ez_vac.npy")
ez_diel = np.load("../data/ez_diel.npy")

min_len = min(len(ez_vac), len(ez_diel))
ez_vac = ez_vac[:min_len]
ez_diel = ez_diel[:min_len]
time = np.arange(min_len)

plt.figure(figsize=(10, 4))
plt.plot(time, ez_vac, label='Vacuum (n=1.0)', linestyle='--')
plt.plot(time, ez_diel, label='Dielectric (n=1.5)', alpha=0.7)
plt.xlabel('Time step')
plt.ylabel('Ez field')
plt.title('Pulse Response at Probe: Vacuum vs Dielectric')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/pulse_response_vacuum_dielectric.png", dpi=300)
plt.show()

numerator = np.linalg.norm(ez_diel - ez_vac)
denominator = np.linalg.norm(ez_vac)
rel_L2 = numerator / denominator

print(f"Relative L² Error between Vacuum and Dielectric: {rel_L2:.4f}")