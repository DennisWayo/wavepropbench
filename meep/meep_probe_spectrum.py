import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

resolution = 100
cell_size = mp.Vector3(20)
default_material = mp.Medium(epsilon=2.25)

fcen = 0.3
df = 0.2

src = mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                center=mp.Vector3(-5),
                component=mp.Ez)

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=[mp.PML(1.0)],
    geometry=[],
    default_material=default_material,
    sources=[src],
    resolution=resolution,
)

probe_location = mp.Vector3(5.0)
ez_data = []

def save_field(sim):
    ez = sim.get_array(center=probe_location,
                       size=mp.Vector3(),
                       component=mp.Ez)
    ez_data.append(ez)

sim.run(mp.at_every(1, save_field), until=200)

Ez_arr = np.array(ez_data)
np.save("../data/ez_probe.npy", Ez_arr)

plt.figure()
plt.plot(Ez_arr)
plt.tight_layout()
plt.savefig("../figures/meep_probe_time.png", dpi=300)
plt.show()

spectrum = np.abs(np.fft.fftshift(np.fft.fft(Ez_arr)))**2
freqs = np.fft.fftshift(np.fft.fftfreq(len(Ez_arr), d=1))

plt.figure()
plt.plot(freqs, spectrum/np.max(spectrum))
plt.tight_layout()
plt.savefig("../figures/meep_probe_spectrum.png", dpi=300)
plt.show()