import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

resolution = 100
cell = mp.Vector3(20)
pml_layers = [mp.PML(1.0)]

src = [mp.Source(mp.GaussianSource(0.15, fwidth=0.1),
                 component=mp.Ez,
                 center=mp.Vector3(-7))]

probe_pt = mp.Vector3(5)
probe_data = []

def save_probe(sim):
    probe_data.append(sim.get_field_point(mp.Ez, probe_pt))

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    sources=src,
                    resolution=resolution,
                    default_material=mp.Medium(epsilon=1.0))

sim.run(mp.at_every(0.5, save_probe), until=100)

probe_data = np.array(probe_data)
np.save("../data/ez_vac.npy", probe_data)

plt.figure()
plt.plot(probe_data)
plt.tight_layout()
plt.savefig("../figures/meep_vacuum.png", dpi=300)
plt.show()