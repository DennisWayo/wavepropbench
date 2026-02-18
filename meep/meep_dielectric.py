import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

resolution = 100
dpml = 2
sx = 20
cell = mp.Vector3(sx)

n = 1.5
geometry = [mp.Block(size=mp.Vector3(mp.inf),
                     material=mp.Medium(index=n))]

lambda_c = 1.55
f_c = 1 / lambda_c
tau_ps = 1.0
tau_um = tau_ps * 0.3
width = tau_um

sources = [mp.Source(mp.GaussianSource(frequency=f_c,
                                       fwidth=1/width),
                     component=mp.Ez,
                     center=mp.Vector3(-0.4*sx))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=[mp.PML(dpml)],
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

probe_loc = mp.Vector3(0.3*sx)
ez_data = []

def get_field(sim):
    ez_data.append(sim.get_field_point(mp.Ez, probe_loc))

sim.run(mp.at_every(0.1, get_field), until=100)

ez_data = np.array(ez_data)
np.save("../data/ez_diel.npy", ez_data)

plt.figure()
plt.plot(ez_data)
plt.tight_layout()
plt.savefig("../figures/meep_dielectric.png", dpi=300)
plt.show()