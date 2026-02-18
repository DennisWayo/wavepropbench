import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os

os.makedirs("../data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

resolution = 100
sx = 20
dpml = 1.0
cell = mp.Vector3(sx)
pml_layers = [mp.PML(dpml)]
eps = 2.25

fcen = 0.3
df = 0.2

src = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                 component=mp.Ez,
                 center=mp.Vector3(-5))]

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    boundary_layers=pml_layers,
    resolution=resolution,
    default_material=mp.Medium(epsilon=eps),
    sources=src,
)

ez_xt = []
monitor_center = mp.Vector3()
monitor_size = mp.Vector3(sx)

def store_ez(sim):
    ez_line = sim.get_array(center=monitor_center,
                            size=monitor_size,
                            component=mp.Ez)
    ez_xt.append(ez_line)

sim.run(mp.at_every(1, store_ez), until=200)

ez_xt = np.array(ez_xt)
x_points = ez_xt.shape[1]
x_coords = np.linspace(-sx/2, sx/2, x_points)
dt = sim.fields.dt
t_array = np.arange(ez_xt.shape[0]) * dt

np.save("../data/ez_xt_meep.npy", ez_xt)
np.save("../data/x_coords.npy", x_coords)
np.save("../data/t_xt.npy", t_array)

analytic = hilbert(ez_xt, axis=0)
envelope = np.abs(analytic)
np.save("../data/envelope_xt.npy", envelope)

plt.figure(figsize=(8,4))
plt.imshow(ez_xt, extent=[x_coords[0], x_coords[-1],
                          t_array[-1], t_array[0]],
           aspect='auto', cmap='RdBu')
plt.colorbar(label='$E_z$')
plt.tight_layout()
plt.savefig("../figures/meep_xt_field.png", dpi=300)
plt.show()