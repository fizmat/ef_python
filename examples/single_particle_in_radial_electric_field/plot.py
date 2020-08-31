import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from math import pi, sqrt

pos_x = 3.0
pos_y = 2.5
charge = -4.8e-10
mass = 9.8e-28
momentum_y = 3.92e-19
field_x = lambda x, y: 8 * (x - 2.5) / ( (x - 2.5)**2 + (y - 2.5)**2 )**(3/2)
field_y = lambda x, y: 8 * (y - 2.5) / ( (x - 2.5)**2 + (y - 2.5)**2 )**(3/2)
pot_en = lambda x, y: 8 * charge * ((x - 2.5)**2 + (y - 2.5)**2) ** (-1/2)
orbit_radius = pos_x - 2.5
force = abs(charge*field_x(pos_x, pos_y))
v = sqrt(force/mass * orbit_radius)
good_momentum = v*mass

fname = "single_particle_in_radial_el_field_history.h5"
with h5py.File(fname, mode="r") as h5:
    h = h5['/history']
    t = np.array(h['time'])
    p = np.array(h['particles/position'][0])
    m = np.array(h['particles/momentum'][0])
    M = float(h['particles/mass'][0])
    c = float(h['particles/charge'][0])
df = pd.DataFrame.from_dict({'time': t, 
                             'x': p[:, 0], 'y': p[:, 1], 'z': p[:, 2],
                             'px': m[:, 0], 'py': m[:, 1], 'pz': m[:, 2]})

p0 = p[0]
v0 = m[0]/M
r = 0.5
mom = m[0][1]
period = 2*pi*r/v0[1]
df['an_x'] = 2.5 + r * np.cos(2 * pi * t / period) 
df['an_y'] = 2.5 + r * np.sin(2 * pi * t / period) 
df['an_z'] = p0[2] + v0[2] * t
df['an_px'] = - mom * np.sin(2 * pi * t / period) 
df['an_py'] = mom * np.cos(2 * pi * t / period)
df['an_pz'] = m[0][2]
df['kinetic'] = (df.px**2 + df.py**2 + df.pz**2) / ( 2 * M )
df['an_kinetic'] = (df.an_px**2 + df.an_py**2 + df.an_pz**2) / ( 2 * M )
df['potential'] = df.apply(lambda row: pot_en(row['x'], row['y']), axis=1)
df['an_potential'] = df.apply(lambda row: pot_en(row['an_x'], row['an_y']), axis=1)
df['E'] = df.kinetic + df.potential
df['an_E'] = df.an_kinetic + df.an_potential
df.head()

def plot_3d(df):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(df.x, df.z, df.y, 'b.', markersize = 12, label = "Num")
    ax.plot(df.an_x, df.an_z, df.an_y,  'g-', linewidth = 3, label = "An")
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Z [cm]')
    ax.set_zlabel('Y [cm]')
    plt.legend(loc = 'upper left', title="3d")
    plt.show()
    plt.savefig('3d.png')

plot_3d(df)

ax = df.plot.scatter(x='x', y='y', label="Num")
ax.legend(title="XY")
df.plot('an_x', 'an_y', ax=ax, label="An", color='red')
ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
fig = ax.get_figure()
fig.savefig("XY.png")


ax = df.plot.scatter(x='z', y='x', label="Num")
ax.legend(title="ZX")
df.plot('an_z', 'an_x', ax=ax, label="An", color='red')
ax.set_xlabel('Z [cm]')
ax.set_ylabel('X [cm]')
fig = ax.get_figure()
fig.savefig("ZX.png")

ax = df.plot.scatter(x='z', y='y', label="Num")
ax.legend(title="ZY")
df.plot('an_z', 'an_y', ax=ax, label="An", color='red')
ax.set_xlabel('Z [cm]')
ax.set_ylabel('Y [cm]')
fig = ax.get_figure()
fig.savefig("ZY.png")

ax = df.plot('time', ['E', 'an_E'], label=["Num", "An"])
ax.set_xlabel('time [s]')
ax.set_ylabel('E [erg]')
fig = ax.get_figure()
fig.savefig("E.png")