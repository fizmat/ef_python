from matplotlib.pyplot import *
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from matplotlib import colors
import pandas as pd

sgse_to_volts = 300


def get_mesh_parameters(h5file):
    x_coord = h5file["/SpatialMesh/node_coordinates_x"][:]
    y_coord = h5file["/SpatialMesh/node_coordinates_y"][:]
    z_coord = h5file["/SpatialMesh/node_coordinates_z"][:]
    den=h5file["/SpatialMesh/charge_density"][:]
    pot=h5file["/SpatialMesh/potential"][:]*sgse_to_volts
    return [x_coord, y_coord, z_coord, pot, den]

filename = 'tsii_0**.h5'
#filename = 'examples/axially_symmetric_beam_contour/contour_jup_0000100.h5'
#filename = 'tsii_0473210.h5'

for f in sorted(glob.glob(filename))[:]:

    # fig, axs = plt.subplots(1,2,gridspec_kw={'width_ratios': [1, 3]})
    # (ax1,ax2) = axs
    h5 = h5py.File( f, mode="r")
    t = h5["TimeGrid/"].attrs['current_time'][0]    
    # fig.suptitle('Potential [V] and Density;     Time step='+str("{0:.3g}".format(t))+'s')
    # fig.set_size_inches(24, 12)
    pot = np.transpose(get_mesh_parameters(h5))
    df = pd.DataFrame(pot, columns=list('xyzpd'))
    df.plot('x')
# #    pot[:,3] = pot[:,3]  - pot_0[:,3]
#     pot = pot[np.where(pot[:, 2] == 1.2)]
#     x_nodes = h5["/SpatialMesh"].attrs['x_n_nodes'][0]
#     y_nodes = h5["/SpatialMesh"].attrs['y_n_nodes'][0]
#     data = np.reshape(pot[:,3], (x_nodes,y_nodes))
#     im1=ax1.imshow(data, aspect='auto', cmap='RdBu')#,clim=(-100,0))
#     print('max potential = ', max(pot[:,3]))
#     pot = np.transpose(get_mesh_parameters(h5))
# #    pot[:,3] = pot[:,3]  - pot_0[:,3]
#     pot = pot[np.where(pot[:, 1] == 0.2)]
#     x_nodes = h5["/SpatialMesh"].attrs['x_n_nodes'][0]
#     y_nodes = h5["/SpatialMesh"].attrs['z_n_nodes'][0]
#     data_2 = np.reshape(pot[:,3], (x_nodes,y_nodes))
#     im2=ax2.imshow(data_2, aspect='auto', cmap='RdBu')#,clim=(-100,0))
#     ax2.set_xlim(0,6)
#     ax1.set_xticks([])
#     ax2.set_yticks([])
#     ax2.set_xticks([])
#
#
#
#     ax1.set_xlabel('x_nodes')
#     ax2.set_xlabel('z_nodes')
#     ax1.set_ylabel('y_nodes')
#     fig.colorbar(im2,ax=ax2)
#     fig.colorbar(im1,ax=ax1)
#
#     fig.savefig('tet3_' + str(f[-9:-3]) + 'png')
#     plt.close()
    h5.close()

