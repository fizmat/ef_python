import numpy as np
import matplotlib.pyplot as plt
import h5py
from ef.output.reader import Reader

SGSE_conv_unit_current_to_A = 3e10 * 0.1;     #from current units SGSE to A
SI_conv_cm_to_m = 0.01;
SI_conv_g_to_kg = 0.001
SI_conv_Fr_to_C = 3.3356409519815207e-10

filename = 'contour_0000050.h5'
with h5py.File( filename, mode="r") as h5:
    sim_PIC = Reader.read_simulation(h5)
    
particles_per_step = sim_PIC.particle_sources[0].particles_to_generate_each_step
current = np.abs(
    sim_PIC.particle_sources[0].particles_to_generate_each_step *
    sim_PIC.particle_sources[0].charge / 
    sim_PIC.time_grid.time_step_size
)
current = current / SGSE_conv_unit_current_to_A
print("source current = {} A".format(current))

mass = sim_PIC.particle_sources[0].mass * SI_conv_g_to_kg
charge = sim_PIC.particle_sources[0].charge * SI_conv_Fr_to_C
momentum_z = sim_PIC.particle_sources[0].mean_momentum[2] * SI_conv_g_to_kg * SI_conv_cm_to_m
print("mass = {} kg".format(mass))
print("charge = {} Coloumb".format(charge))
print("z-momentum = {} kg*m/s".format(momentum_z))
print("charge/mass = {:e} C/kg".format(charge/mass))

length_of_cathode = sim_PIC.particle_sources[0].shape.size[1] * SI_conv_cm_to_m
half_width = sim_PIC.particle_sources[0].shape.size[0] / 2 * SI_conv_cm_to_m
center_of_beam = sim_PIC.particle_sources[0].shape.origin[0] * SI_conv_cm_to_m + half_width
print("lenghth of cathode (in y) = {} m".format(length_of_cathode))
print("half width of cathode (in x) = {} m".format(half_width))
print("center of beam (in x) = {} m".format(center_of_beam))

start_z = sim_PIC.particle_sources[0].shape.origin[2] * SI_conv_cm_to_m 
end_z = sim_PIC.mesh.size[2] * SI_conv_cm_to_m
print("start of beam (in z) = {} m".format(start_z))
print("end of beam (in z) = {} m".format(end_z))

energy = (momentum_z * momentum_z) / (2 * mass)
voltage = energy / np.abs(charge)
print("voltage = {} V".format(voltage))    

current_dens = current / length_of_cathode 
print("current density (along y) = {} A/m".format(current_dens))

eps = 8.85e-12
p_const = 1 / (4*eps*(np.abs(2*charge/mass))**0.5) * current_dens / voltage**1.5 ;   
print("p constant = {}".format(p_const))

def contour(z_position, half_width, angle, p_const):
    y = half_width + np.tan(angle) * z_position + p_const / 2 * (z_position * z_position)         
    return y

filename = 'contour_bin_0000015.h5'
with h5py.File( filename, mode="r") as h5:
    sim_BIN = Reader.read_simulation(h5)
    
particles_per_step = sim_BIN.particle_sources[0].particles_to_generate_each_step
current = np.abs(
    sim_BIN.particle_sources[0].particles_to_generate_each_step *
    sim_BIN.particle_sources[0].charge / 
    sim_BIN.time_grid.time_step_size
)
current = current / SGSE_conv_unit_current_to_A
print("source current = {} A".format(current))

mass = sim_BIN.particle_sources[0].mass * SI_conv_g_to_kg
charge = sim_BIN.particle_sources[0].charge * SI_conv_Fr_to_C
momentum_z = sim_BIN.particle_sources[0].mean_momentum[2] * SI_conv_g_to_kg * SI_conv_cm_to_m
print("mass = {} kg".format(mass))
print("charge = {} Coloumb".format(charge))
print("z-momentum = {} kg*m/s".format(momentum_z))
print("charge/mass = {:e} C/kg".format(charge/mass))

length_of_cathode = sim_BIN.particle_sources[0].shape.size[1] * SI_conv_cm_to_m
half_width = sim_BIN.particle_sources[0].shape.size[0] / 2 * SI_conv_cm_to_m
center_of_beam = sim_BIN.particle_sources[0].shape.origin[0] * SI_conv_cm_to_m + half_width
print("lenghth of cathode (in y) = {} m".format(length_of_cathode))
print("half width of cathode (in x) = {} m".format(half_width))
print("center of beam (in x) = {} m".format(center_of_beam))

start_z = sim_BIN.particle_sources[0].shape.origin[2] * SI_conv_cm_to_m 
end_z = sim_BIN.mesh.size[2] * SI_conv_cm_to_m
print("start of beam (in z) = {} m".format(start_z))
print("end of beam (in z) = {} m".format(end_z))

energy = (momentum_z * momentum_z) / (2 * mass)
voltage = energy / np.abs(charge)
print("voltage = {} V".format(voltage))   

current_dens = current / length_of_cathode 
print("current density (along y) = {} A/m".format(current_dens))

eps = 8.85e-12
p_const = 1 / (4*eps*(np.abs(2*charge/mass))**0.5) * current_dens / voltage**1.5 ;   
print("p constant = {}".format(p_const))

def contour(z_position, half_width, angle, p_const):
    y = half_width + np.tan(angle) * z_position + p_const / 2 * (z_position * z_position)         
    return y

conv_grad_to_rad = np.pi/180
angle = 0 * conv_grad_to_rad          #angle of beam
steps_z = 100000
position_z = np.arange(start_z,end_z,(end_z-start_z)/steps_z) 
# points in z direction, from 0 to 0.01 m with step 0,00001 m 

contour_y = contour(position_z, half_width, angle, p_const)
# countour calculation, m
plt.figure(figsize=(5,5), dpi = (100))
plt.xlabel("Z position, [mm]")
plt.ylabel("X position, [mm]")

plt.plot(sim_PIC.particle_arrays[0].positions[..., 2]*SI_conv_cm_to_m*1000,
         ((sim_PIC.particle_arrays[0].positions[..., 0]*SI_conv_cm_to_m - center_of_beam)*1000),
             '.',label="PIC calculated_points") #plot particles

plt.plot(sim_BIN.particle_arrays[0].positions[..., 2]*SI_conv_cm_to_m*1000,
         ((sim_BIN.particle_arrays[0].positions[..., 0]*SI_conv_cm_to_m - center_of_beam)*1000),
             'o',label="BIN calculated_points") #plot particles

plt.plot(position_z*1000, contour_y*1000, color = 'g', lw = 3, label="analytic_curve") # plot countour in cm and move to left z of beam and top x of beam neat cathode
plt.plot(position_z*1000, -1 * contour_y*1000, color = 'g', lw = 3)
plt.legend(bbox_to_anchor=(0.32, 1), loc=1, borderaxespad=0.)
plt.xlim(0,30)
plt.ylim(-7.5,7.5)
plt.savefig('countour_beam.png')        # save png picture
h5.close() 