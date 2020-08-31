import pandas as pd
import  glob
import h5py
import matplotlib.pyplot as plt 

def main():
    num = extract_num_trajectory_from_out_files()
    an = eval_an_trajectory_at_num_time_points( num )
    plot_trajectories( num , an )

def extract_num_trajectory_from_out_files():
    out_files = glob.glob("single_particle_free_space_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].h5")

    num_trajectory = []
    for f in out_files:
        num_trajectory.append( extract_time_pos_mom( f ) )    

    num_trajectory = sorted([ x for x in num_trajectory if x ], key = lambda x: x[0])
    return pd.DataFrame.from_records(num_trajectory, columns=('t', 'x', 'y', 'z', 'px', 'py', 'pz'))
    
def extract_time_pos_mom( h5file ):
    h5 = h5py.File( h5file, mode="r")
    t = h5["/TimeGrid"].attrs["current_time"][0]
    t_pos_mom = ()
    if ( len(h5["/ParticleSources/emit_single_particle/particle_id"]) > 0 ):
        x = h5["/ParticleSources/emit_single_particle/position_x"][0]
        y = h5["/ParticleSources/emit_single_particle/position_y"][0]
        z = h5["/ParticleSources/emit_single_particle/position_z"][0]
        px = h5["/ParticleSources/emit_single_particle/momentum_x"][0]
        py = h5["/ParticleSources/emit_single_particle/momentum_y"][0]
        pz = h5["/ParticleSources/emit_single_particle/momentum_z"][0]
        t_pos_mom = (t, x, y, z, px, py, pz)
    h5.close()
    return( t_pos_mom )

def eval_an_trajectory_at_num_time_points( num_trajectory ):
    global particle_mass
    particle_mass, pos0, mom0 =  get_mass_and_initial_pos_and_mom()

    an_trajectory = []
    for t in num_trajectory.t:
        pos = coords( particle_mass, t, pos0, mom0 )
        mom = momenta( t, mom0 )
        an_trajectory.append([t] + list(pos) + list(mom))

    return pd.DataFrame.from_records(an_trajectory, columns=('t', 'x', 'y', 'z', 'px', 'py', 'pz'))

def get_mass_and_initial_pos_and_mom():
    initial_out_file = "single_particle_free_space_0000000.h5"
    h5 = h5py.File( initial_out_file, mode="r")
    m = h5["/ParticleSources/emit_single_particle"].attrs["mass"][0]
    x0 = h5["/ParticleSources/emit_single_particle/position_x"][0]
    y0 = h5["/ParticleSources/emit_single_particle/position_y"][0]
    z0 = h5["/ParticleSources/emit_single_particle/position_z"][0]
    px0 = h5["/ParticleSources/emit_single_particle/momentum_x"][0]
    py0 = h5["/ParticleSources/emit_single_particle/momentum_y"][0]
    pz0 = h5["/ParticleSources/emit_single_particle/momentum_z"][0]
    h5.close()
    return( m, [x0, y0, z0], [px0, py0, pz0] )


def momenta( t, mom0 ):    
    return mom0

def coords( m, t, pos0, mom0 ):
    pos = pos0 + mom0 / m * t
    return pos

def plot_trajectories( num , an ):
    plot_3d( num, an )
    plot_2d( num, an )
    plot_kin_en( num , an )

def plot_3d( num, an ):
    fig = plt.figure()
    ax = fig.gca( projection='3d' )
    ax.plot( num.x, num.z, num.y, 'b.', markersize = 12, label = "Num" )
    ax.plot( an.x, an.z, an.y,  'g-', linewidth = 3, label = "An" )
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Z [cm]')
    ax.set_zlabel('Y [cm]')
    plt.legend( loc = 'upper left', title="3d" )
    print( 'Saving 3d trajectory plot to "3d.png"' )
    plt.savefig('3d.png')
    plt.show()

def plot_2d( num, an ):
    plt.figure( figsize=( 16, 6 ) )
    plt.subplots_adjust( left = None, bottom = None,
                         right = None, top = None,
                         wspace = 0.4, hspace = None )
    #XY
    ax = plt.subplot(131)
    plt.plot( num.x, num.y,
              linestyle='', marker='o',
              label = "Num" )
    plt.plot( an.x, an.y,
              linestyle='-', marker='', lw = 2,
              label = "An" )
    ax.set_xlabel('X [cm]') 
    ax.set_ylabel('Y [cm]') 
    plt.legend( loc = 'upper left', title="XY" )
    #ZX
    ax = plt.subplot(132)
    plt.plot( num.z, num.x,
        linestyle='', marker='o',
        label = "Num" )
    plt.plot( an.z, an.x,
              linestyle='-', marker='', lw = 2,
              label = "An" )
    ax.set_xlabel('Z [cm]') 
    ax.set_ylabel('X [cm]') 
    plt.legend( loc = 'upper left', title="ZX" )
    #ZY
    ax = plt.subplot(133)
    plt.plot( num.z, num.y,
        linestyle='', marker='o',
        label = "Num" )
    plt.plot( an.z, an.y,
              linestyle='-', marker='', lw = 2,
              label = "An" )
    ax.set_xlabel('Z [cm]') 
    ax.set_ylabel('Y [cm]') 
    plt.legend( loc = 'upper left', title="ZY" )
    print( 'Saving 2d trajectory projection plots to "2d.png"' )
    plt.savefig('2d.png')
    plt.show()
    
def plot_kin_en( num , an ):
    global particle_mass
    E_num = (num.px**2 + num.py**2 + num.pz**2) / ( 2 * particle_mass )
    E_an = (an.px**2 + an.py**2 + an.pz**2) / ( 2 * particle_mass )
    t = num.t
    plt.figure()
    axes = plt.gca()
    axes.set_xlabel( "t [s]" )
    axes.set_ylabel( "E [erg]" )
    # axes.set_ylim( [min( E_an.min(), E_num.min() ),
    #                 max( E_an.max(), E_num.max() ) ] )
    line, = plt.plot( t, E_num, 'o' )
    line.set_label( "Num" )
    line, = plt.plot( t, E_an, ls = 'solid', lw = 3 )
    line.set_label( "An" )
    plt.legend( loc = 'upper right' )
    print( 'Saving kinetic energy comparison plot to "kin_en.png"' )
    plt.savefig('kin_en.png')
    plt.show()


main()