import HyperDrive

input = {}

######## USER INPUT ########
# 1 = normal 
# 2 = oblique -> Ignore calculations for heat flux, the script will output conditions at each shock
# 3 = flight sim
__RUN__ = 2

if __RUN__ == 1:
    ''' For part 1 & 2'''
    input['Run 3D Sim']     = False                     # True/False
    input['Calc Aero']      = True                      # True/False
    input['Use Mach']       = True                      # True/False
    input['Coordinate File'] = ["Nose_cone_geometry.txt"]

    input['Nose Radius']    = 0.0348                  # meters
    input['Mach']           =[7.0, 10.0]               # non-dim.
    input['Altitude']       =[12.2, 18.28]             # km
    input['Aoa']            = list(range(0,11))         # deg
    # For only normal shock
    HyperDrive.run_point_design(
        input=input,
        calc_only_normal_shock=True,
        show_output=True,
        save_output=False,
        show_plot=True,
        save_plot=True
    )

if __RUN__ == 2:
    # Including Oblique shock
    input['Run 3D Sim']     = False                     # True/False
    input['Calc Aero']      = True                      # True/False
    input['Use Mach']       = True                      # True/False
    input['Coordinate File'] = ["Nose_cone_geometry.txt"]

    input['Nose Radius']    = 0.0348                  # meters
    input['Mach']           = [10.0]               # non-dim.
    input['Altitude']       = [18.28]             # km
    input['Aoa']            = list(range(0,11))         # deg
    print()
    print("************************************************************************************************")
    print("                                 With oblique shocks:                                           ")
    print("************************************************************************************************")
    HyperDrive.run_point_design(
        input=input,
        calc_only_normal_shock=False,
        show_output=True,
        save_output=False,
        show_plot=True,
        save_plot=False)

if __RUN__ == 3:
    print()
    print("************************************************************************************************")
    print("                                   Flight Simulation:                                           ")
    print("************************************************************************************************")
    input = {}                                  # reset 
    input['Coordinate File'] = ["Nose_cone_geometry.txt"]
    input['Run 3D Sim']     = True              #  True/False
    input['Calc Aero']      = True             #  True/False
    input['Use Mach']       = False             #  True/False

    input['Nose Radius']    = 0.0348                  # meters
    input['Radius']         = 6371.            # Earth's radius, km
    input['gamma']          = 20.              # degree
    input['omega']          = 7.292E-5        # rad/sec
    input['theta']          = 30.0
    input['phi']            = 90.0 - input['theta']
    input['psi']            = 10.0
    input['epsilon']        = 5.0               # deg
    input['Thrust']         = 22250.           # N
    input['Throttle']       = 100.             # %
    input['Mass']           = 1100.            # kg
    input['Isp']            = 310.             # sec
    input['CL']             = 0.1411
    input['CD']             = 0.0921
    input['Flight Aoa']     = 3.0             # degrees
    input['sigma']          = 0.0               # deg, expected to change if flight aoa changes
    input['Engine Cutoff']  = 130.0           # sec    
    input['g']              = 9.81           # m/s^2
    input['go']             = 9.81           # m/s^2

    input['Target Mach']    = 6.0             
    input['Target Altitude']     = (60000/3.281)/1000            # km

    # initial values (all of these must be lists)
    input['Velocity']       = [208.0]             # m/s
    input['Altitude']       = [10.668]             # km
    input['Aoa']            = list(range(0,11))         # deg
    HyperDrive.run_flight_simulation(
        input=input,
        calc_only_normal_shock=True,
        show_output=True,
        save_output=False,
        show_plot=True,
        save_plot=False)
######## USER INPUT ########