
import numpy as np
import matplotlib.pyplot as plt

from StagnationHeatFlux import fay_riddell_CPG, discretized_reference_temperature, reference_temperature
from EquationsofMotion import vdot, gammadot, hdot, psidot, thetadot, phidot, massdot
from EquilibriumShock import equilibrium_shock, oblique_shock_angle
from StandardAtmosphere1962 import standard_atmosphere_1962, altitude
from NewtonianAerodynamics import blunted_biconical_coefficients
from TransportSpecies import kinetic_gas_theory

# for Mach 10 at 18.25Km
species_model = {
    'N2': {'mf':0.74423,   'Cp':(36.727/28.014)*1000, 'mw':28.014, 'sigma': 3.798, 'e_k': 71.4},
    'O2': {'mf':0.15558, 'Cp':(39.114/31.998)*1000, 'mw':31.998, 'sigma': 3.467, 'e_k': 106.7},
    'N' : {'mf':2.606E-5,  'Cp':(21.135/14.007)*1000, 'mw':14.007,  'sigma': 3.298, 'e_k': 71.4},
    'O' : {'mf':0.04128,   'Cp':(21.027/15.999)*1000, 'mw':15.999,  'sigma': 3.05, 'e_k': 106.7},
    'NO': {'mf':0.05887, 'Cp':(36.963/30.006)*1000, 'mw':30.006,  'sigma': 3.492, 'e_k': 116.7},
}

def get_downstream_oblique_shock_conditions(alt:float, M:float, u_1: float, theta_c: np.array, show_output: bool=False):
    beta = oblique_shock_angle(theta_c=theta_c, u_1=u_1, alt=np.array([alt]), M=np.array([M]))
    P = h = T = rho = u = Cp = np.array([])
    if show_output:
        print("\n{:12s}\t{:15s} {:15s} {:15s} {:15s} {:15s} {:15s}".format("ObSA (deg.)", "P E5 (Pa)", "T (K)", "h (kJ/kg)", "rho E-1 (kg/m^3)", "u (m/s)", "Cp (kJ/kg)"))
        print("=========================================================================================================================")
    for B in beta:
        [P_2, h_2, T_2, rho_2, u_2, Cp_2] = equilibrium_shock(alt=np.array([alt]), M=np.array([M]), beta=B)
        if show_output:
            print("{:2.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f}".format(B, P_2[-1]/1E5, T_2, h_2[-1], rho_2[-1]*1E1, u_2[-1], Cp_2/1000))
        P = np.append(P, P_2)
        h = np.append(h, h_2)
        T = np.append(T,T_2)
        rho = np.append(rho, rho_2)
        u = np.append(u, u_2)
        Cp = np.append(Cp, Cp_2)
    return [beta, P, h*1000, T, rho, u, Cp/1000]

def _get_radii_and_body_angles(coords: np.array):
    # Take the X and Y coordinates and turn them
    # into radii and body angles
    #X = coords[:,0]; Y = coords[:,1]
    X = coords[1:,0]; Y = coords[1:,1]
    lastidx = len(X)-1
    
    X_prev = np.delete(X, (lastidx), axis=0)
    X_curr = np.delete(X, (0), axis=0)
    Y_prev = np.delete(Y, (lastidx), axis=0)
    Y_curr = np.delete(Y, (0), axis=0)

    theta = np.rad2deg(np.arctan((Y_curr-Y_prev) / (X_curr-X_prev)))
    theta = np.append(np.pi/2, theta)
    radii = Y

    geom = np.column_stack((radii, theta))
    return geom

def _get_geometry(file, R_n):
    coords = np.loadtxt(file)
    geom = _get_radii_and_body_angles(coords=coords)
    
    nose_geom = geom[(geom[:,0] > 0.0) & (geom[:,0] < R_n), :]
    R_b = geom[len(geom)-1,0]
    delta_c = geom[0,1]

    nose_coords = coords[(coords[:,1] > 0.0) & (coords[:,1] < R_n), :]
    body_coords = coords[coords[:,1] > 0.0, :]

    return geom, nose_geom, nose_coords, body_coords, R_b, delta_c

def run_point_design(
    input: dict,
    calc_only_normal_shock: bool=True,
    show_output:bool=True,
    save_output:bool=False,
    show_plot:bool=True,
    save_plot:bool=False
):
    '''
    '''
    # Constants:
    gamma_air = 1.4

    # settings:
    run_3d_sim = input['Run 3D Sim']
    calc_aero = input['Calc Aero']
    use_mach = input['Use Mach']
    coord_file = input['Coordinate File']

    # initialize 
    CL =  CD = L_D = flight_speed = np.nan
    if not use_mach:
        flightspeed = input['Velocity']

    alt = np.array(input['Altitude'])
    aoa = np.array(input['Aoa'])   
    R_n = input['Nose Radius']

    T_inf = P_inf = rho_inf = R = None
    P_2 = h_2 = T_2 = rho_2 = u_2 = Cp_2 = None
    # for part 1.a
    for file in coord_file:
        geom, nose_geom, nose_coords, body_coords, R_b, delta_c = _get_geometry(file, R_n)

        for h in alt:
            valt = np.array(altitude(h,'p')) # from geometric to geopotential  
            [T_inf, P_inf, rho_inf, R] = standard_atmosphere_1962(np.array([valt]))
            R = R * 1000

            M = np.array([])
            if use_mach: 
                M = np.array(input['Mach'])
            else:
                M = np.array(input['Velocity'])/np.sqrt(gamma_air * T_inf * R)
            for mach in M:
                [P_2, h_2, T_2, rho_2, u_2, Cp_2] = equilibrium_shock(alt=np.array([valt]), M=np.array([mach]))
                Cp_2 = Cp_2 / 1000
                h_2 = h_2 * 1000
                
                if use_mach: 
                    flight_speed = mach * np.sqrt(T_inf * gamma_air * R)
                else:
                    flight_speed = flightspeed[0]
                    flightspeed = flightspeed[1:]
                if show_output:
                    if use_mach: print("\nFor Mach ", mach, " at ", round(h,3), " km:")
                    else: print("For flight velocity ", flight_speed, " m/s at ", h," km:")

                eps = np.array([])
                bP_2 = bh_2 = bT_2 = brho_2 = bu_2 = bCp_2 = np.array([])   # downstream conditions at the body
                if not calc_only_normal_shock:
                    if show_output:
                        print("\nStagnation Point Conditions: ")
                        print("{:15s} {:15s} {:15s} {:15s} {:15s} {:15s}".format("P E5 (Pa)", "T (K)", "h (kJ/kg)", "rho E-1 (kg/m^3)", "u (m/s)", "Cp (kJ/kg)"))
                        print("========================================================================================")
                        print("{:2.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f}".format(P_2[-1]/1E5, T_2, h_2[-1]/1000, rho_2[-1]*1E1, u_2[-1], Cp_2))
                    u_1 = mach * np.sqrt(T_inf * gamma_air * R)
                    [_, oP_2, oh_2, oT_2, orho_2, ou_2, oCp_2] = get_downstream_oblique_shock_conditions(alt=valt,M=mach,u_1=u_1,theta_c=geom[1:,1],show_output=show_output)
                    eps = np.append(rho_inf/rho_2, rho_inf / orho_2)
                    bP_2 = oP_2; bh_2 = oh_2; bT_2 = oT_2; bu_2 = ou_2; bCp_2 = oCp_2; brho_2 = orho_2
                    # Since the since the oblique shock is 0 deg at the end point of the geometry, the downstream velocity, 'ou_2', = 0.
                    # This will give 'inf' or 'nan' as answer for heat flux in the reference method calculation, so will copy the 2nd to last
                    # downstream conditions to give the illusion that the downstream conditions behind the vehicle continue on 
                else:
                    eps = (rho_inf/rho_2) * np.ones(len(geom))
                    bP_2 = P_2 * np.ones(len(body_coords))
                    bh_2 = h_2 * np.ones(len(body_coords))
                    bT_2 = T_2 * np.ones(len(body_coords))
                    bu_2 = u_2 * np.ones(len(body_coords))
                    bCp_2 = Cp_2 * np.ones(len(body_coords))
                    brho_2 = rho_2 * np.ones(len(body_coords))

                bbc = blunted_biconical_coefficients(geom=geom, aoas=aoa, eps=eps, rn=R_n)
                L_D = bbc[0]; CL = bbc[1]; CD = bbc[2]; CN = bbc[3]; CA = bbc[4]; cpmax = bbc[5]
                
                if show_output or save_output:
                    print("T inf: ", T_inf[0], " K")
                    print("P_inf: ", P_inf[0], " Pa")
                    print()
                    print("{:15s} {:15s} {:15s} {:15s} {:15s} {:15s}".format("AOA (deg.)", "CA", "CN", "CL", "CD", "L/D"))
                    print("========================================================================================")
                    for idx in range(0, len(aoa)):
                        print("{:2.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f}".format(aoa.item(idx), CA.item(idx), CN.item(idx), CL.item(idx), CD.item(idx), L_D.item(idx)))
                    print("\nAvg. Cp max: ", round(np.average(cpmax),5))
                if show_plot or save_plot:
                    plt.figure(figsize=(7,5))
                    plt.plot(aoa, CL, label="CL", linestyle='dashed')
                    plt.plot(aoa, CD, label="CD", linestyle='dotted')
                    plt.plot(aoa, CN, label="CN", linestyle='solid')
                    plt.plot(aoa, CA, label="CA", linestyle='dashdot')

                    plt.xlabel("Angle of Attack (Deg.)")
                    plt.ylabel("Coefficient")
                    plt_title = f"Mach {mach} at {h} km"
                    plt.title(plt_title)
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.0, 0.), loc="lower right", ncol=4)

                    if save_plot: 
                        plt.savefig(plt_title.replace(' ','_')+".png")
                        plt.show()
                        plt.close()
                    if show_plot: 
                        plt.show()

                # for part 1.b
                # remember to use Fay & Riddell at X = 0, Y = 0
                # then use the discretized reference temperature method
                [u_kgt, k_kgt, _, _] = kinetic_gas_theory(P_2/101325, T_2, species_model)
                Pr = u_kgt * (Cp_2*1000)/ k_kgt
                h_o = (h_2 + ((u_2**2)/2))/1000
                # only use the condiations at the stagnation region (normal shock)
                q_fr=fay_riddell_CPG(
                    R_n=R_n,         # m
                    T_w=293,            # K, assume room temperature
                    P_inf=P_inf,       # Pa
                    P_2=P_2,  # Pa
                    rho_2=rho_2,     #kg/m^3
                    h_o=h_o,        #kJ/kg      
                    Cp=Cp_2,          #kJ/kg*K
                    mu_2=u_kgt,     #kg/m*s
                    Pr=Pr,          #dimensionless
                )
                print()
                print("dq\" stag:  ", round(q_fr[0]/(100**2),5), " W/cm^2")
                print("Pr:         ", round(Pr,5))
                print("mu_e:       ", round(u_kgt*1E5,5), " x10^-5 kg/m*s")
                print("rho_e:      ", round(rho_2[-1],5), " kg/m^3")
                print()
                [S, C_H, q_wx, C_f, Re_starx, rhoe_ue_CH, T_star, Pr_star, H_r] = discretized_reference_temperature(
                    geom=body_coords,
                    delta_c=delta_c, # deg
                    R_n=R_n, # m
                    R_b=R_b, # m
                    T_inf=T_inf, # K
                    P_inf=P_inf, # Pa
                    rho_inf=rho_inf, # kg/m^3
                    T_2=bT_2, # K
                    P_2=bP_2, # Pa
                    rho_2=brho_2,    # kg/m^3
                    u_2=bu_2,  # m/s
                    h_0=h_o,  # J/kg
                    h_2=bh_2,    # J/kg
                    Cp_2=bCp_2*1000,  #J/kg*K 
                    T_w=293,  # K
                )
                if show_output:
                    print("T*:  ", round(np.average(T_star),5), " K")
                    print("Pr*: ", round(np.average(Pr_star),5))
                    print("H_r: ", round(np.average(H_r)/1000,5), " kJ/kg")
                    print("\n{:12s} {:15s} {:15s} {:15s} {:15s} {:22s} {:15s}".format("Point", "Length (m)", "RE* (x10^6)", "Cf (x10^-3)", "CH (x10^-3)", "puCH (kg/m^2*s)", "dq\" (W/cm^2)"))
                    print("=================================================================================================================")
                    # tabulting from the discretized reference temperature
                    for idx in range(0, len(S)):
                        print("{:5d} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f}".format(idx+2, S.item(idx), Re_starx.item(idx)/1E6, C_f.item(idx)*1E3, C_H.item(idx)*1E3, rhoe_ue_CH.item(idx), q_wx.item(idx)/(100**2)))
                if show_plot:
                    None

def _flight_simulation_3D(sim_state:dict, rn:float, geom:np.array, aoa:np.array, dt:float=0.01, show_output:bool=False):
    '''
    You can assume you have the ability to change either, or both, the
    thrust offset angle and pitch angle.

    TODO:
    change aoa over time if necessary
    add ability to change elevon (pitch angle) and thrust offset angle (epsilon) more dynamically
    add a steady-state mechanism (perhaps using previous and current states) 
    add glide capabilities
    
    Parameters:
    
    INPUT:
    ------
    t_burn: float - main propulsion burn time (sec)
    '''
    gamma = psi = time = phi = theta = velocity = alt = mass = mach = xdistance = np.array([])
    target_mach = sim_state['Target Mach']
    target_alt = sim_state['Target Altitude']
    t_burn = sim_state['Burn Time']
    thrust_offset_angle = sim_state['epsilon']
    full_thrust = sim_state['T']
    throttle = sim_state['throttle']
    CL_wing = sim_state['CL']
    CD_wing = sim_state['CD']

    valt = altitude(alt=sim_state['alt'], convert_to='p')
    [T_update, _, rho_update, R] = standard_atmosphere_1962(valt=valt)
    sim_state['Temp'] = T_update[-1]
    sim_state['rho']= rho_update       # atmospheric density, kg/m**3
    sim_state['R'] = R*1000

    x_total = np.zeros(1)
    y_total = np.zeros(1)
    z_total = np.zeros(1)
    t_total: float = 0.0
    v_dot = h_dot = gamma_dot = mass_dot = 0.0
    if show_output:
        print("\n{:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}".format(\
            "t (sec)", "x (km)", "y (km)", "z (km)", "Alt (km)", "Mach #", "V (m/s)", "phi (deg)", "psi (deg)", "theta (deg)", "sig (deg)", "eps (deg)", "mass (kg)", "throttle (%)"))
        print("===================================================================================================================================")
    
    while t_total <= t_burn:
        if show_output:
            _t_total = t_total; _x_total = x_total[0]/1000; _y_total = y_total[0]/1000; _z_total = z_total[0]/1000
            _alt = sim_state['alt'][0]; _V = sim_state['Velocity'][0]; _mach = _V / np.sqrt(1.4 * sim_state['Temp'] * sim_state['R'])
            _phi = np.rad2deg(sim_state['phi'])[0]; _psi = np.rad2deg(sim_state['psi'])[0]; _theta = np.rad2deg(sim_state['theta'])[0]
            _sigma = np.rad2deg(sim_state['sigma'])[0]; _eps = np.rad2deg(sim_state['epsilon'])[0]; _mass = sim_state['m'][0]; _throttle = sim_state['throttle'][0]*100
            print("{:7.3f} {:7.3f}\t{:7.3f}\t{:7.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t\t{:2.3f}\t{:2.3f}".format(\
                _t_total, _x_total, _y_total, _z_total, _alt, _mach, _V, _phi, _psi, _theta, _sigma, _eps, _mass, _throttle))
                
        # update altitude and mach
        curr_alt = sim_state['alt']
        curr_mach = sim_state['Velocity'] / np.sqrt(1.4 * sim_state['Temp'] * sim_state['R'])

        valt = altitude(alt=sim_state['alt'], convert_to='p')
        [T_update, rho_update, rho_update, R] = standard_atmosphere_1962(valt=valt)
        [_, _, _, rho_2, _, _] = equilibrium_shock(alt=np.array([valt]), M=np.array([curr_mach]))
        eps = (rho_update/rho_2) * np.ones(len(geom))
        bbc = blunted_biconical_coefficients(geom=geom, aoas=aoa, eps=eps, rn=rn)
        _CL = bbc[1]; _CD = bbc[2]
        CL = np.column_stack((aoa, _CL))
        CD = np.column_stack((aoa, _CD))

        sim_state['CL'] = CL_wing + CL[CL[:,0]==sim_state['Flight Aoa'], 1]
        sim_state['CD'] = CD_wing + CL[CD[:,0]==sim_state['Flight Aoa'], 1]

        # check if we reach target Mach and height. If so, make v_dot = 0 and h_dot = 0
        # and change the thrust offset angle and flight path angle if necessary
        if curr_mach[0] <= target_mach[0]:
            if throttle < 1.0:
                throttle += 0.05
        else:
            throttle -= 0.05
        if throttle > 1.0: throttle = np.array([1.0])
        if throttle < 0.0: throttle = np.array([0.002])
        sim_state['T'] = full_thrust * throttle
        sim_state['throttle'] = throttle
        v_dot = vdot(sim_state)

        if curr_alt[0] >= target_alt[0]:
            sim_state['gamma'] = 0.0
            sim_state['Flight Aoa'] = 0.0
            sim_state['sigma'] = np.array([0.0])
            thrust_offset_angle = np.array([0.0])
            sim_state['CL'] = CL_wing + CL[CL[:,0]==sim_state['Flight Aoa'], 1]
            sim_state['CD'] = CD_wing + CL[CD[:,0]==sim_state['Flight Aoa'], 1]
            gamma_dot = 0.0
            psi_dot = 0.0
            theta_dot = 0.0
            phi_dot = 0.0

        gamma_dot = gammadot(sim_state)
        psi_dot = psidot(sim_state)
        theta_dot = thetadot(sim_state)
        phi_dot = phidot(sim_state)
        h_dot = hdot(sim_state)     # it's in meters
        mass_dot = massdot(sim_state)

        # update states
        sim_state['Temp'] = T_update[0]
        sim_state['rho']= rho_update       # atmospheric density, kg/m**3
        sim_state['R'] = R * 1000
        sim_state['Velocity'] += v_dot * dt         # flight velocity, m/s
        sim_state['alt'] += (h_dot * dt)/1000
        sim_state['m'] += mass_dot * dt        # total mass, kg
        sim_state['epsilon'] = thrust_offset_angle   # angle b/w thrust and velocity vector, deg.
        sim_state['theta'] += theta_dot * dt    # bank angle/longitude, deg.
        sim_state['phi'] += phi_dot * dt      # latitude, deg.
        sim_state['psi'] += psi_dot * dt      # heading angle (b/w local line of lat. and projection of V vector on the local horizontal plane), deg.
        sim_state['gamma'] += gamma_dot * dt     # flight angle, deg.

        xdistance = np.append(xdistance, x_total/1000)
        time = np.append(time, t_total)
        gamma = np.append(gamma, np.rad2deg(sim_state['gamma'])[0])
        psi = np.append(psi, np.rad2deg(sim_state['psi'])[0])
        phi = np.append(phi, np.rad2deg(sim_state['phi'])[0])
        theta = np.append(theta, np.rad2deg(sim_state['theta'])[0])
        velocity = np.append(velocity, sim_state['Velocity'])
        alt = np.append(alt, sim_state['alt'])
        mass = np.append(mass, sim_state['m'])
        mach = np.append(mach, curr_mach)

        x_total += sim_state['Velocity'][0] * np.cos(sim_state['gamma']) * dt 
        y_total += 0.0 
        z_total += h_dot * dt 
        t_total += dt
    return gamma, psi, time, phi, theta, xdistance, velocity, alt, mass, mach

def run_flight_simulation(
    input: dict,
    calc_only_normal_shock: bool=True,
    show_output:bool=True,
    save_output:bool=False,
    show_plot:bool=True,
    save_plot:bool=False):
    '''
    Setup to run 1 mach and 1 altitude at a time
    '''
    # all angles must be converted from degress to radians
    m =             np.array([input['Mass']])
    gamma =         np.deg2rad(input['gamma'])
    sigma =         np.deg2rad(input['sigma'])
    epsilon =       np.deg2rad(input['epsilon'])
    flight_aoa =    input['Flight Aoa']
    theta =         np.deg2rad(input['theta'])
    phi =           np.deg2rad(input['phi'])
    psi =           np.deg2rad(input['psi'])
    T =             np.array([input['Thrust']])
    throttle =      np.array([input['Throttle']/100])
    I_sp =          np.array([input['Isp']])
    aoas =          np.array(input['Aoa'])
    engine_cutoff = np.array([input['Engine Cutoff']])
    radius =        np.array([input['Radius']])
    target_M =      np.array([input['Target Mach']])
    target_h =      np.array([input['Target Altitude']])
    r =             np.array(input['Altitude']) + radius
    Aref =          np.pi * (input['Nose Radius'] ** 2)

    sim_state = {}
    sim_state['T'] = T
    sim_state['throttle'] = throttle
    sim_state['Aref']  = np.array([Aref])
    sim_state['CL'] = np.array([input['CL']])
    sim_state['CD'] = np.array([input['CD']])
    sim_state['Target Mach'] = target_M
    sim_state['Target Altitude'] = target_h
    sim_state['alt'] = np.array(input['Altitude'])
    sim_state['r'] = r
    sim_state['theta'] = np.array([theta])
    sim_state['phi'] = np.array([phi])
    sim_state['psi'] = np.array([psi])
    sim_state['radius'] = radius
    sim_state['gamma'] = np.array([gamma])
    sim_state['omega'] = np.array([input['omega']])
    sim_state['m'] = m
    sim_state['Isp'] = I_sp
    sim_state['Burn Time'] = engine_cutoff
    sim_state['sigma'] = np.array([sigma])
    sim_state['epsilon'] = np.array([epsilon])
    sim_state['g'] = np.array([input['g']])
    sim_state['go'] = np.array([input['go']])
    sim_state['Flight Aoa'] = np.array([flight_aoa])
    sim_state['Velocity'] = np.array(input['Velocity'])

    R_n = input['Nose Radius']
    coord_file = input['Coordinate File'][0]

    geom, _, _, _, _, _ = _get_geometry(coord_file, R_n)

    _gamma, _psi, _time, _phi, _theta, _x, _velocity, _alt, _mass, _mach = _flight_simulation_3D(sim_state=sim_state, rn=R_n, geom=geom, aoa=aoas, show_output=show_output)

    if show_plot:
        fig, axs = plt.subplots(3, 3)
        
        axs[0,0].plot(_time, _alt)
        axs[0,0].set_title("Altitude (km)")

        axs[1,0].plot(_time, _mach)
        axs[1,0].set_title("Mach No.")

        axs[2,0].plot(_time, _mass)
        axs[2,0].set_title("Mass (kg)")

        axs[0,1].plot(_time, _velocity)
        axs[0,1].set_title("Velocity (m/s)")

        axs[1,1].plot(_time, _gamma)
        axs[1,1].set_title("Flight Path Angle (deg)")

        axs[2,1].plot(_time, _x)
        axs[2,1].set_title("Horiz. Displ. (km)")

        axs[0,2].plot(_time, _phi)
        axs[0,2].set_title("Latitude (deg)")

        axs[1,2].plot(_time, _psi)
        axs[1,2].set_title("heading angle (deg)")

        axs[2,2].plot(_time, _theta)
        axs[2,2].set_title("Longitude (deg)")

        fig.tight_layout()

        plt.show()