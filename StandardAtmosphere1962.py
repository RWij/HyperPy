# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:39:27 2022

@author: ryant
"""
import numpy as np

def standard_atmosphere_1962(valt:np.array):
    '''
     From the 1962 U.S Standard Atmosphere Model:
     This assumes altitude, 'alt', is geopotential height for
     0 < alt < 86. km. Past 86. km is assumed to be geometric.
     Model breaksdown past towards 90 km and nearly complete
     past 120 km from the 1976 U.S Standard Atmosphere model.
     
     Parameters:
     ----------
    
     INPUTS:
     alt - 1xn vector of current altitude (km)
    
     OUTPUTS:
     T_inf - 1xn vector of freestream temperature (K)
     P_inf - 1xn vector of freestream pressure (N/m**2)
     rho_inf - 1xn vector of freestream density (kg/m**3)
     R       - 1xn vector of real gas constant for air (J/kmol*kg)
    '''
    
    ncols = valt.shape[-1];

    T_inf = np.zeros(ncols)
    P_inf = np.zeros(ncols)
    rho_inf = np.zeros(ncols)

    # values pertaining to Earth
    mw_0 = 28.9644; # air molecular weight, kmol
    R_bar = 8.31432; # J / kmol * kg
    g_0 = 9.806; # m/s**2
    b = 3.31E-7; # m**-1

    P = 0.0;
    rho = 0.0;
    for i in range(0, ncols):
        alt = valt[i];

        if alt >= 0.0 and alt <= 86:
            alt = 1 * (6378 * alt) / (6378 + alt);
        
        # set temperature references and lapse rates
        ############################################
        if alt >= 0.0 and alt <= 11.0:
            P_i = 101325; # N/m**2
            rho_i = 1.225; # kg/m**3
            T_mi = 288.15; # K
            h_i = 0.0; # km
            L_hi = -6.5; # k/km'
        

        if alt > 11.0 and alt <= 20.0:
            P_i = 22631.95; # N/m**2
            rho_i = 0.3639; # kg/m**3
            T_mi = 216.650; # K
            h_i = 11.0; # km
            L_hi = 0.0; # k/km'
        

        if alt > 20.0 and alt <= 32.0:
            P_i = 5474.79; # N/m**2
            rho_i = 0.0880354; # kg/m**3
            T_mi = 216.650; # K
            h_i = 20.0; # km
            L_hi = 1.0; # k/km'
        
    
        if alt > 32.0 and alt <= 47.0:
            P_i = 868.01; # N/m**2
            T_mi = 228.65; # K
            rho_i = 0.01355; # kg/m**3
            h_i = 32.0; # km
            L_hi = 2.8; # k/km'
        

        if alt > 47.0 and alt <= 51.0:
            P_i = 110.9; # N/m**2
            T_mi = 270.65; # K
            rho_i = 1.427487E-3; # kg/m**3
            h_i = 47.0; # km
            L_hi = 0.0; # k/km'
        

        if alt > 51.0 and alt <= 71.0:
            P_i = 66.94; # N/m**2
            T_mi = 270.65; # K
            rho_i = 8.616407E-4; # kg/m**3
            h_i = 51.0; # km
            L_hi = -2.8; # k/km'
        

        if alt > 71.0 and alt <= 86.0:
            P_i = 3.956; # N/m**2
            T_mi = 214.65; # K
            rho_i = 6.420575E-5; # kg/m**3
            h_i = 71.0; # km
            L_hi = -2.0; # k/km'
        

        # Warning: goes further away from 1976 model, 
        # especially past 120 km
        if alt > 86.0 and alt <= 100:
            P_i = 0.344180; # N/m**2
            T_mi = 186.945; # K
            rho_i = 6.41396E-6; # kg/m**3
            h_i = 86.0; # km
            L_hi = 1.6481; # k/km'
        

        if alt > 100 and alt <= 110:
            P_i = 0.29073; # N/m**2
            T_mi = 210.65; # K
            rho_i = 4.794193E-6; # kg/m**3
            h_i = 100.0; # km
            L_hi = 5.0; # k/km'
            mw_0 = 28.88;
        

        if alt > 110.0 and alt <= 120:
            P_i = 0.006801; # N/m**2
            T_mi = 260.65; # K
            rho_i = 8.963203E-8; # kg/m**3
            h_i = 110.0; # km
            L_hi = 10.0; # k/km'
            mw_0 = 28.56;
        

        if alt > 120.0 and alt <= 150.0:
            P_i = 0.002247;
            T_mi = 360.65;
            rho_i = 2.104283E-8;
            h_i = 120.0;
            L_hi = 20.0;
            mw_0 = 28.08;
        
        ##########################################################

        R = R_bar / mw_0; # J / kg

        T_m = T_mi + (L_hi * (alt - h_i));
        if alt >= 0 and alt < 86.0:
            if L_hi == 0:
                P = P_i * np.exp(-g_0 * mw_0 * (alt - h_i) / (R_bar * T_mi));
            else:
                P = P_i * ((T_mi / T_m) ** (g_0 * mw_0 / (R_bar * L_hi)));
            rho = P * mw_0 / (R_bar * 1000 * T_m);
        
        if alt >= 86.0:
            L_zi = L_hi;
            z_i = h_i;
            pow_p = -((g_0 / (R * L_zi) * (1 + b * ((T_mi / L_zi) - z_i))));
            P = P_i * ((T_m / T_mi) ** (pow_p)) * np.exp(g_0 * b * (alt - z_i) / (R * L_zi));
            pow_rho = -((g_0 / (R * L_zi) * ((R * L_zi / g_0) + 1 + b * ((T_mi / L_zi) - z_i))));
            rho = rho_i * ((T_m / T_mi) ** (pow_rho)) * np.exp(g_0 * b * (alt - z_i) / (R * L_zi));
        
    
        T_inf[i] = T_m;
        P_inf[i] = P;
        rho_inf[i] = rho;

    return [T_inf, P_inf, rho_inf, R]


def altitude(alt:np.array, convert_to:str):
    # Converts geopotential altitude to geometric altitude and vice-versa
    # Parameters:
    # -----------
    #
    # INPUT:
    # alt - altitude, geopotential (km') and geometric (km)
    # convert_to - string, what to convert to. 'p' - to geopotential, 
    #                                          'm' - to geometric

    L = 1.0; # km'/km
    r_0 = 6378.0; # Earth's radius, km
    # convert into geopotential altitude
    if convert_to == 'p':
        z = alt;
        calt = L * (r_0 * z) / (r_0 + z);
    # convert to geometric altitude
    if convert_to == 'm':
        h = alt;
        z = (h / L) / (1 - (h / (L * r_0)));
        calt = z;
        
    return calt
    
    
def exponential_atmosphere_model(h, h_0, g_0, P_0, rho_0, T_0, mw):
    # This assumes an ideal, calorically-perfect gas: p = rho * R * T
    # and no temperature variation and constant acceleratio of gravity
    # Parameters:
    # Input:
    # h - geopotential altitude (meters)
    # h_0 - reference geoppotential altitude (meters)
    # g_0 - gravitational acceleration (m/s**2)
    # P_0 - atmospheric pressure at sea level (N/m**2
    # rho_0 - atmospheric density at sea level (kg/m**3)
    # T_0 - reference temperature (K)
    # mw - molecular weight (kg / m**3)
    #
    # Output:
    # T_inf - freestream temperature (K)
    # P_inf - freestream pressure (N/m**2)
    # rho_inf - freestream rho (kg/m**3)
    
    R_bar = 8.314 * 1000;
    R = R_bar / mw;
    
    P_inf = P_0 * np.exp(g_0 * (h_0 - h) / (R * T_0));
    rho_inf = rho_0 * np.exp(g_0 * (h_0 - h) / (R * T_0));
    T_inf = P_inf / (rho_inf * R);
    
    return [T_inf, P_inf, rho_inf]


def exponential_density_scaled(H, rho_0, h):
    # Parameters:
    # INPUT:
    # If scale height, H, needs to be calculated provide:
    #   T - average atmosphere temperature (K)
    #   mw - molecular weight of atmosphere (kg/kmol)
    #   go - gravitaional acceleration (m/s**2)
    #
    # Else, just provide 'H' as the first input
    #
    # For both options:
    #   rho_0 - density at sea level
    #   h - evaluated height
    #
    # OUTPUT:
    #   rho - calculated atmospheric density (kg/m**3)

    rho = rho_0 * (np.exp(1) ** (-h / H))
    return rho

'''
def exponential_density_scaled(T, mw, g_0, rho_0, h):
    # Parameters:
    # INPUT:
    # If scale height, H, needs to be calculated provide:
    #   T - average atmosphere temperature (K)
    #   mw - molecular weight of atmosphere (kg/kmol)
    #   go - gravitaional acceleration (m/s**2)
    #
    # Else, just provide 'H' as the first input
    #
    # For both options:
    #   rho_0 - density at sea level
    #   h - evaluated height
    #
    # OUTPUT:
    #   rho - calculated atmospheric density (kg/m**3)
    R_bar = 8314; # J / gmol * k
    H = R_bar * T / (mw * g_0);

    rho = rho_0 * (np.exp(1) ** (-h / H));
    return rho
'''

if __name__ == "__main__":
    # HW 1, Question 1
    alt_115 = 115.0;
    alt_35 = altitude(35, 'm');
    alts = np.array([alt_35, alt_115])
    [T, P, rho, _] = standard_atmosphere_1962(alts)
    rho_0 = 868.01/(8.314 * 228.65);
    rho_e = exponential_density_scaled(7250.0, rho_0, 1000 * alts)
    
    print(f"Altitude (km): {alt_35}\t{alt_115}")
    print("Temperature (K): ")
    print(T)
    print("Pressure (Pa): ")
    print(P)
    print("Density (kg/m**3): ")
    print(rho)
    print("np.exponential Density Model: ")
    print(rho_e)
