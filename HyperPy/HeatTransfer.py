# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:41:11 2022

@author: ryant
"""
import numpy as np
from scipy import special

def surface_temperature(Tinf, T1, Tenv, k, h, emiss, clength, calculate_radiation:bool=True):
    # Assumes a 1-D energy balance problem
    #
    # Parameters:
    # -----------
    # INPUTS:
    # Tinf      - 1xn vector, temperature of air touching surface [K]
    # T1        - 1xn vector, approximate temp. behind surface [K]
    # Tevn      - 1xn vector, environment temp. heat is radiated towards [K]
    # k         - 1xn vector, thermal conductivity (W/m*k)
    # h         - 1xn vector, enthalpy (W/m**2*k)
    # emiss     - 1xn vector, emissivity
    # clength   - 1xn vector, characteristic length the heat transfer travels (m)
    # consrad   - 1x1, calculate surface temp. with radiation (bool)
    # 
    # OUTPUTS:
    # qconv     - 1xn, convection heat transfer [w/m**2]
    # qcond     - 1xn, conduction heat transfer [w/m**2]
    # qrad      - 1xn, radiation heat transfer [w/m**2]
    # Tsurf     - 1xn, surface temperature [K]
    Tsurf = np.nan;
    qrad = np.nan;
    qconv = np.nan;
    qcond = np.nan;
    emol = 1;
    sigma = 5.67E-8;    # stefan-boltzmann constant, W/m*K**4

    nTsurf = T1;
    # use Newton-Ralphson iteration
    while emol >= 1E-7:
        Tsurf = nTsurf;
        qcond = k * (Tsurf - T1) / clength;
        qconv = h * (Tinf - Tsurf);

        if calculate_radiation:
            qrad = emiss * sigma * ((Tsurf**4) - (Tenv**4));
        else:
            qrad = 0;

        f = qconv - qcond - qrad;

        if calculate_radiation:
            fdot = (-k / clength) - h - (4 * emiss * sigma * (Tsurf**3));
        else:
            fdot = (-k / clength) - h;

        pTsurf = Tsurf;
        nTsurf = Tsurf - (f / fdot);
        emol = abs((pTsurf - nTsurf) / pTsurf);

    return [qconv, qcond, qrad, Tsurf]

def composite_wall_temperature(T_inf: float,\
                               T_surr: float,\
                               q_cond: float,\
                               h: float,\
                               hr: float,\
                               k_list: list,\
                               L_list: list):
    # use surface energy balance to find T1 or surface temperature
    q_cond = -q_cond
    T1 = (q_cond + (T_inf * h) + (T_surr * hr)) / (h + hr)
    T2 = (q_cond * L_list[0] / k_list[0]) + T1
    T3 = ((q_cond*2) / (((L_list[2]*k_list[1]) +\
                         (L_list[1]*k_list[2]))/(L_list[1]*L_list[2]))) + T2
    T4 = ((q_cond * L_list[-1]) / k_list[-1]) + T3

    return [T1, T2, T3, T4]

def semi_inf_solid_approx_const_heat_flux(x: float,\
                                          t: float,\
                                          k: float,\
                                          alpha: float,\
                                          q_d2: float,\
                                          Ti: float
                                         ):
    A = 2 * q_d2 * np.sqrt(alpha * t / np.pi) / k
    B = -(x**2) / (4 * alpha * t)
    C = -q_d2 * x / k
    D = x / (2*np.sqrt(alpha * t))
    return Ti + (A * np.exp(B)) + (C * special.erfc(D))


if __name__ == "__main__":
    # HW 2 Question 3
    Tenv:float = 2469.8
    T1:float = 150. + 273.
    Tinf:float = 216.0
    k:float = 0.75
    h:float = 60.0
    emiss:float = 0.88
    clength:float = 0.0762

    [qconv, qcond, qrad, Tsurf] = surface_temperature(Tenv, T1, Tinf, k, h, emiss, clength, False)    
    print("Only conduction and convection: ")
    print("Qconv: ", qconv, " W/m**2")
    print("Qcond: ", qcond, " W/m**2")
    print("Qrad: ", qrad, " W/m**2")
    print("Tsurf: ", Tsurf, " K")
    print("=======================================")
    [qconv, qcond, qrad, Tsurf] = surface_temperature(Tenv, T1, Tinf, k, h, emiss, clength)
    print("Taking into account radiation: ")
    print("Qconv: ", qconv, " W/m**2")
    print("Qcond: ", qcond, " W/m**2")
    print("Qrad: ", qrad, " W/m**2")
    print("Tsurf: ", Tsurf, " K")

    # HW3 Question 1
    h:float = 95.0 # W/m^2*k
    h_r:float = 36.2 # W/m^2*k
    T_inf1:float = 500 + 273.15 # K
    T_surr: float = 20 + 273.14 # K
    q_cond:float = 1000.0 # W/m^2
    ks = [5.6, 0.1, 0.2, 0.4] # W /m*K
    ls = [1/100, 2/100, 2/100, 2/100] # m
    temps = composite_wall_temperature(T_inf=T_inf1,\
                               T_surr=T_surr,\
                               q_cond=q_cond,\
                               h=h,\
                               hr=h_r,\
                               k_list=ks,\
                               L_list=ls)
    print("Question 1: ")
    print("Wall temperatures from left to right: ")
    print(temps, " K")
    print(" or ")
    print([T-273.15 for T in temps], " degC")

    # Hw 3 Question 3, find the thermal conductivity using secant method
    iterations = 6
    target_T = 175.0+273.15 # K
    t = 60.0 # sec
    Lc = 0.02 # m 
    k0 = 0.1  # W/m*K
    k1 = 1    # W/m*K
    k2 = -1   # W/m*K
    rho = 275.0 # kg/m**2
    Cp = 919.8 # J /kg * K  
    Ti = 21.0+273.15 # K
    q_d2 = 30. * (100**2) # W / m**2
    for i in range(iterations):
        f1 = semi_inf_solid_approx_const_heat_flux(x=Lc,\
                                                     t=t,\
                                                     k=k1,\
                                                     alpha=k1/(rho*Cp),\
                                                     q_d2=q_d2,\
                                                     Ti=Ti) - target_T
        f0 = semi_inf_solid_approx_const_heat_flux(x=Lc,\
                                                     t=t,\
                                                     k=k0,\
                                                     alpha=k0/(rho*Cp),\
                                                     q_d2=q_d2,\
                                                     Ti=Ti) - target_T
        k2 = k1 - (f1 * (k1 - k0) / (f1 - f0))
        k0, k1 = k1, k2
    print("\nFor Question 2:")
    print("K = ",k2, " W/m*K")
    alpha = k2 / (rho * Cp)
    Fo = alpha * t / Lc**2
    print("Fourier Number: ", Fo)

