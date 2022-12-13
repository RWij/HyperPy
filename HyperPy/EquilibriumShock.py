# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:03:11 2022

@author: ryant
"""
from StandardAtmosphere1962 import standard_atmosphere_1962
import numpy as np
from scipy import optimize
from scipy.special import cotdg, sindg, cosdg, tandg


def composition_curve_fit(P:float, rho:float, mode:int=1):
    T0 = np.nan
    X, Y = np.nan, np.nan
    h, T, d, c = np.nan, np.nan, np.nan, np.nan
    if mode == 1:
        X = np.log10(P / 1.013E5)[-1]
        Y = np.log10(rho / 1.292)[-1]
    if mode == 2:
        X = np.log10(P / 1.034E5)[-1]
        Y = np.log10(rho / 1.225)[-1]
    Z = X - Y
    # temperature curve fit
    d_curve_fit = np.array([
    [0.27407, 0, 1.00082, 0,	0,	0,	0,	0,	0,	0,	0,	0],
    [0.235869, -0.043304,	1.17619,	0.046498,	-0.143721, -1.3767, 0.160465, 1.08988, -0.083489, -0.217748,	-10, -1.78],
    [0.281611, 0.001267,	0.990406,	0,	0,	0,	0,	0,	0,	0,	0,	0],
    [0.457643, -0.034272,	0.819119,	0.046471,	0,	-0.073233, -0.169816,	0.043264,	0.111854,	0, -15	,-1.28],
    [1.04172, 0.041961	, 0.412752, -0.009329, 0, -0.434074, -0.196914,	0.264883,	0.100599,	0,	-15	,-1.778],
    [0.418298, -0.2521, 0.784048, 0.144576,	0,	-2.00015,	-0.639022, 0.716053, 0.206457, 0, -10, -2.4],
    [2.72964, 0.003725, 0.938851, -0.01192,	0,	0.682406,	0.089153,	-0.646541, -0.070769,	0, -20	, -0.82],
    [2.50246, -0.042827, 1.12924, 0.041517,	0,	1.72067, 0.268008, -1.25038, -0.179711	, 0,	-20, -1.33],
    [2.44531, -0.047722, 1.00488, 0.034349,	0,	1.95893, 0.316244, -1.012, -0.151561, 0, -20, -1.88],
    [2.50342, 0.026825	, 0.83886, -0.009819, 0,	3.58284, 0.533853, -1.36147,	-0.195436, 0, -20, -2.47]])
    # enthalpy curve fit
    c_curve_fit = np.array([
    [1.40000, 0.000000, 0.000000,  0.000000,	0,	0,	0,	0,	0,	0,	0],
    [1.42598, 0.000918,-0.092209, -0.002226, 0.019772 ,-0.0366 ,-0.077469 ,0.043878 ,-15 ,-1 ,-1.04],
    [1.64689,-0.062133,-0.334994,	0.063612,	-0.038332 ,-0.014468 ,0.073421 ,-0.002442 ,-15 ,-1 ,-1.36],
    [1.48558,-0.453562, 0.152096,	0.303350, -0.459282 ,0.448395 ,0.220546 ,-0.292293 ,-10 ,-1 ,-1.6],
    [1.40000, 0.000000, 0.000000,  0.000000, 0,	0,	0 ,0 ,0 ,0 ,0],
    [1.42176,-0.000366,-0.083614,	0.000675,	0.005272 ,-0.115853 ,-0.007363 ,0.146179 ,-20 ,-1 ,-0.86],
    [1.74436,-0.035354,-0.415045,	0.061921,	0.018536 ,0.043582 ,0.044353 ,-0.04975 ,-20 ,-1.04 ,-1.336],
    [1.49674,-0.021583,-0.197008,	0.030886,	-0.157738 ,-0.009158 ,0.123213 ,-0.006553 ,-10 ,-1.05 ,-1.895],
    [1.10421,-0.033664, 0.031768,	0.024335,	-0.176802 ,-0.017456 ,0.080373 ,0.002511 ,-15 ,-1.08 ,-2.65],
    [1.40000, 0.000000, 0.000000,	0.000000, 0 ,0 ,0 ,0 ,0 ,0 ,0],
    [1.47003, 0.007939,-0.244205, -0.025607, 0.872248 ,0.049452 ,-0.764158 ,0.000147 ,-20 ,-1 ,-0.742],
    [3.18652, 0.137930,-1.895290, -0.103490,	-2.14572 ,-0.272717 ,2.06586 ,0.223046 ,-15 ,-1 ,-1.041],
    [1.63963,-0.001004,-0.303549,  0.016464,	-0.852169 ,-0.101237 ,0.503123 ,0.04358 ,-10 ,-1 ,-1.544],
    [1.55889, 0.055932,-0.211764, -0.023548,	-0.549041 ,-0.101758 ,0.276732 ,0.046031 ,-15 ,-1 ,-2.25]])

    if Y > -0.50:
        T0 = 151.78
        # temperature
        if Z > 0.48 and Z <= 0.90: d = d_curve_fit[0]
        else: d = d_curve_fit[1]
        # enthalpy
        if Z<= 0.3: c = c_curve_fit[0]
        elif (Z > 0.3) and (Z <= 1.15): c = c_curve_fit[1]
        elif (Z > 1.15) and (Z <= 1.6): c = c_curve_fit[2]
        else: c = c_curve_fit[3]
    elif Y > -4.5 and Y <= -0.5:
        T0 = 151.78
        # temperature
        if Z > 0.48 and Z <= 0.9165: d = d_curve_fit[2]
        elif Z > 0.9165 and Z <= 1.478: d = d_curve_fit[3]
        elif Z > 1.478 and Z <= 2.176: d = d_curve_fit[4]
        else: d = d_curve_fit[5]
        # enthalpy
        if Z<= 0.3: c = c_curve_fit[4]
        elif (Z > 0.3) and (Z <= 0.98): c = c_curve_fit[5]
        elif (Z > 0.98) and (Z <= 1.38): c = c_curve_fit[6]
        elif (Z > 1.38) and (Z <= 2.04): c = c_curve_fit[7]
        else: c = c_curve_fit[9]
    elif Y > -7.0 and Y <= -4.5:
        #T0 = 1.8
        T0 = 0.5555
        # temperature
        if Z > 0.30 and Z <= 1.07: d = d_curve_fit[6]
        elif Z > 1.07 and Z <= 1.57: d = d_curve_fit[7]
        elif Z > 1.57 and Z <= 2.24: d = d_curve_fit[8]
        else: d = d_curve_fit[9]
        # enthalpy
        if Z<= 0.398: c = c_curve_fit[8]
        elif (Z > 0.398) and (Z <= 0.87): c = c_curve_fit[9]
        elif (Z > 0.87) and (Z <= 1.27): c = c_curve_fit[10]
        elif (Z > 1.27) and (Z <= 1.863): c = c_curve_fit[11]
        else: c = c_curve_fit[12]
    gamma = c[0] + (c[1] * Y) + (c[2] * Z) + (c[3] * Y * Z)\
        + (c[4] + (c[5] * Y) + (c[6] * Z) + (c[7] * Y * Z))\
            / (1 + np.exp(c[8] * (X + (c[9] * Y) + c[10]))) 
    log10_T_T0 = d[0] + (Y*d[1]) + (Z*d[2]) + (Y*Z*d[3]) + ((Z**2)*d[4])\
        + (d[5] + (Y*d[6]) + (Z*d[7]) + (Y*Z*d[8]) + ((Z**2)*d[9]))/ (1 + np.exp(d[10] * (Z + d[11])));
    T = T0 * (10 ** log10_T_T0)
    h = (P/rho) * (gamma / (gamma - 1))
    return [T, h, gamma]


def equilibrium_shock(alt:np.array, M:np.array, beta:np.array=None):
    '''
     Uses the 5-species air model equilibrium flow curve fit
     to calculate the temperature for downstream conditions
     from a normal and oblique shock for Mach no. >= 3, since
     CPG assumption breaksdown. Otherwise, uses the CPG equations.
     This assumes Earth's atmosphere.
    
     Parameters:
     -----------
    
     INPUT:
       Only for 2 & 3 args
       alt - 1xn vector of altitudes (km)
       M - 1xn vector of Mach number
       Only 3 args
       beta -  1xn vector of oblique shock angle (deg.)
     
     OUTPUT:
       P_2 - 1xn vector of pressure (N/m^2)
       h_2 - 1xn vector of enthalpy (kJ/kg)
       T_2 - 1xn vector of temperature (K)
       rho_2 - 1xn vector of density (kg/m^3)
       u_2 - 1xn vector of freestream velocity (m/s)
      '''
   
    gamma = 1.4
    tol = 1e-6; # 1e-5
    eps = 0.1
    h_2 = 0.0

    [T, P, rho, R_air] = standard_atmosphere_1962(alt)
    R_air = R_air * 1000

    a = np.sqrt(gamma * R_air * T)
    u = M * a
    # oblique shock calculations
    if beta is not None:
        beta = np.deg2rad(beta)
        u = u * np.sin(beta)
    Cp = gamma * R_air / (gamma - 1)
    # starting with a cpg assumption
    h_1 = Cp * T
    P_2 = P + (rho * ((u ** 2) * (1 - eps)))
    h_2i = h_1 + (0.5 * ((u ** 2) * (1 - (eps ** 2))))
    [T, h_2j, _] = composition_curve_fit(P_2, rho/eps)
    eps_inc = -0.01 if (h_2j - h_2i) / h_2i > 0 else 0.01
    max_iter = 1000
    iter = 0
    while abs((h_2j - h_2i) / h_2i) >= tol and iter < max_iter:
        eps = eps + eps_inc
        rho2 = rho / eps
        P_2 = P + (rho * ((u ** 2) * (1 - eps)))
        h_2i = h_1 + (0.5 * ((u ** 2) * (1 - (eps ** 2))))
        [_, h_2j, _] = composition_curve_fit(P_2, rho2)
        if (h_2j - h_2i) / h_2i > 0 and eps_inc > 0:
            eps_inc = -0.1 * eps_inc
        elif (h_2j - h_2i) / h_2i < 0 and eps_inc < 0:
            eps_inc = -0.1 * eps_inc
        iter += 1

    if iter >= max_iter:
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,]
    else:
        h_2 = h_2j
        rho_2 = rho / eps
        P_2 = P + (rho * ((u ** 2) * (1 - eps)))
        [T_2,_, gamma] = composition_curve_fit(P_2, rho_2, 2)
        u_2 = u * eps
        h_2 = h_2 / 1000; # to return as kJ/kg
        Cp = R_air * (gamma / (gamma - 1))
        return [P_2, h_2, T_2, rho_2, u_2, Cp]

def oblique_shock_angle(theta_c: np.array, alt:np.array, M:np.array, u_1: float, gamma: float=1.4, show_output: bool=False):
    beta = np.array([])

    for idx, theta in enumerate(np.deg2rad(theta_c)):
        def f(B):
            [_, _, _, _, u_2n, _] = equilibrium_shock(alt=alt, M=M, beta=np.array([B]))
            u_2 = u_2n[0]/np.sin(B)
            return (np.tan(B - theta)/np.tan(B)) - (u_2/u_1)
        B = np.deg2rad(theta)
        B = optimize.root(f, [B], method='hybr')
        sol = np.nan
        if B.success:
            sol = np.rad2deg(B.x)
        else:
            if show_output:
                print("No Solution at point ", idx)
        beta = np.append(beta, sol)
        
    return beta

if __name__ == "__main__":
    # HW 1 Question 2
    # Assumes equilibrium flow
    M = 10.
    alt = 40000 / 3.281 / 1000 # ft to km
    beta = 25.52 # deg
    # Testing: X-43 example from class
    [P_2, h_2, T_2, rho_2, u_2] = equilibrium_shock(np.array([33.5]), np.array([9.64]))
    print(P_2, h_2, T_2, rho_2, u_2)
    # part a: normal shock
    #[P_2n, h_2n, T_2n, rho_2n, u_2n] = equilibrium_shock(np.array([alt]), np.array([M]))
    # part b: oblique shock
    #[P_2o, h_2o, T_2o, rho_2o, u_2o] = equilibrium_shock(np.array([alt]), np.array([M]), np.array([beta]))
