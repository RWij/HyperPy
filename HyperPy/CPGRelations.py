import numpy as np

from StandardAtmosphere1962 import standard_atmosphere_1962, altitude

def rho_ratio(M:float, gamma:float=1.4):
    return ((gamma + 1) * (M**2)) / (2 + (gamma - 1) * (M**2))

def pressure_ratio(M: float, gamma:float=1.4):
    return 1 + (((2 * gamma) / (gamma + 1)) * ((M**2) - 1))

def M2(M: float, gamma:float=1.4):
    a = 1 + (((gamma - 1)/2) * (M **2))
    b = (gamma * (M ** 2)) - ((gamma - 1) / 2)
    return np.sqrt(a / b)

def CPG(alt: np.array, M: np.array, beta: np.array=None):
    Cp_air = 1004.5
    valt = np.array(altitude(alt,'p')) # from geometric to geopotential  
    [T_inf, P_inf, rho_inf, R] = standard_atmosphere_1962(np.array([valt]))

    if beta is not None:
        M = M * np.sin(np.deg2rad(beta))

    h_1 = Cp_air * T_inf
    rho2_1 = rho_ratio(M)
    P2_1 = pressure_ratio(M)
    M_2 = M2(M)

    T2_1 = P2_1/rho2_1
    T_2 = T2_1 * T_inf

    rho_2 = rho2_1 * rho_inf
    P_2 = P2_1 * P_inf
    a_2 = np.sqrt(1.4 * R * 1000 * T_2)
    u_2 = a_2 * M_2
    h_2 = T2_1 * h_1

    return [P_2, h_2/1000, T_2, rho_2, u_2, Cp_air]