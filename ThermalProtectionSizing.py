# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:05:25 2022

@author: ryant
"""

import numpy as np
from HeatTransfer import semi_inf_solid_approx_const_heat_flux

def nominal_required_thickness_TPS(t_small: float,\
                                   t_large: float,\
                                   T_bl: float,\
                                   time: float,\
                                   k: float,\
                                   q: float,\
                                   alpha: float,\
                                   Ti: float=293,\
                                   tdf: float=1.20,\
                                   ahu: float = 1.00,\
                                   tol: float=1E-6,\
                                   iter_max: float=1000):
    '''
    Uses the "0th-order" approach for thermal protection
    system sizing which assumes a semi-infinite solid
    approximation.
    This method neglects recession, decomposition, and
    pyrolysis

    PARAMETERS:
    INPUT:
    ------
    t_small: float - smallest guessed thickness (m)
    t_large: float - largest guess thickness (m)
    T_bl: float - bondline temperature limit (K)
    time: float - Time heat flux is applied (sec)
    k: float - thermal conductivity (W/m*k)
    q: float - heat flux (W/m^2)
    alpha: float = thermal diffusivity
    Ti: float - Initial temperature (K) (room temp.)
    tdf: float - trajectory dispersion factor
    tol: float - bisection toleration
    iter_max: float - max iterations

    OUTPUT:
    --------
    t: float - nominal thickness (m)
    '''
    q = q * tdf * ahu
    t = np.linspace(t_small, t_large, iter_max)
    dT = np.absolute((T_bl - semi_inf_solid_approx_const_heat_flux(x=t,\
                                          t=time,\
                                          k=k,\
                                          alpha=alpha,\
                                          q_d2=q,\
                                          Ti=Ti)) - 0)
    min_dT = np.min(dT)
    min_dT_index = np.where(dT == min_dT)
    nom_thick = t[min_dT_index]
    return nom_thick[-1]

def margined_thickness_TPS(
    t_case1:float,\
    t_case2:float,\
    t_case3:float,\
    t_R1:float=0.0,\
    t_R2:float=0.0,\
    t_R3:float=0.0,\
    calc_reces_margin:bool=False,\
    rf: float=1.1,\
    ):
    '''
    Calculate the margined thickness of a thermal protection
    system using the semi-infinite solid approximation.
    Neglects recession

    Parameters:
    INPUT:
    ------
    t_case1: float - case 1 TPS thickness (nominal case) (m)
    t_case2: float - case 2 TPS thickness (aerodynamic heating uncertainty) (m)
    t_case3: float - case 3 TPS thickness (lower bondline temperature) (m)
    t_R1: float - case 1 recession thickness margin (m)
    t_R2: float - case 2 recession thickness margin (m)
    t_R3: float - case 3 recession thickness margin (m)
    calc_reces_margin: bool - Take into account thickness changes from recession (m)
    rf: float - recession factor

    OUTPUT:
    -------
    t_margin: float - Margined TPS thickness (m)
    '''
    t_b = t_case1 + np.sqrt((t_case2-t_case1)**2 + (t_case3-t_case1)**2)
    t_margin = rf * t_b
    if calc_reces_margin:
        delta_tR = np.sqrt((rf*t_R1 - t_R1)**2 + (rf*t_R2-t_R2)**2 + (rf*t_R3-t_R3)**2)
        t_margin = np.max(t_margin, t_b + delta_tR)

    return t_margin

def approximate_recession_TPS(rho_v: float,\
                              q: float,\
                              time: float,\
                              eps: float,\
                              B: float,\
                              H_r: float,\
                              T_w_limit: float=1500):
    '''
    Assumes steady-state conditions so q_rad_in = q_rad_out
    Also assumes convection radiation and recession rate of change
    are constant. Assume recover enthalpy >>>> wall enthalpy

    Parameters:
    INPUT:
    ------
    OUTPUT:
    -------
    '''
    S = 0
    lamb = 0.5 # assume Turbulent
    T_w = (q / (5.67E-8 * eps))**0.25
    if T_w < T_w_limit:
        S = 0
    else:
        CH_CHo = np.log(1 + (2*lamb*B))/(2*lamb*B)
        rhoe_ue_CHo = q/H_r
        rhoe_ue_CHo = rhoe_ue_CHo*CH_CHo
        m_T = B * rhoe_ue_CHo
        S_dot = m_T / rho_v
        S = S_dot * time
    return S


if __name__ == "__main__":
    # Question 3
    print("Question 3")
    q = 350 * (100**2)  # constant, W/m**2
    t = 300. # Sec.
    rho = 0.265 * (100**3) / 1000 # kg/m**3
    Cp = 1625 # J/kg*K
    k = 0.963 # W/m*K
    H_r = 35000*1000 # J/kg 
    T_bl = 250 + 273.15 # K
    T_bl_lower = 195 + 273.15 # K
    eps = 0.9
    B = 0.156
    alpha = k / (rho * Cp)
    # part a
    nom_thick = nominal_required_thickness_TPS(t_small=2/100,\
                                t_large=130/100,\
                                T_bl=T_bl,\
                                time=t,\
                                alpha=alpha,\
                                k=k,\
                                q=q,\
                                tdf=1.15)
    print("a. Nominal   Thickness: ", round(nom_thick*100,3), " cm")

    # part b
    t_2 = nominal_required_thickness_TPS(t_small=2/100,\
                                t_large=130/100,\
                                T_bl=T_bl,\
                                time=t,\
                                k=k,\
                                q=q,\
                                alpha=alpha,\
                                tdf=1.15,\
                                ahu=1.25)
    #print("Case 2 Thickness: ", round(t_2*100,3), " cm")

    t_3 = nominal_required_thickness_TPS(t_small=2/100,\
                            t_large=130/100,\
                            T_bl=T_bl_lower,\
                            time=t,\
                            k=k,\
                            q=q,\
                            alpha=alpha,\
                            tdf=1.15)
    #print("Case 3 Thickness: ", round(t_3*100,3), " cm")

    margin_thick =  margined_thickness_TPS(t_case1=nom_thick,\
                                t_case2=t_2,\
                                t_case3=t_3)
    print("b. Margined  Thickness: ", round(margin_thick*100,3), " cm")

    # part c
    recess_amt = approximate_recession_TPS(rho_v=rho,\
                                q=q*1.15,\
                                time=t,\
                                eps=eps,\
                                B=B,\
                                H_r=H_r)
    print("c. Amount of Recession: ", round(recess_amt*100,4), " cm")
