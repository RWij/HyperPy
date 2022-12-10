import numpy as np

def fay_riddell_CPG(R_n: float,
                    T_w: float,
                    P_inf: float,
                    P_2: float,
                    rho_2: float,
                    h_o: float,
                    Cp: float,
                    mu_2: float,
                    Pr: float,
                    Le: float=1.0,
                    ):
    '''
    For a spherical nose in CPG air and fully catalytic wall
    For non-catalytical walls, the exponent for 'Le' is 0.63
    This function also assumes Le ~ 1.0, simplifying the Fay
    -Riddell equation.

    Returns
    -------
    None.

    '''
    h_oe = h_o*1000
    Cp *= 1000
    Cp_w = 1004.5
    h_w = Cp_w*T_w
    R = 8314/28.96
    rho_w = P_2 / (R*T_w)
    mu_w = 1.458E-6*((T_w**1.5)/(T_w+110.4))
    du_e_dx = ((2*(P_2-P_inf)/rho_2)**0.5)/R_n
    q_w = 0.76*(Pr**-0.6)*((rho_2*mu_2)**0.4)*((rho_w*mu_w)**0.1)*\
        (du_e_dx**0.5)*((h_oe-h_w)) #(h_oe-h_w)*(1-(((Le**0.52)-1)*(h_D/h_oe)))
    return q_w

def reference_temperature(x: float,
                          delta_c: float,
                          R_n: float,
                          R_b: float,
                          T_inf: float,
                          P_inf: float,
                          rho_inf: float,
                          T_2: float,
                          P_2: float,
                          rho_2: float,
                          u_2: float,
                          h_0: float,
                          h_2: float,
                          Cp_2: float,
                          T_w: float,
                          Cp_w:float=1004.5,
                          ):
    '''
    Uses the reference temperature method to calculate the stanton number
    and heat flux at a certain location along a blunted cone 

    Returns
    -------
    list
        DESCRIPTION.

    '''
    delta_c = np.deg2rad(delta_c)

    gamma_bare = (2.39683E-19*T_2**5)-(3.0436E-15*T_2**4)+\
        (8.89216E-12*T_2**3)+(2.77835E-8*T_2**2)-(1.46939E-4*T_2)+1.4517
    a_e = (gamma_bare * P_2 / rho_2)**0.5
    M_e =  u_2 / a_e

    T_star = T_2 * (1 + (0.032 * (M_e**2)) + 0.58*((T_w/T_2)-1))
    Cp_star = (8.31935E-12*(T_star**3))-(8.62469E-8*(T_star**2))+\
        (3.14166E-4*T_star)+0.901439
    Cp_star *= 1000
    P_star = P_2
    rho_star = P_star / (287. * T_star)
    mu_star = 1.458E-6 * ((T_star**1.5)/(T_star + 110.4))

    K_star = 1.993E-3 * ((T_star**1.5)/(T_star + 112))

    X1 = R_n * (1 - np.sin(delta_c))
    Y1 = R_n * np.cos(delta_c)
    y = Y1 + (np.tan(delta_c) * (x-X1))
    theta = (np.pi/2) - delta_c
    S = (R_n * theta) + ((x-X1)**2 + (y-Y1)**2)**0.5

    Re_starx = rho_star * u_2 * S/ mu_star
    Pr_star = mu_star * Cp_star / K_star

    mu_edge = 1.458E-6 * ((T_2**1.5)/(T_2 + 110.4))
    K_edge = 1.993E-3 * ((T_2**1.5)/(T_2 + 112))
    Pr_edge = mu_edge * Cp_2 / K_edge

    r = C_f = 0
    if Re_starx < 1E5:
        # laminar
        m_f = (3)**0.5
        r = (Pr_edge)**0.5
        C_f = 0.664 * m_f / ((Re_starx)**0.5)
    else:
        # turbulent
        m_f = (2)**0.5
        r = (Pr_edge)**(1/3)
        C_f = 0.0592 * m_f / ((Re_starx)**0.2)


    H_r = h_2 + (r * (u_2**2)/2)
    H_w = Cp_w * T_w

    C_H = 0.5 * C_f * (Pr_star**(-2/3))
    rhoe_ue_CH = rho_2 * u_2 * C_H
    q = rhoe_ue_CH * (H_r - H_w)

    return [S, C_H, q, C_f, Re_starx, rhoe_ue_CH, T_star, Pr_star, H_r]

def discretized_reference_temperature(geom: np.array,
                          delta_c: float,
                          R_n: float,
                          R_b: float,
                          T_inf: float,
                          P_inf: float,
                          rho_inf: float,
                          T_2: np.array,
                          P_2: np.array,
                          rho_2: np.array,
                          u_2: np.array,
                          h_0: float,
                          h_2: np.array,
                          Cp_2: np.array,
                          T_w: float,
                          Cp_w:float=1004.5,
                          ):
    '''
    Uses the reference temperature method to calculate the stanton number
    and heat flux for a discretized geometry along the entire shape axially

    Returns
    -------
    list
        DESCRIPTION.

    '''
    delta_c = np.deg2rad(delta_c)

    gamma_bare = (2.39683E-19*T_2**5)-(3.0436E-15*T_2**4)+\
        (8.89216E-12*T_2**3)+(2.77835E-8*T_2**2)-(1.46939E-4*T_2)+1.4517
    a_e = (gamma_bare * P_2 / rho_2)**0.5
    M_e =  u_2 / a_e

    T_star = T_2 * (1 + (0.032 * (M_e**2)) + 0.58*((T_w/T_2)-1))
    Cp_star = (8.31935E-12*(T_star**3))-(8.62469E-8*(T_star**2))+\
        (3.14166E-4*T_star)+0.901439
    Cp_star *= 1000
    P_star = P_2
    rho_star = P_star / (287. * T_star)
    mu_star = 1.458E-6 * ((T_star**1.5)/(T_star + 110.4))

    K_star = 1.993E-3 * ((T_star**1.5)/(T_star + 112))

    theta = (np.pi/2) - delta_c

    X = geom[:,0]; Y = geom[:,1]
    lastidx = len(X)-1
    X_prev = np.delete(X, (lastidx), axis=0)
    X_curr = np.delete(X, (0), axis=0)
    Y_prev = np.delete(Y, (lastidx), axis=0)
    Y_curr = np.delete(Y, (0), axis=0)
    S = np.array([R_n * theta])
    S = np.append(S, (R_n * theta) + np.cumsum(np.sqrt((X_curr - X_prev)**2 + (Y_curr - Y_prev)**2)))

    Re_starx = rho_star * u_2 * S/ mu_star
    Pr_star = mu_star * Cp_star / K_star

    mu_edge = 1.458E-6 * ((T_2**1.5)/(T_2 + 110.4))
    K_edge = 1.993E-3 * ((T_2**1.5)/(T_2 + 112))
    Pr_edge = mu_edge * Cp_2 / K_edge

    r = C_f = np.array([])
    for idx, REs in enumerate(Re_starx):
        _r = _C_f = np.nan
        _Pr_edge = Pr_edge[idx]
        if REs < 1E5:
            # laminar
            m_f = (3)**0.5
            _r = (_Pr_edge)**0.5
            _C_f = 0.664 * m_f / ((REs)**0.5)
        else:
            # turbulent
            m_f = (2)**0.5
            _r = (_Pr_edge)**(1/3)
            _C_f = 0.0592 * m_f / ((REs)**0.2)
        C_f = np.append(C_f, _C_f)
        r = np.append(r, _r)

    H_r = h_2 + (r * (u_2**2)/2)
    H_w = Cp_w * T_w

    C_H = 0.5 * C_f * (Pr_star**(-2/3))
    rhoe_ue_CH = rho_2 * u_2 * C_H
    q = rhoe_ue_CH * (H_r - H_w)

    return [S, C_H, q, C_f, Re_starx, rhoe_ue_CH, T_star, Pr_star, H_r]

if __name__ == "__main__":
    # Question 1
    M=8.0
    a=(1.4*287*216.5)**0.5
    q_w=fay_riddell_CPG(
        R_n=0.0381,         # m
        T_w=300,            # K
        P_inf=5694.5,       # Pa
        P_2=4.3283*101325,  # Pa
        rho_2=0.617686,     #kg/m^3
        h_o=2943.43,        #kJ/kg
        Cp=1.2684,          #kJ/kg*K
        mu_2=7.5522E-5,     #kg/m*s
        Pr=0.9259,          #dimensionless
        )
    '''
    # values from lecture example
    q_w=fay_riddell_CPG(R_n=0.3048,         # m
                        T_w=1500,            # K
                        T_inf=212.,        # K
                        P_inf=3.202,       # Pa
                        rho_inf=9.2878E-4/17.71738,   #kg/m^3
                        T_2=5898.98,         #K
                        P_2=2650.18,  # Pa
                        rho_2=9.2878E-4,     #kg/m^3
                        h_o=27280.66,        #kJ/kg
                        h_2=27280.66,       #kJ/kg
                        Cp=1.5591,          #kJ/kg*K
                        mu_2=1.5636E-4,     #kg/m*s
                        Pr=0.7935,        #dimensionless
                        u_e=7316.2)             # m/s
    '''
    print("Question 1")
    print("q_w: ", round(q_w/(100**2),3), " W/cm**2")
    print("")

    # Question 2
    [S, C_H, q, C_f, Re_starx, rhoe_ue_CH, T_star, Pr_star, H_r] = reference_temperature(x=0.4, # m
                          delta_c=35.0, # deg
                          R_n=0.0381, # m
                          R_b=0.329, # m
                          T_inf=216.5, # K
                          P_inf=5694.5, # Pa
                          rho_inf=0.091566, # kg/m^3
                          T_2=1434.29, # K
                          P_2=2.1206*101325, # Pa
                          rho_2=0.52148,    # kg/m^3
                          u_2=1696.43,  # m/s
                          h_0=2943.43*1000,  # J/kg
                          h_2=1564.517*1000,    # J/kg
                          Cp_2=1.2047*1000,  #J/kg*K 
                          T_w=300,  # K
                          )

    '''
    # from lecture example:
    [C_H, q_w] = reference_temperature(x=2.7065, # m
                          delta_c=21.3, # deg
                          R_n=0.725, # m
                          R_b=2.6, # m
                          T_inf=198.6, # K
                          P_inf=1.0521*101325, # Pa
                          rho_inf=1.8452E-3, # kg/m^3
                          T_2=1054.9, # K
                          P_2=29.28, # Pa
                          rho_2=9.6344E-5,    # kg/m^3
                          u_2=2480.2,  # m/s
                          h_0=4190.1*1000,  # J/kg
                          h_2=1114.4*1000,    # J/kg
                          Cp_2=301350/300,  #J/kg*K
                          T_w=300,  # K
                          )
    '''
    print("Question 2")
    print("C_H: ", round(C_H,7))
    print("q_w: ", round(q/(100**2),3), " W/cm**2")
    print("")

