# TODO make sure all use numpy instead of math
import math

def sutherlands_law_air(P: float, T: float):
    mw_ref = mw_o2 = 31.998
    sigma_ref = sigma_o2 = 3.433  # angstroms
    mw_o = 15.999
    sigma_o = 2.67
    omega_ij = 0.6235

    u = 1.458e-6 * (math.pow(T, 1.5) / (T + 110.4))
    k = 1.993e-3 * (math.pow(T, 1.5) / (T + 112))

    D_bar = 2.628e-3 * T * math.sqrt(T / mw_ref) / (P * (sigma_ref**2) * omega_ij)
    F_i = math.pow(mw_o / 26, 0.461)
    F_j = math.pow(mw_o2 / 26, 0.461)
    D = D_bar / (F_i * F_j)

    return [u, k, D]

def omega_K_and_N(Tref:float):
    omega_K = omega_N = (1.16145 / (Tref ** 0.14874)) + (0.52487 / math.exp(0.77320*Tref)) + (2.16178/math.exp(2.48787*Tref))
    return [omega_K, omega_N]

def omega_Dij(Tref:float):
    omega_Dij = (1.06036 / (Tref ** 0.15610)) + (0.1930 / math.exp(0.47635*Tref))\
        + (1.03587/math.exp(1.52996*Tref))+(1.76474/math.exp(3.89411*Tref))
    return omega_Dij

def phi_ij(mw_i:float, u_i:float, mw_j:float, u_j:float):
    a = 1/(8**0.5)
    b = (1 + (mw_i / mw_j))**-0.5
    c = (u_i / u_j)**0.5
    d = (mw_j / mw_i)**0.25
    e = (1 + (c * d))**2
    return  a * b * e

def D_ij(P:float, T:float, mw_A:float, mw_B:float, sigma_AB:float, ohm_D:float):
    return 0.0018583 * math.sqrt((T**3) * ((1/mw_A) + (1/mw_B))) / (P * (sigma_AB**2) * ohm_D)

def kinetic_gas_theory(P: float, T: float, species_model_dict: dict):
    # applying Wilke's Rule
    u = k = 0
    D_im = {}
    Dij = []
    species_model_dict_keys = list(species_model_dict.keys())
    for i in range(len(species_model_dict_keys)):
        species = species_model_dict_keys[i]

        # initialize for each species
        k_i = 0
        X_i = species_model_dict[species]['mf']
        mw_i = species_model_dict[species]['mw']
        Cp_i = species_model_dict[species]['Cp']
        sigma_i = species_model_dict[species]['sigma']
        Tstar_i = T / species_model_dict[species]['e_k']

        [omega_k, omega_N] = omega_K_and_N(Tstar_i)
        # species viscosity
        u_i = (2.6693e-5) * (math.sqrt(mw_i * T) / ((sigma_i ** 2) * omega_N))
        # species thermal conductivity
        if len(species) < 2: # if monotonic
            k_i = ((1.9891e-4) * math.sqrt(T / mw_i) / ((sigma_i ** 2) * omega_k)) * 4.186 * 100
        else: # if non-monotonic gas
            k_i = (u_i/10) * (Cp_i + ((5/4)*8314/mw_i))

        # 1. compute Xi*ui
        X_u = X_i * u_i
        X_k = X_i * k_i

        X_eps = Tstar_j = D_ij_ = D_im_denom = 0.0
        species_Dij_list = []
        for j in range(len(species_model_dict_keys)):
            # 2. compute eps_ij for 2 species (4 values)
            speciesj = species_model_dict_keys[j]

            X_j = species_model_dict[speciesj]['mf']
            mw_j = species_model_dict[speciesj]['mw']
            sigma_j = species_model_dict[speciesj]['sigma']
            Tstar_j = T / species_model_dict[speciesj]['e_k']
            [_, omega_Nj] = omega_K_and_N(Tstar_j)

            u_j = 2.6693e-5 * (math.sqrt(mw_j * T) / ((sigma_j ** 2) * omega_Nj)) #/ 10

            phi_ij_ = phi_ij(mw_i, u_i, mw_j, u_j)
            omega_Di = omega_Dij(Tstar_i); omega_Dj = omega_Dij(Tstar_j)
            omega_D = math.sqrt(omega_Di * omega_Dj)
            D_ij_ = D_ij(P, T, mw_i, mw_j, (sigma_i + sigma_j)/2, omega_D)

            # by-products of inner loop:
            X_eps += X_j * phi_ij_
            D_im_denom += X_j / D_ij_

            species_Dij_list.append(D_ij_)

        # overall viscosity
        u += X_u / X_eps
        # overall thermal conductivity
        k += X_k / X_eps
        # average diffusion of species through mixture
        D_im_i = (1 - X_i) / (D_im_denom)
        D_im[species] = D_im_i

        Dij.append(species_Dij_list)

    return [u/10, k, D_im, Dij]

if __name__ == "__main__":
    # HW3 Question 4
    ## for Testing ##
    #P: float = 0.35 #atm 
    #T: float = 4500 #K 
    ## for testing ##

    P: float = 0.10 # atm
    T: float = 5500.0 # K
    species_model = {
        'N2': {'mf':0.47158,   'Cp':(37.157/28.014)*1000, 'mw':28.014, 'sigma': 3.798,\
               'e_k': 71.4},
        'O2': {'mf':5.6911e-5, 'Cp':(39.848/31.998)*1000, 'mw':31.998, 'sigma': 3.467,\
               'e_k': 106.7},
        'N' : {'mf':0.215627,  'Cp':(24.461/14.007)*1000, 'mw':14.007,  'sigma': 3.298,\
               'e_k': 71.4},
        'O' : {'mf':0.30919,   'Cp':(22.044/15.999)*1000, 'mw':15.999,  'sigma': 3.05,\
               'e_k': 106.7},
        'NO': {'mf':0.0032146, 'Cp':(37.247/30.006)*1000, 'mw':30.006,  'sigma': 3.492,\
               'e_k': 116.7},
    }

    print("")
    print("For question 4:")
    [u_suther, k_suther, D_suther] = sutherlands_law_air(P, T)
    print("From simple model: ")
    # only this is agrees:
    print("u: ", u_suther, " kg/m*s")

    # other values below disagrees:
    print("k: ", k_suther, " W/m*k")
    print("D: ", D_suther, " cm**2/s")
    print("")

    [u_kgt, k_kgt, D_im, Dij] = kinetic_gas_theory(P, T, species_model)
    print("From sophisticated model: ")
    print("u: ", u_kgt, " kg/m*s")
    print("k: ", k_kgt, " W/m*k")
    print("Dim: ", D_im, " cm**2/s")
    print("\nDij matrix: ")
    print(Dij)

    print("\nDifference between sophisticated and simple models")
    u_err = abs((u_kgt - u_suther) / u_kgt) * 100
    k_err = abs((k_kgt - k_suther) / k_kgt) * 100
    print("u: ", u_err, " %")
    print("k: ", k_err, " %")

