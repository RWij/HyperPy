# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:05:12 2022

@author: ryant
"""
import numpy as np

def sharp_cone_coefficients(theta:np.array, angle:np.array):
    '''
     Calculates the Newtonian aerodynamic coefficients and lift and
     drag coefficients for a sharp cone unp.sing its cone angle, 
     base diameter, and angle of attack.
     Assumes a 2-D, equilibrium, attached shock, Mach > 5 flow.
    
     Parameters:
     -----------
    
     Inputs:
       theta - 1xn vector of body angle(s) (deg.)
       angle - 1xn vector of angle(s) of attack (deg.)
    
     Outputs:
       CN - 1xn vector of Normal Force Coefficient(s)
       CA - 1xn vector of Axial Force Coefficient(s)
       CL - 1xn vector of Lift Coefficient(s)
       CD - 1xn vector of Drag Coefficient(s)
    '''
    
    # Newtonian flow, attached shock
    Cpmax = 2.0
    
    theta = np.deg2rad(theta)
    angle = np.deg2rad(angle)

    CN = Cpmax * ((np.cos(theta) ** 2) * np.sin(angle) * np.cos(angle))
    CA = (Cpmax * (np.sin(theta) ** 2) * (np.cos(angle) ** 2)) + (0.5 * Cpmax * (np.cos(theta) ** 2) * (np.sin(angle) ** 2))

    CL = (CN * np.cos(angle)) - (CA * np.sin(angle))
    CD = (CN * np.sin(angle)) + (CA * np.cos(angle))

    return [CN, CA, CL, CD]

def blunted_biconical_coefficients(geom:np.array, aoas:np.array, eps:np.array, rn: float):
    # 
    # Parameters:
    # -----------
    # INPUTS:
    # geom - MxN matrix of the biconical geometry (m, degrees)
    #        beginning/middle row:   cone radius, cone angle
    #        ending row:             base radius, nan
    # aoa - Mx1 matrix of angles of attack (degrees)
    #
    # OUTPUTS:
    # L_D - 1xM Matrix of L/D
    # CL - 1xM Matrix of Lift Coefficients
    # CD - 1xM Matrix of Drag Coefficients
    # CN_all - 1xM Matrix of Normal Coefficients
    # CA_all - 1XM Matrix of Axial Coefficients

    CL = CD = L_D = CN_all = CA_all = np.array([])

    thetac = np.deg2rad(geom[:,1])
    r = geom[:,0]

    lastidx = len(geom)-1
    angles = np.deg2rad(aoas)
    thetac1 = thetac[1]
    thetac = thetac[1:lastidx+1]
    rb = geom[lastidx,0]
    bp = rn / rb # bluntness parameter
    Cpmax = np.array([])
    if eps.size == 0:
        Cpmax = 2.0 * np.ones(len(geom))
    else:
        Cpmax = 2.0 - eps

    rprev = np.delete(r, (lastidx), axis=0)
    rcurr = np.delete(r, (0), axis=0)

    # Nose CA and CN
    Cpmax_nose = Cpmax[0]
    Cpmax = np.delete(Cpmax, (0), axis=0)

    CAnose = 0.5 * Cpmax_nose * (1 - (np.sin(thetac1)**4))
    CN = 0

    for aoa in angles:
        CA = ((bp**2) * CAnose)
        CN = 0
        # body CA and CN along each point
        CAi = (1 - ((rprev/rcurr)**2)*(np.cos(thetac)**2)) * ((Cpmax*(np.sin(thetac)**2)*(np.cos(aoa)**2))+\
            (0.5*Cpmax*(np.cos(thetac)**2)*(np.sin(aoa)**2)))
        CNi = Cpmax * (1 - ((rprev/rcurr)**2)*(np.cos(thetac)**2)) * (np.cos(thetac)**2) * np.sin(aoa) * np.cos(aoa)
        
        # total CA and CN
        CA += np.sum(CAi * ((rcurr**2) - (rprev**2)) / (rb**2))
        CN += np.sum(CNi * ((rcurr**2) - (rprev**2)) / (rb**2))
            
        current_CL = (CN * np.cos(aoa)) - (CA * np.sin(aoa))
        current_CD = (CN * np.sin(aoa)) + (CA * np.cos(aoa))

        CL = np.append(CL,current_CL)
        CD = np.append(CD,current_CD)
        L_D = np.append(L_D, current_CL/current_CD)
        CN_all = np.append(CN_all, CN)
        CA_all = np.append(CA_all, CA)

    return [L_D, CL, CD, CN_all, CA_all, Cpmax]

    
def blunt_cone_lift_over_drag(theta, aoa, bp):
    '''
    function [LD, LDmax, aoaLDmax] = LDbluntcone(aoas, theta, blunt)
    #
    # Parameters:
    # -----------
    # INPUTS:
    # aoas - 1xn vector of angles of attack (deg)
    # theta - current body angle (deg)
    # blunt - 1xn vector of bluntness parameters
    #
    # OUTPUTS:
    # LD - 1xn vector of lift-over-drags with each row for each aoa and
    #       each column for each bluntness parameters
    LD = zeros(length(aoas), length(blunt));
    rb = 1.0;
    rn = blunt;
    LDmax = zeros(1, length(blunt));
    aoaLDmax = zeros(1, length(blunt));
    
    for idx = 1:length(aoas)
        aoa = aoas(idx);        

        N1 = (cosd(aoa)**2) * cosd(deg2rad(2)*theta);
        N2 = (rn**2)-(2*(rb**2))+((rn**2)*cos(2*deg2rad(theta)));
        N3 = (rb**2)*(cosd(theta)**2)*(sind(aoa)**2);
        N4 = (rn**2)*(cosd(theta)**4)*(sind(aoa)**2);
        N5 = (rn**2)*((sind(theta)**4)-1);
    
        D1 = 3*(rn**3)*(cosd(theta)**4)*(sind(theta)**2);
        D2 = (rb**2)*(cosd(aoa)**2)*(sind(theta)**2);
        D3 = 2*(rn**2)*(cosd(aoa)**2)*(sind(theta)**2);
        D4 = 3*(rb**2)*(sind(aoa)**2);
        D5 = N5;

        LD(idx,:) = tand(aoa)*(N1*N2+N3-N4-N5)/(D1-D2+(cosd(theta)**2)*(D3-D4)+D5);
    end
    # find the max LD and aoa for each bluntness parameter
    for idx = 1:length(blunt)
        cLDmax = max(LD(:,idx));
        [row, ~] = find(LD(:,idx)==cLDmax, 1, 'first');
        caoaLDmax = aoas(row);
    
        LDmax(idx) = cLDmax;
        aoaLDmax(idx) = caoaLDmax;
    '''
    None
    
def blunt_cone_small_angle_approximation(theta, blunt):
    # Calculates the max L/D and the angle of attack it happens via
    # small-angle approximation
    # Parameters:
    # -----------
    # INPUTS:
    # theta     - 1xn vector of body angles (deg)
    # blunt     - 1xn vector of bluntness parameters
    #
    # OUTPUTS:
    # aoa_LDmax    - 1xn vector of aoa at max L/D
    # LDmax     - 1xn vector of max L/D

    # equation 42 from notes
    aoa_LDmax = np.sqrt(((blunt**2) + (2*(theta**2)*(1 - (blunt**2)))) / (3 * (1 - (blunt**2))));
    # equation 43 from notes
    LDmax = np.sqrt(((1-(blunt**2))*((1-(theta**2))**2)) / ((3*(blunt**2))+(6*(theta**2)*((1-blunt)**2))));

    return [aoa_LDmax, LDmax]

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # HW 1 Question 3
    '''
    aoa_min = 0
    aoa_max = 20
    aoa = np.linspace(aoa_min, aoa_max,10);
    cone_20_deg = (20.0);
    cone_60_deg = (60.0);
    
    [cn_20, ca_20, cl_20, cd_20] = sharp_cone_coefficients(cone_20_deg, aoa)
    [cn_60, ca_60, cl_60, cd_60] = sharp_cone_coefficients(cone_60_deg, aoa)
    
    fig = plt.figure(figsize=(10,6))
    ax_cl= fig.add_subplot(1, 2, 1)
    ax_cl.plot(aoa, cl_20, aoa, cl_60)
    ax_cl.set_title('CL vs. AOA')
    ax_cl.set_xlabel('AOA (deg.)')
    ax_cl.set_ylabel('CL')
    ax_cl.legend(['20 Deg.', '60 Deg.'])
    ax_cl.grid(True)
    
    ax_cd = fig.add_subplot(1, 2, 2)
    ax_cd.plot(aoa, cd_20, aoa, cd_60)
    ax_cd.set_title('CD vs. AOA')
    ax_cd.set_xlabel('AOA (deg.)')
    ax_cd.set_ylabel('CD')
    ax_cd.legend(['20 Deg.', '60 Deg.'])
    ax_cd.grid(True)
    
    plt.show()
    '''

    # HW 2 Question 1
    print("for 3, 6, 9 degrees aoa top to bottom")
    geom = np.array([[1.5, 20.0], [2.85, 10.0], [3.605, 0.0]]);
    [L_D, CL, CD] = blunted_biconical_coefficients(geom, np.array([3., 6., 9.]))
    print("For Blunted Biconical")
    print("L/D: ", L_D)
    print("Cl: ", CL)
    print("CD: ", CD)
    
    # HW 2 Question 2
    '''
    disp("For QUESTION 2")
    disp("NOTE: Each column in each output is for each of the bluntness parameters")
    disp("in order list below:")
    blunt = [0 0.2 0.5 0.75];
    aoa = linspace(-20,60,50);
    
    # part a
    theta = 10.0;
    disp("Derivative method for 10 deg.:")
    [LD, LDmax, aoaLDmax] = LDbluntcone(aoa, theta, blunt);
    LDmax
    aoaLDmax
    disp("Small angle approximation for 10 deg.: ")
    [aoa_10, LD_10] = smallanglebluntcone(theta, blunt)
    
    figure;
    plot(aoa, LD(:,1))
    hold on
    xlabel("AOA (deg.)")
    ylabel("L/D")
    title("10 deg")
    plot(aoa, LD(:,2))
    plot(aoa, LD(:,3))
    plot(aoa, LD(:,4))
    lgd_cl = legend('0.0', '0.2', '0.5', '0.75');
    lgd_cl.Location = 'best';
    grid on
    hold off
    
    # part b
    theta = 20.0;
    disp("Derivative method for 20 deg.:")
    [LD, LDmax, aoaLDmax] = LDbluntcone(aoa, theta, blunt);
    LDmax
    aoaLDmax
    disp("Small angle approximation for 20 deg.: ")
    [aoa_20, LD_20] = smallanglebluntcone(theta, blunt)
    
    figure;
    plot(aoa, LD(:,1))
    hold on
    xlabel("AOA (deg.)")
    ylabel("L/D")
    title("20 deg")
    plot(aoa, LD(:,2))
    plot(aoa, LD(:,3))
    plot(aoa, LD(:,4))
    lgd_cl = legend('0.0', '0.2', '0.5', '0.75');
    lgd_cl.Location = 'best';
    grid on
    hold off
    
    # part c
    theta = 45.0;
    disp("Derivative method for 45 deg.:")
    [LD, LDmax, aoaLDmax] = LDbluntcone(aoa, theta, blunt);
    LDmax
    aoaLDmax
    disp("Small angle approximation for 45 deg.: ")
    [aoa_45, LD_45] = smallanglebluntcone(theta, blunt)
    
    figure;
    plot(aoa, LD(:,1))
    hold on
    xlabel("AOA (deg.)")
    ylabel("L/D")
    title("45 deg")
    plot(aoa, LD(:,2))
    plot(aoa, LD(:,3))
    plot(aoa, LD(:,4))
    lgd_cl = legend('0.0', '0.2', '0.5', '0.75');
    lgd_cl.Location = 'best';
    grid on
    hold off
    
    disp("small angle approx breaks down at body angles and/or bluntness ratios")
    disp("based on function outputs, it seems that the approximations breakdown")
    disp("at a body angle of ~30 degrees and at bluntness parameters of ~0.6")
    
    # part d
    disp("I believe the team lead gave the correct approach since not all the cone designs")
    disp("don't follow the limits of the small angle approximation.")
    disp("#############################################################")
    '''