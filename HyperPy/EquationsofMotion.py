# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:59:07 2022

@author: ryant
"""
import numpy as np

from StandardAtmosphere1962 import standard_atmosphere_1962, altitude

state = {
    'T': -99.9,         # thrust, kg
    'rho': -99.9,       # atmospheric density, kg/m**3
    'V': -99.9,         # flight velocity, m/s
    'g': -99.9,         # local earth's gravitational acceleration, m/s**2
    'go': -99.9,        # earth's gravitational acceleration, m/s**2
    'r': -99.9,         # earth's reference radius (m)
    'Aref': -99.9,      # reference area, m**2
    'm': -99.9,         # total mass, kg
    'epsilon': -99.9,   # angle b/w thrust and velocity vector, deg.
    'theta': -99.9,     # bank angle/longitude, deg.
    'phi': -99.9,       # latitude, deg.
    'psi': -99.9,       # heading angle (b/w local line of lat. and projection of V vector on the 
                        # local horizontal plane), deg.
    'gamma': -99.9,     # flight angle, deg.
    'omega': -99.9,     # earth's rotation speed, rad/s
    'CD': -99.9,        
    'CL': -99.9,
    'sigma': -99.9
    }

# all angles should be in radians

def vdot(state:dict):
    thrust_term = state['T'] * np.cos(state['epsilon']) / state['m']
    aero_term = state['rho'] * (state['Velocity'] ** 2) * state['CD'] * state['Aref'] / (2 * state['m'])
    gravity_term = state['g'] * np.sin(state['gamma']) 
    rotation_term = (state['omega'] ** 2) * state['r'] * np.cos(state['phi'])
    threed_term = state['omega'] ** 2 * state['r'] * np.cos(state['phi'])\
        * ((np.sin(state['gamma']) * np.cos(state['phi']))\
           - (np.cos(state['gamma']) * np.sin(state['phi']) * np.sin(state['psi'])))
    return thrust_term - aero_term - gravity_term + rotation_term + threed_term

def gammadot(state:dict):
    thrust_term = (state['T'] * np.sin(state['epsilon']) * np.cos(state['sigma']))/(state['m']*state['Velocity'])
    lift_term = (state['rho'] * state['Velocity'] * state['CL'] * state['Aref'] * np.cos(state['sigma'])) / (2 * state['m'])
    velocity_term = (state['Velocity'] * np.cos(state['gamma'])) / state['r']
    gravity_term = state['g'] * np.cos(state['gamma']) / state['Velocity'] 
    rotation_term_1 = 2 * state['omega'] * np.cos(state['phi']) * np.cos(state['psi'])
    rotation_term_2 = (((state['omega']**2 * state['r'])/(state['Velocity']) * np.cos(state['phi']))) * ((np.cos(state['gamma'])\
        * np.cos(state['phi'])) * ( np.sin(state['gamma']) * np.sin(state['phi']) * np.sin(state['psi'])))
    return thrust_term + velocity_term + lift_term - gravity_term + rotation_term_1 + rotation_term_2
    
def hdot(state:dict):
    return state['Velocity'] * np.sin(state['gamma'])
    
def psidot(state:dict):
    thrust_term = (state['T'] * np.sin(state['epsilon']) * np.sin(state['sigma']))/(state['m']*state['Velocity'])
    lift_term = (state['rho'] * state['Velocity'] * state['CL'] * state['Aref'] * np.sin(state['sigma'])) / (2 * state['m'] * np.cos(state['gamma']))
    velocity_term = (state['Velocity'] * np.cos(state['gamma']) * np.cos(state['psi']) * np.tan(state['phi'])) / state['r']
    rotation_term_1 = 2 * state['omega'] * ((np.tan(state['gamma']) * np.cos(state['phi']) * np.sin(state['psi'])) - np.sin(state['phi']))
    rotation_term_2 = ((state['omega']**2 * state['r'])/(state['Velocity'] * np.cos(state['gamma']))) * np.sin(state['phi']) * np.cos(state['phi']) * np.cos(state['psi'])
    return thrust_term + lift_term - velocity_term + rotation_term_1 - rotation_term_2
    
def thetadot(state:dict):
    return state['Velocity'] * np.cos(state['gamma']) * np.cos(state['psi']) / (state['r'] * np.cos(state['phi']))
    
def phidot(state:dict):
    return state['Velocity'] * np.cos(state['gamma']) * np.sin(state['psi']) / (state['r'])
    
def massdot(state:dict):
    return -state['T'] / (state['Isp'] * state['go'])


if __name__ == "__main__":
    None
