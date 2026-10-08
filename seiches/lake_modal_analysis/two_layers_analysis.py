import pandas as pd
import numpy as np
import os

import pylake

def compute_two_layers_analysis(config, date):
    stratification = pd.read_csv(config['stratification_csv']).set_index('Z')[date]

    lat0 = config.get('lat0', None)
    if lat0 == 'None' or lat0 is None:
        lat_grid = np.load(os.path.join(config['mitgcm_grid_folder'], 'lat_grid.npy'))
        lat0 = lat_grid[0,0]
        print(f'Using latitude from MITgcm grid: {lat0:.2f} degrees')

    lat_rad = np.deg2rad(lat0)# Latitude of Lake Geneva (degrees)
    f = 2 * config['omega'] * np.sin(lat_rad) # Coriolis parameter
    f0 = abs(f/(2*np.pi)) # conversion from rad/s to 1/s

    rho = pylake.water_density(S=0.2, T=stratification.values[:-1])
    thermocline_depth, thermocline_idx = pylake.thermocline(stratification.values[:-1], -1*stratification.index.values[:-1])

    dz = pd.read_csv(os.path.join(config['mitgcm_grid_folder'], 'dz.csv'), header=None).to_numpy().flatten()
    rho1 = np.average(rho[:thermocline_idx], weights=dz[:thermocline_idx])
    rho2 = np.average(rho[thermocline_idx:], weights=dz[thermocline_idx:-1])

    H_mean = 80 #geneva: 150
    h1 = np.array(thermocline_depth)
    h2 = H_mean - h1

    g_prime = config['g']*(rho2-rho1)/rho2
    H_bar = h1*h2/(h1+h2)
    c_phase = np.sqrt(g_prime*H_bar)
    T_iw = 2*config['Lx']/c_phase
    T_iw_in_hour = T_iw / 3600
    Rd = c_phase / f0
    Bu = (Rd / config['Lx'])**2

    print(f'Thermocline depth  = {thermocline_depth:.2f} m')
    print(f'Upper layer density rho1 = {rho1:.2f} kg/m3')
    print(f'Lower layer density rho2 = {rho2:.2f} kg/m3')
    print(f'Rossby radius of deformation = {Rd:.2f} m2/s')
    print(f'Wave phase velocity = {c_phase:.2f} m/s')
    print(f'Wave period = {T_iw_in_hour:.2f} hours')
    print(f'Burger Number = {Bu:.2f} m2/s2')

    return {
        'thermocline_depth': thermocline_depth,
        'rho1': rho1,
        'rho2': rho2,
        'c_phase':c_phase,
        'T_iw_in_hour': T_iw_in_hour,
        'Rd': Rd,
        'Bu': Bu,
    }