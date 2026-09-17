# -*- encoding: utf-8 -*-
'''
@File    :    Jmatrix_cal.py
@Time    :    2023/11/02 10:01:10
@Author  :    ly
@Desc    :    None
'''
# %%
import sympy as sy
from Tab0_constants import *
from Tab1_ion_pos_cal import *
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root, newton, minimize

# %%
N_ions = 4 # number of ions
zfreq = 2*np.pi * np.array([0.422, 0.747, 1.052, 1.332]) * MHz # input the frequency measured by exp

result = minimize(Loss_function_axial, 
                  x0=np.array([-1/2 * mca * (2*np.pi*0.1*1e6)**2,1e-20, 0, 4e-4]),            # initial value
                  args=(N_ions, zfreq), tol=1e-30, method='Nelder-Mead',
                  bounds=((-1e-12, 1e-12), (-1e-18, 1e-18), (-1e-10, 1e-10), (-1e-3, 1e-3)))  # bound value

print(result.x)

zpos = equ_pos_fast(N_ions, *result.x)  # calculate ion's position

# plot ion's position and axial potential
zlist = np.linspace(1.2*float(min(np.array(zpos).flatten())), 1.2*float(max(np.array(zpos).flatten())), 200)
plt.plot(zpos*1e6,np.zeros(N_ions), "o")
plt.plot(1e6*zlist, result.x[1] * zlist + result.x[0] * zlist ** 2 + result.x[2] * zlist ** 3 + result.x[3] * zlist ** 4)
plt.show()

zfreq_cal, zmode, _ = axial_mode_spectrum(N_ions, zpos, *result.x) # calculate ion's axial frequency

# plot ion's axial frequency (exp and sim)
plt.vlines(zfreq/ (2 * np.pi * MHz), 0, 0.5, "C0", label='exp')
plt.vlines(zfreq_cal/ (2 * np.pi * MHz), 0.5, 1, "C1", label='sim')
plt.legend()
plt.show()

# plot ion's axial mode amplitude b(m, i) for each ion, m is motional index, i is ion's index
fig, ax = plt.subplots(figsize=(8, 8))
pcm = plt.matshow(zmode, fignum=0, cmap='bwr', vmax=0.7, vmin=-0.7)
plt.xlabel('mode index')
plt.ylabel('ion index')
fig.colorbar(pcm, fraction=0.04)

# %%
N_ions = 50
zpos_exp = np.array([-10.8902308366952, -10.321072010599867, -9.781369817698291, -9.319319766828963, -8.879323587792086, -8.495287010789031, -8.080808506123807, -7.700094326245581, -7.3085010523772755, -6.9194661659355825, -6.511272791269749, -6.132124031742321, -5.71478794406797, -5.325033048622722, -4.899809736381242, -4.466495136285614, -4.01468305584205, -3.5969521194302816, -3.092542007297724, -2.622308664895311, -2.1173088708055925, -1.5710389818226642, -1.0373786961124931, -0.49237625927067596, 0.08109629406260165, 0.6583451494563898, 1.1775819109140573, 1.751487768853598, 2.237810888287077, 2.776460369872228, 3.265555434433704, 3.721590543550136, 4.1618915541247645, 4.594267566752949, 4.970124065544661, 5.381906603055585, 5.788112700394946, 6.135585501538531, 6.5319849578163405, 6.876750312257411, 7.247858979150279, 7.628506881558993, 7.962309488071958, 8.340488783724926, 8.709817016659978, 9.112709991691471, 9.477323208970766, 9.918169657494078, 10.385639652878272, 10.949002913618514])/38*653*1e-6

result = minimize(Loss_function_position, 
                  x0=np.array([-1/2 * mca * (2*np.pi*0.1*1e6)**2,0, 0, 5e-7]),            # initial value
                  args=(N_ions, zpos_exp), tol=1e-30, method='Nelder-Mead',
                  bounds=((-1e-12, 1e-12), (-1e-18, 1e-18), (-1e-10, 1e-10), (-1e-3, 1e-3)))  # bound value

print(result.x)

zpos = equ_pos_fast(N_ions, *result.x)  # calculate ion's position

# %%
# result.x = np.array([-1/2 * mca * (2*np.pi*0.1*1e6)**2,0, 0, 5e-7])
zpos = equ_pos_fast(N_ions, *result.x)
plt.plot(zpos*1e6,np.zeros(N_ions), "o")
plt.plot(zpos_exp*1e6,np.ones(N_ions), "o")
print(Loss_function_position(result.x, N_ions, zpos_exp))
plt.show()

# %%
omega_y = 2*np.pi * 2.513 * MHz
yfreq_cal, ymode, _ = radial_mode_spectrum(N_ions, omega_y, zpos)
omega_x = 2*np.pi * 2.465 * MHz
xfreq_cal, xmode, _ = radial_mode_spectrum(N_ions, omega_x, zpos)
plt.vlines(xfreq_cal/ (2 * np.pi * MHz), 0, 0.5, "C0")
plt.vlines(yfreq_cal/ (2 * np.pi * MHz), 0.5, 1, "C1")
plt.show()
# plot ion's axial mode amplitude b(m, i) for each ion, m is motional index, i is ion's index
fig, ax = plt.subplots(figsize=(8, 8))
pcm = plt.matshow(ymode, fignum=0, cmap='bwr', vmax=0.7, vmin=-0.7)
plt.xlabel('mode index')
plt.ylabel('ion index')
fig.colorbar(pcm, fraction=0.04)
# %%
ionlist = [6, 7, 8, 9, 10]
modelist = []
for ii in range(N_ions):
   flag = 0
   for jj in ionlist:
      if np.abs(ymode[jj][ii]) > 0.1 and flag < 1:
         modelist.append(ii)
      else:
         continue

plt.vlines([xfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist], 0, 0.5, "C0")
plt.vlines([yfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist], 0.5, 1, "C1")
# %%
zfreq_cal, zmode, _ = axial_mode_spectrum(N_ions, zpos, *result.x) # calculate ion's axial frequency
plt.vlines(zfreq_cal/ (2 * np.pi * MHz), 0, 0.5, "C0", label='exp')
plt.legend()
plt.show()
# plot ion's axial mode amplitude b(m, i) for each ion, m is motional index, i is ion's index
fig, ax = plt.subplots(figsize=(8, 8))
pcm = plt.matshow(zmode, fignum=0, cmap='bwr', vmax=0.7, vmin=-0.7)
plt.xlabel('mode index')
plt.ylabel('ion index')
fig.colorbar(pcm, fraction=0.04)

# %%
zd = []
for ii in range(N_ions-1):
    zd.append((zpos[ii+1] - zpos[ii])*1e6)
print(zd)
# %%
