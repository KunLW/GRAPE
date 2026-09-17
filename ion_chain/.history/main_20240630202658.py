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
zpos_exp = np.array([-6.00642487760957, -5.3728382614025065, -4.835048297943007, -4.367564905011956, -3.9071510256612187, -3.466362431515591, -3.047062786550269, -2.6291433286856916, -2.2334138302720654, -1.7934413807479808, -1.3836281496843885, -0.9243521141570117, -0.5254190869734938, -0.05678916939677783, 0.4164813633694036, 0.8860813813213869, 1.3813338609147976, 1.9142132751482455, 2.4675135631358325, 3.018292349351868, 3.6009072447569537, 4.1820706544513735, 4.767312077847167, 5.341011940661461, 5.888265579215416, 6.4070611635010914, 6.92403309641396, 7.3918354712387515, 7.848469823612925, 8.300722535753676, 8.72627253909401, 9.118347343757382, 9.494574257230552, 9.895980307443299, 10.253327592283675, 10.605207878619819, 10.956651513409678, 11.317860362388243, 11.608762123344952, 11.975140791309816, 12.295954868227136, 12.615991472113116, 12.93487585712472, 13.26903978269964, 13.580515600373635, 14.013555821518256, 14.396295286692334, 14.677917671320936, 15.15698447803056, 15.766939990406806])/38*640*1e-6
zpos_exp= zpos_exp-np.mean(zpos_exp)
result = minimize(Loss_function_position, 
                  x0=np.array([-8.92371360e-15,-6.24263632e-19, 2.50378286e-11, 3.50184232e-07]),            # initial value
                  args=(N_ions, zpos_exp), tol=1e-30, method='Nelder-Mead',
                  bounds=((-1e-12, 1e-12), (-1e-18, 1e-18), (-1e-10, 1e-10), (-1e-3, 1e-3)))  # bound value

print(result.x)

zpos = equ_pos_fast(N_ions, *result.x)  # calculate ion's position

# %%
# result.x = np.array([-1/2 * mca * (2*np.pi*0.1*1e6)**2,0, 0, 5e-7])
zpos = equ_pos_fast(N_ions, *result.x)
plt.plot(zpos*1e6,np.zeros(N_ions), "o", label="fit")
plt.plot(zpos_exp*1e6,np.ones(N_ions), "o", label="exp")
print(Loss_function_position(result.x, N_ions, zpos_exp))
plt.legend()
plt.show()

# %%
omega_y = 2*np.pi * 2.56 * MHz
yfreq_cal, ymode, _ = radial_mode_spectrum(N_ions, omega_y, zpos)
omega_x = 2*np.pi * 2.48 * MHz
xfreq_cal, xmode, _ = radial_mode_spectrum(N_ions, omega_x, zpos)
plt.vlines(xfreq_cal/ (2 * np.pi * MHz), 0, 0.5, "C0")
plt.vlines(yfreq_cal/ (2 * np.pi * MHz), 0.5, 1, "C1")
plt.show()
# plot ion's axial mode amplitude b(m, i) for each ion, m is motional index, i is ion's index
fig, ax = plt.subplots(figsize=(8, 8))
pcm = plt.matshow(xmode, fignum=0, cmap='bwr', vmax=0.7, vmin=-0.7)
plt.xlabel('mode index')
plt.ylabel('ion index')
fig.colorbar(pcm, fraction=0.04)
plt.show()
# %%
omega_x_list = 2*np.pi * (2.4255 + np.linspace(0.01, -0.01, N_ions)) * MHz
xfreq_cal, xmode = radial_mode_spectrum_nonuni(N_ions, omega_x_list, zpos)
plt.vlines(xfreq_cal/ (2 * np.pi * MHz), 0, 0.5, "C0")
# plt.vlines(yfreq_cal/ (2 * np.pi * MHz), 0.5, 1, "C1")
plt.show()
# plot ion's axial mode amplitude b(m, i) for each ion, m is motional index, i is ion's index
fig, ax = plt.subplots(figsize=(8, 8))
pcm = plt.matshow(xmode, fignum=0, cmap='bwr', vmax=0.7, vmin=-0.7)
plt.xlabel('mode index')
plt.ylabel('ion index')
fig.colorbar(pcm, fraction=0.04)
plt.show()
# %%
ionlist = [6, 7, 8]
modelist = []
for ii in range(N_ions):
   flag = 0
   for jj in ionlist:
      if np.abs(ymode[jj][ii]) > 0.05 and flag < 1:
         modelist.append(ii)
         flag += 1
      else:
         continue

plt.vlines([xfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist[:10]], 0, 1, "C0")
# plt.vlines([yfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist[:5]], 0, 1, "C1")
plt.xlim(2.2, 2.4)
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
