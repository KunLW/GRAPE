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
N_ions = 51
zpos_exp = np.array([-6.023694757034449, -5.419356873804367, -4.906460570329802, -4.4345729829600575, -4.017641268343308, -3.606129248793309, -3.1905042299738264, -2.7976102557911195, -2.4020245413662344, -2.00722024864986, -1.6071198849499144, -1.2126706688547184, -0.797925701360351, -0.3865938482608832, 0.04088350128624204, 0.476483230412356, 0.9315543000496469, 1.3937824723000851, 1.8751595380939223, 2.3725035957032743, 2.888090636113676, 3.3969715436308308, 3.9485551990119556, 4.4868176460870135, 5.023875098715997, 5.547579506787462, 6.068063635867334, 6.565554336979242, 7.046892747900236, 7.511288059335347, 7.952830598542211, 8.375479150879153, 8.77275095064681, 9.164669351629879, 9.5541889583355, 9.92531226688688, 10.28750567940145, 10.640391441322329, 10.969499422319618, 11.33333267742252, 11.658867394288736, 12.016883958276566, 12.353878751750626, 12.690246354022829, 13.028886128022455, 13.411623512063384, 13.750271868409014, 14.154798843285338, 14.546819868317014, 14.966181848468713, 15.781939111884876]
)/38*653*1e-6
zpos_exp= zpos_exp-np.mean(zpos_exp)
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
omega_y = 2*np.pi * 2.5088 * MHz
yfreq_cal, ymode, _ = radial_mode_spectrum(N_ions, omega_y, zpos)
omega_x = 2*np.pi * 2.422 * MHz
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
         flag += 1
      else:
         continue

plt.vlines([xfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist[:8]], 0, 0.5, "C0")
plt.vlines([yfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist[:4]], 0.5, 1, "C1")
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
