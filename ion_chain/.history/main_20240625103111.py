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
zpos_exp = np.array([-5.841502342641313, -5.2142200080659205, -4.699337675456445, -4.2036329323877135, -3.769844573510081, -3.3266785408253146, -2.890383541454634, -2.4394922292309023, -1.9885059109002117, -1.58282520934692, -1.1830476030512695, -0.7334968478178802, -0.27841133993522754, 0.22473967822673668, 0.7215488143841412, 1.231543552373114, 1.729706539275564, 2.3039351278042774, 2.902350133194928, 3.49665080039734, 4.058109908719091, 4.65790814977667, 5.22742073542949, 5.815715494573574, 6.3555868410969945, 6.874720959714191, 7.344595476164488, 7.808082903931982, 8.253776810036117, 8.67012553915017, 9.040582335772386, 9.43052567319049, 9.813826167354007, 10.176538162205167, 10.504353388347592, 10.856648244283669, 11.188502277705222, 11.513675509418455, 11.828795759034827, 12.169423846383225, 12.476455927602387, 12.805617102882364, 13.120960288122305, 13.450322788664192, 13.804402608381102, 14.02170900541195, 14.470197617037718, 14.853558074710286, 15.320775330372356, 15.878883357621037]
)/38*640*1e-6
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
omega_y = 2*np.pi * 2.56 * MHz
yfreq_cal, ymode, _ = radial_mode_spectrum(N_ions, omega_y, zpos)
omega_x = 2*np.pi * 2.4255 * MHz
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
ionlist = [20, 21, 22]
modelist = []
for ii in range(N_ions):
   flag = 0
   for jj in ionlist:
      if np.abs(ymode[jj][ii]) > 0.05 and flag < 1:
         modelist.append(ii)
         flag += 1
      else:
         continue

plt.vlines([xfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist], 0, 1, "C0")
plt.vlines([yfreq_cal[int(ii)] / (2 * np.pi * MHz) for ii in modelist], 0, 1, "C1")
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
