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
N_ions = 53
zpos_exp = np.array([-10.233176019449871, -9.697433541203893, -9.228151386484448, -8.80556306073385, -8.422359619381796, -8.0120033940306, -7.6672283853871726, -7.294738942365462, -6.936818500865506, -6.580602213679286, -6.207979415382237, -5.8422218435312505, -5.45484832923636, -5.068975883391454, -4.709871705397407, -4.2904142127885425, -3.848394914296123, -3.444295522813383, -2.9784875796355306, -2.5072089351435562, -2.0484327450464463, -1.5209246319582437, -1.0141030826089594, -0.4341446386099136, 0.020577893821913783, 0.5914742226870842, 1.144052733996694, 1.6271853804487044, 2.1361759513504337, 2.6178418177492357, 3.04436728477264, 3.467192165115765, 3.8515983678793284, 4.281984339246114, 4.606018976160647, 5.026921306781267, 5.324548712601541, 5.641374833777336, 5.997958561586822, 6.3234434519331515, 6.606667238657821, 6.9242269035092745, 7.212752274465825, 7.539055136907321, 7.8512355405015075, 8.160768926070768, 8.451632948916368, 8.759450030433927, 9.129438243991462, 9.454568799281589, 9.811020421921485, 10.211052037707507, 10.72645881174857])/38*653*1e-6

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
omega_ra = 2*np.pi * 2.5 * MHz
yfreq_cal, ymode, _ = radial_mode_spectrum(N_ions, omega_ra, zpos)

plt.vlines(yfreq_cal/ (2 * np.pi * MHz), 0, 0.5, "C0", label='exp')
plt.legend()
plt.show()
# plot ion's axial mode amplitude b(m, i) for each ion, m is motional index, i is ion's index
fig, ax = plt.subplots(figsize=(8, 8))
pcm = plt.matshow(ymode, fignum=0, cmap='bwr', vmax=0.7, vmin=-0.7)
plt.xlabel('mode index')
plt.ylabel('ion index')
fig.colorbar(pcm, fraction=0.04)
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
