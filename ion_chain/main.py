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
# use ion position to fit the electrical potential
N_ions = 58
zpos_exp = np.array([-6.26566795747128, -5.59964251722938, 
                     -5.10147810288158, -4.6277053969431705, 
                     -4.185189732649032, -3.744033689546192, 
                     -3.3346501178786743, -2.9203828151329665, 
                     -2.492515029013504, -2.083273354578818, 
                     -1.6663694612070294, -1.233255673032396, 
                     -0.8029583902207562, -0.3525933480852453, 
                     0.09898978961325867, 0.5740326866630566, 
                     1.0898932142133968, 1.6045475748084679, 
                     .1309710340988945, 2.683541972785637, 
                     3.2403313412909784, 3.828815195107811, 
                     4.402553981635625, 4.952196583755355, 
                     5.494925567942459, 6.016970938790197, 
                     6.5044017304015345, 6.980987379317148, 
                     7.427426060016152, 7.854072219268418, 
                     8.243930408247182, 8.628663126710592, 
                     8.992453388018394, 9.347716362499249, 
                     9.691605543874855, 10.018581729639465, 
                     10.353964909576721, 10.66314621250206, 
                     10.962749553063073, 11.26548247504936, 
                     11.550151422065454, 11.830079931683219, 
                     12.120406233136853, 12.410817847802484, 
                     12.668406250099718, 12.974022178526688, 
                     13.252140348823405, 13.491299130895507, 
                     13.80971170371858, 14.082166457469997, 
                     14.350029705149309, 14.654832186717083, 
                     14.978997267672954, 15.29117706777991, 
                     15.62622779154524, 15.921664410449951, 
                     16.437421115734054, 16.890433717688552])/38*640*1e-6
zpos_exp= zpos_exp-np.mean(zpos_exp) # normalize the position
## why not using curve_fit?
result = minimize(Loss_function_position, 
                  x0=np.array([-7.66734435e-15, -1.00000000e-18,  3.52488470e-11,  2.93564624e-07]),            # initial value
                  args=(N_ions, zpos_exp), tol=1e-30, method='Nelder-Mead',
                  bounds=((-1e-12, 1e-12), (-1e-17, 1e-17), (-1e-10, 1e-10), (-1e-3, 1e-3)))  # bound value

print(result.x)

zpos = equ_pos_fast(N_ions, *result.x)  # calculate ion's position
# result.x = np.array([-1/2 * mca * (2*np.pi*0.1*1e6)**2,0, 0, 5e-7])
zpos = equ_pos_fast(N_ions, *result.x)
plt.plot(zpos*1e6,np.zeros(N_ions), "o", label="fit")
plt.plot(zpos_exp*1e6,np.ones(N_ions), "o", label="exp")
print(Loss_function_position(result.x, N_ions, zpos_exp))
plt.legend()
plt.show()

# %%
# calculate x and y frequency and mode pattern
omega_y = 2*np.pi * 2.56 * MHz
yfreq_cal, ymode, _ = radial_mode_spectrum(N_ions, omega_y, zpos)
omega_x = 2*np.pi * 2.425 * MHz
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

ionlist = [6, 7, 8, 9, 10]
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
# plot axial mode frequency and pattern
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
# calculate axial position
zd = []
for ii in range(N_ions-1):
    zd.append((zpos[ii+1] - zpos[ii])*1e6)
print(zd)

# %%
# calculate ion coupling
laser_freq = (2 * np.pi * MHz) * 2.2

J_matrix = np.zeros((N_ions, N_ions))
Omega = 2 * np.pi * MHz * 0.1
tau = 200 * us
for ii in range(N_ions):
    for jj in range(N_ions):
        J_matrix[ii][jj] = Jij([ii, jj], laser_freq - xfreq_cal, xmode)


fig, ax = plt.subplots(figsize=(8, 8))
pcm=plt.matshow(J_matrix * Omega**2/2*0.05**2/(2*np.pi*MHz)*1e3, fignum=0, cmap='bwr')
fig.colorbar(pcm, fraction=0.04)
plt.show()
# %%
