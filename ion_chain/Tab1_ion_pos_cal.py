# -*- encoding: utf-8 -*-
'''
@File    :    ion_pos_cal.py
@Time    :    2023/10/30 11:57:59
@Author  :    ly
@Desc    :    None
'''
# %%
import sympy as sy
from Tab0_constants import *
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root, newton, minimize
# %%
def equ_pos_fast(N_ions, axial_sq, axial_1=0, axial_tr=0, axial_qu=0):
    '''
    calculate ion's equilibrium position by numberically solving the force for each ion, f_i = 0

    Parameters
    ----------
    N_ions:   int
              number of ions

    axial_sq: axial 2 coefficient
              1/2 m \omega^2, \omega need to times 2\pi MHz
            
    axial_1:  axial 1 coefficient

    axial_tr: axial 3 coefficient

    axial_qu: axial 4 coefficient

    the axial trapping potential (unit J) is axial_1 x + axial_sq x^2 + axial_tr x^3 + axial_qu x^4

    Return
    ---------
    result.x: list
              equilibrium position
    '''
    def force_func(x):
        function_list = []
        for ii in range(N_ions):
            f = 0
            for jj in range(0, ii):
                f = f - k_cou / (x[ii] - x[jj])**2
            for jj in range(ii+1, N_ions):
                f = f + k_cou / (x[ii] - x[jj])**2
            f = f + axial_1 + 2 * axial_sq * x[ii] + 3 * axial_tr * x[ii] ** 2 + 4 * axial_qu * x[ii] ** 3
            function_list.append(f)
        return function_list
    initialguess = np.linspace(-N_ions/2, N_ions/2, N_ions) * 5e-6
    result = root(force_func, initialguess)
    # print(result.x)
    return result.x


def radial_mode_spectrum(N_ions, omega_ra, pos):
    """
    calculate the axial mode frequency and coupling matrix by diagnolize the Hessian matrix
    reference: Appl. Phys. B 66, 181-190 (1998)

    Parameters
    -------------
    N_ions:   int
              number of ions

    omega_ra: float
              radial frequency, need to times 2\pi MHz

    pos:      list
              ion's equilibrium position 
    
    Return
    -------
    X_freqcal: ndarray(n)
               radial frequency,  2\pi MHz

    X_mode   : ndarray(n*n)
               mode matrix b(m, i), m is mode index, i is ion index

    flag     : Bool
               if Ture, means the ion's position is stable 

    """
    hessian_radial = [[0 for j in range(0, N_ions)] for i in range(0, N_ions)]
    def X_matrix_11(i):
        s = 0
        for ii in range(0, N_ions):
            if ii != i:
                s = s + 1 / abs(pos[i] - pos[ii]) ** 3
        return -s * k_cou + mca * omega_ra ** 2
    def X_matrix_12(i, j):
        if i == j:
            return 0
        return k_cou * 1 / abs(pos[i] - pos[j]) ** 3

    for i in range(0, N_ions):
        hessian_radial[i][i] = X_matrix_11(i) / mca / omega_ra**2
    for i in range(0, N_ions):
        for j in range(0, N_ions):
            if i != j:
                hessian_radial[i][j] = X_matrix_12(i, j) / mca / omega_ra**2

    X_freq, X_modes = linalg.eigh(  np.array(hessian_radial,dtype=float)  )
    X_freqcal = (X_freq ** 0.5) * omega_ra 
    X_modes = np.array(X_modes)     
    if min(X_freqcal) <0:
        flag = False
    else:
        flag = True
    return X_freqcal, X_modes, flag

def axial_mode_spectrum(N_ions, pos, axial_sq, axial_1=0, axial_tr=0, axial_qu=0):
    """
    calculate the axial mode frequency and coupling matrix by diagnolize the Hessian matrix
    reference: Appl. Phys. B 66, 181-190 (1998)

    Parameters
    -----------
    N_ions:   int
              number of ions

    pos:      list
              ion's equilibrium position
    
    axial_sq: axial 2 coefficient
              1/2 m \omega^2, \omega unit: 2\pi MHz
            
    axial_1:  axial 1 coefficient

    axial_tr: axial 3 coefficient

    axial_qu: axial 4 coefficient

    the axial trapping potential (unit J) is axial_1 x + axial_sq x^2 + axial_tr x^3 + axial_qu x^4
    
    Return
    -------
    Z_freqcal: ndarray(n)
               axial frequency,  2\pi MHz

    Z_mode   : ndarray(n*n)
               mode matrix b(m, i), m is mode index, i is ion index

    flag     : Bool
               if Ture, means the ion's position is stable               
    """
    hessian_axial = [[0 for j in range(0, N_ions)] for i in range(0, N_ions)]
    def Z_matrix_11(i):
        s = 0
        for ii in range(0, N_ions):
            if ii != i:
                s = s + 2 / abs(pos[i] - pos[ii]) ** 3
        return s * k_cou + 2 * axial_sq + 6 * axial_tr * pos[i] + 12 * axial_qu * pos[i] ** 2
    def Z_matrix_12(i, j):
        if i == j:
            return 0
        return -2 * k_cou / abs(pos[i] - pos[j]) ** 3

    for i in range(0, N_ions):
        hessian_axial[i][i] = Z_matrix_11(i) / mca 
    for i in range(0, N_ions):
        for j in range(0, N_ions):
            if i != j:
                hessian_axial[i][j] = Z_matrix_12(i, j) / mca 

    Z_freq, Z_modes = linalg.eigh(  np.array(hessian_axial,dtype=float)  )
    Z_freqcal = (Z_freq ** 0.5)
    Z_modes = np.array(Z_modes)  

    if min(Z_freqcal) <0:
        flag = False
    else:
        flag = True
    return Z_freqcal, Z_modes, flag

# Loss function of fitting
def Loss_function_axial(axlist, N_ions, freqlist):
    '''
    Fitting the axial trapping potential according to all the frequency measured by experiment
   
    Parameters
    -----------
    axlist:   list
              give the axial potential coefficient

    N_ions:   int
              number of ions
    
    freqlist: list
              axial frequency measured in exp, sorted from low to high
    
    Return
    ----------
    value:    distance betweeen exp and sim frequency

    '''
    value = 0
    z = equ_pos_fast(N_ions, axlist[0], axlist[1], axlist[2], axlist[3])
    freq, _, flag = axial_mode_spectrum(N_ions, z, axlist[0], axlist[1], axlist[2], axlist[3])
    # print(freq)
    if flag == False:
        value = 999999999
    else:
        value = np.sum([((freq[ii] - freqlist[ii])) ** 2/MHz for ii in range(N_ions)])
    return value

# Loss function of fitting 
def Loss_function_position(axlist, N_ions, poslist):
    '''
    Fitting the axial trapping potential according to all the position measured by experiment
   
    Parameters
    -----------
    axlist:   list
              give the axial potential coefficient

    N_ions:   int
              number of ions
    
    freqlist: list
              axial frequency measured in exp, sorted from low to high
    
    Return
    ----------
    value:    distance betweeen exp and sim frequency

    '''
    value = 0
    z = equ_pos_fast(N_ions, axlist[0], axlist[1], axlist[2], axlist[3])
    freq, _, flag = axial_mode_spectrum(N_ions, z, axlist[0], axlist[1], axlist[2], axlist[3])
    # print(freq)
    if flag == False:
        value = 999999999
    else:
        value = np.sum([((z[ii] - poslist[ii])/1e-6) ** 2 for ii in range(N_ions)])
    return value

def radial_mode_spectrum_nonuni(N_ions, omega_ralist, partialpos):     #  计算X方向振动模式

    hessian_radial = [[0 for j in range(0, N_ions)] for i in range(0, N_ions)]

    def X_matrix_11(i):
        s = 0
        for ii in range(0, N_ions):
            if ii != i:
                s = s + 1 / abs(partialpos[i] - partialpos[ii]) ** 3
        return -s * k_cou + mca * omega_ralist[i] ** 2
    def X_matrix_12(i, j):
        if i == j:
            return 0
        return k_cou * 1 / abs(partialpos[i] - partialpos[j]) ** 3

    for i in range(0, N_ions):
        hessian_radial[i][i] = X_matrix_11(i) / mca 
    for i in range(0, N_ions):
        for j in range(0, N_ions):
            if i != j:
                hessian_radial[i][j] = X_matrix_12(i, j) / mca 

    X_freq, X_modes = linalg.eigh(  np.array(hessian_radial,dtype=float)  )
    X_freqcal = (X_freq ** 0.5) 
    X_modes = np.array(X_modes)     #   这里的X_modes[i,j] 即为b-matrix，其中i表示离子的编号，j表示模式的编号
    # print('radial b_j^m matrix is:\n', X_modes)
    return X_freqcal, X_modes

def Jij(ion_index, detuning, b_matrix):
    ii = ion_index[0]
    jj = ion_index[1]
    J = 0
    if ii == jj:
        return J
    try:
        for mm in range(len(detuning)):
            J += b_matrix[ii][mm] * b_matrix[jj][mm]/detuning[mm]
    except:
        J += b_matrix[ii] * b_matrix[jj]/detuning
    return J