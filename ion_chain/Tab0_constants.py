import numpy as np
from scipy import constants

MHz = 1e6
us = 1e-6
k_cou = constants.e ** 2 / (4 * np.pi * constants.epsilon_0)
mca = 40 * constants.m_p

la_729=0.729*1e-6
k = 2*np.pi/la_729
